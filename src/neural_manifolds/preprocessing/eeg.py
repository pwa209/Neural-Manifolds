"""Reproducible EEG harmonisation without dataset-label leakage.

The array functions are dependency-light and unit-testable. ``preprocess_mne_raw``
is the production path and imports MNE lazily so configuration and metric tests do
not require the full electrophysiology environment.
"""

from __future__ import annotations

import re
import warnings
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

import numpy as np
from scipy import signal, stats

LEGACY_CHANNEL_MAP = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}


def canonicalize_channel_name(name: str) -> str:
    """Convert common vendor/BIDS channel labels to canonical 10-20 spelling."""

    value = name.strip().upper()
    value = re.sub(r"^(EEG|POLY)\s*", "", value)
    value = re.sub(r"[-_](REF|LE|RE|AVG|A1|A2)$", "", value)
    value = value.replace(" ", "")
    value = LEGACY_CHANNEL_MAP.get(value, value)
    if value.startswith("FP"):
        return "Fp" + value[2:]
    if value.endswith("Z"):
        return value[:-1].capitalize() + "z"
    if value and value[0] in "FCTPOAI" and value[1:].isdigit():
        return value[0] + value[1:]
    return value


def robust_zscore(values: np.ndarray, *, axis: int | None = None) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    median = np.nanmedian(values, axis=axis, keepdims=True)
    mad = np.nanmedian(np.abs(values - median), axis=axis, keepdims=True)
    scale = 1.4826 * mad
    fallback = np.nanstd(values, axis=axis, keepdims=True)
    scale = np.where(scale > np.finfo(float).eps, scale, fallback)
    scale = np.where(scale > np.finfo(float).eps, scale, 1.0)
    return (values - median) / scale


@dataclass(frozen=True)
class BadChannelResult:
    bad_indices: np.ndarray
    flat_fraction: np.ndarray
    log_variance_z: np.ndarray
    high_frequency_z: np.ndarray
    kurtosis_z: np.ndarray
    correlation_z: np.ndarray


def detect_bad_channels(
    data: np.ndarray,
    sfreq: float,
    *,
    flat_tolerance_volts: float = 1e-12,
    flat_fraction_limit: float = 0.05,
    robust_threshold: float = 5.0,
) -> BadChannelResult:
    """Flag channels using label-blind robust signal-quality statistics.

    Parameters
    ----------
    data
        Continuous data shaped ``(channels, samples)`` in volts.
    sfreq
        Sampling frequency in hertz.
    """

    x = np.asarray(data, dtype=float)
    if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] < max(20, int(sfreq)):
        raise ValueError("data must be a channels x samples array of adequate length")
    if not np.all(np.isfinite(x)):
        raise ValueError("data contains non-finite samples")

    differences = np.diff(x, axis=1)
    flat_fraction = np.mean(np.abs(differences) <= flat_tolerance_volts, axis=1)
    variance = np.var(x, axis=1, ddof=1)
    log_variance_z = robust_zscore(np.log(np.maximum(variance, np.finfo(float).tiny)))

    frequencies, psd = signal.welch(
        x,
        fs=sfreq,
        nperseg=min(x.shape[1], max(256, int(2 * sfreq))),
        axis=1,
    )
    high = frequencies >= min(40.0, 0.35 * sfreq)
    reference = (frequencies >= 1.0) & (frequencies <= min(75.0, 0.45 * sfreq))
    high_ratio = np.sum(psd[:, high], axis=1) / np.maximum(
        np.sum(psd[:, reference], axis=1), np.finfo(float).tiny
    )
    high_frequency_z = robust_zscore(np.log(np.maximum(high_ratio, np.finfo(float).tiny)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        kurtosis = stats.kurtosis(x, axis=1, fisher=True, bias=False)
        correlation = np.corrcoef(x)
    kurtosis_z = robust_zscore(np.nan_to_num(kurtosis, nan=0.0))
    np.fill_diagonal(correlation, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        median_correlation = np.nanmedian(np.abs(correlation), axis=1)
    median_correlation = np.nan_to_num(median_correlation, nan=0.0)
    correlation_z = robust_zscore(median_correlation)

    bad = (
        (flat_fraction > flat_fraction_limit)
        | (np.abs(log_variance_z) > robust_threshold)
        | (high_frequency_z > robust_threshold)
        | (np.abs(kurtosis_z) > robust_threshold)
        | (correlation_z < -robust_threshold)
    )
    return BadChannelResult(
        bad_indices=np.flatnonzero(bad),
        flat_fraction=flat_fraction,
        log_variance_z=log_variance_z,
        high_frequency_z=high_frequency_z,
        kurtosis_z=kurtosis_z,
        correlation_z=correlation_z,
    )


def make_windows(
    data: np.ndarray,
    sfreq: float,
    window_seconds: float,
    step_seconds: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Create deterministic channel-by-time windows and sample start indices."""

    x = np.asarray(data)
    if x.ndim != 2:
        raise ValueError("data must have shape channels x samples")
    window = round(window_seconds * sfreq)
    step = round(step_seconds * sfreq)
    if window <= 0 or step <= 0:
        raise ValueError("window and step must contain at least one sample")
    if x.shape[1] < window:
        return np.empty((0, x.shape[0], window), dtype=x.dtype), np.empty(0, dtype=int)
    starts = np.arange(0, x.shape[1] - window + 1, step, dtype=int)
    windows = np.stack([x[:, start : start + window] for start in starts])
    return windows, starts


@dataclass(frozen=True)
class ArtifactWindowResult:
    keep: np.ndarray
    peak_to_peak_z: np.ndarray
    maximum_step_z: np.ndarray
    high_frequency_z: np.ndarray


def detect_artifact_windows(
    windows: np.ndarray,
    sfreq: float,
    *,
    robust_threshold: float = 6.0,
) -> ArtifactWindowResult:
    """Reject windows by recording-relative, label-blind robust thresholds."""

    x = np.asarray(windows, dtype=float)
    if x.ndim != 3:
        raise ValueError("windows must have shape windows x channels x samples")
    if x.shape[0] == 0:
        empty = np.empty(0, dtype=float)
        return ArtifactWindowResult(np.empty(0, dtype=bool), empty, empty, empty)
    if not np.all(np.isfinite(x)):
        raise ValueError("windows contain non-finite samples")

    peak_to_peak = np.max(np.ptp(x, axis=2), axis=1)
    maximum_step = np.max(np.abs(np.diff(x, axis=2)), axis=(1, 2))
    frequencies, psd = signal.welch(
        x,
        fs=sfreq,
        nperseg=min(x.shape[2], max(64, int(sfreq))),
        axis=2,
    )
    high = frequencies >= min(40.0, 0.35 * sfreq)
    full = (frequencies >= 1.0) & (frequencies <= min(75.0, 0.45 * sfreq))
    high_ratio = np.sum(psd[:, :, high], axis=(1, 2)) / np.maximum(
        np.sum(psd[:, :, full], axis=(1, 2)), np.finfo(float).tiny
    )
    p2p_z = robust_zscore(np.log(np.maximum(peak_to_peak, np.finfo(float).tiny)))
    step_z = robust_zscore(np.log(np.maximum(maximum_step, np.finfo(float).tiny)))
    high_z = robust_zscore(np.log(np.maximum(high_ratio, np.finfo(float).tiny)))
    keep = (p2p_z <= robust_threshold) & (step_z <= robust_threshold) & (high_z <= robust_threshold)
    return ArtifactWindowResult(keep, p2p_z, step_z, high_z)


def resample_array(data: np.ndarray, source_hz: float, target_hz: float) -> np.ndarray:
    if source_hz <= 0 or target_hz <= 0:
        raise ValueError("sampling rates must be positive")
    if np.isclose(source_hz, target_hz):
        return np.asarray(data).copy()
    ratio = Fraction(target_hz / source_hz).limit_denominator(10_000)
    return signal.resample_poly(data, ratio.numerator, ratio.denominator, axis=-1)


def preprocess_mne_raw(
    raw: Any,
    *,
    canonical_channels: Sequence[str],
    target_sampling_hz: float = 200.0,
    highpass_hz: float = 0.1,
    lowpass_hz: float = 75.0,
    notch_hz: float | None = None,
    maximum_interpolation_fraction: float = 0.15,
) -> tuple[Any, dict[str, Any]]:
    """Apply the primary harmonisation track to a preloaded MNE Raw object.

    The caller remains responsible for dataset-specific event recovery and for ICA
    decisions based on EOG/ECG evidence. The function copies ``raw`` and never
    modifies source data.
    """

    try:
        import mne
    except ImportError as exc:  # pragma: no cover - exercised in EEG environment
        raise RuntimeError("install neural-manifolds[eeg] for MNE preprocessing") from exc

    clean = raw.copy().load_data()
    original_names = list(clean.ch_names)
    rename = {name: canonicalize_channel_name(name) for name in original_names}
    clean.rename_channels(rename, allow_duplicates=False)
    available = [name for name in canonical_channels if name in clean.ch_names]
    if not available:
        raise ValueError("none of the canonical channels are present")
    clean.pick(available, ordered=True)

    quality = detect_bad_channels(clean.get_data(), float(clean.info["sfreq"]))
    bad_names = [clean.ch_names[index] for index in quality.bad_indices]
    maximum_bad = int(np.floor(maximum_interpolation_fraction * len(clean.ch_names)))
    if len(bad_names) > maximum_bad:
        raise ValueError(f"{len(bad_names)} bad channels exceeds interpolation limit {maximum_bad}")
    clean.info["bads"] = bad_names
    clean.filter(highpass_hz, lowpass_hz, method="fir", phase="zero-double")
    if notch_hz is not None and notch_hz < clean.info["sfreq"] / 2:
        harmonics = np.arange(notch_hz, clean.info["sfreq"] / 2, notch_hz)
        clean.notch_filter(harmonics, method="fir", phase="zero-double")
    if bad_names:
        clean.interpolate_bads(reset_bads=True, mode="accurate")
    clean.set_eeg_reference("average", projection=False)
    if not np.isclose(clean.info["sfreq"], target_sampling_hz):
        clean.resample(target_sampling_hz, method="polyphase")

    provenance = {
        "mne_version": mne.__version__,
        "original_channels": original_names,
        "renamed_channels": rename,
        "selected_channels": list(clean.ch_names),
        "bad_channels": bad_names,
        "source_sampling_hz": float(raw.info["sfreq"]),
        "target_sampling_hz": float(clean.info["sfreq"]),
        "highpass_hz": highpass_hz,
        "lowpass_hz": lowpass_hz,
        "notch_hz": notch_hz,
    }
    return clean, provenance


def common_channel_order(
    channel_sets: Iterable[Sequence[str]], canonical_channels: Sequence[str]
) -> list[str]:
    common = set(canonical_channels)
    for names in channel_sets:
        common.intersection_update(canonicalize_channel_name(name) for name in names)
    return [name for name in canonical_channels if name in common]
