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
NATIVE_AVERAGE_REFERENCE_BRANCH = "native_full_montage_average_reference"
NATIVE_CSD_BRANCH = "native_full_montage_csd"
SLEEP_HIGHPASS_BRANCH = "sleep_highpass_sensitivity"
SENSITIVITY_BRANCHES = (
    NATIVE_AVERAGE_REFERENCE_BRANCH,
    NATIVE_CSD_BRANCH,
    SLEEP_HIGHPASS_BRANCH,
)
SENSITIVITY_PROCESSING_ERRORS = (
    AttributeError,
    KeyError,
    NotImplementedError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
    np.linalg.LinAlgError,
)


@dataclass(frozen=True)
class SensitivityBranchResult:
    """One optional preprocessing branch with an explicit availability state."""

    raw: Any | None
    status: str
    reason: str | None
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        if self.status not in {"available", "unavailable", "not_applicable", "disabled"}:
            raise ValueError(f"invalid sensitivity status: {self.status}")
        if self.status == "available" and self.raw is None:
            raise ValueError("an available sensitivity branch requires a signal object")
        if self.status != "available" and self.raw is not None:
            raise ValueError("an unavailable sensitivity branch cannot contain signal")
        if self.status == "available" and self.reason is not None:
            raise ValueError("an available sensitivity branch cannot have an unavailable reason")
        if self.status != "available" and not self.reason:
            raise ValueError("an unavailable sensitivity branch requires a reason")


def canonicalize_channel_name(name: str) -> str:
    """Convert common vendor/BIDS channel labels to canonical 10-20 spelling."""

    value = name.strip().upper()
    value = re.sub(r"^(EEG|POLY)\s*", "", value)
    value = re.sub(r"[-_:](REF|LE|RE|AVG|A1|A2)$", "", value)
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


def auxiliary_channel_inventory(raw: Any) -> dict[str, Any]:
    """Audit auxiliary channels without claiming that they were used for ICA."""

    names = list(getattr(raw, "ch_names", []))
    getter = getattr(raw, "get_channel_types", None)
    try:
        channel_types = list(getter()) if callable(getter) else []
    except (RuntimeError, ValueError, TypeError):
        channel_types = []
    if len(channel_types) != len(names):
        return {
            "metadata_status": "unavailable_channel_type_metadata",
            "channels": {"eog": [], "ecg": [], "emg": []},
            "ica_support_status": "unavailable_channel_type_metadata",
            "ica_status": "not_performed_unavailable_channel_type_metadata",
            "auxiliary_artifact_control_support_status": ("unavailable_channel_type_metadata"),
            "auxiliary_artifact_control_status": (
                "not_performed_unavailable_channel_type_metadata"
            ),
            "auxiliary_channels_used_for_cleaning": False,
        }
    channels = {
        kind: [
            name for name, observed in zip(names, channel_types, strict=True) if observed == kind
        ]
        for kind in ("eog", "ecg", "emg")
    }
    support = bool(channels["eog"] or channels["ecg"])
    return {
        "metadata_status": "available",
        "channels": channels,
        "ica_support_status": (
            "available_eog_or_ecg_reference" if support else "unavailable_no_eog_or_ecg_reference"
        ),
        "ica_status": (
            "not_performed_policy_report_only_with_auxiliary_support"
            if support
            else "not_performed_no_eog_or_ecg_reference"
        ),
        "auxiliary_artifact_control_support_status": (
            "available_eog_ecg_or_emg_reference"
            if any(channels.values())
            else "unavailable_no_auxiliary_reference"
        ),
        "auxiliary_artifact_control_status": (
            "not_performed_policy_report_only_with_auxiliary_support"
            if any(channels.values())
            else "not_performed_no_auxiliary_reference"
        ),
        "auxiliary_channels_used_for_cleaning": False,
    }


def _effective_lowpass(sfreq: float, requested_hz: float) -> float:
    nyquist = sfreq / 2.0
    # Leave transition-band room instead of requesting an unrealizable cutoff
    # infinitesimally below Nyquist on low-sampling-rate source recordings.
    effective = min(float(requested_hz), 0.95 * nyquist)
    if effective <= 0:
        raise ValueError("sampling frequency does not permit a positive low-pass cutoff")
    return effective


def _filter_and_resample(
    raw: Any,
    *,
    highpass_hz: float,
    lowpass_hz: float,
    notch_hz: float | None,
    target_sampling_hz: float,
) -> dict[str, Any]:
    source_hz = float(raw.info["sfreq"])
    effective_lowpass = _effective_lowpass(source_hz, lowpass_hz)
    if highpass_hz >= effective_lowpass:
        raise ValueError("high-pass cutoff must be below the effective low-pass cutoff")
    raw.filter(highpass_hz, effective_lowpass, method="fir", phase="zero-double")
    applied_notches: list[float] = []
    if notch_hz is not None and 0 < notch_hz < source_hz / 2:
        applied_notches = np.arange(notch_hz, source_hz / 2, notch_hz).astype(float).tolist()
        if applied_notches:
            raw.notch_filter(applied_notches, method="fir", phase="zero-double")
    if not np.isclose(source_hz, target_sampling_hz):
        raw.resample(target_sampling_hz, method="polyphase")
    return {
        "source_sampling_hz": source_hz,
        "target_sampling_hz": float(raw.info["sfreq"]),
        "requested_highpass_hz": float(highpass_hz),
        "applied_highpass_hz": float(highpass_hz),
        "requested_lowpass_hz": float(lowpass_hz),
        "applied_lowpass_hz": float(effective_lowpass),
        "requested_notch_hz": notch_hz,
        "applied_notch_hz": applied_notches,
    }


def _position_inventory(raw: Any) -> dict[str, Any]:
    names = list(raw.ch_names)
    try:
        montage = raw.get_montage()
        positions = {} if montage is None else montage.get_positions().get("ch_pos", {})
    except (AttributeError, RuntimeError, ValueError, TypeError):
        positions = {}
    positioned = []
    for name in names:
        value = positions.get(name)
        if value is not None and np.asarray(value).shape == (3,) and np.all(np.isfinite(value)):
            positioned.append(name)
    return {
        "channel_count": len(names),
        "positioned_channel_count": len(positioned),
        "position_fraction": len(positioned) / len(names) if names else 0.0,
        "missing_position_channels": [name for name in names if name not in set(positioned)],
    }


def _native_average_reference(
    raw: Any,
    *,
    target_sampling_hz: float,
    highpass_hz: float,
    lowpass_hz: float,
    notch_hz: float | None,
    maximum_interpolation_fraction: float,
) -> tuple[Any, dict[str, Any]]:
    native = raw.copy().load_data()
    original_names = list(native.ch_names)
    getter = getattr(native, "get_channel_types", None)
    if not callable(getter):
        raise ValueError("channel type metadata are unavailable")
    channel_types = list(getter())
    if len(channel_types) != len(original_names):
        raise ValueError("channel type metadata do not align with channel names")
    eeg_original = [
        name for name, kind in zip(original_names, channel_types, strict=True) if kind == "eeg"
    ]
    if len(eeg_original) < 2:
        raise ValueError("native sensitivity requires at least two EEG channels")
    rename = {name: canonicalize_channel_name(name) for name in eeg_original}
    if len(set(rename.values())) != len(rename):
        raise ValueError("canonicalized native EEG channel names are not unique")
    native.rename_channels(rename, allow_duplicates=False)
    eeg_names = [rename[name] for name in eeg_original]
    native.pick(eeg_names, ordered=True)
    quality = detect_bad_channels(native.get_data(), float(native.info["sfreq"]))
    bad_names = [native.ch_names[index] for index in quality.bad_indices]
    maximum_bad = int(np.floor(maximum_interpolation_fraction * len(native.ch_names)))
    if len(bad_names) > maximum_bad:
        raise ValueError(
            f"{len(bad_names)} native bad channels exceeds interpolation limit {maximum_bad}"
        )
    filter_metadata = _filter_and_resample(
        native,
        highpass_hz=highpass_hz,
        lowpass_hz=lowpass_hz,
        notch_hz=notch_hz,
        target_sampling_hz=target_sampling_hz,
    )
    if bad_names:
        native.info["bads"] = bad_names
        try:
            native.interpolate_bads(reset_bads=True, mode="accurate")
        except SENSITIVITY_PROCESSING_ERRORS as error:
            raise ValueError(
                "native bad-channel interpolation requires valid electrode positions"
            ) from error
    native.set_eeg_reference("average", projection=False)
    return native, {
        "branch": NATIVE_AVERAGE_REFERENCE_BRANCH,
        "original_channels": original_names,
        "selected_eeg_channels": list(native.ch_names),
        "renamed_eeg_channels": rename,
        "bad_channels": bad_names,
        "reference": "average",
        **filter_metadata,
        "position_inventory": _position_inventory(native),
    }


def preprocess_mne_raw(
    raw: Any,
    *,
    canonical_channels: Sequence[str],
    target_sampling_hz: float = 200.0,
    highpass_hz: float = 0.1,
    lowpass_hz: float = 75.0,
    notch_hz: float | None = None,
    maximum_interpolation_fraction: float = 0.15,
    require_complete_canonical: bool = True,
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
    missing_names = [name for name in canonical_channels if name not in available]
    interpolated_names = [*bad_names, *(missing_names if require_complete_canonical else [])]
    denominator = len(canonical_channels) if require_complete_canonical else len(available)
    maximum_bad = int(np.floor(maximum_interpolation_fraction * denominator))
    if len(interpolated_names) > maximum_bad:
        raise ValueError(
            f"{len(interpolated_names)} bad/missing channels exceeds interpolation limit "
            f"{maximum_bad}"
        )
    filter_metadata = _filter_and_resample(
        clean,
        highpass_hz=highpass_hz,
        lowpass_hz=lowpass_hz,
        notch_hz=notch_hz,
        target_sampling_hz=target_sampling_hz,
    )
    if require_complete_canonical and missing_names:
        mne.add_reference_channels(clean, missing_names, copy=False)
    if interpolated_names:
        montage = mne.channels.make_standard_montage("standard_1020")
        clean.set_montage(montage, match_case=False, on_missing="raise")
        clean.info["bads"] = interpolated_names
        clean.interpolate_bads(reset_bads=True, mode="accurate")
    if require_complete_canonical:
        clean.pick(list(canonical_channels), ordered=True)
        if list(clean.ch_names) != list(canonical_channels):
            raise RuntimeError("harmonised preprocessing did not produce the configured montage")
    clean.set_eeg_reference("average", projection=False)

    provenance = {
        "mne_version": mne.__version__,
        "original_channels": original_names,
        "renamed_channels": rename,
        "selected_channels": list(clean.ch_names),
        "bad_channels": bad_names,
        "missing_canonical_channels": missing_names,
        "interpolated_channels": interpolated_names,
        "maximum_interpolated_channels": maximum_bad,
        "canonical_montage_complete": list(clean.ch_names) == list(canonical_channels),
        "reference": "average",
        "auxiliary_channel_audit": auxiliary_channel_inventory(raw),
        **filter_metadata,
    }
    return clean, provenance


def preprocess_mne_sensitivity_branches(
    raw: Any,
    *,
    canonical_channels: Sequence[str],
    target_sampling_hz: float,
    primary_highpass_hz: float,
    sleep_highpass_hz: float,
    lowpass_hz: float,
    notch_hz: float | None,
    maximum_interpolation_fraction: float,
    require_complete_canonical: bool,
    native_montage_sensitivity: bool,
    csd_sensitivity: bool,
    csd_minimum_channels: int,
    csd_minimum_position_fraction: float,
    is_sleep_recording: bool,
) -> dict[str, SensitivityBranchResult]:
    """Create configured label-free preprocessing sensitivities independently.

    A failed sensitivity never removes an otherwise valid primary derivative. Every
    branch is returned with a fixed availability state and auditable reason.
    """

    try:
        import mne
    except ImportError as exc:  # pragma: no cover - exercised in EEG environment
        raise RuntimeError("install neural-manifolds[eeg] for MNE preprocessing") from exc

    results: dict[str, SensitivityBranchResult] = {}
    native: Any | None = None
    native_metadata: dict[str, Any] = {}
    if native_montage_sensitivity:
        try:
            native, native_metadata = _native_average_reference(
                raw,
                target_sampling_hz=target_sampling_hz,
                highpass_hz=primary_highpass_hz,
                lowpass_hz=lowpass_hz,
                notch_hz=notch_hz,
                maximum_interpolation_fraction=maximum_interpolation_fraction,
            )
        except SENSITIVITY_PROCESSING_ERRORS as error:
            results[NATIVE_AVERAGE_REFERENCE_BRANCH] = SensitivityBranchResult(
                raw=None,
                status="unavailable",
                reason=f"{type(error).__name__}: {error}",
                metadata={"branch": NATIVE_AVERAGE_REFERENCE_BRANCH},
            )
        else:
            results[NATIVE_AVERAGE_REFERENCE_BRANCH] = SensitivityBranchResult(
                raw=native,
                status="available",
                reason=None,
                metadata=native_metadata,
            )
    else:
        results[NATIVE_AVERAGE_REFERENCE_BRANCH] = SensitivityBranchResult(
            raw=None,
            status="disabled",
            reason="native_montage_sensitivity_disabled_by_configuration",
            metadata={"branch": NATIVE_AVERAGE_REFERENCE_BRANCH},
        )

    if not csd_sensitivity:
        results[NATIVE_CSD_BRANCH] = SensitivityBranchResult(
            raw=None,
            status="disabled",
            reason="csd_sensitivity_disabled_by_configuration",
            metadata={"branch": NATIVE_CSD_BRANCH},
        )
    elif native is None:
        results[NATIVE_CSD_BRANCH] = SensitivityBranchResult(
            raw=None,
            status="unavailable",
            reason="native_average_reference_dependency_unavailable",
            metadata={"branch": NATIVE_CSD_BRANCH},
        )
    else:
        position_inventory = _position_inventory(native)
        if len(native.ch_names) < csd_minimum_channels:
            results[NATIVE_CSD_BRANCH] = SensitivityBranchResult(
                raw=None,
                status="unavailable",
                reason=(
                    f"requires_at_least_{csd_minimum_channels}_channels;"
                    f"observed_{len(native.ch_names)}"
                ),
                metadata={
                    "branch": NATIVE_CSD_BRANCH,
                    "position_inventory": position_inventory,
                },
            )
        elif position_inventory["position_fraction"] < csd_minimum_position_fraction:
            results[NATIVE_CSD_BRANCH] = SensitivityBranchResult(
                raw=None,
                status="unavailable",
                reason=(
                    "insufficient_montage_positions;"
                    f"required_fraction_{csd_minimum_position_fraction};"
                    f"observed_fraction_{position_inventory['position_fraction']:.6f}"
                ),
                metadata={
                    "branch": NATIVE_CSD_BRANCH,
                    "position_inventory": position_inventory,
                },
            )
        else:
            try:
                csd = mne.preprocessing.compute_current_source_density(native.copy())
            except SENSITIVITY_PROCESSING_ERRORS as error:
                results[NATIVE_CSD_BRANCH] = SensitivityBranchResult(
                    raw=None,
                    status="unavailable",
                    reason=f"{type(error).__name__}: {error}",
                    metadata={
                        "branch": NATIVE_CSD_BRANCH,
                        "position_inventory": position_inventory,
                    },
                )
            else:
                results[NATIVE_CSD_BRANCH] = SensitivityBranchResult(
                    raw=csd,
                    status="available",
                    reason=None,
                    metadata={
                        "branch": NATIVE_CSD_BRANCH,
                        "transform": "mne.preprocessing.compute_current_source_density",
                        "input_branch": NATIVE_AVERAGE_REFERENCE_BRANCH,
                        "position_inventory": position_inventory,
                    },
                )

    if not is_sleep_recording:
        results[SLEEP_HIGHPASS_BRANCH] = SensitivityBranchResult(
            raw=None,
            status="not_applicable",
            reason="unit_modality_not_configured_as_sleep",
            metadata={
                "branch": SLEEP_HIGHPASS_BRANCH,
                "configured_highpass_hz": float(sleep_highpass_hz),
            },
        )
    else:
        try:
            sleep, sleep_metadata = preprocess_mne_raw(
                raw,
                canonical_channels=canonical_channels,
                target_sampling_hz=target_sampling_hz,
                highpass_hz=sleep_highpass_hz,
                lowpass_hz=lowpass_hz,
                notch_hz=notch_hz,
                maximum_interpolation_fraction=maximum_interpolation_fraction,
                require_complete_canonical=require_complete_canonical,
            )
        except SENSITIVITY_PROCESSING_ERRORS as error:
            results[SLEEP_HIGHPASS_BRANCH] = SensitivityBranchResult(
                raw=None,
                status="unavailable",
                reason=f"{type(error).__name__}: {error}",
                metadata={
                    "branch": SLEEP_HIGHPASS_BRANCH,
                    "configured_highpass_hz": float(sleep_highpass_hz),
                },
            )
        else:
            results[SLEEP_HIGHPASS_BRANCH] = SensitivityBranchResult(
                raw=sleep,
                status="available",
                reason=None,
                metadata={
                    **sleep_metadata,
                    "branch": SLEEP_HIGHPASS_BRANCH,
                    "configured_highpass_hz": float(sleep_highpass_hz),
                    "sleep_identification": "label_free_modality_membership",
                },
            )
    if tuple(results) != SENSITIVITY_BRANCHES:
        raise RuntimeError("preprocessing sensitivities did not return the fixed branch contract")
    return {
        branch: SensitivityBranchResult(
            raw=result.raw,
            status=result.status,
            reason=result.reason,
            metadata={**result.metadata, "mne_version": mne.__version__},
        )
        for branch, result in results.items()
    }


def common_channel_order(
    channel_sets: Iterable[Sequence[str]], canonical_channels: Sequence[str]
) -> list[str]:
    common = set(canonical_channels)
    for names in channel_sets:
        common.intersection_update(canonicalize_channel_name(name) for name in names)
    return [name for name in canonical_channels if name in common]
