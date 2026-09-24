"""Label-blind window QC and reusable time-series measurements.

No cohorts are concatenated in time. EEG is in volts. Arrays and detailed
exclusions stay on cluster storage; the frozen encoder never receives labels.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from scipy import signal

from neural_manifolds.benchmarks import (
    normalized_lempel_ziv,
    permutation_entropy,
    relative_band_power,
    spectral_exponent,
)
from neural_manifolds.continuous.signal_qc import materialize_recording
from neural_manifolds.preprocessing.eeg import resample_array
from neural_manifolds.provenance import atomic_write_json, sha256_file
from neural_manifolds.revised.cohort import normalized_channel
from neural_manifolds.stage_processing import read_raw_recording


def window_qc(values: np.ndarray, sfreq: float, policy: dict) -> tuple[np.ndarray, list[str]]:
    """One-second decisions from prespecified physical thresholds, no label fitting."""
    x = np.asarray(values, dtype=float)
    if x.ndim != 2 or sfreq != int(sfreq):
        raise ValueError("integer-sampled channels-by-time EEG required")
    count = x.shape[1] // int(sfreq)
    windows = x[:, : count * int(sfreq)].reshape(x.shape[0], count, int(sfreq)).transpose(1, 0, 2)
    reasons = []
    for w in windows:
        reason = []
        if not np.isfinite(w).all():
            reason.append("nonfinite")
        if np.any(np.ptp(w, axis=1) > policy["peak_to_peak_volts"]):
            reason.append("amplitude")
        if np.any(np.std(w, axis=1) < policy["flat_standard_deviation_volts"]):
            reason.append("flat")
        reasons.append(";".join(reason))
    return windows, reasons


def conventional_features(windows: np.ndarray, sfreq: float) -> dict[str, float]:
    """Average per-contiguous-window features, never join separated signal samples."""
    rows = []
    for x in windows:
        bands = relative_band_power(
            x,
            sfreq,
            bands={
                "delta": (0.5, 4),
                "theta": (4, 8),
                "alpha": (8, 13),
                "beta": (13, 30),
            },
        )
        rows.append(
            {
                **bands,
                "spectral_exponent": spectral_exponent(x, sfreq, frequency_range=(2, 30)),
                "lz": normalized_lempel_ziv(x),
                "permutation_entropy": float(np.mean([permutation_entropy(c) for c in x])),
                "log_rms": float(np.log(np.sqrt(np.mean(x * x)) + np.finfo(float).tiny)),
            }
        )
    if not rows:
        raise ValueError("no_clean_windows")
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}


def sensor_trajectory(windows: np.ndarray) -> np.ndarray:
    """Fixed channel-wise log RMS per second, independent of training/evaluation labels."""
    return np.log(np.sqrt(np.mean(windows**2, axis=-1)) + np.finfo(float).tiny)


def time_frequency_trajectory(windows: np.ndarray, sfreq: float) -> np.ndarray:
    """Fixed channel-band log power, with no sample-fitted representation."""
    frequency, power = signal.welch(windows, fs=sfreq, nperseg=int(sfreq), axis=-1)
    bands = []
    for low, high in [(0.5, 4), (4, 8), (8, 13), (13, 30)]:
        selected = (frequency >= low) & (frequency < high)
        bands.append(np.log(power[..., selected].mean(axis=-1) + np.finfo(float).tiny))
    return np.concatenate(bands, axis=1)


def prepare_window(raw, unit: dict, channels: list[str], policy: dict) -> tuple[np.ndarray, dict]:
    sfreq = float(raw.info["sfreq"])
    aliases = unit.get("channel_aliases", {})
    mapped = [normalized_channel(aliases.get(n, n)) for n in raw.ch_names]
    if any(mapped.count(c.upper()) != 1 for c in channels):
        raise ValueError("required_observed_channel_missing_or_ambiguous")
    picks = [mapped.index(c.upper()) for c in channels]
    declared_units = getattr(raw, "_orig_units", {})
    if any(
        str(declared_units.get(raw.ch_names[p], "V")).strip().lower()
        in {"", "n/a", "na", "unknown"}
        for p in picks
    ):
        raise ValueError("source_voltage_units_not_documented_no_amplitude_guess")
    stop = min(round(unit["stop_seconds"] * sfreq), raw.n_times)
    padding = 0
    if unit.get("timing_policy") == "preawakening_with_edf_padding":
        tail = raw.get_data(picks=picks, start=max(0, stop - round(2 * sfreq)), stop=stop)
        # EDF padding is an exactly constant suffix across every selected sensor.
        constant = np.all(np.diff(tail, axis=1) == 0, axis=0)
        for flag in constant[::-1]:
            if not flag:
                break
            padding += 1
        if padding > round(sfreq):
            raise ValueError("terminal_flat_run_exceeds_documented_edf_padding")
        if padding >= max(2, round(0.002 * sfreq)):
            stop -= padding
        else:
            padding = 0
    samples = round(policy["dream_seconds"] * sfreq)
    start = stop - samples
    if start < 0:
        raise ValueError("insufficient_preawakening_samples")
    # Get annotation-based rejections on the original time grid before filtering.
    x = raw.get_data(picks=picks, start=start, stop=stop, reject_by_annotation="NaN")
    annotation_bad = ~np.isfinite(x).all(axis=0)
    reference_mode = policy.get("reference_mode", "observed_common_average")
    if reference_mode == "observed_common_average":
        x = x - np.mean(x, axis=0, keepdims=True)
    elif reference_mode != "native_bipolar_no_rereference":
        raise ValueError("unknown_reference_mode")
    finite = ~annotation_bad
    # Filter independent finite stretches, never through gaps; mark filter edges.
    clean = np.full((len(picks), round(policy["dream_seconds"] * policy["sampling_hz"])), np.nan)
    boundaries = np.flatnonzero(np.diff(np.r_[False, finite, False]))
    for a, b in zip(boundaries[::2], boundaries[1::2], strict=True):
        if b - a < round(2 * sfreq):
            continue
        sos = signal.butter(4, policy["filter_hz"], btype="bandpass", fs=sfreq, output="sos")
        part = signal.sosfiltfilt(sos, x[:, a:b], axis=1)
        part = resample_array(part, sfreq, policy["sampling_hz"])
        dest = round(a * policy["sampling_hz"] / sfreq)
        take = min(part.shape[1], clean.shape[1] - dest)
        clean[:, dest : dest + take] = part[:, :take]
        # Gap boundaries cannot be treated as intact dynamical transitions.
        if a > 0:
            clean[:, dest : dest + policy["sampling_hz"]] = np.nan
        if b < len(finite):
            clean[:, max(dest, dest + take - policy["sampling_hz"]) : dest + take] = np.nan
    return clean, {
        "source_sample_start": start,
        "source_sample_stop_exclusive": stop,
        "source_sfreq": sfreq,
        "padding_samples_removed": padding,
        "channels": channels,
        "native_channels": [raw.ch_names[p] for p in picks],
        "position_equivalence": unit.get("position_equivalence", "native_named_channel"),
        "position_error_cm": unit.get("position_error_cm", {}),
        "channel_interpolation": False,
        "reference": reference_mode,
        "filter": "4th_order_butterworth_zero_phase_offline_not_prospective_timing",
        "filter_hz": policy["filter_hz"],
        "annotation_rejected_samples": int(annotation_bad.sum()),
    }


def measure_cohort(cohort_path: Path, output: Path, policy: dict, encoder=None) -> dict:
    cohort = json.loads(cohort_path.read_text())
    identity = sha256_file(cohort_path)
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for unit in cohort["units"]:
        for track in unit["tracks"]:
            key = f"{unit['unit_id']}-{track}"
            checkpoint = output / "records" / f"{key}.json"
            if checkpoint.exists():
                row = json.loads(checkpoint.read_text())
                if row["cohort_sha256"] != identity:
                    raise ValueError("measurement_checkpoint_input_changed")
                if row.get("array_path") and sha256_file(row["array_path"]) != row["array_sha256"]:
                    raise ValueError("measurement_array_checksum_mismatch")
                records.append(row)
                continue
            row = {
                k: unit[k]
                for k in [
                    "unit_id",
                    "dataset_id",
                    "participant_id",
                    "study_group",
                    "report_code",
                    "sleep_stage",
                    "experience",
                    "primary_eligible",
                    "participant_alias_status",
                ]
            }
            row.update(track=track, cohort_sha256=identity)
            try:
                release = Path(unit["release"]).resolve(strict=True)
                if (
                    sha256_file(release / ".acquisition/COMPLETE.json")
                    != unit["completion_marker_sha256"]
                ):
                    raise ValueError("raw_completion_marker_changed")
                with materialize_recording(
                    release, unit["recording"], Path(os.environ["NM_SCRATCH_ROOT"]) / "revised-qc"
                ) as source:
                    raw = read_raw_recording(source)
                    try:
                        values, audit = prepare_window(
                            raw, unit, policy[f"{track}_channels"], policy
                        )
                    finally:
                        raw.close()
                windows, reasons = window_qc(values, policy["sampling_hz"], policy)
                keep = np.array([not reason for reason in reasons])
                row.update(
                    window_reasons=reasons, clean_seconds=int(keep.sum()), preprocessing=audit
                )
                if keep.sum() < policy["minimum_clean_seconds"]:
                    raise ValueError("insufficient_clean_seconds")
                selected = windows[keep]
                indices = np.flatnonzero(keep)
                segments = np.cumsum(np.r_[True, np.diff(indices) != 1])
                row["conventional"] = conventional_features(selected, policy["sampling_hz"])
                arrays = {
                    "sensor": sensor_trajectory(selected),
                    "time_frequency": time_frequency_trajectory(selected, policy["sampling_hz"]),
                    "segments": segments,
                    "seconds": indices,
                    "windows_volts": selected.astype(np.float32),
                    "channels": np.asarray(audit["channels"]),
                }
                if encoder is not None:
                    encoded = encoder.encode(selected, audit["channels"], metadata_fields=())
                    arrays["encoder"] = encoded.global_states
                    arrays.update(
                        {
                            f"regional__encoder__{name}": value
                            for name, value in encoded.regional_states.items()
                        }
                    )
                    row["encoder_provenance"] = encoded.metadata
                else:
                    row["encoder_status"] = "not_requested_sensor_track_only"
                path = output / "arrays" / f"{key}.npz"
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.with_suffix(".tmp").open("wb") as stream:
                    np.savez_compressed(stream, **arrays)
                path.with_suffix(".tmp").replace(path)
                row.update(status="measured", array_path=str(path), array_sha256=sha256_file(path))
            except (ValueError, OSError, RuntimeError) as exc:
                row.update(status="unavailable", reason=f"{type(exc).__name__}:{exc}")
            atomic_write_json(checkpoint, row)
            records.append(row)
    result = {
        "records": records,
        "cohort_sha256": identity,
        "labels_used_for_measurement": [],
        "scientific_gates": False,
        "no_consciousness_probability": True,
    }
    atomic_write_json(output / "measurement.json", result)
    return result
