"""Independent subsecond sensor track with participant-level matched contrasts."""

from __future__ import annotations

import hashlib
import io
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd
from scipy import signal
from scipy.io import loadmat

from neural_manifolds.adapters.datasets import (
    SomatosensoryReportTaskAdapter,
    TactileDetectionAdapter,
)
from neural_manifolds.continuous.audit import safe_member
from neural_manifolds.provenance import atomic_write_json, sha256_file
from neural_manifolds.stage_processing import read_raw_recording

INTERVALS = {
    "prestimulus": (-0.2, 0.0),
    "early": (0.02, 0.08),
    "intermediate": (0.08, 0.2),
    "late": (0.2, 0.4),
    "response_adjacent": (0.4, 0.6),
}


def causal_filter_audit(sfreq):
    sos = signal.butter(4, 40, fs=sfreq, output="sos")
    impulse = np.zeros(round(sfreq * 2))
    impulse[0] = 1
    response = signal.sosfilt(sos, impulse)
    energy = np.cumsum(response**2)
    energy /= energy[-1]
    return sos, {
        "filter": "causal_fourth_order_40Hz_lowpass",
        "future_sample_support": 0,
        "peak_delay_seconds": float(np.argmax(np.abs(response)) / sfreq),
        "energy_95_seconds": float(np.searchsorted(energy, 0.95) / sfreq),
    }


def fast_features(epoch, times, sfreq, *, response_seconds=None):
    x = np.asarray(epoch, float)
    if x.ndim != 2 or x.shape[1] != len(times) or not np.isfinite(x).all():
        raise ValueError("nonfinite_or_malformed_fast_epoch")
    baseline = (times >= -0.3) & (times < -0.05)
    if baseline.sum() < 10:
        raise ValueError("insufficient_fast_baseline")
    # Dimensionless features avoid guessing the scale of public MAT arrays.
    x = x - x.mean(axis=0, keepdims=True)
    scale = float(np.sqrt(np.mean(x[:, baseline] ** 2)))
    if scale <= np.finfo(float).tiny:
        raise ValueError("flat_fast_baseline")
    x /= scale
    if np.max(np.abs(x)) > 100:
        raise ValueError("extreme_baseline_normalized_amplitude")
    sos, audit = causal_filter_audit(sfreq)
    x = signal.sosfilt(sos, x, axis=1)
    x -= x[:, baseline].mean(axis=1, keepdims=True)
    result = []
    for interval, (start, stop) in INTERVALS.items():
        mask = (times >= start) & (times < stop)
        if mask.sum() < 5:
            continue
        values = x[:, mask]
        energy = np.square(values.mean(axis=1))
        denominator = float(np.sum(energy**2))
        result.append(
            {
                "interval": interval,
                "start_seconds": start,
                "stop_seconds": stop,
                "normalized_rms": float(np.sqrt(np.mean(values**2))),
                "spatial_participation": float(energy.sum() ** 2 / denominator)
                if denominator > 0
                else None,
                "normalized_roughness": float(np.sqrt(np.mean(np.diff(values, axis=1) ** 2))),
                "pre_response": response_seconds is not None and stop <= response_seconds,
                "response_onset_available": response_seconds is not None,
            }
        )
    return result, audit


def matched_contrasts(frame, *, contrasts, repetitions=1000, seed=20260916):
    if frame.empty:
        return []
    results = []
    rng = np.random.default_rng(seed)
    features = ["normalized_rms", "spatial_participation", "normalized_roughness"]
    for label, first, second in contrasts:
        for interval in frame.interval.unique():
            selected = frame[(frame.interval == interval) & frame.condition.isin([first, second])]
            # Match physical intensity within participant before any averaging.
            for feature in features:
                means = (
                    selected.groupby(["participant_id", "intensity", "condition"])[feature]
                    .mean()
                    .unstack("condition")
                )
                if not {first, second}.issubset(means):
                    continue
                delta = (means[first] - means[second]).dropna().groupby("participant_id").mean()
                if len(delta) < 2:
                    results.append(
                        {
                            "contrast": label,
                            "interval": interval,
                            "feature": feature,
                            "status": "unavailable",
                            "participants": len(delta),
                        }
                    )
                    continue
                draws = [
                    rng.choice(delta.to_numpy(), len(delta), replace=True).mean()
                    for _ in range(repetitions)
                ]
                signs = rng.choice([-1, 1], size=(repetitions, len(delta)))
                null = (signs * delta.to_numpy()).mean(axis=1)
                results.append(
                    {
                        "contrast": label,
                        "interval": interval,
                        "feature": feature,
                        "difference": float(delta.mean()),
                        "interval_95": np.quantile(draws, [0.025, 0.975]).tolist(),
                        "p_sign_flip": float(
                            (1 + np.count_nonzero(np.abs(null) >= abs(delta.mean())))
                            / (repetitions + 1)
                        ),
                        "participants": len(delta),
                        "status": "analysed",
                        "sign_flip_assumption": "symmetric_participant_difference_under_null",
                    }
                )
    valid = [r for r in results if "p_sign_flip" in r]
    order = np.argsort([r["p_sign_flip"] for r in valid])
    previous = 0.0
    for rank, index in enumerate(order):
        previous = max(previous, min(1.0, valid[index]["p_sign_flip"] * (len(valid) - rank)))
        valid[index]["p_holm"] = previous
    return results


def read_official_osf_matrix(payload):
    entries = {k: v for k, v in loadmat(io.BytesIO(payload)).items() if not k.startswith("__")}
    matrices = [
        (k, v)
        for k, v in entries.items()
        if isinstance(v, np.ndarray) and v.ndim == 3 and v.shape[:2] == (110, 600)
    ]
    if len(matrices) != 1:
        raise ValueError("official_110x600xtrials_EEG_matrix_not_unique")
    return matrices[0]


def confidence_associations(frame, repetitions=1000, seed=20260916):
    """Within-person, intensity- and detection-stratified confidence slopes."""
    if frame.empty:
        return []
    frame = frame.copy()
    frame["confidence"] = pd.to_numeric(frame.confidence, errors="coerce")
    result = []
    rng = np.random.default_rng(seed)
    for interval, rows in frame.groupby("interval"):
        for feature in ("normalized_rms", "spatial_participation", "normalized_roughness"):
            slopes = []
            for person, block in rows.groupby("participant_id"):
                block = block.dropna(subset=["confidence", feature])
                centered = block[["confidence", feature]] - block.groupby(
                    ["condition", "intensity"]
                )[["confidence", feature]].transform("mean")
                denominator = float(np.sum(centered.confidence**2))
                if len(block) < 5 or denominator <= 0:
                    continue
                slopes.append(
                    {
                        "participant_id": person,
                        "slope": float(
                            np.sum(centered.confidence * centered[feature]) / denominator
                        ),
                    }
                )
            if not slopes:
                continue
            values = np.array([s["slope"] for s in slopes])
            draws = [
                rng.choice(values, len(values), replace=True).mean() for _ in range(repetitions)
            ]
            result.append(
                {
                    "interval": interval,
                    "feature": feature,
                    "participants": len(values),
                    "slope": float(values.mean()),
                    "interval_95": np.quantile(draws, [0.025, 0.975]).tolist()
                    if len(values) > 1
                    else None,
                    "participant_slopes": slopes,
                    "estimand": "confidence_association_within_intensity_and_detection_not_total_effect",
                }
            )
    return result


def run_osf(release: Path, output: Path, policy: dict):
    rows, unavailable = [], []
    archives = list(release.glob("**/DATASET_PREPROCESSED.zip"))
    if len(archives) != 1:
        raise ValueError("official_preprocessed_archive_not_unique")
    seen = set()
    with zipfile.ZipFile(archives[0]) as archive:
        for item in archive.infolist():
            path = PurePosixPath(item.filename)
            if path.name != "EEG_data.mat":
                continue
            if (
                not safe_member(item.filename)
                or item.filename in seen
                or item.file_size > 512 * 1024**2
            ):
                raise ValueError("unsafe_duplicate_or_oversized_MAT_member")
            seen.add(item.filename)
            try:
                condition, _, intensity = SomatosensoryReportTaskAdapter.condition_from_path(
                    item.filename
                )
                parts = path.parts
                block = next(i for i, part in enumerate(parts) if part in {"R", "NR"})
                if block == 0:
                    raise ValueError("missing_subject_folder")
                # Native subject folder is the independent unit; no parsing numerical order.
                participant = parts[block - 1]
                variable, data = read_official_osf_matrix(archive.read(item))
                times = np.arange(600) / 500 - 0.4
                for trial in range(data.shape[2]):
                    key = hashlib.sha256(f"{item.filename}:{trial}".encode()).hexdigest()[:24]
                    try:
                        values, _ = fast_features(data[:, :, trial], times, 500)
                        rows.extend(
                            {
                                "unit_id": key,
                                "participant_id": participant,
                                "condition": condition,
                                "intensity": intensity,
                                "matrix_variable": variable,
                                **v,
                            }
                            for v in values
                        )
                    except ValueError as exc:
                        unavailable.append({"unit_id": key, "reason": str(exc)})
            except ValueError as exc:
                unavailable.append({"member": item.filename, "reason": str(exc)})
    contrasts = [
        ("within_report_task_relevance", "report_task_relevant", "report_task_irrelevant"),
        ("report_minus_no_report_order_confounded", "report_task_irrelevant", "no_report"),
    ]
    result = {
        "rows": rows,
        "contrasts": matched_contrasts(
            pd.DataFrame(rows),
            contrasts=contrasts,
            repetitions=policy["bootstrap_repetitions"],
            seed=policy["seed"],
        ),
        "unavailable": unavailable,
        "filter_audit": causal_filter_audit(500)[1],
        "source": "https://doi.org/10.1038/s41597-025-05970-1",
        "source_archive_sha256": sha256_file(archives[0]),
        "limitations": [
            "NR_precedes_R_order_arousal_and_habituation_confounded",
            "upstream_zero_phase_preprocessing_precludes_prospective_latency_claim",
            "MAT_amplitude_units_not_assumed_features_baseline_normalized",
            "no_report_is_not_verified_identical_experience",
        ],
        "scientific_gates": False,
    }
    atomic_write_json(output / "specificity.json", result)
    return result


def run_tactile(release: Path, output: Path, policy: dict):
    participants = pd.read_csv(release / "participants.tsv", sep="\t")
    event_files = list(release.glob("sub-*/ses-*/eeg/*task-adapt*_events.tsv"))
    rows, unavailable = [], []
    for event_file in event_files:
        relative = event_file.relative_to(release).as_posix()
        events = pd.read_csv(event_file, sep="\t")
        units = TactileDetectionAdapter().adapt(participants, {relative: events})
        if not units:
            continue
        raw = read_raw_recording(release / units[0].source_file)
        try:
            raw.pick("eeg")
            sfreq = float(raw.info["sfreq"])
            response_rows = events[events.trial_type.isin(["hit", "miss", "cr", "fa"])].reset_index(
                drop=True
            )
            if len(response_rows) != len(units):
                raise ValueError("response_marker_alignment_changed")
            for index, unit in enumerate(units):
                try:
                    onset = unit.selector.event_onset_seconds
                    start, stop = round((onset - 0.4) * sfreq), round((onset + 0.8) * sfreq)
                    if start < 0 or stop > raw.n_times:
                        raise ValueError("epoch_outside_recording")
                    epoch = raw.get_data(start=start, stop=stop, reject_by_annotation="NaN")
                    times = np.arange(epoch.shape[1]) / sfreq - 0.4
                    response = float(response_rows.iloc[index].onset) - onset
                    if response < 0:
                        raise ValueError("response_precedes_stimulus")
                    values, _ = fast_features(epoch, times, sfreq, response_seconds=response)
                    rows.extend(
                        {
                            "unit_id": unit.unit_id,
                            "participant_id": unit.participant_id,
                            "condition": unit.condition,
                            "intensity": unit.variables["stimulus_amplitude"],
                            "confidence": unit.variables["confidence"],
                            **v,
                        }
                        for v in values
                    )
                except ValueError as exc:
                    unavailable.append({"unit_id": unit.unit_id, "reason": str(exc)})
        finally:
            raw.close()
    frame = pd.DataFrame(rows)
    contrasts = [
        ("detected_minus_undetected", "tactile_detected", "tactile_undetected"),
        (
            "catch_false_alarm_minus_correct_rejection",
            "catch_false_alarm",
            "catch_correct_rejection",
        ),
    ]
    result = {
        "rows": rows,
        "contrasts": matched_contrasts(
            frame,
            contrasts=contrasts,
            repetitions=policy["bootstrap_repetitions"],
            seed=policy["seed"],
        ),
        "pre_response_contrasts": matched_contrasts(
            frame[frame.pre_response] if not frame.empty else frame,
            contrasts=contrasts,
            repetitions=policy["bootstrap_repetitions"],
            seed=policy["seed"],
        ),
        "confidence_associations": confidence_associations(
            frame, policy["bootstrap_repetitions"], policy["seed"]
        ),
        "unavailable": unavailable,
        "scientific_gates": False,
        "limitations": [
            "undetected_is_not_global_unconsciousness",
            "exact_intensity_matching_can_reduce_sample_size",
            "causal_filter_delay_reported_not_removed",
            "confidence_preserved_not_merged_into_detection_label",
        ],
    }
    atomic_write_json(output / "specificity.json", result)
    return result
