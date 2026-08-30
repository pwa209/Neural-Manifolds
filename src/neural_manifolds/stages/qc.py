"""Label-blind recording, event, channel, and sampled-signal quality control."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from neural_manifolds.config import StudyConfig, config_sha256
from neural_manifolds.preprocessing.eeg import (
    canonicalize_channel_name,
    detect_artifact_windows,
    detect_bad_channels,
    make_windows,
)
from neural_manifolds.provenance import atomic_write_json, sha256_file
from neural_manifolds.recording_provenance import recording_inventory
from neural_manifolds.stage_processing import infer_mains_frequency, read_raw_recording


def _atomic_parquet(frame: pd.DataFrame, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, destination)
    return destination


def _optional_path(value: Any) -> Path | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = str(value).strip()
    return Path(text) if text else None


def _event_summary(path: Path | None, *, duration_seconds: float) -> dict[str, Any]:
    if path is None:
        return {
            "event_sidecar_status": "missing",
            "event_rows": 0,
            "event_invalid_onsets": 0,
            "event_out_of_bounds_onsets": 0,
            "event_negative_durations": 0,
            "event_duplicate_onsets": 0,
            "event_value_columns_consumed_json": "[]",
        }
    source = path.resolve(strict=True)
    header = pd.read_csv(source, sep="\t", dtype=str, keep_default_na=False, nrows=0)
    if "onset" not in header.columns:
        return {
            "event_sidecar_status": "present_without_onset_column",
            "event_rows": 0,
            "event_invalid_onsets": 0,
            "event_out_of_bounds_onsets": 0,
            "event_negative_durations": 0,
            "event_duplicate_onsets": 0,
            "events_sha256": sha256_file(source),
            "event_value_columns_consumed_json": "[]",
        }
    value_columns = ["onset"] + (["duration"] if "duration" in header.columns else [])
    # ``usecols`` is the label firewall: trial type, response, condition, and
    # outcome values are never materialised in the QC process.
    frame = pd.read_csv(
        source,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        usecols=value_columns,
    )
    onset = pd.to_numeric(frame["onset"], errors="coerce")
    valid = onset.notna()
    duration = (
        pd.to_numeric(frame["duration"], errors="coerce")
        if "duration" in frame
        else pd.Series(np.zeros(len(frame)), index=frame.index)
    )
    finite_onset = onset[valid]
    return {
        "event_sidecar_status": "present",
        "event_rows": len(frame),
        "event_invalid_onsets": int((~valid).sum()),
        "event_out_of_bounds_onsets": int(
            ((finite_onset < 0) | (finite_onset > duration_seconds)).sum()
        ),
        "event_negative_durations": int((duration.dropna() < 0).sum()),
        "event_duplicate_onsets": int(finite_onset.duplicated(keep=False).sum()),
        "events_sha256": sha256_file(source),
        "event_value_columns_consumed_json": json.dumps(value_columns, separators=(",", ":")),
    }


def _channel_sidecar_summary(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {
            "channel_sidecar_status": "missing",
            "channel_sidecar_rows": 0,
            "channel_sidecar_bad_rows": 0,
        }
    source = path.resolve(strict=True)
    frame = pd.read_csv(source, sep="\t", dtype=str, keep_default_na=False)
    bad = (
        frame["status"].astype(str).str.casefold().eq("bad")
        if "status" in frame
        else pd.Series(False, index=frame.index)
    )
    return {
        "channel_sidecar_status": "present",
        "channel_sidecar_rows": len(frame),
        "channel_sidecar_bad_rows": int(bad.sum()),
        "channels_sha256": sha256_file(source),
    }


def _sample_chunks(
    raw: Any,
    *,
    segments: int,
    seconds_per_segment: float,
) -> tuple[list[np.ndarray], list[int]]:
    sfreq = float(raw.info["sfreq"])
    samples = min(raw.n_times, max(1, round(seconds_per_segment * sfreq)))
    maximum_start = max(0, raw.n_times - samples)
    starts = np.unique(np.linspace(0, maximum_start, num=segments, dtype=np.int64)).tolist()
    chunks = [
        np.asarray(raw.get_data(start=int(start), stop=int(start) + samples), dtype=float)
        for start in starts
    ]
    return chunks, [int(value) for value in starts]


def _montage_mask(raw: Any) -> np.ndarray:
    channels = raw.info.get("chs", ())
    if len(channels) != len(raw.ch_names):
        return np.zeros(len(raw.ch_names), dtype=bool)
    values = []
    for channel in channels:
        location = np.asarray(channel.get("loc", np.zeros(12)), dtype=float)[:3]
        values.append(bool(np.all(np.isfinite(location)) and np.linalg.norm(location) > 1e-9))
    return np.asarray(values, dtype=bool)


def _inspect_recording(
    row: dict[str, Any],
    *,
    study: StudyConfig,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    source = Path(str(row["source_path"])).resolve(strict=True)
    raw = read_raw_recording(source)
    try:
        source_inventory = recording_inventory(source, raw=raw)
        source_types = list(raw.get_channel_types())
        type_counts = pd.Series(source_types, dtype=str).value_counts().sort_index().to_dict()
        eeg = raw.copy().pick("eeg")
        sfreq = float(eeg.info["sfreq"])
        duration_seconds = float(eeg.n_times / sfreq)
        if len(eeg.ch_names) < study.signal_qc.minimum_eeg_channels:
            raise ValueError(
                f"only {len(eeg.ch_names)} EEG channels; "
                f"need {study.signal_qc.minimum_eeg_channels}"
            )
        if duration_seconds < study.signal_qc.minimum_recording_seconds:
            raise ValueError(
                f"only {duration_seconds:.3f}s; need {study.signal_qc.minimum_recording_seconds}s"
            )
        chunks, sampled_starts = _sample_chunks(
            eeg,
            segments=study.signal_qc.sample_segments,
            seconds_per_segment=study.signal_qc.seconds_per_segment,
        )
        sampled = np.concatenate(chunks, axis=1)
        if not np.all(np.isfinite(sampled)):
            raise ValueError("sampled EEG contains non-finite values")
        quality = detect_bad_channels(
            sampled,
            sfreq,
            flat_tolerance_volts=study.signal_qc.flat_tolerance_volts,
            flat_fraction_limit=study.signal_qc.flat_fraction_limit,
            robust_threshold=study.signal_qc.bad_channel_robust_threshold,
        )
        window_blocks = []
        for chunk in chunks:
            windows, _ = make_windows(
                chunk,
                sfreq,
                study.signal_qc.diagnostic_window_seconds,
                study.signal_qc.diagnostic_window_seconds,
            )
            if len(windows):
                window_blocks.append(windows)
        windows = (
            np.concatenate(window_blocks, axis=0)
            if window_blocks
            else np.empty((0, len(eeg.ch_names), 0), dtype=float)
        )
        artifact = detect_artifact_windows(
            windows,
            sfreq,
            robust_threshold=study.signal_qc.artifact_window_robust_threshold,
        )
        bad_mask = np.zeros(len(eeg.ch_names), dtype=bool)
        bad_mask[quality.bad_indices] = True
        montage_mask = _montage_mask(eeg)
        bad_fraction = float(np.mean(bad_mask))
        artifact_fraction = float(np.mean(~artifact.keep)) if len(artifact.keep) else float("nan")
        montage_fraction = float(np.mean(montage_mask))
        review_flags = []
        if bad_fraction > study.signal_qc.review_bad_channel_fraction:
            review_flags.append("sampled_bad_channel_fraction")
        if np.isfinite(artifact_fraction) and (
            artifact_fraction > study.signal_qc.review_artifact_window_fraction
        ):
            review_flags.append("sampled_artifact_window_fraction")
        if montage_fraction < study.signal_qc.review_montage_position_fraction:
            review_flags.append("montage_position_fraction")
        events = _event_summary(
            _optional_path(row.get("events_path")),
            duration_seconds=duration_seconds,
        )
        channels_sidecar = _channel_sidecar_summary(_optional_path(row.get("channels_path")))
        if events["event_invalid_onsets"] or events["event_out_of_bounds_onsets"]:
            review_flags.append("event_timing_integrity")
        if channels_sidecar["channel_sidecar_status"] == "missing":
            review_flags.append("missing_channel_sidecar")
        channel_rows = []
        for index, name in enumerate(eeg.ch_names):
            channel_rows.append(
                {
                    "recording_id": row["recording_id"],
                    "dataset_id": row["dataset_id"],
                    "source_recording_sha256": source_inventory["combined_sha256"],
                    "channel_index": index,
                    "channel_name": str(name),
                    "canonical_channel_name": canonicalize_channel_name(str(name)),
                    "sampled_bad_channel": bool(bad_mask[index]),
                    "flat_fraction": float(quality.flat_fraction[index]),
                    "log_variance_robust_z": float(quality.log_variance_z[index]),
                    "high_frequency_robust_z": float(quality.high_frequency_z[index]),
                    "kurtosis_robust_z": float(quality.kurtosis_z[index]),
                    "correlation_robust_z": float(quality.correlation_z[index]),
                    "montage_position_available": bool(montage_mask[index]),
                }
            )
        line_frequency = eeg.info.get("line_freq")
        if line_frequency not in {50, 60}:
            line_frequency = infer_mains_frequency(eeg)
        record = {
            **row,
            "source_recording_sha256": source_inventory["combined_sha256"],
            "source_recording_file_count": source_inventory["file_count"],
            "source_members_read_only": all(
                (Path(item["path"]).stat().st_mode & 0o222) == 0
                for item in source_inventory["files"]
            ),
            "technically_eligible": True,
            "qc_status": ("eligible_with_blind_review_flags" if review_flags else "eligible"),
            "technical_exclusion_reason": None,
            "review_flags_json": json.dumps(sorted(set(review_flags)), separators=(",", ":")),
            "label_fields_consumed_json": "[]",
            "sampling_hz": sfreq,
            "duration_seconds": duration_seconds,
            "eeg_channel_count": len(eeg.ch_names),
            "source_channel_type_counts_json": json.dumps(
                type_counts, sort_keys=True, separators=(",", ":")
            ),
            "eog_channel_count": int(type_counts.get("eog", 0)),
            "ecg_channel_count": int(type_counts.get("ecg", 0)),
            "emg_channel_count": int(type_counts.get("emg", 0)),
            "montage_position_fraction": montage_fraction,
            "sampled_seconds": float(sum(chunk.shape[1] for chunk in chunks) / sfreq),
            "sampled_segment_starts_json": json.dumps(sampled_starts, separators=(",", ":")),
            "sampled_bad_channel_count": int(bad_mask.sum()),
            "sampled_bad_channel_fraction": bad_fraction,
            "sampled_window_count": len(artifact.keep),
            "sampled_artifact_window_count": int(np.count_nonzero(~artifact.keep)),
            "sampled_artifact_window_fraction": artifact_fraction,
            "line_frequency_hz": float(line_frequency),
            "custom_reference_applied": str(eeg.info.get("custom_ref_applied", "unknown")),
            "projector_count": len(eeg.info.get("projs", ())),
            "annotation_count": len(eeg.annotations),
            **events,
            **channels_sidecar,
        }
        return record, channel_rows
    finally:
        close = getattr(raw, "close", None)
        if callable(close):
            close()


def run_signal_qc(
    *,
    inventory_path: str | Path,
    output_root: str | Path,
    study: StudyConfig,
) -> tuple[Path, Path, Path]:
    """Inspect all inventoried recordings without opening condition/outcome labels."""

    source_inventory_path = Path(inventory_path).resolve(strict=True)
    inventory = pd.read_parquet(source_inventory_path)
    required = {"recording_id", "dataset_id", "source_path", "events_path", "channels_path"}
    missing = required.difference(inventory.columns)
    if missing:
        raise ValueError(f"recording inventory is missing {sorted(missing)}")
    forbidden = {
        "condition",
        "label",
        "outcome",
        "diagnosis",
        "binary_target",
        "continuous_target",
        "experience_code",
    }.intersection(inventory.columns)
    if forbidden:
        raise ValueError(f"label-blind QC inventory contains forbidden fields {sorted(forbidden)}")
    if inventory["recording_id"].duplicated().any():
        raise ValueError("recording inventory contains duplicate recording IDs")
    records: list[dict[str, Any]] = []
    channels: list[dict[str, Any]] = []
    for row in inventory.sort_values("recording_id").to_dict(orient="records"):
        try:
            record, channel_rows = _inspect_recording(row, study=study)
            records.append(record)
            channels.extend(channel_rows)
        except (
            ValueError,
            RuntimeError,
            OSError,
            KeyError,
            UnicodeError,
            pd.errors.ParserError,
        ) as error:
            records.append(
                {
                    **row,
                    "technically_eligible": False,
                    "qc_status": "excluded_technical",
                    "technical_exclusion_reason": f"{type(error).__name__}: {error}",
                    "review_flags_json": "[]",
                    "label_fields_consumed_json": "[]",
                }
            )
    destination = Path(output_root)
    recording_path = _atomic_parquet(pd.DataFrame(records), destination / "recording-flow.parquet")
    channel_columns = [
        "recording_id",
        "dataset_id",
        "source_recording_sha256",
        "channel_index",
        "channel_name",
        "canonical_channel_name",
        "sampled_bad_channel",
        "flat_fraction",
        "log_variance_robust_z",
        "high_frequency_robust_z",
        "kurtosis_robust_z",
        "correlation_robust_z",
        "montage_position_available",
    ]
    channel_path = _atomic_parquet(
        pd.DataFrame(channels, columns=channel_columns),
        destination / "channel-qc.parquet",
    )
    eligible = sum(bool(row["technically_eligible"]) for row in records)
    audit_path = destination / "signal-qc-audit.json"
    atomic_write_json(
        audit_path,
        {
            "schema_version": 1,
            "inventory_path": str(source_inventory_path),
            "inventory_sha256": sha256_file(source_inventory_path),
            "signal_qc_config_sha256": config_sha256(study.signal_qc),
            "recordings_total": len(records),
            "recordings_technically_eligible": eligible,
            "recordings_technically_excluded": len(records) - eligible,
            "recordings_with_blind_review_flags": sum(
                row["qc_status"] == "eligible_with_blind_review_flags" for row in records
            ),
            "technical_exclusions": [
                {
                    "recording_id": row["recording_id"],
                    "reason": row["technical_exclusion_reason"],
                }
                for row in records
                if not row["technically_eligible"]
            ],
            "sampling_strategy": {
                "segments": study.signal_qc.sample_segments,
                "seconds_per_segment": study.signal_qc.seconds_per_segment,
                "placement": "deterministic_evenly_spaced",
                "artifact_windows_cross_segment_boundaries": False,
            },
            "review_flags_are_exclusions": False,
            "label_fields_consumed": [],
            "condition_or_outcome_values_written": False,
            "scientific_gate_applied": False,
        },
    )
    if eligible == 0:
        raise RuntimeError(
            f"all {len(records)} recordings failed technical QC; inspect {audit_path}"
        )
    return recording_path, channel_path, audit_path
