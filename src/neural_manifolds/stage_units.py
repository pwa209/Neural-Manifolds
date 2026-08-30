"""Analysis-unit preprocessing, label-free encoding, and post-encoding label join."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from neural_manifolds.config import StudyConfig
from neural_manifolds.foundation.labram import OfficialLaBraMEncoder
from neural_manifolds.preprocessing.eeg import (
    detect_artifact_windows,
    make_windows,
    preprocess_mne_raw,
)
from neural_manifolds.provenance import atomic_write_json, sha256_file
from neural_manifolds.stage_processing import (
    _model_environment,
    infer_mains_frequency,
    read_raw_recording,
)


def _atomic_parquet(frame: pd.DataFrame, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, destination)
    return destination


def _safe_key(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()[:24]


def _selector(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("selector_json")
    if not isinstance(value, str):
        raise ValueError("encoder row has no serialized selector")
    selector = json.loads(value)
    if not isinstance(selector, dict) or not isinstance(selector.get("kind"), str):
        raise ValueError("serialized selector is malformed")
    return selector


def _crop_clean(raw: Any, selector: dict[str, Any]) -> Any:
    kind = selector["kind"]
    if kind == "full_recording":
        return raw.copy()
    if kind == "interval_seconds":
        start = float(selector["start_seconds"])
        stop = min(float(selector["stop_seconds"]), float(raw.times[-1]))
    elif kind == "event_epoch":
        onset = selector.get("event_onset_seconds")
        if onset is None and selector.get("event_sample") is not None:
            onset = float(selector["event_sample"]) / float(raw.info["sfreq"])
        if onset is None:
            raise ValueError("event selector has no onset")
        start = float(onset) + float(selector["epoch_start_offset_seconds"])
        stop = float(onset) + float(selector["epoch_stop_offset_seconds"])
        start = max(0.0, start)
        stop = min(stop, float(raw.times[-1]))
    else:
        raise ValueError(f"electrophysiology selector {kind!r} is not implemented")
    if stop <= start:
        raise ValueError("selector lies outside the recording")
    # MNE includes tmax; subtract one sample so adjacent intervals never share a sample.
    stop = max(start, stop - 1.0 / float(raw.info["sfreq"]))
    return raw.copy().crop(tmin=start, tmax=stop, include_tmax=True)


def preprocess_analysis_units(
    *,
    encoder_inputs: str | Path,
    output_root: str | Path,
    study: StudyConfig,
) -> tuple[Path, Path]:
    """Preprocess source recordings once, then apply immutable unit selectors."""

    try:
        import mne
    except ImportError as exc:  # pragma: no cover - EEG runtime extra
        raise RuntimeError("install neural-manifolds[eeg]") from exc
    frame = pd.read_parquet(encoder_inputs)
    required = {"unit_id", "source_path", "modality", "selector_json"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"encoder inputs are missing {sorted(missing)}")
    destination = Path(output_root)
    destination.mkdir(parents=True, exist_ok=True)
    source_cache: dict[str, tuple[Path, dict[str, Any], str]] = {}
    rows: list[dict[str, Any]] = []
    for row in frame.to_dict(orient="records"):
        result = {
            "unit_id": row["unit_id"],
            "modality": row["modality"],
            "selector_json": row["selector_json"],
            "preprocessed_path": None,
            "eligible": False,
        }
        try:
            source = Path(str(row["source_path"])).resolve(strict=True)
            source_key = str(source)
            if source_key not in source_cache:
                source_hash = sha256_file(source)
                cache_key = f"{source}:{source_hash}"
                raw = read_raw_recording(source)
                clean, provenance = preprocess_mne_raw(
                    raw,
                    canonical_channels=study.preprocessing.canonical_channels,
                    target_sampling_hz=study.preprocessing.target_sampling_hz,
                    highpass_hz=study.preprocessing.highpass_hz,
                    lowpass_hz=study.preprocessing.lowpass_hz,
                    notch_hz=infer_mains_frequency(raw),
                    maximum_interpolation_fraction=(
                        study.preprocessing.maximum_interpolation_fraction
                    ),
                )
                if len(clean.ch_names) < study.preprocessing.minimum_canonical_channels:
                    raise ValueError(
                        f"only {len(clean.ch_names)} canonical channels after preprocessing"
                    )
                master = destination / "source-recordings" / f"{_safe_key(cache_key)}-raw.fif"
                master.parent.mkdir(parents=True, exist_ok=True)
                if not master.is_file():
                    temporary = master.with_name(f".{master.name}.tmp.fif")
                    clean.save(temporary, overwrite=False, verbose="ERROR")
                    os.replace(temporary, master)
                source_cache[source_key] = (master, provenance, source_hash)
            master, provenance, source_hash = source_cache[source_key]
            clean = mne.io.read_raw_fif(master, preload=False, verbose="ERROR")
            selector = _selector(row)
            selected = _crop_clean(clean, selector)
            duration = float(selected.n_times / selected.info["sfreq"])
            if (
                selector["kind"] != "event_epoch"
                and duration < study.preprocessing.minimum_rest_seconds
            ):
                raise ValueError(
                    f"only {duration:.1f}s; need {study.preprocessing.minimum_rest_seconds}s"
                )
            unit_path = destination / "units" / f"{row['unit_id']}-raw.fif"
            unit_path.parent.mkdir(parents=True, exist_ok=True)
            if not unit_path.is_file():
                temporary = unit_path.with_name(f".{unit_path.name}.tmp.fif")
                selected.load_data().save(temporary, overwrite=False, verbose="ERROR")
                os.replace(temporary, unit_path)
            provenance_path = destination / "provenance" / f"{row['unit_id']}.json"
            atomic_write_json(
                provenance_path,
                {
                    **provenance,
                    "unit_id": row["unit_id"],
                    "selector": selector,
                    "source_path": str(source),
                    "source_sha256": source_hash,
                    "preprocessed_path": str(unit_path),
                    "preprocessed_sha256": sha256_file(unit_path),
                    "duration_seconds": duration,
                    "label_fields_consumed": [],
                },
            )
            result.update(
                {
                    "preprocessed_path": str(unit_path),
                    "preprocessed_sha256": sha256_file(unit_path),
                    "preprocessing_provenance_path": str(provenance_path),
                    "duration_seconds": duration,
                    "eligible": True,
                    "exclusion_reason": None,
                }
            )
        except (ValueError, RuntimeError, OSError) as error:
            result["exclusion_reason"] = f"{type(error).__name__}: {error}"
        rows.append(result)
    manifest = _atomic_parquet(pd.DataFrame(rows), destination / "preprocessing-manifest.parquet")
    flow = destination / "preprocessing-flow.json"
    eligible = sum(bool(row["eligible"]) for row in rows)
    atomic_write_json(
        flow,
        {
            "schema_version": 1,
            "analysis_units": len(rows),
            "eligible_units": eligible,
            "excluded_units": len(rows) - eligible,
            "unique_source_recordings": len(source_cache),
            "exclusions": [
                {"unit_id": row["unit_id"], "reason": row["exclusion_reason"]}
                for row in rows
                if not row["eligible"]
            ],
            "label_fields_consumed": [],
        },
    )
    if eligible == 0:
        raise RuntimeError(f"all analysis units failed preprocessing; inspect {flow}")
    return manifest, flow


def _encode_windows(
    raw: Any,
    encoder: OfficialLaBraMEncoder,
    *,
    window_seconds: float,
    step_seconds: float,
) -> tuple[Any, np.ndarray, Any]:
    windows, starts = make_windows(
        raw.get_data(), float(raw.info["sfreq"]), window_seconds, step_seconds
    )
    if len(windows) == 0:
        raise ValueError(f"recording is shorter than the {window_seconds}s encoder window")
    artifact = detect_artifact_windows(windows, float(raw.info["sfreq"]))
    kept = windows[artifact.keep]
    if len(kept) == 0:
        raise ValueError("all encoder windows were rejected as artifacts")
    return encoder.encode(kept, raw.ch_names), starts[artifact.keep], artifact


def _aggregate_event_rows(
    frame: pd.DataFrame,
    *,
    destination: Path,
    minimum_trials: int,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    selectors = frame["selector_json"].map(json.loads)
    event_mask = selectors.map(lambda item: item["kind"] in {"event_epoch", "pre_epoched"})
    retained = [row for row in frame[~event_mask].to_dict(orient="records")]
    issues: list[dict[str, Any]] = []
    event = frame[event_mask & frame["encoded"].astype(bool)].copy()
    group_columns = ["participant_id", "dataset_id", "condition"]
    for identity, group in event.groupby(group_columns, dropna=False):
        if len(group) < minimum_trials:
            issues.append(
                {
                    "group": dict(zip(group_columns, identity, strict=True)),
                    "status": "insufficient_trials",
                    "observed": len(group),
                    "required": minimum_trials,
                }
            )
            continue
        archives: list[dict[str, np.ndarray]] = []
        for path in group["trajectory_path"]:
            with np.load(Path(str(path)).resolve(strict=True), allow_pickle=False) as source:
                archives.append({name: np.asarray(source[name]) for name in source.files})
        region_names = set(name for name in archives[0] if name.startswith("alignment_regional_"))
        for archive in archives[1:]:
            region_names.intersection_update(
                name for name in archive if name.startswith("alignment_regional_")
            )
        if len(region_names) < 2:
            issues.append(
                {
                    "group": dict(zip(group_columns, identity, strict=True)),
                    "status": "insufficient_shared_regions",
                }
            )
            continue
        trajectories = [archive["alignment_global_states"] for archive in archives]
        lengths = [len(value) for value in trajectories]
        segments = np.concatenate(
            [np.full(length, index, dtype=np.int32) for index, length in enumerate(lengths)]
        )
        arrays: dict[str, np.ndarray] = {
            "global_states": np.concatenate(trajectories).astype(np.float32),
            "alignment_global_states": np.concatenate(trajectories).astype(np.float32),
            "segment_ids": segments,
            "alignment_segment_ids": segments,
        }
        for encoded_name in sorted(region_names):
            name = encoded_name.removeprefix("alignment_")
            values = np.concatenate([archive[encoded_name] for archive in archives]).astype(
                np.float32
            )
            arrays[name] = values
            arrays[encoded_name] = values
        group_key = _safe_key("|".join(str(value) for value in identity))
        trajectory = destination / "trajectories" / f"group-{group_key}.npz"
        temporary = trajectory.with_name(f".{trajectory.name}.tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        os.replace(temporary, trajectory)
        first = group.iloc[0].to_dict()
        row: dict[str, Any] = {
            key: value
            for key, value in first.items()
            if key
            not in {
                "unit_id",
                "trajectory_path",
                "trajectory_sha256",
                "source_path",
                "source_file",
                "preprocessed_path",
            }
        }
        for column in group.columns:
            if column in row or column in {
                "unit_id",
                "trajectory_path",
                "trajectory_sha256",
                "source_path",
                "source_file",
                "preprocessed_path",
            }:
                continue
            numeric = pd.to_numeric(group[column], errors="coerce")
            if numeric.notna().all():
                row[column] = float(numeric.mean())
        row.update(
            {
                "unit_id": group_key,
                "trajectory_path": str(trajectory),
                "trajectory_sha256": sha256_file(trajectory),
                "encoded": True,
                "encoding_error": None,
                "trial_count": len(group),
                "event_aggregated": True,
                "coarse_windows": int(sum(lengths)),
                "alignment_windows": int(sum(lengths)),
            }
        )
        retained.append(row)
    return pd.DataFrame(retained), issues


def encode_analysis_units(
    *,
    preprocessing_manifest: str | Path,
    labels_manifest: str | Path,
    output_root: str | Path,
    study: StudyConfig,
) -> tuple[Path, Path]:
    """Encode signal-only units and attach labels only after checkpoint inference."""

    try:
        import mne
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("install neural-manifolds[eeg]") from exc
    repository, checkpoint, checkpoint_hash = _model_environment()
    encoder = OfficialLaBraMEncoder(
        repository=repository,
        factory="modeling_finetune:labram_base_patch200_200",
        checkpoint=checkpoint,
        checkpoint_sha256=checkpoint_hash,
        device="cuda",
    )
    frame = pd.read_parquet(preprocessing_manifest)
    destination = Path(output_root)
    destination.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for row in frame[frame["eligible"].astype(bool)].to_dict(orient="records"):
        result = {
            "unit_id": row["unit_id"],
            "encoded": False,
            "trajectory_path": None,
            "trajectory_sha256": None,
            "preprocessed_path": row["preprocessed_path"],
            "preprocessed_sha256": row.get("preprocessed_sha256"),
        }
        try:
            raw = mne.io.read_raw_fif(
                Path(str(row["preprocessed_path"])), preload=True, verbose="ERROR"
            )
            selector = _selector(row)
            is_event = selector["kind"] in {"event_epoch", "pre_epoched"}
            if is_event:
                encoded, starts, artifact = _encode_windows(
                    raw,
                    encoder,
                    window_seconds=study.representation.alignment_window_seconds,
                    step_seconds=study.representation.alignment_step_seconds,
                )
                coarse = encoded
                coarse_starts = starts
            else:
                coarse, coarse_starts, _coarse_artifact = _encode_windows(
                    raw,
                    encoder,
                    window_seconds=study.representation.harmonised_window_seconds,
                    step_seconds=study.representation.harmonised_step_seconds,
                )
                encoded, starts, artifact = _encode_windows(
                    raw,
                    encoder,
                    window_seconds=study.representation.alignment_window_seconds,
                    step_seconds=study.representation.alignment_step_seconds,
                )
                if len(coarse.global_states) < study.preprocessing.minimum_valid_windows:
                    raise ValueError(
                        f"only {len(coarse.global_states)} coarse windows; need "
                        f"{study.preprocessing.minimum_valid_windows}"
                    )
            arrays: dict[str, np.ndarray] = {
                "global_states": np.asarray(coarse.global_states, dtype=np.float32),
                "window_start_samples": coarse_starts,
                "alignment_global_states": np.asarray(encoded.global_states, dtype=np.float32),
                "alignment_window_start_samples": starts,
                "segment_ids": np.zeros(len(coarse.global_states), dtype=np.int32),
                "alignment_segment_ids": np.zeros(len(encoded.global_states), dtype=np.int32),
            }
            arrays.update(
                {
                    f"regional_{name}": np.asarray(value, dtype=np.float32)
                    for name, value in coarse.regional_states.items()
                }
            )
            arrays.update(
                {
                    f"alignment_regional_{name}": np.asarray(value, dtype=np.float32)
                    for name, value in encoded.regional_states.items()
                }
            )
            trajectory = destination / "trajectories" / f"{row['unit_id']}.npz"
            trajectory.parent.mkdir(parents=True, exist_ok=True)
            temporary = trajectory.with_name(f".{trajectory.name}.tmp")
            with temporary.open("wb") as stream:
                np.savez_compressed(stream, **arrays)
            os.replace(temporary, trajectory)
            result.update(
                {
                    "trajectory_path": str(trajectory),
                    "trajectory_sha256": sha256_file(trajectory),
                    "encoded": True,
                    "encoding_error": None,
                    "coarse_windows": len(coarse.global_states),
                    "alignment_windows": len(encoded.global_states),
                    "rejected_alignment_windows": int(np.count_nonzero(~artifact.keep)),
                    "label_fields_consumed": "",
                }
            )
        except (ValueError, RuntimeError, OSError) as error:
            result["encoding_error"] = f"{type(error).__name__}: {error}"
        rows.append(result)
    signal = pd.DataFrame(rows)
    labels = pd.read_parquet(labels_manifest)
    if set(signal["unit_id"]) - set(labels["unit_id"]):
        raise RuntimeError("encoded units are absent from the post-encoding label manifest")
    joined = signal.merge(labels, on="unit_id", how="left", validate="one_to_one")
    joined, aggregation_issues = _aggregate_event_rows(
        joined,
        destination=destination,
        minimum_trials=study.preprocessing.minimum_event_trials_per_condition,
    )
    manifest = _atomic_parquet(joined, destination / "encoding-manifest.parquet")
    flow = destination / "encoding-flow.json"
    encoded_count = int(signal["encoded"].sum()) if len(signal) else 0
    atomic_write_json(
        flow,
        {
            "schema_version": 1,
            "eligible_units": len(signal),
            "encoded_units": encoded_count,
            "failed_units": len(signal) - encoded_count,
            "checkpoint_sha256": checkpoint_hash,
            "labels_joined_after_encoding": True,
            "encoder_label_fields_consumed": [],
            "event_aggregation_issues": aggregation_issues,
            "errors": [
                {"unit_id": row["unit_id"], "error": row.get("encoding_error")}
                for row in rows
                if not row["encoded"]
            ],
        },
    )
    if encoded_count == 0:
        raise RuntimeError(f"no analysis units encoded; inspect {flow}")
    return manifest, flow
