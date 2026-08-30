"""Recording-level preprocessing and frozen-encoding stage drivers."""

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


def _record_key(recording_id: str) -> str:
    return hashlib.sha256(recording_id.encode("utf-8")).hexdigest()[:20]


def read_raw_recording(path: str | Path) -> Any:
    try:
        import mne
    except ImportError as exc:  # pragma: no cover - server EEG environment
        raise RuntimeError("install neural-manifolds[eeg]") from exc
    source = Path(path)
    suffix = source.suffix.lower()
    readers = {
        ".vhdr": mne.io.read_raw_brainvision,
        ".edf": mne.io.read_raw_edf,
        ".bdf": mne.io.read_raw_bdf,
        ".set": mne.io.read_raw_eeglab,
        ".fif": mne.io.read_raw_fif,
    }
    if suffix not in readers:
        raise ValueError(f"unsupported recording format: {source}")
    return readers[suffix](source, preload=False, verbose="ERROR")


def infer_mains_frequency(raw: Any) -> float:
    line = raw.info.get("line_freq")
    if line in {50, 60}:
        return float(line)
    data = raw.get_data(start=0, stop=min(raw.n_times, int(raw.info["sfreq"] * 60)))
    sfreq = float(raw.info["sfreq"])
    from scipy.signal import welch

    frequencies, psd = welch(data, fs=sfreq, nperseg=min(data.shape[1], int(sfreq * 4)), axis=1)
    candidates: dict[float, float] = {}
    for frequency in (50.0, 60.0):
        if frequency >= sfreq / 2:
            continue
        narrow = np.abs(frequencies - frequency) <= 0.75
        flanks = (np.abs(frequencies - frequency) >= 2.0) & (np.abs(frequencies - frequency) <= 4.0)
        candidates[frequency] = float(
            np.median(psd[:, narrow]) / max(np.median(psd[:, flanks]), np.finfo(float).tiny)
        )
    return max(candidates, key=candidates.get) if candidates else 50.0


def preprocess_inventory(
    *,
    inventory_path: Path,
    output_root: Path,
    study: StudyConfig,
) -> tuple[Path, Path]:
    """Preprocess every inventoried recording, preserving an explicit exclusion flow."""

    frame = pd.read_parquet(inventory_path)
    output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for row in frame.to_dict(orient="records"):
        key = _record_key(str(row["recording_id"]))
        destination = output_root / "recordings" / f"{key}-harmonised-raw.fif"
        provenance_path = output_root / "provenance" / f"{key}.json"
        result = {**row, "record_key": key, "preprocessed_path": None, "eligible": False}
        try:
            source = Path(str(row["source_path"]))
            if destination.is_file() and provenance_path.is_file():
                provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
                if provenance.get("source_sha256") != sha256_file(source):
                    raise RuntimeError("source changed after preprocessing")
            else:
                raw = read_raw_recording(source)
                clean, provenance = preprocess_mne_raw(
                    raw,
                    canonical_channels=study.preprocessing.canonical_channels,
                    target_sampling_hz=study.preprocessing.target_sampling_hz,
                    highpass_hz=study.preprocessing.highpass_hz,
                    lowpass_hz=study.preprocessing.lowpass_hz,
                    notch_hz=infer_mains_frequency(raw),
                    maximum_interpolation_fraction=study.preprocessing.maximum_interpolation_fraction,
                )
                if len(clean.ch_names) < study.preprocessing.minimum_canonical_channels:
                    raise ValueError(
                        f"only {len(clean.ch_names)} canonical channels; "
                        f"need {study.preprocessing.minimum_canonical_channels}"
                    )
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = destination.with_name(f".{destination.name}.tmp.fif")
                clean.save(temporary, overwrite=False, verbose="ERROR")
                os.replace(temporary, destination)
                provenance_path.parent.mkdir(parents=True, exist_ok=True)
                atomic_write_json(
                    provenance_path,
                    {
                        **provenance,
                        "recording_id": row["recording_id"],
                        "source_path": str(source),
                        "source_sha256": sha256_file(source),
                        "output_path": str(destination),
                        "output_sha256": sha256_file(destination),
                    },
                )
            result.update(
                {
                    "preprocessed_path": str(destination),
                    "preprocessed_sha256": sha256_file(destination),
                    "provenance_path": str(provenance_path),
                    "eligible": True,
                    "exclusion_reason": None,
                }
            )
        except Exception as error:
            result.update(
                {
                    "exclusion_reason": f"{type(error).__name__}: {error}",
                    "provenance_path": None,
                }
            )
        rows.append(result)

    manifest = output_root / "preprocessing-manifest.parquet"
    temporary_manifest = manifest.with_name(f".{manifest.name}.tmp")
    pd.DataFrame(rows).to_parquet(temporary_manifest, index=False)
    os.replace(temporary_manifest, manifest)
    flow = output_root / "preprocessing-flow.json"
    eligible = sum(bool(row["eligible"]) for row in rows)
    atomic_write_json(
        flow,
        {
            "schema_version": 1,
            "recordings_total": len(rows),
            "recordings_eligible": eligible,
            "recordings_excluded": len(rows) - eligible,
            "exclusions": [
                {"recording_id": row["recording_id"], "reason": row["exclusion_reason"]}
                for row in rows
                if not row["eligible"]
            ],
        },
    )
    if eligible == 0:
        raise RuntimeError(f"all {len(rows)} recordings failed preprocessing; inspect {flow}")
    return manifest, flow


def _model_environment() -> tuple[Path, Path, str]:
    values = {
        "repository": os.environ.get("NEURAL_MANIFOLDS_LABRAM_SOURCE")
        or os.environ.get("NEURAL_MANIFOLDS_LABRAM_REPOSITORY"),
        "checkpoint": os.environ.get("NEURAL_MANIFOLDS_LABRAM_CHECKPOINT"),
        "sha256": os.environ.get("NEURAL_MANIFOLDS_LABRAM_SHA256"),
    }
    if not values["sha256"] and (
        manifest_value := os.environ.get("NEURAL_MANIFOLDS_MODEL_MANIFEST")
    ):
        manifest_path = Path(manifest_value).resolve(strict=True)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        artifact = manifest.get("models", {}).get("labram_base", {}).get("checkpoint", {})
        if artifact.get("path") == values["checkpoint"]:
            values["sha256"] = artifact.get("sha256")
    missing = [key for key, value in values.items() if not value]
    if missing:
        raise RuntimeError(f"LaBraM model bootstrap is incomplete: {missing}")
    return Path(str(values["repository"])), Path(str(values["checkpoint"])), str(values["sha256"])


def encode_preprocessed(
    *,
    preprocessing_manifest: Path,
    output_root: Path,
    study: StudyConfig,
) -> tuple[Path, Path]:
    """Encode all eligible recordings with the same frozen LaBraM checkpoint."""

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
    rows: list[dict[str, Any]] = []
    output_root.mkdir(parents=True, exist_ok=True)
    for row in frame[frame["eligible"]].to_dict(orient="records"):
        key = str(row["record_key"])
        destination = output_root / "trajectories" / f"{key}.npz"
        metadata_path = output_root / "provenance" / f"{key}.json"
        try:
            source = Path(str(row["preprocessed_path"]))
            if destination.is_file() and metadata_path.is_file():
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                if metadata.get("source_sha256") != sha256_file(source):
                    raise RuntimeError("preprocessed source changed after encoding")
            else:
                raw = mne.io.read_raw_fif(source, preload=True, verbose="ERROR")
                windows, starts = make_windows(
                    raw.get_data(),
                    float(raw.info["sfreq"]),
                    study.representation.harmonised_window_seconds,
                    study.representation.harmonised_step_seconds,
                )
                artifact = detect_artifact_windows(windows, float(raw.info["sfreq"]))
                windows = windows[artifact.keep]
                starts = starts[artifact.keep]
                if windows.shape[0] < study.preprocessing.minimum_valid_windows:
                    raise ValueError(
                        f"only {windows.shape[0]} retained windows; "
                        f"need {study.preprocessing.minimum_valid_windows}"
                    )
                encoded = encoder.encode(windows, raw.ch_names)

                alignment_windows, alignment_starts = make_windows(
                    raw.get_data(),
                    float(raw.info["sfreq"]),
                    study.representation.alignment_window_seconds,
                    study.representation.alignment_step_seconds,
                )
                alignment_artifact = detect_artifact_windows(
                    alignment_windows, float(raw.info["sfreq"])
                )
                alignment_windows = alignment_windows[alignment_artifact.keep]
                alignment_starts = alignment_starts[alignment_artifact.keep]
                if alignment_windows.shape[0] < study.preprocessing.minimum_valid_windows:
                    raise ValueError(
                        f"only {alignment_windows.shape[0]} retained fine-lag windows; "
                        f"need {study.preprocessing.minimum_valid_windows}"
                    )
                alignment_encoded = encoder.encode(alignment_windows, raw.ch_names)
                arrays: dict[str, np.ndarray] = {
                    "global_states": encoded.global_states.astype(np.float32),
                    "window_start_samples": starts,
                    "alignment_window_start_samples": alignment_starts,
                }
                arrays.update(
                    {
                        f"regional_{name}": values.astype(np.float32)
                        for name, values in encoded.regional_states.items()
                    }
                )
                arrays.update(
                    {
                        f"alignment_regional_{name}": values.astype(np.float32)
                        for name, values in alignment_encoded.regional_states.items()
                    }
                )
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = destination.with_name(f".{destination.name}.tmp")
                with temporary.open("wb") as stream:
                    np.savez_compressed(stream, **arrays)
                os.replace(temporary, destination)
                metadata_path.parent.mkdir(parents=True, exist_ok=True)
                atomic_write_json(
                    metadata_path,
                    {
                        **dict(encoded.metadata),
                        "recording_id": row["recording_id"],
                        "participant_id": row["participant_id"],
                        "dataset_id": row["dataset_id"],
                        "channel_names": list(encoded.channel_names),
                        "source_path": str(source),
                        "source_sha256": sha256_file(source),
                        "trajectory_path": str(destination),
                        "trajectory_sha256": sha256_file(destination),
                        "retained_windows": int(windows.shape[0]),
                        "rejected_windows": int(np.count_nonzero(~artifact.keep)),
                        "alignment_window_seconds": (study.representation.alignment_window_seconds),
                        "alignment_step_seconds": study.representation.alignment_step_seconds,
                        "alignment_retained_windows": int(alignment_windows.shape[0]),
                        "alignment_rejected_windows": int(
                            np.count_nonzero(~alignment_artifact.keep)
                        ),
                    },
                )
            rows.append(
                {
                    **row,
                    "trajectory_path": str(destination),
                    "trajectory_sha256": sha256_file(destination),
                    "encoding_metadata_path": str(metadata_path),
                    "encoded": True,
                    "encoding_error": None,
                }
            )
        except Exception as error:
            rows.append(
                {
                    **row,
                    "trajectory_path": None,
                    "trajectory_sha256": None,
                    "encoding_metadata_path": None,
                    "encoded": False,
                    "encoding_error": f"{type(error).__name__}: {error}",
                }
            )
    manifest = output_root / "encoding-manifest.parquet"
    temporary = manifest.with_name(f".{manifest.name}.tmp")
    pd.DataFrame(rows).to_parquet(temporary, index=False)
    os.replace(temporary, manifest)
    flow = output_root / "encoding-flow.json"
    encoded_count = sum(bool(row["encoded"]) for row in rows)
    atomic_write_json(
        flow,
        {
            "schema_version": 1,
            "eligible_recordings": len(rows),
            "encoded_recordings": encoded_count,
            "failed_recordings": len(rows) - encoded_count,
            "checkpoint_sha256": checkpoint_hash,
            "errors": [
                {"recording_id": row["recording_id"], "error": row["encoding_error"]}
                for row in rows
                if not row["encoded"]
            ],
        },
    )
    if encoded_count == 0:
        raise RuntimeError(f"no recordings encoded; inspect {flow}")
    return manifest, flow
