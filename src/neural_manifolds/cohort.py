"""Build strict analysis-unit manifests from immutable acquired releases."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

from neural_manifolds.adapters import (
    AnalysisUnit,
    DreamSerialAwakeningsAdapter,
    FigshareDoCRestingAdapter,
    MendeleyDoCPSGAdapter,
    PropofolTMSEEGAdapter,
    PsiConnectAdapter,
    SchemaError,
    TactileDetectionAdapter,
    UnresolvedMetadataError,
    encoding_view,
)
from neural_manifolds.provenance import atomic_write_json
from neural_manifolds.tms_separation import DIRECT_TMS_MODALITY

DATASET_IDS = (
    "propofol_tms_eeg",
    "dream_tononi_serial_awakenings",
    "tactile_detection",
    "somatosensory_report_task",
    "cogitate_meeg",
    "psiconnect",
    "doc_resting_eeg",
    "doc_polysomnography",
    "propofol_fmri",
)


def _release_root(raw_root: Path, dataset_id: str) -> Path | None:
    dataset = raw_root / dataset_id
    if not dataset.is_dir():
        return None
    releases = sorted(
        path
        for path in dataset.iterdir()
        if path.is_dir() and (path / ".acquisition" / "COMPLETE.json").is_file()
    )
    if len(releases) > 1:
        raise RuntimeError(f"multiple immutable releases are present for {dataset_id}")
    return releases[0] if releases else None


def _relative_files(root: Path, pattern: str = "*") -> list[str]:
    return [
        path.relative_to(root).as_posix() for path in sorted(root.rglob(pattern)) if path.is_file()
    ]


def _one(root: Path, name: str) -> Path:
    matches = sorted(path for path in root.rglob(name) if path.is_file())
    if len(matches) != 1:
        raise SchemaError(f"expected one {name} below {root}, found {len(matches)}")
    return matches[0]


def _read_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)


def _build_dataset(dataset_id: str, root: Path) -> list[AnalysisUnit]:
    if dataset_id == "propofol_tms_eeg":
        participants = _read_tsv(_one(root, "participants.tsv"))
        recordings = [value for value in _relative_files(root, "*.vhdr") if "/eeg/" in f"/{value}"]
        return PropofolTMSEEGAdapter().adapt(participants, recordings)
    if dataset_id == "dream_tononi_serial_awakenings":
        records = pd.read_csv(_one(root, "Records.csv"), dtype=str, keep_default_na=False)
        return DreamSerialAwakeningsAdapter().adapt(records)
    if dataset_id == "tactile_detection":
        participants = _read_tsv(_one(root, "participants.tsv"))
        event_paths = _relative_files(root, "*_events.tsv")
        events = {path: _read_tsv(root / path) for path in event_paths if "task-adapt" in path}
        return TactileDetectionAdapter().adapt(participants, events)
    if dataset_id == "psiconnect":
        participants = _read_tsv(_one(root, "participants.tsv"))
        recordings = [value for value in _relative_files(root, "*.vhdr") if "task-series" in value]
        return PsiConnectAdapter().adapt(participants, recordings)
    if dataset_id == "doc_resting_eeg":
        files = [
            value
            for value in _relative_files(root)
            if Path(value).suffix.lower() in {".dat", ".vhdr", ".vmrk"}
        ]
        return FigshareDoCRestingAdapter().adapt(files)
    if dataset_id == "doc_polysomnography":
        return MendeleyDoCPSGAdapter().adapt(_relative_files(root, "*.edf"))
    if dataset_id == "somatosensory_report_task":
        raise UnresolvedMetadataError(
            "OSF MAT signal files/variables require an audited post-extraction inventory"
        )
    if dataset_id == "cogitate_meeg":
        raise UnresolvedMetadataError(
            "Cogitate event schema is unavailable until the account-gated bundle is approved"
        )
    if dataset_id == "propofol_fmri":
        # The fMRI stage constructs volume-interval units only after ESC/LOR/ROR
        # sidecars have passed their separate timing audit.
        return []
    raise SchemaError(f"no cohort builder for {dataset_id}")


def _flatten_unit(unit: AnalysisUnit, root: Path) -> dict[str, Any]:
    payload = unit.model_dump(mode="json")
    payload["participant_id"] = f"{unit.dataset_id}:{unit.participant_id}"
    selector = payload.pop("selector")
    variables = payload.pop("variables")
    collisions = set(payload).intersection(variables)
    if collisions:
        raise SchemaError(f"adapter variable names collide with core fields: {sorted(collisions)}")
    source_path = (root / unit.source_file).resolve(strict=True)
    return {
        **payload,
        **variables,
        "source_path": str(source_path),
        "selector_json": json.dumps(selector, sort_keys=True, separators=(",", ":")),
        "secondary_fmri": unit.modality == "fmri",
    }


def build_cohort_manifest(
    *,
    raw_root: str | Path,
    output_root: str | Path,
    dataset_ids: tuple[str, ...] | None = None,
) -> tuple[Path, Path, Path]:
    """Write separate label and encoder views; unresolved datasets stay explicit."""

    root = Path(raw_root).resolve(strict=True)
    destination = Path(output_root)
    destination.mkdir(parents=True, exist_ok=True)
    labels: list[dict[str, Any]] = []
    encoder_rows: list[dict[str, Any]] = []
    issues: list[dict[str, str]] = []
    direct_tms_units: list[dict[str, str]] = []
    selected_ids = DATASET_IDS if dataset_ids is None else dataset_ids
    if not selected_ids or len(selected_ids) != len(set(selected_ids)):
        raise ValueError("dataset_ids must be a non-empty unique tuple")
    unknown = sorted(set(selected_ids).difference(DATASET_IDS))
    if unknown:
        raise ValueError(f"unknown cohort datasets: {unknown}")
    for dataset_id in selected_ids:
        release = _release_root(root, dataset_id)
        if release is None:
            issues.append({"dataset_id": dataset_id, "status": "not_acquired"})
            continue
        try:
            units = _build_dataset(dataset_id, release)
            for unit in units:
                flattened = _flatten_unit(unit, release)
                labels.append(flattened)
                if unit.modality == DIRECT_TMS_MODALITY:
                    direct_tms_units.append(
                        {
                            "unit_id": unit.unit_id,
                            "dataset_id": unit.dataset_id,
                            "source_path": flattened["source_path"],
                        }
                    )
                    continue
                view = encoding_view(unit).model_dump(mode="json")
                encoder_rows.append(
                    {
                        **view,
                        "source_path": flattened["source_path"],
                        "selector_json": flattened["selector_json"],
                    }
                )
        except (SchemaError, UnresolvedMetadataError, FileNotFoundError) as error:
            issues.append(
                {
                    "dataset_id": dataset_id,
                    "status": "metadata_blocked",
                    "error": f"{type(error).__name__}: {error}",
                }
            )
    unit_ids = [row["unit_id"] for row in labels]
    if len(unit_ids) != len(set(unit_ids)):
        raise RuntimeError("cohort manifest contains duplicate unit IDs")
    if not labels:
        raise RuntimeError("no analysis units could be built from acquired releases")

    def write(frame: pd.DataFrame, path: Path) -> Path:
        temporary = path.with_name(f".{path.name}.tmp")
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, path)
        return path

    labels_path = write(pd.DataFrame(labels), destination / "cohort-labels.parquet")
    encoder_path = write(pd.DataFrame(encoder_rows), destination / "encoder-inputs.parquet")
    issues_path = destination / "cohort-issues.json"
    atomic_write_json(
        issues_path,
        {
            "schema_version": 1,
            "analysis_units": len(labels),
            "general_encoder_units": len(encoder_rows),
            "datasets_represented": sorted({row["dataset_id"] for row in labels}),
            "issues": issues,
            "labels_joined_after_encoding": True,
            "direct_tms_separation": {
                "status": "omitted_from_general_encoder_inputs",
                "reason": "requires_dedicated_pulse_interpolation_and_tms_preprocessing",
                "retained_in_cohort_labels": True,
                "raw_lineage_retained": True,
                "dedicated_tms_units": direct_tms_units,
            },
            "scientific_gate_applied": False,
        },
    )
    return labels_path, encoder_path, issues_path
