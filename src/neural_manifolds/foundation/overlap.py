"""Conservative provenance for foundation-model pretraining overlap.

Frozen inference prevents target-study fine-tuning, but it does not by itself
establish that a target dataset was absent from foundation-model pretraining.
This module keeps those two claims separate in every downstream artifact.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

OVERLAP_ARTIFACT_COLUMNS = (
    "representation_model_id",
    "pretraining_overlap_status",
    "pretraining_overlap_configured_status",
    "pretraining_overlap_target_covered",
    "pretraining_overlap_evidence_json",
    "pretraining_overlap_source",
    "pretraining_overlap_control",
    "pretraining_overlap_limitation",
    "zero_shot_classification",
    "zero_shot_verified",
)
OVERLAP_OUTPUT_COLUMNS = (
    "pretraining_overlap_status",
    "zero_shot_classification",
    "zero_shot_verified",
    "pretraining_overlap_control",
    "pretraining_overlap_limitation",
    "representation_model_ids_json",
    "pretraining_overlap_dataset_classifications_json",
)

_UNRESOLVED_CLASSIFICATION = "unresolved_not_verified_zero_shot"
_CONFIRMED_CLASSIFICATION = "confirmed_overlap_not_zero_shot"
_VERIFIED_CLASSIFICATION = "verified_zero_shot_no_pretraining_overlap"


def _default_models_path() -> Path:
    return Path(__file__).resolve().parents[3] / "configs" / "models.yaml"


def _canonical_status(value: Any) -> str:
    normalized = str(value or "unresolved").strip().lower().replace("-", "_")
    aliases = {
        "confirmed": "confirmed_overlap",
        "overlap_confirmed": "confirmed_overlap",
        "verified_absent": "verified_no_overlap",
        "no_overlap_verified": "verified_no_overlap",
        "verified_zero_shot": "verified_no_overlap",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"unresolved", "confirmed_overlap", "verified_no_overlap"}:
        return "unresolved"
    return normalized


def _json_text(value: Any) -> str:
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return "null"
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            parsed = stripped
        return json.dumps(parsed, sort_keys=True, separators=(",", ":"))
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _has_evidence(value: Any) -> bool:
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped or stripped.lower() in {"null", "none", "unresolved", "unknown"}:
            return False
        try:
            value = json.loads(stripped)
        except json.JSONDecodeError:
            return True
    if value is None:
        return False
    if isinstance(value, float) and pd.isna(value):
        return False
    if isinstance(value, Mapping):
        return bool(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return bool(value)
    return bool(value)


def _artifact_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value == 1
    if isinstance(value, float) and not pd.isna(value):
        return value == 1.0
    return False


def classify_pretraining_overlap(
    *,
    model_id: str,
    dataset_id: str,
    audit: Mapping[str, Any] | None,
    trainable: bool | None,
    source: str,
) -> dict[str, Any]:
    """Classify one model/dataset pair without upgrading missing evidence."""

    audit = dict(audit or {})
    configured_status = _canonical_status(audit.get("status"))
    raw_targets = audit.get("target_dataset_ids", [])
    targets = (
        {str(value) for value in raw_targets}
        if isinstance(raw_targets, Sequence) and not isinstance(raw_targets, (str, bytes))
        else set()
    )
    target_covered = dataset_id in targets
    evidence = audit.get("evidence")
    evidence_available = _has_evidence(evidence)

    if configured_status == "confirmed_overlap" and target_covered:
        status = "confirmed_overlap"
    elif configured_status == "verified_no_overlap" and target_covered and evidence_available:
        status = "verified_no_overlap"
    else:
        status = "unresolved"

    if status == "verified_no_overlap":
        classification = _VERIFIED_CLASSIFICATION
        limitation = (
            f"Documented evidence verifies no pretraining overlap for {dataset_id}; "
            "this claim remains limited to the cited evidence and model revision."
        )
    elif status == "confirmed_overlap":
        classification = _CONFIRMED_CLASSIFICATION
        limitation = (
            f"Pretraining overlap is confirmed for {dataset_id}; analyses using this "
            "representation are not zero-shot for that dataset."
        )
    else:
        classification = _UNRESOLVED_CLASSIFICATION
        limitation = (
            f"Pretraining overlap with {dataset_id} is unresolved; frozen inference "
            "alone does not verify zero-shot transfer."
        )
    configured_limitation = audit.get("limitation")
    if isinstance(configured_limitation, str) and configured_limitation.strip():
        limitation = f"{limitation} {configured_limitation.strip()}"

    if trainable is False:
        control = "frozen_weights_no_study_finetuning"
    elif trainable is True:
        control = "representation_weights_trainable"
    else:
        control = "representation_training_status_unresolved"
    configured_control = audit.get("control")
    if isinstance(configured_control, str) and configured_control.strip():
        control = configured_control.strip()

    return {
        "representation_model_id": str(model_id),
        "pretraining_overlap_status": status,
        "pretraining_overlap_configured_status": configured_status,
        "pretraining_overlap_target_covered": bool(target_covered),
        "pretraining_overlap_evidence_json": _json_text(evidence),
        "pretraining_overlap_source": source,
        "pretraining_overlap_control": control,
        "pretraining_overlap_limitation": limitation,
        "zero_shot_classification": classification,
        "zero_shot_verified": status == "verified_no_overlap",
    }


def load_pretraining_overlap(
    *,
    model_id: str,
    dataset_ids: Sequence[str],
    models_path: str | Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Load conservative dataset classifications from the pinned model registry."""

    path = Path(models_path) if models_path is not None else _default_models_path()
    with path.open("r", encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    if not isinstance(document, Mapping) or document.get("schema_version") != 1:
        raise ValueError("model configuration must use schema_version 1")
    models = document.get("models")
    if not isinstance(models, Mapping):
        raise ValueError("model configuration has no models mapping")
    spec = models.get(model_id)
    if not isinstance(spec, Mapping):
        raise ValueError(f"model configuration has no {model_id!r} entry")
    audit = spec.get("pretraining_overlap_audit")
    if audit is not None and not isinstance(audit, Mapping):
        raise ValueError(f"pretraining_overlap_audit for {model_id!r} must be a mapping")
    source = f"{path.resolve()}#models.{model_id}.pretraining_overlap_audit"
    return {
        str(dataset_id): classify_pretraining_overlap(
            model_id=model_id,
            dataset_id=str(dataset_id),
            audit=audit,
            trainable=spec.get("trainable") if isinstance(spec.get("trainable"), bool) else None,
            source=source,
        )
        for dataset_id in dataset_ids
    }


def _normalize_artifact_row(row: Mapping[str, Any], *, default_model_id: str) -> dict[str, Any]:
    dataset_id = str(row.get("dataset_id", "unknown_dataset"))
    model_id = str(row.get("representation_model_id") or default_model_id)
    configured = _canonical_status(
        row.get("pretraining_overlap_configured_status", row.get("pretraining_overlap_status"))
    )
    status = _canonical_status(row.get("pretraining_overlap_status"))
    evidence = row.get("pretraining_overlap_evidence_json")
    target_covered = _artifact_bool(row.get("pretraining_overlap_target_covered", False))
    if status == "verified_no_overlap" and not (target_covered and _has_evidence(evidence)):
        status = "unresolved"
    if status == "verified_no_overlap":
        classification = _VERIFIED_CLASSIFICATION
        default_limitation = (
            f"Documented evidence verifies no pretraining overlap for {dataset_id}; "
            "this claim remains limited to the cited evidence and model revision."
        )
    elif status == "confirmed_overlap":
        classification = _CONFIRMED_CLASSIFICATION
        default_limitation = (
            f"Pretraining overlap is confirmed for {dataset_id}; analyses using this "
            "representation are not zero-shot for that dataset."
        )
    else:
        classification = _UNRESOLVED_CLASSIFICATION
        default_limitation = (
            f"Pretraining overlap with {dataset_id} is unresolved; frozen inference "
            "alone does not verify zero-shot transfer."
        )
    control = row.get("pretraining_overlap_control")
    limitation = row.get("pretraining_overlap_limitation")
    source = row.get("pretraining_overlap_source")
    return {
        "representation_model_id": model_id,
        "pretraining_overlap_status": status,
        "pretraining_overlap_configured_status": configured,
        "pretraining_overlap_target_covered": target_covered,
        "pretraining_overlap_evidence_json": _json_text(evidence),
        "pretraining_overlap_source": str(source or "artifact_metadata"),
        "pretraining_overlap_control": str(control or "representation_training_status_unresolved"),
        "pretraining_overlap_limitation": str(limitation or default_limitation),
        "zero_shot_classification": classification,
        "zero_shot_verified": status == "verified_no_overlap",
    }


def ensure_pretraining_overlap_columns(
    frame: pd.DataFrame,
    *,
    default_model_id: str = "labram_base",
    models_path: str | Path | None = None,
) -> pd.DataFrame:
    """Attach or normalize overlap metadata on an artifact table."""

    output = frame.copy()
    if "dataset_id" not in output:
        raise ValueError("overlap provenance requires dataset_id")
    if output.empty:
        for column in OVERLAP_ARTIFACT_COLUMNS:
            output[column] = pd.Series(dtype="object")
        return output

    has_artifact_status = "pretraining_overlap_status" in output
    if has_artifact_status:
        records = [
            _normalize_artifact_row(row, default_model_id=default_model_id)
            for row in output.to_dict(orient="records")
        ]
    else:
        model_ids = (
            output["representation_model_id"].fillna(default_model_id).astype(str)
            if "representation_model_id" in output
            else pd.Series(default_model_id, index=output.index)
        )
        classifications: dict[tuple[str, str], dict[str, Any]] = {}
        for model_id in sorted(set(model_ids)):
            datasets = sorted(set(output.loc[model_ids.eq(model_id), "dataset_id"].astype(str)))
            for dataset_id, classification in load_pretraining_overlap(
                model_id=model_id,
                dataset_ids=datasets,
                models_path=models_path,
            ).items():
                classifications[(model_id, dataset_id)] = classification
        records = [
            classifications[(model_id, str(dataset_id))]
            for model_id, dataset_id in zip(model_ids, output["dataset_id"], strict=True)
        ]
    metadata = pd.DataFrame(records, index=output.index)
    for column in OVERLAP_ARTIFACT_COLUMNS:
        output[column] = metadata[column]
    return output


def summarize_pretraining_overlap(frame: pd.DataFrame) -> dict[str, Any]:
    """Return an artifact-level conservative status plus dataset-level evidence."""

    if frame.empty:
        return {
            "pretraining_overlap_status": "unresolved",
            "zero_shot_classification": _UNRESOLVED_CLASSIFICATION,
            "zero_shot_verified": False,
            "pretraining_overlap_control": "no_rows_available",
            "pretraining_overlap_limitation": "No rows were available for overlap assessment.",
            "dataset_classifications": [],
        }
    normalized = ensure_pretraining_overlap_columns(frame)
    unique = (
        normalized[["dataset_id", *OVERLAP_ARTIFACT_COLUMNS]]
        .drop_duplicates()
        .sort_values(["representation_model_id", "dataset_id", "pretraining_overlap_status"])
    )
    records = unique.to_dict(orient="records")
    statuses = {str(record["pretraining_overlap_status"]) for record in records}
    if "confirmed_overlap" in statuses:
        status = "confirmed_overlap"
        classification = _CONFIRMED_CLASSIFICATION
    elif statuses == {"verified_no_overlap"}:
        status = "verified_no_overlap"
        classification = _VERIFIED_CLASSIFICATION
    else:
        status = "unresolved"
        classification = _UNRESOLVED_CLASSIFICATION
    controls = sorted({str(record["pretraining_overlap_control"]) for record in records})
    limitations = sorted({str(record["pretraining_overlap_limitation"]) for record in records})
    return {
        "pretraining_overlap_status": status,
        "zero_shot_classification": classification,
        "zero_shot_verified": status == "verified_no_overlap",
        "pretraining_overlap_control": " | ".join(controls),
        "pretraining_overlap_limitation": " | ".join(limitations),
        "representation_model_ids": sorted(
            {str(record["representation_model_id"]) for record in records}
        ),
        "dataset_classifications": records,
    }


def overlap_output_fields(frame: pd.DataFrame) -> dict[str, Any]:
    """Flatten the conservative summary for Parquet model-result rows."""

    summary = summarize_pretraining_overlap(frame)
    fields = {key: summary[key] for key in OVERLAP_OUTPUT_COLUMNS[:5]}
    fields["representation_model_ids_json"] = _json_text(
        summary.get("representation_model_ids", [])
    )
    fields["pretraining_overlap_dataset_classifications_json"] = _json_text(
        summary["dataset_classifications"]
    )
    return fields
