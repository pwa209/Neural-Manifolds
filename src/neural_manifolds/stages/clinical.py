"""Technically locked, non-diagnostic transfer to held-out DoC cohorts."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy import stats

from neural_manifolds.config import StudyConfig
from neural_manifolds.manifold.profile import AXIS_NAMES, FiveAxisProfileEstimator
from neural_manifolds.provenance import atomic_write_json, sha256_file
from neural_manifolds.stages.metrics import _load_unit, _record


def _atomic_parquet(frame: pd.DataFrame, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, destination)
    return destination


def _validate_lock(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version": 1,
        "kind": "technical_clinical_transfer_snapshot",
        "project_status": "exploratory_non_preregistered",
        "scientific_gate": False,
    }
    for key, expected in required.items():
        if payload.get(key) != expected:
            raise ValueError(f"clinical lock has invalid {key}")
    markers = payload.get("healthy_success_markers")
    artifacts = payload.get("healthy_validated_artifacts")
    if not isinstance(markers, dict):
        raise ValueError("clinical lock has no healthy success-marker hashes")
    if not isinstance(artifacts, dict):
        raise ValueError("clinical lock has no validated healthy artifact inventory")
    excluded = {"basis_sha256", "created_at", "notice"}
    basis = {key: value for key, value in payload.items() if key not in excluded}
    observed_basis = hashlib.sha256(
        json.dumps(basis, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()
    if payload.get("basis_sha256") != observed_basis:
        raise ValueError("clinical lock basis hash is invalid")
    for phase, expected_marker_hash in markers.items():
        marker_path = path.parent / "phases" / str(phase) / "success.json"
        if not marker_path.is_file() or sha256_file(marker_path) != expected_marker_hash:
            raise ValueError(f"healthy success marker changed after lock: {phase}")
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        marker_artifacts = marker.get("artifacts")
        if not isinstance(marker_artifacts, list) or not marker_artifacts:
            raise ValueError(f"healthy success marker has no artifacts: {phase}")
        observed_artifacts: dict[str, str] = {}
        for item in marker_artifacts:
            if not isinstance(item, dict):
                raise ValueError(f"invalid healthy artifact entry: {phase}")
            artifact_path = Path(str(item.get("path", "")))
            expected_hash = item.get("sha256")
            expected_size = item.get("size")
            if (
                not artifact_path.is_file()
                or artifact_path.stat().st_size != expected_size
                or sha256_file(artifact_path) != expected_hash
            ):
                raise ValueError(f"healthy artifact changed after lock: {artifact_path}")
            observed_artifacts[str(artifact_path)] = str(expected_hash)
        if artifacts.get(phase) != observed_artifacts:
            raise ValueError(f"clinical lock artifact inventory differs for {phase}")
    return payload


def _association_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    participant = frame.groupby("participant_id", as_index=False).agg(
        {
            **{axis: "mean" for axis in AXIS_NAMES},
            **({"crs_r_total": "first"} if "crs_r_total" in frame else {}),
            **({"diagnosis": "first"} if "diagnosis" in frame else {}),
            "dataset_id": "first",
        }
    )
    if "crs_r_total" in participant:
        score = pd.to_numeric(participant["crs_r_total"], errors="coerce")
        for axis in AXIS_NAMES:
            mask = score.notna() & participant[axis].notna()
            if mask.sum() >= 5:
                coefficient, p_value = stats.spearmanr(
                    score[mask].to_numpy(), participant.loc[mask, axis].to_numpy()
                )
                rows.append(
                    {
                        "endpoint": "crs_r_total",
                        "axis": axis,
                        "test": "spearman_participant_level",
                        "estimate": float(coefficient),
                        "p_value": float(p_value),
                        "n_participants": int(mask.sum()),
                    }
                )
    if "diagnosis" in participant:
        diagnosis = participant["diagnosis"].dropna().astype(str)
        groups = sorted(diagnosis.unique())
        if len(groups) >= 2:
            for axis in AXIS_NAMES:
                values = [
                    participant.loc[participant["diagnosis"].astype(str) == group, axis]
                    .dropna()
                    .to_numpy()
                    for group in groups
                ]
                if all(len(value) >= 2 for value in values):
                    statistic, p_value = stats.kruskal(*values)
                    rows.append(
                        {
                            "endpoint": "diagnosis",
                            "axis": axis,
                            "test": "kruskal_participant_level",
                            "estimate": float(statistic),
                            "p_value": float(p_value),
                            "n_participants": int(sum(map(len, values))),
                            "groups": "|".join(groups),
                        }
                    )
    return rows


def run_clinical_transfer(
    *,
    encoding_manifest: str | Path,
    state_dictionary_path: str | Path,
    profile_estimator_path: str | Path,
    clinical_lock_path: str | Path,
    output_root: str | Path,
    study: StudyConfig,
) -> tuple[Path, Path, Path]:
    """Apply frozen healthy objects without refitting or individual diagnosis."""

    if any(
        (
            study.clinical_transfer.retrain_representation,
            study.clinical_transfer.retrain_scaler,
            study.clinical_transfer.retrain_state_dictionary,
            study.clinical_transfer.individual_diagnostic_reclassification,
        )
    ):
        raise ValueError("clinical transfer configuration is not locked")
    lock_path = Path(clinical_lock_path).resolve(strict=True)
    lock = _validate_lock(lock_path)
    dictionary_path = Path(state_dictionary_path).resolve(strict=True)
    estimator_path = Path(profile_estimator_path).resolve(strict=True)
    metrics_artifacts = lock["healthy_validated_artifacts"].get("metrics")
    if not isinstance(metrics_artifacts, dict):
        raise ValueError("clinical lock has no validated metrics artifacts")
    for frozen_path in (dictionary_path, estimator_path):
        expected = metrics_artifacts.get(str(frozen_path))
        if not isinstance(expected, str) or sha256_file(frozen_path) != expected:
            raise ValueError(
                f"frozen clinical object is not bound to the metrics receipt: {frozen_path}"
            )
    dictionary = joblib.load(dictionary_path)
    estimator = joblib.load(estimator_path)
    if not isinstance(estimator, FiveAxisProfileEstimator):
        raise TypeError("profile estimator artifact has the wrong type")
    frame = pd.read_parquet(encoding_manifest)
    if "clinical_holdout" in frame:
        selected = frame[frame["clinical_holdout"].fillna(False).astype(bool)]
    else:
        selected = frame[frame["dataset_id"].isin(["doc_resting_eeg", "doc_polysomnography"])]
    selected = selected[selected["encoded"].astype(bool)]
    if selected.empty:
        raise RuntimeError("no encoded held-out clinical units are available")
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for raw in selected.to_dict(orient="records"):
        try:
            unit = _load_unit(raw)
            profile = estimator.profile(_record(unit, dictionary))
            row = {
                **unit.metadata,
                "unit_id": unit.unit_id,
                "participant_id": unit.participant_id,
                "dataset_id": unit.dataset_id,
                "transfer_status": "frozen",
            }
            for index, axis in enumerate(AXIS_NAMES):
                row[axis] = float(profile.values[index])
                row[f"{axis}_raw"] = float(profile.raw_values[index])
            if "diagnosis" not in row:
                condition = row.get("condition")
                row["diagnosis"] = (
                    None if condition in {None, "unresolved_clinical_group"} else str(condition)
                )
            if "crs_r_total" not in row and "crs_r" in row:
                row["crs_r_total"] = row.get("crs_r")
            row["regime_preservation_score"] = float(np.mean([row[axis] for axis in AXIS_NAMES]))
            rows.append(row)
        except (ValueError, RuntimeError, OSError, np.linalg.LinAlgError) as error:
            failures.append(
                {
                    "unit_id": raw.get("unit_id", raw.get("recording_id")),
                    "error": f"{type(error).__name__}: {error}",
                }
            )
    if not rows:
        raise RuntimeError("all clinical transfer units failed")
    destination = Path(output_root)
    profiles_path = _atomic_parquet(pd.DataFrame(rows), destination / "clinical-profiles.parquet")
    associations_path = _atomic_parquet(
        pd.DataFrame(_association_rows(pd.DataFrame(rows))),
        destination / "clinical-associations.parquet",
    )
    audit_path = destination / "clinical-transfer-audit.json"
    atomic_write_json(
        audit_path,
        {
            "schema_version": 1,
            "technical_lock": str(lock_path),
            "technical_lock_sha256": sha256_file(lock_path),
            "technical_lock_basis_sha256": lock.get("basis_sha256"),
            "state_dictionary_sha256": sha256_file(state_dictionary_path),
            "profile_estimator_sha256": sha256_file(profile_estimator_path),
            "units_selected": len(selected),
            "units_transferred": len(rows),
            "failures": failures,
            "representation_refit": False,
            "scaler_refit": False,
            "state_dictionary_refit": False,
            "individual_diagnostic_reclassification": False,
            "project_status": "exploratory_non_preregistered",
            "scientific_gate_applied": False,
        },
    )
    return profiles_path, associations_path, audit_path
