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
from neural_manifolds.manifold.clinical_reference import (
    WAKE_REGIME_LLR,
    FrozenWakePropofolLikelihoodRatio,
)
from neural_manifolds.manifold.profile import AXIS_NAMES, FiveAxisProfileEstimator
from neural_manifolds.provenance import atomic_write_json, sha256_file
from neural_manifolds.stage_units import CLINICAL_LOW_CHANNEL_BRANCH
from neural_manifolds.stages.metrics import _load_unit, _record
from neural_manifolds.statistics.multivariate import benjamini_hochberg


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


def _property_scope_from_metadata(metadata: dict[str, Any]) -> dict[str, str]:
    """Resolve per-axis clinical support and fail closed for the sparse PSG branch."""

    branch = str(metadata.get("analysis_branch", ""))
    encoded = metadata.get("property_scope_json")
    if not isinstance(encoded, str) or not encoded.strip():
        if branch == CLINICAL_LOW_CHANNEL_BRANCH:
            raise ValueError("clinical low-channel unit lacks explicit property-scope metadata")
        return {axis: "available_frozen_healthy_axis" for axis in AXIS_NAMES}
    try:
        raw_scope = json.loads(encoded)
    except json.JSONDecodeError as error:
        raise ValueError("clinical property-scope metadata is not valid JSON") from error
    if not isinstance(raw_scope, dict):
        raise ValueError("clinical property-scope metadata must be an object")
    missing = set(AXIS_NAMES).difference(raw_scope)
    if missing:
        raise ValueError(f"clinical property-scope metadata lacks axes {sorted(missing)}")
    scope = {axis: raw_scope[axis] for axis in AXIS_NAMES}
    if any(not isinstance(status, str) or not status for status in scope.values()):
        raise ValueError("clinical property-scope statuses must be non-empty strings")
    return scope


def validate_clinical_lock(path: str | Path) -> dict[str, Any]:
    """Validate the immutable healthy-workflow snapshot before clinical signal access."""

    return _validate_lock(Path(path).resolve(strict=True))


def _association_rng(seed: int, *parts: object) -> np.random.Generator:
    payload = json.dumps([seed, *parts], ensure_ascii=True, separators=(",", ":"))
    derived = int(hashlib.sha256(payload.encode()).hexdigest()[:16], 16) % (2**32)
    return np.random.default_rng(derived)


def _interval(values: list[float]) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if len(finite) < 2:
        return float("nan"), float("nan")
    low, high = np.quantile(finite, [0.025, 0.975])
    return float(low), float(high)


def _epsilon_squared(statistic: float, n: int, groups: int) -> float:
    if n <= groups:
        return float("nan")
    return float(max(0.0, (statistic - groups + 1) / (n - groups)))


def _association_rows(
    frame: pd.DataFrame,
    *,
    bootstrap_repetitions: int,
    permutation_repetitions: int,
    seed: int,
    fdr_alpha: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    participant = frame.groupby(["dataset_id", "participant_id"], as_index=False).agg(
        {
            **{axis: "mean" for axis in AXIS_NAMES},
            WAKE_REGIME_LLR: "mean",
            **({"crs_r_total": "first"} if "crs_r_total" in frame else {}),
            **({"diagnosis": "first"} if "diagnosis" in frame else {}),
        }
    )
    for dataset_id, dataset in participant.groupby("dataset_id", sort=True):
        if "crs_r_total" in dataset:
            score = pd.to_numeric(dataset["crs_r_total"], errors="coerce")
            for metric in (*AXIS_NAMES, WAKE_REGIME_LLR):
                mask = score.notna() & dataset[metric].notna()
                if mask.sum() >= 5:
                    x = score[mask].to_numpy(dtype=float)
                    y = dataset.loc[mask, metric].to_numpy(dtype=float)
                    if np.ptp(x) == 0 or np.ptp(y) == 0:
                        continue
                    coefficient = float(stats.spearmanr(x, y).statistic)
                    rng = _association_rng(seed, dataset_id, "crs_r_total", metric)
                    bootstrap: list[float] = []
                    for _ in range(bootstrap_repetitions):
                        indices = rng.integers(0, len(x), size=len(x))
                        if np.ptp(x[indices]) > 0 and np.ptp(y[indices]) > 0:
                            bootstrap.append(
                                float(stats.spearmanr(x[indices], y[indices]).statistic)
                            )
                    ci_low, ci_high = _interval(bootstrap)
                    extreme = 0
                    valid_permutations = 0
                    for _ in range(permutation_repetitions):
                        permuted = rng.permutation(x)
                        if np.ptp(permuted) == 0 or np.ptp(y) == 0:
                            continue
                        null = float(stats.spearmanr(permuted, y).statistic)
                        valid_permutations += 1
                        extreme += int(abs(null) >= abs(coefficient))
                    p_value = float((extreme + 1) / (valid_permutations + 1))
                    rows.append(
                        {
                            "dataset_id": dataset_id,
                            "endpoint": "crs_r_total",
                            "metric": metric,
                            "test": "spearman_participant_level",
                            "estimate": coefficient,
                            "ci_low": ci_low,
                            "ci_high": ci_high,
                            "p_value": p_value,
                            "p_value_method": "participant_label_permutation_plus_one",
                            "bootstrap_repetitions": bootstrap_repetitions,
                            "bootstrap_valid_repetitions": len(bootstrap),
                            "permutation_repetitions": valid_permutations,
                            "n_participants": int(mask.sum()),
                        }
                    )
        if "diagnosis" in dataset:
            diagnosis = dataset["diagnosis"].dropna().astype(str)
            groups = sorted(diagnosis.unique())
            if len(groups) >= 2:
                for metric in (*AXIS_NAMES, WAKE_REGIME_LLR):
                    values = [
                        dataset.loc[dataset["diagnosis"].astype(str) == group, metric]
                        .dropna()
                        .to_numpy()
                        for group in groups
                    ]
                    if all(len(value) >= 2 for value in values):
                        try:
                            statistic = float(stats.kruskal(*values).statistic)
                        except ValueError:
                            continue
                        n_participants = int(sum(map(len, values)))
                        effect = _epsilon_squared(statistic, n_participants, len(values))
                        rng = _association_rng(seed, dataset_id, "diagnosis", metric)
                        bootstrap_effects: list[float] = []
                        for _ in range(bootstrap_repetitions):
                            sampled = [
                                value[rng.integers(0, len(value), size=len(value))]
                                for value in values
                            ]
                            try:
                                sampled_statistic = float(stats.kruskal(*sampled).statistic)
                            except ValueError:
                                continue
                            bootstrap_effects.append(
                                _epsilon_squared(
                                    sampled_statistic,
                                    n_participants,
                                    len(values),
                                )
                            )
                        ci_low, ci_high = _interval(bootstrap_effects)
                        pooled = np.concatenate(values)
                        labels = np.concatenate(
                            [np.full(len(value), index) for index, value in enumerate(values)]
                        )
                        extreme = 0
                        for _ in range(permutation_repetitions):
                            permuted = rng.permutation(labels)
                            null_values = [
                                pooled[permuted == index] for index in range(len(values))
                            ]
                            try:
                                null_statistic = float(stats.kruskal(*null_values).statistic)
                            except ValueError:
                                continue
                            extreme += int(null_statistic >= statistic)
                        p_value = float((extreme + 1) / (permutation_repetitions + 1))
                        rows.append(
                            {
                                "dataset_id": dataset_id,
                                "endpoint": "diagnosis",
                                "metric": metric,
                                "test": "kruskal_epsilon_squared_participant_level",
                                "estimate": effect,
                                "test_statistic": statistic,
                                "ci_low": ci_low,
                                "ci_high": ci_high,
                                "p_value": p_value,
                                "p_value_method": "participant_label_permutation_plus_one",
                                "bootstrap_repetitions": bootstrap_repetitions,
                                "bootstrap_valid_repetitions": len(bootstrap_effects),
                                "permutation_repetitions": permutation_repetitions,
                                "n_participants": n_participants,
                                "groups": "|".join(groups),
                            }
                        )
    result = pd.DataFrame(rows)
    if result.empty:
        return []
    result["p_value_fdr"] = np.nan
    result["fdr_reject"] = False
    for _, indices in result.groupby(["dataset_id", "endpoint"], sort=True).groups.items():
        positions = list(indices)
        adjusted, rejected = benjamini_hochberg(
            result.loc[positions, "p_value"].to_numpy(dtype=float),
            alpha=fdr_alpha,
        )
        result.loc[positions, "p_value_fdr"] = adjusted
        result.loc[positions, "fdr_reject"] = rejected
    result["fdr_alpha"] = fdr_alpha
    result["scientific_gate_applied"] = False
    result = result.astype(object).where(pd.notna(result), None)
    return result.to_dict(orient="records")


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
    lock = validate_clinical_lock(lock_path)
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
    clinical_reference = getattr(estimator, "wake_propofol_reference_", None)
    if not isinstance(clinical_reference, FrozenWakePropofolLikelihoodRatio):
        raise ValueError(
            "frozen healthy wake-versus-propofol likelihood-ratio reference is unavailable"
        )
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
            scope = _property_scope_from_metadata(row)
            unavailable_axes: list[str] = []
            for index, axis in enumerate(AXIS_NAMES):
                status = scope[axis]
                row[f"{axis}_status"] = status
                if status.startswith("unavailable"):
                    row[axis] = np.nan
                    row[f"{axis}_raw"] = np.nan
                    unavailable_axes.append(axis)
                else:
                    row[axis] = float(profile.values[index])
                    row[f"{axis}_raw"] = float(profile.raw_values[index])
            row["clinical_metric_branch"] = (
                "clinical_low_channel_supported_property_transfer"
                if row.get("analysis_branch") == CLINICAL_LOW_CHANNEL_BRANCH
                else "frozen_five_axis_transfer"
            )
            if "diagnosis" not in row:
                condition = row.get("condition")
                row["diagnosis"] = (
                    None if condition in {None, "unresolved_clinical_group"} else str(condition)
                )
            if "crs_r_total" not in row and "crs_r" in row:
                row["crs_r_total"] = row.get("crs_r")
            if unavailable_axes:
                row[WAKE_REGIME_LLR] = np.nan
                row["wake_regime_score_status"] = (
                    "unavailable_incomplete_five_axis_profile:" + "|".join(unavailable_axes)
                )
                row["wake_regime_score_interpretation"] = (
                    "not_computed_when_any_frozen_axis_is_unavailable"
                )
            else:
                row[WAKE_REGIME_LLR] = clinical_reference.score(profile.values)
                row["wake_regime_score_status"] = (
                    "available_frozen_healthy_wake_vs_propofol_reference"
                )
                row["wake_regime_score_interpretation"] = (
                    "positive_is_more_wake_like_not_probability_of_consciousness"
                )
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
        pd.DataFrame(
            _association_rows(
                pd.DataFrame(rows),
                bootstrap_repetitions=study.statistics.participant_bootstrap_repetitions,
                permutation_repetitions=study.statistics.permutation_repetitions,
                seed=study.random_seeds[0],
                fdr_alpha=study.statistics.false_discovery_rate,
            )
        ),
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
            "wake_propofol_reference": clinical_reference.audit(),
            "secondary_clinical_similarity_field": WAKE_REGIME_LLR,
            "clinical_metric_branch_counts": pd.Series(
                [row["clinical_metric_branch"] for row in rows]
            )
            .value_counts()
            .sort_index()
            .to_dict(),
            "units_selected": len(selected),
            "units_transferred": len(rows),
            "failures": failures,
            "representation_refit": False,
            "scaler_refit": False,
            "state_dictionary_refit": False,
            "wake_propofol_reference_refit": False,
            "individual_diagnostic_reclassification": False,
            "association_inference": {
                "unit": "participant",
                "bootstrap_repetitions": study.statistics.participant_bootstrap_repetitions,
                "permutation_repetitions": study.statistics.permutation_repetitions,
                "permutation_p_value": "plus_one",
                "multiple_comparison_correction": "benjamini_hochberg_within_dataset_endpoint",
                "false_discovery_rate": study.statistics.false_discovery_rate,
                "fdr_controls_phase_execution": False,
            },
            "project_status": "exploratory_non_preregistered",
            "scientific_gate_applied": False,
        },
    )
    return profiles_path, associations_path, audit_path
