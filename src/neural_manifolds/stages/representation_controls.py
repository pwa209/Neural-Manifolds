"""Audited representation controls and participant-safe dataset diagnostics.

This stage never creates substitute coordinate trajectories.  It reports which
branches were actually materialised under a hash-pinned backend and evaluates
dataset identity only from scalar participant-condition cells.  Participant
identifiers and row-level predictions remain in memory and are not published.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from neural_manifolds.config import StudyConfig, config_sha256
from neural_manifolds.manifold.profile import AXIS_NAMES
from neural_manifolds.provenance import atomic_write_json, sha256_file
from neural_manifolds.stages.benchmarks import CONVENTIONAL_FEATURES
from neural_manifolds.statistics.folds import (
    maximum_participant_stratified_splits,
    participant_stratified_test_sets,
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_REVISION = re.compile(r"^[0-9a-f]{40}$")
_ENCODED_UNIT_MARKER = "neural_manifolds.encoded_unit.v1"
_ENCODED_EVENT_GROUP_MARKER = "neural_manifolds.encoded_event_group.v1"
_PRIVATE_OUTPUT_FIELDS = frozenset(
    {
        "participant_id",
        "participant_key",
        "unit_id",
        "trajectory_path",
        "preprocessed_path",
        "source_path",
    }
)


def _atomic_parquet(frame: pd.DataFrame, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, destination)
    return destination


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def _set_hash(values: Sequence[str]) -> str:
    return _canonical_hash(sorted(str(value) for value in values))


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read valid {label}: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return payload


def _load_models(path: Path) -> dict[str, dict[str, Any]]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f"cannot read model configuration: {path}") from error
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("model configuration must use schema_version 1")
    models = payload.get("models")
    if not isinstance(models, dict) or not models:
        raise ValueError("model configuration has no models mapping")
    if any(
        not isinstance(key, str) or not isinstance(value, dict) for key, value in models.items()
    ):
        raise ValueError("model configuration contains a malformed model entry")
    return models


def _validate_encoding_receipt(
    *,
    receipt_path: Path,
    expected_receipt_sha256: str,
    expected_trajectory_sha256: str,
    expected_representation_sha256: str,
    expected_study_sha256: str,
    expected_checkpoint_sha256: str,
    expected_model_fingerprint_sha256: str,
) -> list[str]:
    if receipt_path.is_symlink() or not receipt_path.is_file():
        raise ValueError(f"encoding receipt is missing or unsafe: {receipt_path}")
    observed_receipt_sha256 = sha256_file(receipt_path)
    if observed_receipt_sha256 != expected_receipt_sha256:
        raise ValueError(f"encoding receipt checksum changed: {receipt_path}")
    payload = _load_json_object(receipt_path, label="encoding receipt")
    output = payload.get("output")
    if not isinstance(output, dict) or output.get("sha256") != expected_trajectory_sha256:
        raise ValueError("encoding receipt does not bind its declared trajectory")
    inputs = payload.get("inputs")
    if not isinstance(inputs, dict):
        raise ValueError("encoding receipt has no input fingerprint mapping")
    marker = payload.get("provenance_marker")
    if marker == _ENCODED_UNIT_MARKER:
        if inputs.get("representation_config_sha256") != expected_representation_sha256:
            raise ValueError("encoding receipt representation configuration changed")
        if inputs.get("study_config_sha256") != expected_study_sha256:
            raise ValueError("encoding receipt study configuration changed")
        model = inputs.get("model")
        if not isinstance(model, dict) or model.get("checkpoint_sha256") != (
            expected_checkpoint_sha256
        ):
            raise ValueError("encoding receipt is not bound to the runtime checkpoint")
        if _canonical_hash(model) != expected_model_fingerprint_sha256:
            raise ValueError("encoding receipt model fingerprint differs from the encoding flow")
        return [observed_receipt_sha256]
    if marker != _ENCODED_EVENT_GROUP_MARKER:
        raise ValueError("encoding receipt has an unknown provenance marker")
    trials = inputs.get("trials")
    if not isinstance(trials, list) or not trials:
        raise ValueError("event-group encoding receipt has no nested trial inventory")
    nested = [observed_receipt_sha256]
    for trial in trials:
        if not isinstance(trial, dict):
            raise ValueError("event-group encoding receipt has a malformed trial entry")
        nested_path = trial.get("encoding_provenance_path")
        nested_receipt_hash = str(trial.get("encoding_provenance_sha256", "")).lower()
        nested_trajectory_hash = str(trial.get("trajectory_sha256", "")).lower()
        if (
            not isinstance(nested_path, str)
            or not _SHA256.fullmatch(nested_receipt_hash)
            or not _SHA256.fullmatch(nested_trajectory_hash)
        ):
            raise ValueError("event-group encoding receipt has incomplete trial hashes")
        nested.extend(
            _validate_encoding_receipt(
                receipt_path=Path(nested_path),
                expected_receipt_sha256=nested_receipt_hash,
                expected_trajectory_sha256=nested_trajectory_hash,
                expected_representation_sha256=expected_representation_sha256,
                expected_study_sha256=expected_study_sha256,
                expected_checkpoint_sha256=expected_checkpoint_sha256,
                expected_model_fingerprint_sha256=expected_model_fingerprint_sha256,
            )
        )
    return nested


def _trajectory_inventory(
    frame: pd.DataFrame,
    *,
    study: StudyConfig,
    checkpoint_sha256: str,
    model_fingerprint_sha256: str,
) -> dict[str, Any]:
    required = {
        "encoded",
        "trajectory_path",
        "trajectory_sha256",
        "encoding_provenance_path",
        "encoding_provenance_sha256",
    }
    missing = required.difference(frame.columns)
    if missing:
        return {
            "status": "unavailable",
            "reason": f"encoding_manifest_missing_fields:{','.join(sorted(missing))}",
            "trajectory_count": 0,
            "trajectory_inventory_sha256": None,
        }
    encoded = frame[frame["encoded"].fillna(False).astype(bool)]
    if encoded.empty:
        return {
            "status": "unavailable",
            "reason": "encoding_manifest_has_no_encoded_trajectories",
            "trajectory_count": 0,
            "trajectory_inventory_sha256": None,
        }
    if not _SHA256.fullmatch(checkpoint_sha256) or not _SHA256.fullmatch(model_fingerprint_sha256):
        return {
            "status": "unavailable",
            "reason": "encoding_flow_lacks_exact_checkpoint_or_model_fingerprint_sha256",
            "trajectory_count": 0,
            "trajectory_inventory_sha256": None,
            "encoding_receipt_count": 0,
            "encoding_receipt_inventory_sha256": None,
            "representation_config_sha256": None,
        }
    expected_representation_sha256 = _canonical_hash(study.representation.model_dump(mode="json"))
    expected_study_sha256 = config_sha256(study)
    observed: list[str] = []
    receipt_hashes: list[str] = []
    for row in encoded.to_dict(orient="records"):
        expected = str(row.get("trajectory_sha256", "")).lower()
        if not _SHA256.fullmatch(expected):
            raise ValueError("encoded trajectory has no valid SHA-256")
        raw_path = row.get("trajectory_path")
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError("encoded trajectory has no path")
        path = Path(raw_path)
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"encoded trajectory is missing or unsafe: {path}")
        if sha256_file(path) != expected:
            raise ValueError(f"encoded trajectory checksum changed: {path}")
        receipt_value = row.get("encoding_provenance_path")
        receipt_sha256 = str(row.get("encoding_provenance_sha256", "")).lower()
        if not isinstance(receipt_value, str) or not _SHA256.fullmatch(receipt_sha256):
            raise ValueError("encoded trajectory has no valid provenance receipt fingerprint")
        receipt_hashes.extend(
            _validate_encoding_receipt(
                receipt_path=Path(receipt_value),
                expected_receipt_sha256=receipt_sha256,
                expected_trajectory_sha256=expected,
                expected_representation_sha256=expected_representation_sha256,
                expected_study_sha256=expected_study_sha256,
                expected_checkpoint_sha256=checkpoint_sha256,
                expected_model_fingerprint_sha256=model_fingerprint_sha256,
            )
        )
        observed.append(expected)
    return {
        "status": "available",
        "reason": None,
        "trajectory_count": len(observed),
        "trajectory_inventory_sha256": _set_hash(observed),
        "encoding_receipt_count": len(receipt_hashes),
        "encoding_receipt_inventory_sha256": _set_hash(receipt_hashes),
        "representation_config_sha256": expected_representation_sha256,
    }


def _availability_rows(
    *,
    study: StudyConfig,
    models: Mapping[str, Mapping[str, Any]],
    encoding_flow: Mapping[str, Any],
    trajectory_inventory: Mapping[str, Any],
    input_hashes: Mapping[str, str],
) -> list[dict[str, Any]]:
    primary_id = study.representation.primary
    primary = models.get(primary_id)
    checkpoint_sha256 = str(encoding_flow.get("checkpoint_sha256", "")).lower()
    fingerprint_sha256 = str(encoding_flow.get("model_fingerprint_sha256", "")).lower()
    revision = str(primary.get("revision", "")).lower() if primary is not None else ""
    factory = str(primary.get("factory", "")) if primary is not None else ""
    exact_primary = bool(
        primary is not None
        and primary.get("trainable") is False
        and study.representation.weights_frozen
        and study.representation.label_free
        and _GIT_REVISION.fullmatch(revision)
        and factory
        and _SHA256.fullmatch(checkpoint_sha256)
        and _SHA256.fullmatch(fingerprint_sha256)
        and trajectory_inventory.get("status") == "available"
    )
    if exact_primary:
        primary_reason = "hash_pinned_backend_and_trajectory_inventory_verified"
    elif primary is None:
        primary_reason = "unavailable_primary_model_absent_from_models_config"
    elif trajectory_inventory.get("status") != "available":
        primary_reason = str(trajectory_inventory.get("reason"))
    else:
        primary_reason = "unavailable_primary_backend_or_runtime_hash_not_exactly_pinned"

    shared = {
        "primary_model_id": primary_id,
        "checkpoint_sha256": checkpoint_sha256 if _SHA256.fullmatch(checkpoint_sha256) else None,
        "model_fingerprint_sha256": (
            fingerprint_sha256 if _SHA256.fullmatch(fingerprint_sha256) else None
        ),
        "source_revision": revision if _GIT_REVISION.fullmatch(revision) else None,
        "factory": factory or None,
        "trajectory_inventory_sha256": trajectory_inventory.get("trajectory_inventory_sha256"),
        "trajectory_count": int(trajectory_inventory.get("trajectory_count", 0)),
        "encoding_receipt_inventory_sha256": trajectory_inventory.get(
            "encoding_receipt_inventory_sha256"
        ),
        "representation_config_sha256": trajectory_inventory.get("representation_config_sha256"),
        "input_artifact_hashes_sha256": _canonical_hash(dict(input_hashes)),
        "scientific_gate_applied": False,
        "result_selection_applied": False,
    }
    comparator_reference = json.dumps(
        [
            "relative_band_power",
            "spectral_exponent",
            "permutation_entropy",
            "normalized_lempel_ziv",
            "weighted_phase_lag_index",
            "weighted_symbolic_mutual_information",
        ],
        separators=(",", ":"),
    )
    rows = [
        {
            **shared,
            "control_id": "primary_layer_pooling",
            "control_family": "primary_coordinate_trajectory",
            "requested_layer": study.representation.layer,
            "requested_pooling": study.representation.pooling,
            "requested_checkpoint_size": (
                str(primary.get("parameters_reported"))
                if primary and primary.get("parameters_reported") is not None
                else None
            ),
            "status": "available" if exact_primary else "unavailable",
            "reason": primary_reason,
            "exact_hash_pinned_backend_configured": exact_primary,
            "trajectory_generated": exact_primary,
            "implemented_comparator_reference_json": "[]",
        },
        {
            **shared,
            "control_id": "secondary_pooling",
            "control_family": "pooling_sensitivity",
            "requested_layer": study.representation.layer,
            "requested_pooling": study.representation.secondary_pooling,
            "requested_checkpoint_size": (
                str(primary.get("parameters_reported"))
                if primary and primary.get("parameters_reported") is not None
                else None
            ),
            "status": "unavailable",
            "reason": "unavailable_no_hash_pinned_backend_branch_for_secondary_pooling",
            "exact_hash_pinned_backend_configured": False,
            "trajectory_generated": False,
            "implemented_comparator_reference_json": "[]",
        },
        {
            **shared,
            "control_id": "non_primary_layer_sensitivity",
            "control_family": "layer_sensitivity",
            "requested_layer": "configured_non_primary_layers",
            "requested_pooling": study.representation.pooling,
            "requested_checkpoint_size": (
                str(primary.get("parameters_reported"))
                if primary and primary.get("parameters_reported") is not None
                else None
            ),
            "status": "unavailable",
            "reason": "unavailable_no_hash_pinned_non_primary_layer_extraction_backend",
            "exact_hash_pinned_backend_configured": False,
            "trajectory_generated": False,
            "implemented_comparator_reference_json": "[]",
        },
        {
            **shared,
            "control_id": "checkpoint_size_sensitivity",
            "control_family": "checkpoint_size_sensitivity",
            "requested_layer": study.representation.layer,
            "requested_pooling": study.representation.pooling,
            "requested_checkpoint_size": "alternate_parameter_counts",
            "status": "unavailable",
            "reason": "unavailable_no_exact_alternate_labram_checkpoint_backend_and_trajectory",
            "exact_hash_pinned_backend_configured": False,
            "trajectory_generated": False,
            "implemented_comparator_reference_json": "[]",
        },
        {
            **shared,
            "control_id": "pca_coordinate_control",
            "control_family": "classical_coordinate_control",
            "requested_layer": None,
            "requested_pooling": None,
            "requested_checkpoint_size": None,
            "status": "unavailable",
            "reason": "unavailable_no_leakage_safe_full_coordinate_trajectory_backend",
            "exact_hash_pinned_backend_configured": False,
            "trajectory_generated": False,
            "implemented_comparator_reference_json": comparator_reference,
        },
        {
            **shared,
            "control_id": "time_frequency_coordinate_control",
            "control_family": "time_frequency_coordinate_control",
            "requested_layer": None,
            "requested_pooling": None,
            "requested_checkpoint_size": None,
            "status": "unavailable",
            "reason": "unavailable_no_full_time_frequency_coordinate_trajectory_backend",
            "exact_hash_pinned_backend_configured": False,
            "trajectory_generated": False,
            "implemented_comparator_reference_json": comparator_reference,
        },
    ]
    return rows


def _identity_text(value: Any) -> str:
    if value is None:
        return "__missing__"
    try:
        if bool(pd.isna(value)):
            return "__missing__"
    except (TypeError, ValueError):
        pass
    return str(value)


def _required_identity_text(value: Any, *, field: str) -> str:
    output = _identity_text(value)
    if output in {"", "__missing__"}:
        raise ValueError(f"participant-condition table has a missing {field}")
    return output


def _cell_key(dataset_id: Any, participant_id: Any, condition: Any) -> str:
    return _canonical_hash(
        [_identity_text(dataset_id), _identity_text(participant_id), _identity_text(condition)]
    )


def _profile_cells(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"dataset_id", "participant_id", "condition", *AXIS_NAMES}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"profile table is missing {sorted(missing)}")
    working = frame.loc[:, list(required)].copy()
    working["dataset_id"] = working["dataset_id"].map(
        lambda value: _required_identity_text(value, field="dataset_id")
    )
    working["participant_id"] = working["participant_id"].map(
        lambda value: _required_identity_text(value, field="participant_id")
    )
    working["condition"] = working["condition"].map(_identity_text)
    for feature in AXIS_NAMES:
        working[feature] = pd.to_numeric(working[feature], errors="coerce")
    grouped = (
        working.groupby(["dataset_id", "participant_id", "condition"], sort=True, dropna=False)[
            list(AXIS_NAMES)
        ]
        .mean()
        .reset_index()
    )
    grouped["participant_key"] = (
        grouped["dataset_id"].astype(str) + "\x1f" + grouped["participant_id"].astype(str)
    )
    grouped["cell_key_sha256"] = [
        _cell_key(row.dataset_id, row.participant_id, row.condition)
        for row in grouped.itertuples(index=False)
    ]
    if grouped["cell_key_sha256"].duplicated().any():
        raise ValueError("profile participant-condition cells are not unique after aggregation")
    return grouped


def _conventional_cells(
    frame: pd.DataFrame, profile_cells: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, Any]]:
    required = {
        "dataset_id",
        "participant_id",
        "condition",
        "benchmark_status",
        "benchmark_cell_expected",
        "participant_condition_cell_key_sha256",
        *CONVENTIONAL_FEATURES,
    }
    missing = required.difference(frame.columns)
    if missing:
        return pd.DataFrame(), {
            "status": "unavailable",
            "reason": f"benchmark_table_missing_fields:{','.join(sorted(missing))}",
            "complete_cells": 0,
            "incomplete_cells": 0,
            "matched_cells": 0,
            "excluded_profile_cells": len(profile_cells),
            "matched_cell_inventory_sha256": None,
        }
    expected = frame[frame["benchmark_cell_expected"].fillna(False).astype(bool)].copy()
    if expected.empty:
        return pd.DataFrame(), {
            "status": "unavailable",
            "reason": "benchmark_table_has_no_expected_participant_condition_cells",
            "complete_cells": 0,
            "incomplete_cells": 0,
            "matched_cells": 0,
            "excluded_profile_cells": len(profile_cells),
            "matched_cell_inventory_sha256": None,
        }
    expected["dataset_id"] = expected["dataset_id"].map(
        lambda value: _required_identity_text(value, field="dataset_id")
    )
    expected["participant_id"] = expected["participant_id"].map(
        lambda value: _required_identity_text(value, field="participant_id")
    )
    expected["condition"] = expected["condition"].map(_identity_text)
    derived_keys = [
        _cell_key(row.dataset_id, row.participant_id, row.condition)
        for row in expected.itertuples(index=False)
    ]
    declared_keys = expected["participant_condition_cell_key_sha256"].astype(str).tolist()
    if derived_keys != declared_keys:
        raise ValueError("benchmark participant-condition cell hash does not match its identity")
    for feature in CONVENTIONAL_FEATURES:
        expected[feature] = pd.to_numeric(expected[feature], errors="coerce")
    complete_keys: list[str] = []
    incomplete_keys: list[str] = []
    rows: list[dict[str, Any]] = []
    for key, group in expected.groupby("participant_condition_cell_key_sha256", sort=True):
        complete = bool(
            group["benchmark_status"].astype(str).eq("computed").all()
            and np.isfinite(group[list(CONVENTIONAL_FEATURES)].to_numpy(dtype=float)).all()
        )
        if not complete:
            incomplete_keys.append(str(key))
            continue
        complete_keys.append(str(key))
        first = group.iloc[0]
        rows.append(
            {
                "dataset_id": str(first["dataset_id"]),
                "participant_id": str(first["participant_id"]),
                "condition": str(first["condition"]),
                "participant_key": f"{first['dataset_id']}\x1f{first['participant_id']}",
                "cell_key_sha256": str(key),
                **{feature: float(group[feature].mean()) for feature in CONVENTIONAL_FEATURES},
            }
        )
    output = pd.DataFrame(rows)
    profile_keys = set(profile_cells["cell_key_sha256"].astype(str))
    complete_set = set(complete_keys)
    extra_complete = complete_set.difference(profile_keys)
    if extra_complete:
        return pd.DataFrame(), {
            "status": "unavailable",
            "reason": "complete_benchmark_cells_absent_from_five_axis_profiles",
            "complete_cells": len(complete_keys),
            "incomplete_cells": len(incomplete_keys),
            "matched_cells": 0,
            "excluded_profile_cells": len(profile_cells),
            "matched_cell_inventory_sha256": None,
        }
    matched_set = complete_set.intersection(profile_keys)
    if not output.empty:
        output = output[output["cell_key_sha256"].isin(matched_set)].copy()
    return output, {
        "status": "available" if matched_set else "unavailable",
        "reason": None if matched_set else "no_complete_exact_benchmark_profile_cells",
        "complete_cells": len(complete_keys),
        "incomplete_cells": len(incomplete_keys),
        "matched_cells": len(matched_set),
        "excluded_profile_cells": len(profile_keys - matched_set),
        "matched_cell_inventory_sha256": _set_hash(sorted(matched_set)) if matched_set else None,
    }


def _participant_row_weights(
    participant_keys: np.ndarray,
    labels: np.ndarray,
    *,
    balance_classes: bool,
) -> np.ndarray:
    groups = np.asarray(participant_keys, dtype=str)
    targets = np.asarray(labels, dtype=int)
    weights = np.zeros(len(groups), dtype=float)
    unique_groups = np.unique(groups)
    group_labels: dict[str, int] = {}
    for group in unique_groups:
        mask = groups == group
        values = np.unique(targets[mask])
        if len(values) != 1:
            raise ValueError("a participant key maps to more than one dataset class")
        group_labels[str(group)] = int(values[0])
        weights[mask] = 1.0 / int(np.count_nonzero(mask))
    if balance_classes:
        counts: dict[int, int] = {}
        for value in group_labels.values():
            counts[value] = counts.get(value, 0) + 1
        for group, value in group_labels.items():
            weights[groups == group] /= counts[value]
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0):
        raise ValueError("participant weights are invalid")
    return weights / weights.mean()


def _metric_values(
    labels: np.ndarray,
    probabilities: np.ndarray,
    participant_keys: np.ndarray,
    classes: np.ndarray,
) -> dict[str, Any]:
    y = np.asarray(labels, dtype=int)
    proba = np.asarray(probabilities, dtype=float)
    groups = np.asarray(participant_keys, dtype=str)
    class_values = np.asarray(classes, dtype=int)
    if proba.shape != (len(y), len(class_values)) or not np.all(np.isfinite(proba)):
        raise ValueError("probability matrix is invalid")
    present = np.unique(y)
    weights = _participant_row_weights(groups, y, balance_classes=False)
    hard = class_values[np.argmax(proba, axis=1)]
    if len(present) < 2:
        balanced = np.nan
        balanced_status = "unavailable_test_partition_has_fewer_than_two_classes"
    else:
        balanced = float(balanced_accuracy_score(y, hard, sample_weight=weights))
        balanced_status = "available"
    auroc = np.nan
    auroc_status = "unavailable_test_partition_has_incomplete_classes"
    if set(present) == set(class_values):
        try:
            if len(class_values) == 2:
                positive = int(class_values[1])
                auroc = float(
                    roc_auc_score(
                        (y == positive).astype(int),
                        proba[:, 1],
                        sample_weight=weights,
                    )
                )
            else:
                auroc = float(
                    roc_auc_score(
                        y,
                        proba,
                        labels=class_values,
                        multi_class="ovr",
                        average="macro",
                        sample_weight=weights,
                    )
                )
            auroc_status = "available"
        except ValueError as error:
            auroc_status = f"unavailable:{type(error).__name__}:{error}"
    return {
        "balanced_accuracy": balanced,
        "balanced_accuracy_status": balanced_status,
        "auroc": auroc,
        "auroc_status": auroc_status,
    }


def _pipeline(seed: int) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    solver="lbfgs",
                    max_iter=2000,
                    random_state=seed,
                ),
            ),
        ]
    )


def _cross_validate(
    values: np.ndarray,
    labels: np.ndarray,
    participant_keys: np.ndarray,
    dataset_names: np.ndarray,
    *,
    n_splits: int,
    seed: int,
    feature_set: str,
    publish_folds: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]], np.ndarray]:
    x = np.asarray(values, dtype=float)
    y = np.asarray(labels, dtype=int)
    groups = np.asarray(participant_keys, dtype=str)
    names = np.asarray(dataset_names, dtype=str)
    classes = np.unique(y)
    probabilities = np.full((len(y), len(classes)), np.nan, dtype=float)
    fold_rows: list[dict[str, Any]] = []
    for fold_index, test_groups in enumerate(
        participant_stratified_test_sets(groups, y, n_splits=n_splits, seed=seed)
    ):
        test_mask = np.isin(groups, test_groups)
        train_mask = ~test_mask
        train_groups = set(groups[train_mask])
        heldout_groups = set(groups[test_mask])
        if train_groups & heldout_groups:
            raise AssertionError("participant leakage detected in dataset-identity folds")
        if set(np.unique(y[train_mask])) != set(classes):
            raise ValueError("training fold does not contain every dataset class")
        if np.any(np.all(~np.isfinite(x[train_mask]), axis=0)):
            raise ValueError("a feature has no finite training value in at least one fold")
        fit_weights = _participant_row_weights(
            groups[train_mask], y[train_mask], balance_classes=True
        )
        model = _pipeline(seed + fold_index)
        model.fit(
            x[train_mask],
            y[train_mask],
            classifier__sample_weight=fit_weights,
        )
        predicted = model.predict_proba(x[test_mask])
        fitted_classes = np.asarray(model.named_steps["classifier"].classes_, dtype=int)
        if set(fitted_classes) != set(classes):
            raise ValueError("classifier fold lacks one or more dataset classes")
        for source_index, value in enumerate(fitted_classes):
            target_index = int(np.flatnonzero(classes == value)[0])
            probabilities[test_mask, target_index] = predicted[:, source_index]
        if publish_folds:
            fold_metric = _metric_values(
                y[test_mask], probabilities[test_mask], groups[test_mask], classes
            )
            fold_rows.append(
                {
                    "feature_set": feature_set,
                    "fold": fold_index,
                    "status": "available",
                    "reason": None,
                    "train_cells": int(np.count_nonzero(train_mask)),
                    "test_cells": int(np.count_nonzero(test_mask)),
                    "train_participants": len(train_groups),
                    "test_participants": len(heldout_groups),
                    "train_participant_set_sha256": _set_hash(sorted(train_groups)),
                    "test_participant_set_sha256": _set_hash(sorted(heldout_groups)),
                    "participant_sets_disjoint": True,
                    "test_dataset_counts_json": json.dumps(
                        {
                            name: int(np.count_nonzero(names[test_mask] == name))
                            for name in sorted(np.unique(names[test_mask]))
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    **fold_metric,
                    "imputer_fit_scope": "fold_training_participants_only",
                    "scaler_fit_scope": "fold_training_participants_only",
                    "classifier_fit_scope": "fold_training_participants_only",
                    "participant_level_predictions_published": False,
                }
            )
    if np.any(~np.isfinite(probabilities)):
        raise ValueError("out-of-fold prediction matrix is incomplete")
    return _metric_values(y, probabilities, groups, classes), fold_rows, probabilities


def _bootstrap_intervals(
    labels: np.ndarray,
    probabilities: np.ndarray,
    participant_keys: np.ndarray,
    *,
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    y = np.asarray(labels, dtype=int)
    groups = np.asarray(participant_keys, dtype=str)
    classes = np.unique(y)
    by_class: dict[int, np.ndarray] = {}
    group_label: dict[str, int] = {}
    for group in np.unique(groups):
        values = np.unique(y[groups == group])
        if len(values) != 1:
            raise ValueError("participant spans dataset classes")
        group_label[str(group)] = int(values[0])
    for value in classes:
        by_class[int(value)] = np.asarray(
            sorted(group for group, label in group_label.items() if label == value), dtype=str
        )
    rng = np.random.default_rng(seed)
    distributions: dict[str, list[float]] = {"balanced_accuracy": [], "auroc": []}
    for _ in range(repetitions):
        multiplicities: dict[str, int] = {}
        for value in classes:
            candidates = by_class[int(value)]
            for group in rng.choice(candidates, size=len(candidates), replace=True):
                key = str(group)
                multiplicities[key] = multiplicities.get(key, 0) + 1
        selected = np.asarray([group in multiplicities for group in groups], dtype=bool)
        base_weights = _participant_row_weights(
            groups[selected], y[selected], balance_classes=False
        )
        replicate_weights = base_weights * np.asarray(
            [multiplicities[group] for group in groups[selected]], dtype=float
        )
        hard = classes[np.argmax(probabilities[selected], axis=1)]
        distributions["balanced_accuracy"].append(
            float(balanced_accuracy_score(y[selected], hard, sample_weight=replicate_weights))
        )
        try:
            if len(classes) == 2:
                value = roc_auc_score(
                    (y[selected] == int(classes[1])).astype(int),
                    probabilities[selected, 1],
                    sample_weight=replicate_weights,
                )
            else:
                value = roc_auc_score(
                    y[selected],
                    probabilities[selected],
                    labels=classes,
                    multi_class="ovr",
                    average="macro",
                    sample_weight=replicate_weights,
                )
            distributions["auroc"].append(float(value))
        except ValueError:
            continue
    output: dict[str, Any] = {
        "participant_bootstrap_repetitions": repetitions,
        "participant_bootstrap_scheme": "stratified_participant_cluster_resampling",
    }
    minimum = min(repetitions, max(20, repetitions // 2))
    for metric, values in distributions.items():
        output[f"{metric}_bootstrap_successful_repetitions"] = len(values)
        if len(values) >= minimum:
            low, high = np.quantile(np.asarray(values, dtype=float), [0.025, 0.975])
            output[f"{metric}_bootstrap_status"] = "available"
            output[f"{metric}_ci_low"] = float(low)
            output[f"{metric}_ci_high"] = float(high)
        else:
            output[f"{metric}_bootstrap_status"] = "unavailable_insufficient_valid_resamples"
            output[f"{metric}_ci_low"] = np.nan
            output[f"{metric}_ci_high"] = np.nan
    return output


def _permutation_null(
    values: np.ndarray,
    labels: np.ndarray,
    participant_keys: np.ndarray,
    dataset_names: np.ndarray,
    observed: Mapping[str, Any],
    *,
    n_splits: int,
    repetitions: int,
    seed: int,
    feature_set: str,
) -> dict[str, Any]:
    y = np.asarray(labels, dtype=int)
    groups = np.asarray(participant_keys, dtype=str)
    unique_groups = np.asarray(sorted(np.unique(groups)), dtype=str)
    group_labels = np.asarray([np.unique(y[groups == group])[0] for group in unique_groups])
    rng = np.random.default_rng(seed)
    nulls: dict[str, list[float]] = {"balanced_accuracy": [], "auroc": []}
    for repeat in range(repetitions):
        shuffled = rng.permutation(group_labels)
        mapping = dict(zip(unique_groups, shuffled, strict=True))
        permuted = np.asarray([mapping[group] for group in groups], dtype=int)
        try:
            metrics, _, _ = _cross_validate(
                values,
                permuted,
                groups,
                dataset_names,
                n_splits=n_splits,
                seed=seed + 10_007 * (repeat + 1),
                feature_set=feature_set,
                publish_folds=False,
            )
        except (RuntimeError, ValueError, np.linalg.LinAlgError):
            continue
        for metric in nulls:
            value = metrics[metric]
            if isinstance(value, (int, float)) and np.isfinite(value):
                nulls[metric].append(float(value))
    output: dict[str, Any] = {
        "label_permutation_repetitions": repetitions,
        "label_permutation_scheme": "dataset_labels_permuted_across_participants",
        "label_permutation_plus_one": True,
    }
    minimum = min(repetitions, max(20, repetitions // 2))
    for metric, values in nulls.items():
        observed_value = observed[metric]
        successful = len(values)
        output[f"{metric}_permutation_successful_repetitions"] = successful
        if np.isfinite(observed_value) and successful >= minimum:
            extreme = int(np.count_nonzero(np.asarray(values) >= float(observed_value)))
            output[f"{metric}_permutation_status"] = "available"
            output[f"{metric}_permutation_extreme_count"] = extreme
            output[f"{metric}_permutation_pvalue_plus_one"] = (extreme + 1) / (successful + 1)
        else:
            output[f"{metric}_permutation_status"] = "unavailable_insufficient_valid_permutations"
            output[f"{metric}_permutation_extreme_count"] = np.nan
            output[f"{metric}_permutation_pvalue_plus_one"] = np.nan
    return output


def _unavailable_diagnostic(
    *,
    feature_set: str,
    reason: str,
    cells: pd.DataFrame,
    features: Sequence[str],
    cell_contract: Mapping[str, Any],
    seed: int,
    requested_folds: int,
    bootstrap_repetitions: int,
    permutation_repetitions: int,
) -> dict[str, Any]:
    datasets = sorted(cells["dataset_id"].astype(str).unique()) if "dataset_id" in cells else []
    participants = int(cells["participant_key"].nunique()) if "participant_key" in cells else 0
    return {
        "feature_set": feature_set,
        "status": "unavailable",
        "reason": reason,
        "n_cells": len(cells),
        "n_participants": participants,
        "n_datasets": len(datasets),
        "n_features": len(features),
        "feature_names_json": json.dumps(list(features), separators=(",", ":")),
        "dataset_ids_json": json.dumps(datasets, separators=(",", ":")),
        "cell_inventory_sha256": (
            _set_hash(cells["cell_key_sha256"].astype(str).tolist())
            if "cell_key_sha256" in cells and len(cells)
            else None
        ),
        "cell_contract_json": json.dumps(
            dict(cell_contract), sort_keys=True, separators=(",", ":")
        ),
        "requested_folds": requested_folds,
        "realized_folds": 0,
        "random_seed": seed,
        "balanced_accuracy": np.nan,
        "balanced_accuracy_status": "unavailable_diagnostic_not_fitted",
        "balanced_accuracy_ci_low": np.nan,
        "balanced_accuracy_ci_high": np.nan,
        "balanced_accuracy_bootstrap_status": "unavailable_diagnostic_not_fitted",
        "balanced_accuracy_bootstrap_successful_repetitions": 0,
        "balanced_accuracy_permutation_status": "unavailable_diagnostic_not_fitted",
        "balanced_accuracy_permutation_successful_repetitions": 0,
        "balanced_accuracy_permutation_extreme_count": np.nan,
        "balanced_accuracy_permutation_pvalue_plus_one": np.nan,
        "auroc": np.nan,
        "auroc_status": "unavailable_diagnostic_not_fitted",
        "auroc_ci_low": np.nan,
        "auroc_ci_high": np.nan,
        "auroc_bootstrap_status": "unavailable_diagnostic_not_fitted",
        "auroc_bootstrap_successful_repetitions": 0,
        "auroc_permutation_status": "unavailable_diagnostic_not_fitted",
        "auroc_permutation_successful_repetitions": 0,
        "auroc_permutation_extreme_count": np.nan,
        "auroc_permutation_pvalue_plus_one": np.nan,
        "participant_bootstrap_repetitions": bootstrap_repetitions,
        "participant_bootstrap_scheme": "stratified_participant_cluster_resampling",
        "label_permutation_repetitions": permutation_repetitions,
        "label_permutation_scheme": "dataset_labels_permuted_across_participants",
        "label_permutation_plus_one": True,
        "participant_key_scope": "dataset_id_plus_participant_id",
        "participant_equal_weighting": True,
        "classifier": "fixed_l2_logistic_regression",
        "imputation": "training_fold_median",
        "scaling": "training_fold_standard_scaler",
        "participant_level_predictions_published": False,
        "scientific_gate_applied": False,
        "result_selection_applied": False,
    }


def _diagnose(
    cells: pd.DataFrame,
    *,
    feature_set: str,
    features: Sequence[str],
    cell_contract: Mapping[str, Any],
    requested_folds: int,
    bootstrap_repetitions: int,
    permutation_repetitions: int,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    unavailable = {
        "feature_set": feature_set,
        "cells": cells,
        "features": features,
        "cell_contract": cell_contract,
        "seed": seed,
        "requested_folds": requested_folds,
        "bootstrap_repetitions": bootstrap_repetitions,
        "permutation_repetitions": permutation_repetitions,
    }
    if cells.empty:
        return _unavailable_diagnostic(
            reason="unavailable_no_eligible_participant_condition_cells", **unavailable
        ), []
    missing = set(features).difference(cells.columns)
    if missing:
        return _unavailable_diagnostic(
            reason=f"unavailable_missing_features:{','.join(sorted(missing))}", **unavailable
        ), []
    values = cells[list(features)].to_numpy(dtype=float)
    entirely_missing = [
        feature
        for feature, column in zip(features, values.T, strict=True)
        if not np.isfinite(column).any()
    ]
    if entirely_missing:
        return _unavailable_diagnostic(
            reason=f"unavailable_features_have_no_finite_values:{','.join(entirely_missing)}",
            **unavailable,
        ), []
    datasets = np.asarray(cells["dataset_id"].astype(str), dtype=str)
    dataset_names = np.asarray(sorted(np.unique(datasets)), dtype=str)
    if len(dataset_names) < 2:
        return _unavailable_diagnostic(
            reason="unavailable_fewer_than_two_dataset_classes", **unavailable
        ), []
    label_mapping = {name: index for index, name in enumerate(dataset_names)}
    labels = np.asarray([label_mapping[name] for name in datasets], dtype=int)
    participants = np.asarray(cells["participant_key"].astype(str), dtype=str)
    maximum_folds = maximum_participant_stratified_splits(participants, labels)
    if maximum_folds < 2:
        return _unavailable_diagnostic(
            reason="unavailable_fewer_than_two_participants_in_at_least_one_dataset",
            **unavailable,
        ), []
    realized_folds = min(requested_folds, maximum_folds)
    try:
        observed, fold_rows, probabilities = _cross_validate(
            values,
            labels,
            participants,
            datasets,
            n_splits=realized_folds,
            seed=seed,
            feature_set=feature_set,
            publish_folds=True,
        )
        bootstrap = _bootstrap_intervals(
            labels,
            probabilities,
            participants,
            repetitions=bootstrap_repetitions,
            seed=seed + 1_000_003,
        )
        permutation = _permutation_null(
            values,
            labels,
            participants,
            datasets,
            observed,
            n_splits=realized_folds,
            repetitions=permutation_repetitions,
            seed=seed + 2_000_003,
            feature_set=feature_set,
        )
    except (RuntimeError, ValueError, np.linalg.LinAlgError) as error:
        return _unavailable_diagnostic(
            reason=f"unavailable_model_fit_failed:{type(error).__name__}:{error}", **unavailable
        ), []
    row = {
        "feature_set": feature_set,
        "status": "available",
        "reason": None,
        "n_cells": len(cells),
        "n_participants": int(cells["participant_key"].nunique()),
        "n_datasets": len(dataset_names),
        "n_features": len(features),
        "feature_names_json": json.dumps(list(features), separators=(",", ":")),
        "dataset_ids_json": json.dumps(dataset_names.tolist(), separators=(",", ":")),
        "cell_inventory_sha256": _set_hash(cells["cell_key_sha256"].astype(str).tolist()),
        "cell_contract_json": json.dumps(
            dict(cell_contract), sort_keys=True, separators=(",", ":")
        ),
        "requested_folds": requested_folds,
        "realized_folds": realized_folds,
        "random_seed": seed,
        **observed,
        **bootstrap,
        **permutation,
        "participant_key_scope": "dataset_id_plus_participant_id",
        "participant_equal_weighting": True,
        "classifier": "fixed_l2_logistic_regression",
        "imputation": "training_fold_median",
        "scaling": "training_fold_standard_scaler",
        "participant_level_predictions_published": False,
        "scientific_gate_applied": False,
        "result_selection_applied": False,
    }
    return row, fold_rows


def run_representation_controls(
    *,
    profiles_path: str | Path,
    benchmarks_path: str | Path,
    encoding_manifest_path: str | Path,
    encoding_flow_path: str | Path,
    models_path: str | Path,
    output_root: str | Path,
    study: StudyConfig,
    participant_folds: int | None = None,
    bootstrap_repetitions: int | None = None,
    permutation_repetitions: int | None = None,
) -> tuple[Path, Path, Path, Path]:
    """Publish honest branch availability and dataset-identity diagnostics."""

    paths = {
        "profiles": Path(profiles_path).resolve(strict=True),
        "benchmarks": Path(benchmarks_path).resolve(strict=True),
        "encoding_manifest": Path(encoding_manifest_path).resolve(strict=True),
        "encoding_flow": Path(encoding_flow_path).resolve(strict=True),
        "models": Path(models_path).resolve(strict=True),
    }
    requested_folds = int(
        study.statistics.participant_stratified_folds
        if participant_folds is None
        else participant_folds
    )
    boot_repetitions = int(
        study.statistics.participant_bootstrap_repetitions
        if bootstrap_repetitions is None
        else bootstrap_repetitions
    )
    perm_repetitions = int(
        study.statistics.permutation_repetitions
        if permutation_repetitions is None
        else permutation_repetitions
    )
    if requested_folds < 2 or boot_repetitions < 1 or perm_repetitions < 1:
        raise ValueError("fold and resampling counts must be positive and folds at least two")
    input_hashes = {name: sha256_file(path) for name, path in paths.items()}
    profiles = pd.read_parquet(paths["profiles"])
    benchmarks = pd.read_parquet(paths["benchmarks"])
    encoding_manifest = pd.read_parquet(paths["encoding_manifest"])
    encoding_flow = _load_json_object(paths["encoding_flow"], label="encoding flow")
    models = _load_models(paths["models"])
    checkpoint_sha256 = str(encoding_flow.get("checkpoint_sha256", "")).lower()
    model_fingerprint_sha256 = str(encoding_flow.get("model_fingerprint_sha256", "")).lower()
    trajectory_inventory = _trajectory_inventory(
        encoding_manifest,
        study=study,
        checkpoint_sha256=checkpoint_sha256,
        model_fingerprint_sha256=model_fingerprint_sha256,
    )
    availability_rows = _availability_rows(
        study=study,
        models=models,
        encoding_flow=encoding_flow,
        trajectory_inventory=trajectory_inventory,
        input_hashes=input_hashes,
    )

    profile_cells = _profile_cells(profiles)
    conventional_cells, conventional_contract = _conventional_cells(benchmarks, profile_cells)
    primary_contract = {
        "role": "all_available_five_axis_participant_condition_cells",
        "source_cells": len(profile_cells),
        "cell_inventory_sha256": _set_hash(profile_cells["cell_key_sha256"].astype(str).tolist()),
    }
    five_axis, five_axis_folds = _diagnose(
        profile_cells,
        feature_set="five_axis_participant_condition",
        features=AXIS_NAMES,
        cell_contract=primary_contract,
        requested_folds=requested_folds,
        bootstrap_repetitions=boot_repetitions,
        permutation_repetitions=perm_repetitions,
        seed=int(study.random_seeds[0]) + 300_007,
    )
    if conventional_contract["status"] == "available":
        conventional, conventional_folds = _diagnose(
            conventional_cells,
            feature_set="conventional_exact_eligible_cells",
            features=CONVENTIONAL_FEATURES,
            cell_contract={
                "role": "complete_benchmark_cells_exactly_present_in_five_axis_profiles",
                **conventional_contract,
            },
            requested_folds=requested_folds,
            bootstrap_repetitions=boot_repetitions,
            permutation_repetitions=perm_repetitions,
            seed=int(study.random_seeds[0]) + 600_011,
        )
    else:
        conventional = _unavailable_diagnostic(
            feature_set="conventional_exact_eligible_cells",
            reason=str(conventional_contract["reason"]),
            cells=conventional_cells,
            features=CONVENTIONAL_FEATURES,
            cell_contract=conventional_contract,
            seed=int(study.random_seeds[0]) + 600_011,
            requested_folds=requested_folds,
            bootstrap_repetitions=boot_repetitions,
            permutation_repetitions=perm_repetitions,
        )
        conventional_folds = []

    destination = Path(output_root)
    availability_path = _atomic_parquet(
        pd.DataFrame(availability_rows), destination / "representation-availability.parquet"
    )
    diagnostics_path = _atomic_parquet(
        pd.DataFrame([five_axis, conventional]),
        destination / "dataset-identity-diagnostics.parquet",
    )
    folds_path = _atomic_parquet(
        pd.DataFrame(five_axis_folds + conventional_folds),
        destination / "dataset-identity-folds.parquet",
    )
    for frame in (pd.DataFrame(availability_rows), pd.DataFrame([five_axis, conventional])):
        forbidden = _PRIVATE_OUTPUT_FIELDS.intersection(frame.columns)
        if forbidden:
            raise AssertionError(
                f"representation control output exposes private fields: {forbidden}"
            )
    audit_path = destination / "representation-control-audit.json"
    atomic_write_json(
        audit_path,
        {
            "schema_version": 1,
            "study_config_sha256": config_sha256(study),
            "input_artifact_sha256": input_hashes,
            "trajectory_inventory": trajectory_inventory,
            "availability_rows": len(availability_rows),
            "available_coordinate_trajectory_controls": [
                row["control_id"]
                for row in availability_rows
                if row["status"] == "available" and row["trajectory_generated"]
            ],
            "unavailable_coordinate_trajectory_controls": [
                {
                    "control_id": row["control_id"],
                    "reason": row["reason"],
                }
                for row in availability_rows
                if row["status"] != "available"
            ],
            "conventional_cell_contract": conventional_contract,
            "diagnostic_feature_sets": [five_axis["feature_set"], conventional["feature_set"]],
            "output_artifact_sha256": {
                "availability": sha256_file(availability_path),
                "diagnostics": sha256_file(diagnostics_path),
                "folds": sha256_file(folds_path),
            },
            "preprocessing_imputation_scaling_classifier_fit_scope": (
                "participant_separated_training_folds_only"
            ),
            "participant_level_predictions_published": False,
            "participant_identifiers_published": False,
            "raw_or_coordinate_arrays_published": False,
            "full_pca_coordinate_trajectory_claimed": False,
            "full_time_frequency_coordinate_trajectory_claimed": False,
            "scientific_gate_applied": False,
            "result_selection_applied": False,
        },
    )
    return availability_path, diagnostics_path, folds_path, audit_path
