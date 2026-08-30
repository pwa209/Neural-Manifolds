"""Participant-level contrasts and leakage-safe predictive comparisons."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from neural_manifolds.config import StudyConfig
from neural_manifolds.foundation.overlap import (
    ensure_pretraining_overlap_columns,
    overlap_output_fields,
    summarize_pretraining_overlap,
)
from neural_manifolds.manifold.profile import AXIS_NAMES
from neural_manifolds.provenance import atomic_write_json, sha256_file
from neural_manifolds.stages.benchmarks import CONVENTIONAL_FEATURES
from neural_manifolds.statistics.equivalence import participant_bootstrap_tost_interval
from neural_manifolds.statistics.folds import (
    maximum_participant_stratified_splits,
    participant_stratified_test_sets,
)
from neural_manifolds.statistics.mixed import fit_participant_mixed_model
from neural_manifolds.statistics.multivariate import benjamini_hochberg
from neural_manifolds.statistics.prediction import participant_bootstrap_prediction_metrics
from neural_manifolds.tms_separation import assert_no_direct_tms

REPEATED_EQUAL_WINDOW_PRIMARY = "repeated_equal_window_primary"
ALL_AVAILABLE_SENSITIVITY = "all_available_participant_condition_sensitivity"
CONTINUOUS_ALL_AVAILABLE_PRIMARY = "continuous_all_available_primary"


def _atomic_parquet(frame: pd.DataFrame, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, destination)
    return destination


def _load_contrasts(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    if not isinstance(document, dict) or document.get("schema_version") != 1:
        raise ValueError("contrast configuration must use schema_version 1")
    contrasts = document.get("contrasts")
    if contrasts is None:
        contrasts = []
        datasets = document.get("datasets")
        if not isinstance(datasets, dict):
            raise ValueError("contrast configuration has no datasets mapping")
        for dataset_id, dataset in datasets.items():
            if not isinstance(dataset, dict):
                continue
            for raw in dataset.get("contrasts", []):
                if not isinstance(raw, dict):
                    raise ValueError(f"contrast for {dataset_id} must be a mapping")
                item = dict(raw)
                item["datasets"] = [str(dataset_id)]
                item["design"] = (
                    "within_participant"
                    if "match_within" in item or "match_on" in item
                    else "between_participant"
                )
                if "continuous_covariate" in item:
                    item["analysis_type"] = "continuous"
                    item["label_column"] = item["continuous_covariate"]
                    item["positive"] = None
                    item["negative"] = None
                else:
                    item["analysis_type"] = "binary"
                    item["label_column"] = "condition"
                    if "positive" not in item:
                        conditions = item.get("conditions")
                        if not isinstance(conditions, list) or len(conditions) != 2:
                            raise ValueError(
                                f"contrast {item.get('id')} has no explicit two-level mapping"
                            )
                        item["positive"], item["negative"] = conditions
                    else:
                        item["negative"] = item.get("reference")
                contrasts.append(item)
    if not isinstance(contrasts, list) or not contrasts:
        raise ValueError("contrast configuration has no contrasts")
    output: list[dict[str, Any]] = []
    for item in contrasts:
        if not isinstance(item, dict):
            raise ValueError("each contrast must be a mapping")
        required = {
            "id",
            "datasets",
            "label_column",
            "positive",
            "negative",
            "design",
        }
        missing = required.difference(item)
        if missing:
            raise ValueError(f"contrast is missing {sorted(missing)}")
        if item["design"] not in {"within_participant", "between_participant"}:
            raise ValueError(f"contrast {item['id']} has an invalid design")
        normalized = dict(item)
        normalized.setdefault("analysis_type", "binary")
        output.append(normalized)
    return output


def _select_contrast(frame: pd.DataFrame, contrast: Mapping[str, Any]) -> pd.DataFrame:
    datasets = contrast["datasets"]
    if isinstance(datasets, str):
        datasets = [datasets]
    column = str(contrast["label_column"])
    if column not in frame:
        raise ValueError(f"label column {column!r} is absent")
    subset = frame[frame["dataset_id"].isin([str(value) for value in datasets])].copy()
    for key, expected in (contrast.get("subset") or {}).items():
        if key not in subset:
            raise ValueError(f"subset column {key!r} is absent")
        allowed = expected if isinstance(expected, list) else [expected]
        subset = subset[subset[key].isin(allowed)]
    if contrast.get("analysis_type") == "continuous":
        subset["continuous_target"] = pd.to_numeric(subset[column], errors="coerce")
        return subset[subset["continuous_target"].notna()]
    positive = contrast["positive"]
    negative = contrast["negative"]
    positive_values = positive if isinstance(positive, list) else [positive]
    negative_values = negative if isinstance(negative, list) else [negative]
    subset = subset[subset[column].isin([*positive_values, *negative_values])]
    subset["binary_target"] = subset[column].isin(positive_values).astype(int)
    return subset


def _matched_contrast_subset(
    matched: pd.DataFrame,
    *,
    contrast: Mapping[str, Any],
    profile_subset: pd.DataFrame,
) -> pd.DataFrame:
    """Convert repeated equal-window summaries into a model-ready contrast table."""

    required = {
        "contrast_id",
        "participant_id",
        "dataset_id",
        "contrast_arm",
        "successful_repeats",
        *(f"{axis}_mean" for axis in AXIS_NAMES),
    }
    missing = required.difference(matched.columns)
    if missing:
        raise ValueError(f"matched sampling profiles are missing {sorted(missing)}")
    subset = matched[matched["contrast_id"].astype(str) == str(contrast["id"])].copy()
    if subset.empty:
        raise ValueError("no equal-window profiles are available for this contrast")
    successful_repeats = pd.to_numeric(subset["successful_repeats"], errors="coerce")
    if successful_repeats.isna().any() or (successful_repeats <= 0).any():
        raise ValueError("an equal-window profile has no successful repeats")
    arms = set(subset["contrast_arm"].astype(str))
    if {"positive", "reference"} <= arms:
        positive_arms = {"positive"}
        negative_arms = {"reference"}
    else:
        positive = contrast["positive"]
        negative = contrast["negative"]
        positive_arms = {
            str(value) for value in (positive if isinstance(positive, list) else [positive])
        }
        negative_arms = {
            str(value) for value in (negative if isinstance(negative, list) else [negative])
        }
    subset = subset[subset["contrast_arm"].astype(str).isin(positive_arms | negative_arms)].copy()
    subset["binary_target"] = subset["contrast_arm"].astype(str).isin(positive_arms).astype(int)
    if set(subset["binary_target"].unique()) != {0, 1}:
        raise ValueError("equal-window profiles do not contain both contrast arms")
    subset = subset.rename(columns={f"{axis}_mean": axis for axis in AXIS_NAMES})
    subset[list(AXIS_NAMES)] = subset[list(AXIS_NAMES)].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(subset[list(AXIS_NAMES)].to_numpy(dtype=float)).all():
        raise ValueError("equal-window profiles contain non-finite axis estimates")
    eligibility = profile_subset[
        ["participant_id", "dataset_id", "prediction_evaluation_eligible"]
    ].drop_duplicates()
    if eligibility.duplicated(["participant_id", "dataset_id"], keep=False).any():
        raise ValueError("profile rows disagree on participant prediction eligibility")
    subset = subset.merge(
        eligibility,
        on=["participant_id", "dataset_id"],
        how="inner",
        validate="many_to_one",
    )
    if subset.empty:
        raise ValueError("equal-window profiles do not map to the frozen representation split")
    return subset


def _continuous_inference(
    frame: pd.DataFrame,
    *,
    contrast_id: str,
    repetitions: int,
    seed: int,
    bootstrap_repetitions: int | None = None,
    equivalence_margin: float | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not np.isfinite(frame[list(AXIS_NAMES)].to_numpy(dtype=float)).all():
        raise ValueError("continuous contrast contains non-finite axis estimates")
    collapsed = (
        frame.groupby(["participant_id", "continuous_target"], as_index=False)[list(AXIS_NAMES)]
        .mean()
        .sort_values(["participant_id", "continuous_target"])
    )
    counts = collapsed.groupby("participant_id").size()
    keep = counts[counts >= 2].index
    collapsed = collapsed[collapsed["participant_id"].isin(keep)].copy()
    if collapsed["participant_id"].nunique() < 4:
        raise ValueError("continuous contrast needs four participants with repeated observations")
    bootstrap_count = repetitions if bootstrap_repetitions is None else bootstrap_repetitions
    if bootstrap_count < 20:
        raise ValueError("at least 20 participant bootstrap repetitions are required")
    bootstrap_rng = np.random.default_rng(seed + 1009)
    permutation_rng = np.random.default_rng(seed + 2003)
    participants = collapsed["participant_id"].unique()
    by_participant = {name: collapsed[collapsed["participant_id"] == name] for name in participants}
    participant_slope_map: dict[str, np.ndarray] = {}
    for name in participants:
        group = by_participant[name]
        gx = group["continuous_target"] - group["continuous_target"].mean()
        gy = group[list(AXIS_NAMES)] - group[list(AXIS_NAMES)].mean()
        denom = float(np.sum(gx.to_numpy() ** 2))
        if denom > 0:
            participant_slope_map[str(name)] = (gx.to_numpy()[:, None] * gy.to_numpy()).sum(
                0
            ) / denom
    if len(participant_slope_map) != len(participants):
        raise ValueError("a participant has no within-participant covariate variation")
    # The observed estimate, cluster bootstrap, and permutation distribution all
    # use the same estimand: the equally weighted mean participant-specific slope.
    observed_participant_slopes = np.stack(
        [participant_slope_map[str(name)] for name in participants]
    )
    slopes = np.mean(observed_participant_slopes, axis=0)
    bootstrap_values = np.stack(
        [
            observed_participant_slopes[
                bootstrap_rng.integers(0, len(participants), len(participants))
            ].mean(axis=0)
            for _ in range(bootstrap_count)
        ]
    )
    null = []
    for _ in range(repetitions):
        permuted_parts = []
        for name in participants:
            group = by_participant[name]
            permuted = permutation_rng.permutation(group["continuous_target"].to_numpy())
            gx = permuted - permuted.mean()
            gy = group[list(AXIS_NAMES)].to_numpy(copy=True)
            gy -= gy.mean(0)
            denom = float(np.sum(gx**2))
            if denom > 0:
                permuted_parts.append((gx[:, None] * gy).sum(0) / denom)
        null.append(np.mean(permuted_parts, axis=0))
    null_values = np.stack(null)
    p_values = (np.count_nonzero(np.abs(null_values) >= np.abs(slopes), axis=0) + 1) / (
        repetitions + 1
    )
    adjusted, rejected = benjamini_hochberg(p_values)
    rows = []
    for index, axis in enumerate(AXIS_NAMES):
        low, high = np.quantile(bootstrap_values[:, index], [0.025, 0.975])
        equivalence: dict[str, Any]
        if equivalence_margin is None:
            equivalence = {
                "equivalence_status": "unavailable_no_configured_smallest_effect_size",
                "equivalence_unavailable_reason": (
                    "continuous_smallest_effect is required for TOST equivalence"
                ),
            }
        else:
            interval = participant_bootstrap_tost_interval(
                bootstrap_values[:, index],
                estimate=float(slopes[index]),
                smallest_effect_size=equivalence_margin,
            )
            equivalence = {
                "equivalence_status": ("equivalent" if interval.equivalent else "not_equivalent"),
                "equivalence_unavailable_reason": None,
                "equivalence_ci_low": interval.ci_low,
                "equivalence_ci_high": interval.ci_high,
                "equivalence_alpha": interval.alpha,
                "equivalence_smallest_effect_size": interval.smallest_effect_size,
                "equivalence_lower_test_p_value": interval.lower_test_p_value,
                "equivalence_upper_test_p_value": interval.upper_test_p_value,
                "equivalence_method": interval.method,
            }
        rows.append(
            {
                "contrast_id": contrast_id,
                "axis": axis,
                "effect": float(slopes[index]),
                "ci_low": float(low),
                "ci_high": float(high),
                "participant_bootstrap_ci_low": float(low),
                "participant_bootstrap_ci_high": float(high),
                "participant_bootstrap_repetitions": bootstrap_count,
                "participant_bootstrap_unit": "participant",
                "p_value": float(p_values[index]),
                "q_value": float(adjusted[index]),
                "fdr_reject": bool(rejected[index]),
                "n_participants": len(participants),
                "design": "within_participant_continuous",
                **equivalence,
            }
        )
    observed_norm = float(np.linalg.norm(slopes))
    null_norm = np.linalg.norm(null_values, axis=1)
    bootstrap_norm = np.linalg.norm(bootstrap_values, axis=1)
    omnibus_low, omnibus_high = np.quantile(bootstrap_norm, [0.025, 0.975])
    return rows, {
        "contrast_id": contrast_id,
        "profile_distance": observed_norm,
        "p_value": float((np.count_nonzero(null_norm >= observed_norm) + 1) / (repetitions + 1)),
        "n_participants": len(participants),
        "permutations": repetitions,
        "participant_bootstrap_ci_low": float(omnibus_low),
        "participant_bootstrap_ci_high": float(omnibus_high),
        "participant_bootstrap_repetitions": bootstrap_count,
        "participant_bootstrap_unit": "participant",
        "equivalence_status": "unavailable_no_configured_multivariate_smallest_effect_size",
        "equivalence_unavailable_reason": (
            "no multivariate smallest-effect region is configured for profile distance"
        ),
    }


def _participant_rows(frame: pd.DataFrame) -> pd.DataFrame:
    group = ["participant_id", "dataset_id", "binary_target"]
    return frame.groupby(group, as_index=False)[list(AXIS_NAMES)].mean()


def _observed_effects(frame: pd.DataFrame, *, design: str) -> np.ndarray:
    if design == "within_participant":
        participant = frame.groupby(["participant_id", "binary_target"])[list(AXIS_NAMES)].mean()
        wide = participant.unstack("binary_target")
        complete = wide.dropna()
        if (
            len(complete) < 3
            or (1 not in complete.columns.levels[1])
            or (0 not in complete.columns.levels[1])
        ):
            raise ValueError("within-participant contrast has fewer than three complete pairs")
        positive = complete.xs(1, axis=1, level=1).to_numpy()
        negative = complete.xs(0, axis=1, level=1).to_numpy()
        return positive - negative
    collapsed = _participant_rows(frame)
    positive = collapsed.loc[collapsed["binary_target"] == 1, list(AXIS_NAMES)].to_numpy()
    negative = collapsed.loc[collapsed["binary_target"] == 0, list(AXIS_NAMES)].to_numpy()
    if min(len(positive), len(negative)) < 3:
        raise ValueError("between-participant contrast needs three participants per group")
    return np.vstack([positive, -negative])


def _contrast_inference(
    frame: pd.DataFrame,
    *,
    contrast_id: str,
    design: str,
    repetitions: int,
    seed: int,
    bootstrap_repetitions: int | None = None,
    equivalence_margin: float | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not np.isfinite(frame[list(AXIS_NAMES)].to_numpy(dtype=float)).all():
        raise ValueError("contrast contains non-finite axis estimates")
    bootstrap_count = repetitions if bootstrap_repetitions is None else bootstrap_repetitions
    if bootstrap_count < 20:
        raise ValueError("at least 20 participant bootstrap repetitions are required")
    bootstrap_rng = np.random.default_rng(seed + 1009)
    permutation_rng = np.random.default_rng(seed + 2003)
    if design == "within_participant":
        grouped = frame.groupby(["participant_id", "binary_target"])[list(AXIS_NAMES)].mean()
        wide = grouped.unstack("binary_target").dropna()
        differences = (
            wide.xs(1, axis=1, level=1).to_numpy() - wide.xs(0, axis=1, level=1).to_numpy()
        )
        n_participants = len(differences)
        if n_participants < 3:
            raise ValueError("within-participant contrast has fewer than three complete pairs")
        observed = differences.mean(axis=0)
        bootstrap = np.stack(
            [
                differences[bootstrap_rng.integers(0, n_participants, n_participants)].mean(axis=0)
                for _ in range(bootstrap_count)
            ]
        )
        signs = permutation_rng.choice((-1.0, 1.0), size=(repetitions, n_participants, 1))
        null = (signs * differences[None]).mean(axis=1)
    else:
        collapsed = _participant_rows(frame)
        values = collapsed[list(AXIS_NAMES)].to_numpy()
        labels = collapsed["binary_target"].to_numpy(dtype=int)
        n_participants = len(collapsed)
        if min(np.count_nonzero(labels == 0), np.count_nonzero(labels == 1)) < 3:
            raise ValueError("between-participant contrast needs three participants per group")
        observed = values[labels == 1].mean(axis=0) - values[labels == 0].mean(axis=0)
        bootstrap_rows = []
        for _ in range(bootstrap_count):
            group_values = []
            for label in (0, 1):
                indices = np.flatnonzero(labels == label)
                group_values.append(
                    values[bootstrap_rng.choice(indices, len(indices), replace=True)].mean(axis=0)
                )
            bootstrap_rows.append(group_values[1] - group_values[0])
        bootstrap = np.stack(bootstrap_rows)
        null_rows = []
        for _ in range(repetitions):
            shuffled = permutation_rng.permutation(labels)
            null_rows.append(values[shuffled == 1].mean(0) - values[shuffled == 0].mean(0))
        null = np.stack(null_rows)
    p_values = (np.count_nonzero(np.abs(null) >= np.abs(observed), axis=0) + 1) / (repetitions + 1)
    adjusted, rejected = benjamini_hochberg(p_values)
    axis_rows = []
    for index, axis in enumerate(AXIS_NAMES):
        low, high = np.quantile(bootstrap[:, index], [0.025, 0.975])
        equivalence: dict[str, Any]
        if equivalence_margin is None:
            equivalence = {
                "equivalence_status": "unavailable_no_configured_smallest_effect_size",
                "equivalence_unavailable_reason": (
                    "continuous_smallest_effect is required for TOST equivalence"
                ),
            }
        else:
            interval = participant_bootstrap_tost_interval(
                bootstrap[:, index],
                estimate=float(observed[index]),
                smallest_effect_size=equivalence_margin,
            )
            equivalence = {
                "equivalence_status": ("equivalent" if interval.equivalent else "not_equivalent"),
                "equivalence_unavailable_reason": None,
                "equivalence_ci_low": interval.ci_low,
                "equivalence_ci_high": interval.ci_high,
                "equivalence_alpha": interval.alpha,
                "equivalence_smallest_effect_size": interval.smallest_effect_size,
                "equivalence_lower_test_p_value": interval.lower_test_p_value,
                "equivalence_upper_test_p_value": interval.upper_test_p_value,
                "equivalence_method": interval.method,
            }
        axis_rows.append(
            {
                "contrast_id": contrast_id,
                "axis": axis,
                "effect": float(observed[index]),
                "ci_low": float(low),
                "ci_high": float(high),
                "participant_bootstrap_ci_low": float(low),
                "participant_bootstrap_ci_high": float(high),
                "participant_bootstrap_repetitions": bootstrap_count,
                "participant_bootstrap_unit": "participant",
                "p_value": float(p_values[index]),
                "q_value": float(adjusted[index]),
                "fdr_reject": bool(rejected[index]),
                "n_participants": n_participants,
                "design": design,
                **equivalence,
            }
        )
    observed_norm = float(np.linalg.norm(observed))
    null_norm = np.linalg.norm(null, axis=1)
    bootstrap_norm = np.linalg.norm(bootstrap, axis=1)
    omnibus_low, omnibus_high = np.quantile(bootstrap_norm, [0.025, 0.975])
    omnibus = {
        "contrast_id": contrast_id,
        "profile_distance": observed_norm,
        "p_value": float((np.count_nonzero(null_norm >= observed_norm) + 1) / (repetitions + 1)),
        "n_participants": n_participants,
        "permutations": repetitions,
        "participant_bootstrap_ci_low": float(omnibus_low),
        "participant_bootstrap_ci_high": float(omnibus_high),
        "participant_bootstrap_repetitions": bootstrap_count,
        "participant_bootstrap_unit": "participant",
        "equivalence_status": "unavailable_no_configured_multivariate_smallest_effect_size",
        "equivalence_unavailable_reason": (
            "no multivariate smallest-effect region is configured for profile distance"
        ),
    }
    return axis_rows, omnibus


def _collapse_prediction_rows(frame: pd.DataFrame, *, features: Sequence[str]) -> pd.DataFrame:
    """Give every participant-condition cell equal weight in prediction."""

    required = {
        "participant_id",
        "dataset_id",
        "binary_target",
        "prediction_evaluation_eligible",
        *features,
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"prediction rows are missing {sorted(missing)}")
    work = frame.copy()
    work["participant_id"] = work["participant_id"].astype(str)
    work["prediction_evaluation_eligible"] = work["prediction_evaluation_eligible"].astype(bool)
    eligibility_counts = work.groupby("participant_id")["prediction_evaluation_eligible"].nunique()
    if (eligibility_counts > 1).any():
        raise ValueError("prediction eligibility must be constant within participant")
    group = [
        "participant_id",
        "dataset_id",
        "binary_target",
        "prediction_evaluation_eligible",
    ]
    return work.groupby(group, as_index=False)[list(features)].mean()


def _classifier(*, c_value: float, seed: int) -> Any:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=c_value,
            l1_ratio=0.0,
            class_weight="balanced",
            max_iter=2000,
            random_state=seed,
        ),
    )


def _select_regularization(
    *,
    values: np.ndarray,
    labels: np.ndarray,
    participants: np.ndarray,
    train_indices: np.ndarray,
    validation_participants: np.ndarray,
    seed: int,
) -> float:
    candidates = np.unique(np.asarray(validation_participants, dtype=str))
    validation_rows = train_indices[np.isin(participants[train_indices], candidates)]
    inner_splits = min(
        3,
        maximum_participant_stratified_splits(
            participants[validation_rows], labels[validation_rows]
        ),
    )
    if inner_splits < 2:
        raise ValueError("inner tuning requires two participant-separated validation folds")
    best_score = -np.inf
    best_c: float | None = None
    for c_value in (0.01, 0.1, 1.0, 10.0):
        scores = []
        for inner_test_participants in participant_stratified_test_sets(
            participants[validation_rows],
            labels[validation_rows],
            n_splits=inner_splits,
            seed=seed + 1,
        ):
            inner_test = train_indices[
                np.isin(participants[train_indices], inner_test_participants)
            ]
            inner_train = train_indices[
                ~np.isin(participants[train_indices], inner_test_participants)
            ]
            if set(participants[inner_train]).intersection(participants[inner_test]):
                raise AssertionError("participant leakage detected in inner prediction fold")
            if len(np.unique(labels[inner_train])) != 2:
                continue
            model = _classifier(c_value=c_value, seed=seed)
            model.fit(values[inner_train], labels[inner_train])
            predicted = model.predict_proba(values[inner_test])[:, 1]
            if len(np.unique(labels[inner_test])) == 2:
                scores.append(roc_auc_score(labels[inner_test], predicted))
        if scores:
            score = float(np.mean(scores))
            if score > best_score:
                best_score, best_c = score, c_value
    if best_c is None:
        raise ValueError("inner participant-separated tuning produced no two-class validation fold")
    return best_c


def _prediction(
    frame: pd.DataFrame,
    *,
    features: Sequence[str],
    folds: int,
    seed: int,
    bootstrap_repetitions: int,
) -> dict[str, Any]:
    collapsed = _collapse_prediction_rows(frame, features=features)
    participant = collapsed["participant_id"].to_numpy(dtype=str)
    labels = collapsed["binary_target"].to_numpy(dtype=int)
    values = collapsed[list(features)].to_numpy(dtype=float)
    eligible = collapsed["prediction_evaluation_eligible"].to_numpy(dtype=bool)
    evaluation_participants = np.unique(participant[eligible])
    fixed_participants = np.unique(participant[~eligible])
    if set(evaluation_participants).intersection(fixed_participants):
        raise ValueError("a participant cannot be both transform-fitting and evaluation eligible")
    scored_labels = labels[eligible]
    n_splits = min(
        folds,
        maximum_participant_stratified_splits(participant[eligible], scored_labels),
    )
    if n_splits < 3 or len(np.unique(scored_labels)) != 2:
        raise ValueError(
            "prediction requires two classes and at least three untouched evaluation participants"
        )
    probabilities = np.full(len(collapsed), np.nan)
    selected_c: list[float] = []
    # Split only untouched evaluation participants. Discovery/validation
    # participants remain fixed training rows and are never scored.
    for test_participants in participant_stratified_test_sets(
        participant[eligible],
        scored_labels,
        n_splits=n_splits,
        seed=seed,
    ):
        test = np.flatnonzero(eligible & np.isin(participant, test_participants))
        train = np.flatnonzero(~np.isin(participant, test_participants))
        if set(participant[train]).intersection(participant[test]):
            raise AssertionError("participant leakage detected in outer prediction fold")
        if len(np.unique(labels[train])) != 2:
            raise ValueError("an outer training fold contains one class")
        inner_candidates = np.unique(participant[train][eligible[train]])
        best_c = _select_regularization(
            values=values,
            labels=labels,
            participants=participant,
            train_indices=train,
            validation_participants=inner_candidates,
            seed=seed,
        )
        selected_c.append(best_c)
        final = _classifier(c_value=best_c, seed=seed)
        final.fit(values[train], labels[train])
        probabilities[test] = final.predict_proba(values[test])[:, 1]
    if not np.all(np.isfinite(probabilities[eligible])):
        raise RuntimeError("outer predictions are incomplete")
    scored_probabilities = probabilities[eligible]
    metrics = participant_bootstrap_prediction_metrics(
        scored_labels,
        scored_probabilities,
        participant[eligible],
        repetitions=bootstrap_repetitions,
        seed=seed + 3001,
        retain_distributions=True,
    )
    return {
        "features": list(features),
        **metrics,
        "n_participants": len(evaluation_participants),
        "n_evaluation_participants": len(evaluation_participants),
        "n_fixed_transform_participants": len(fixed_participants),
        "n_observations": int(eligible.sum()),
        "n_training_observations_total": len(collapsed),
        "selected_c": selected_c,
        "representation_heldout": True,
        "outer_fold_participant_separation_verified": True,
        "preprocessing_and_tuning_fit_within_training_folds": True,
    }


def _leave_one_dataset_out_predictions(
    frame: pd.DataFrame,
    *,
    features: Sequence[str],
    seed: int,
    bootstrap_repetitions: int,
) -> list[dict[str, Any]]:
    collapsed = _collapse_prediction_rows(frame, features=features)
    participants = collapsed["participant_id"].to_numpy(dtype=str)
    datasets = collapsed["dataset_id"].to_numpy(dtype=str)
    labels = collapsed["binary_target"].to_numpy(dtype=int)
    values = collapsed[list(features)].to_numpy(dtype=float)
    eligible = collapsed["prediction_evaluation_eligible"].to_numpy(dtype=bool)
    qualifying = []
    for dataset_id in np.unique(datasets):
        dataset_rows = datasets == dataset_id
        if set(np.unique(labels[dataset_rows])) == {0, 1} and set(
            np.unique(labels[dataset_rows & eligible])
        ) == {0, 1}:
            qualifying.append(str(dataset_id))
    if len(qualifying) < 2:
        raise ValueError(
            "leave-one-dataset-out requires at least two datasets with both arms "
            "and evaluation-eligible observations"
        )

    output: list[dict[str, Any]] = []
    for dataset_index, held_out_dataset in enumerate(qualifying):
        test = np.flatnonzero((datasets == held_out_dataset) & eligible)
        test_participants = np.unique(participants[test])
        train = np.flatnonzero(
            (datasets != held_out_dataset) & ~np.isin(participants, test_participants)
        )
        if set(participants[train]).intersection(participants[test]):
            raise AssertionError("participant leakage detected in leave-one-dataset-out split")
        if set(np.unique(labels[train])) != {0, 1} or set(np.unique(labels[test])) != {0, 1}:
            raise ValueError(
                f"held-out dataset {held_out_dataset} lacks a two-class train/test split"
            )
        validation_participants = np.unique(participants[train][eligible[train]])
        best_c = _select_regularization(
            values=values,
            labels=labels,
            participants=participants,
            train_indices=train,
            validation_participants=validation_participants,
            seed=seed + dataset_index * 101,
        )
        model = _classifier(c_value=best_c, seed=seed + dataset_index * 101)
        model.fit(values[train], labels[train])
        probabilities = model.predict_proba(values[test])[:, 1]
        metrics = participant_bootstrap_prediction_metrics(
            labels[test],
            probabilities,
            participants[test],
            repetitions=bootstrap_repetitions,
            seed=seed + dataset_index * 101 + 3001,
        )
        output.append(
            {
                "features": list(features),
                **metrics,
                "generalization_scheme": "leave_one_dataset_out",
                "held_out_dataset_id": held_out_dataset,
                "n_participants": len(test_participants),
                "n_evaluation_participants": len(test_participants),
                "n_training_participants": len(np.unique(participants[train])),
                "n_observations": len(test),
                "n_training_observations_total": len(train),
                "selected_c": [best_c],
                "representation_heldout": True,
                "outer_fold_participant_separation_verified": True,
                "preprocessing_and_tuning_fit_within_training_folds": True,
            }
        )
    return output


def _mixed_model_diagnostics(
    frame: pd.DataFrame,
    *,
    analysis_type: str,
) -> dict[str, dict[str, Any]]:
    fixed_effect = "continuous_target" if analysis_type == "continuous" else "binary_target"
    repeated_counts = frame.groupby("participant_id").size()
    if int((repeated_counts >= 2).sum()) < 3:
        reason = "random-intercept mixed model requires three participants with repeated rows"
        return {
            axis: {
                "mixed_model_status": "unavailable_insufficient_repeated_observations",
                "mixed_model_unavailable_reason": reason,
                "mixed_model_random_effect": "participant_intercept",
            }
            for axis in AXIS_NAMES
        }
    if fixed_effect not in frame or frame[fixed_effect].nunique(dropna=True) < 2:
        reason = f"mixed model fixed effect {fixed_effect!r} has no variation"
        return {
            axis: {
                "mixed_model_status": "unavailable_no_fixed_effect_variation",
                "mixed_model_unavailable_reason": reason,
                "mixed_model_random_effect": "participant_intercept",
            }
            for axis in AXIS_NAMES
        }
    output: dict[str, dict[str, Any]] = {}
    for axis in AXIS_NAMES:
        try:
            fitted = fit_participant_mixed_model(
                frame,
                outcome=axis,
                fixed_effects=[fixed_effect],
            )
            low, high = fitted.confidence_intervals[fixed_effect]
            output[axis] = {
                "mixed_model_status": (
                    "available" if fitted.converged else "unavailable_nonconvergence"
                ),
                "mixed_model_unavailable_reason": (
                    None if fitted.converged else "statsmodels optimizer did not converge"
                ),
                "mixed_model_effect": fitted.parameters[fixed_effect],
                "mixed_model_standard_error": fitted.standard_errors[fixed_effect],
                "mixed_model_ci_low": low,
                "mixed_model_ci_high": high,
                "mixed_model_p_value": fitted.p_values[fixed_effect],
                "mixed_model_converged": fitted.converged,
                "mixed_model_formula": fitted.formula,
                "mixed_model_optimizer": fitted.optimizer,
                "mixed_model_random_effect": "participant_intercept",
                "mixed_model_random_intercept_variance": fitted.random_intercept_variance,
                "mixed_model_n_participants": fitted.n_participants,
                "mixed_model_n_observations": fitted.n_observations,
            }
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
            output[axis] = {
                "mixed_model_status": "unavailable_fit_failure",
                "mixed_model_unavailable_reason": f"{type(error).__name__}: {error}",
                "mixed_model_random_effect": "participant_intercept",
            }
    return output


def _axis_redundancy_diagnostics(frame: pd.DataFrame) -> dict[str, Any]:
    if "binary_target" in frame:
        group = ["participant_id", "dataset_id", "binary_target"]
    elif "continuous_target" in frame:
        group = ["participant_id", "dataset_id", "continuous_target"]
    else:
        return {
            "status": "unavailable_missing_target",
            "reason": "axis redundancy requires a binary or continuous target column",
            "pairs": [],
        }
    collapsed = frame.groupby(group, as_index=False)[list(AXIS_NAMES)].mean()
    if len(collapsed) < 4:
        return {
            "status": "unavailable_insufficient_participant_cells",
            "reason": "axis redundancy requires at least four participant-condition cells",
            "n_participant_cells": len(collapsed),
            "pairs": [],
        }
    pearson = collapsed[list(AXIS_NAMES)].corr(method="pearson")
    spearman = collapsed[list(AXIS_NAMES)].corr(method="spearman")
    pairs = []
    for first_index, first in enumerate(AXIS_NAMES):
        for second in AXIS_NAMES[first_index + 1 :]:
            pearson_value = float(pearson.loc[first, second])
            spearman_value = float(spearman.loc[first, second])
            pairs.append(
                {
                    "axis_a": first,
                    "axis_b": second,
                    "pearson": pearson_value if np.isfinite(pearson_value) else None,
                    "spearman": spearman_value if np.isfinite(spearman_value) else None,
                }
            )
    finite = [abs(float(pair["spearman"])) for pair in pairs if pair["spearman"] is not None]
    standardized = collapsed[list(AXIS_NAMES)].to_numpy(dtype=float)
    standard_deviation = standardized.std(axis=0, ddof=1)
    if np.any(standard_deviation <= np.finfo(float).eps):
        condition_number = None
        condition_status = "unavailable_constant_axis"
    else:
        standardized = (standardized - standardized.mean(axis=0)) / standard_deviation
        singular_values = np.linalg.svd(standardized, compute_uv=False)
        if singular_values[-1] <= np.finfo(float).eps:
            condition_number = None
            condition_status = "available_singular_axis_system"
        else:
            condition_number = float(singular_values[0] / singular_values[-1])
            condition_status = "available"
    if not finite:
        return {
            "status": "unavailable_no_finite_correlations",
            "reason": "all pairwise axis correlations are undefined",
            "n_participant_cells": len(collapsed),
            "condition_number": condition_number,
            "condition_number_status": condition_status,
            "pairs": pairs,
        }
    return {
        "status": "available",
        "reason": None,
        "n_participant_cells": len(collapsed),
        "max_absolute_spearman": float(max(finite)),
        "condition_number": condition_number,
        "condition_number_status": condition_status,
        "pairs": pairs,
    }


def _attach_leave_one_property_out_deltas(
    rows: list[dict[str, Any]],
    *,
    bootstrap_distributions: Mapping[str, Mapping[str, Sequence[float]]],
    smallest_auc_difference: float,
) -> None:
    full = next((row for row in rows if row.get("model") == "five_axis"), None)
    if full is None:
        for row in rows:
            if str(row.get("model", "")).startswith("without_"):
                row["leave_one_property_out_status"] = "unavailable_missing_full_reference"
                row["leave_one_property_out_unavailable_reason"] = (
                    "the corresponding five-axis prediction was unavailable"
                )
        return
    metrics = ("auroc", "auprc", "balanced_accuracy", "brier", "ece")
    for row in rows:
        model = str(row.get("model", ""))
        if model == "five_axis":
            row["leave_one_property_out_status"] = "reference_full_model"
            row["omitted_property"] = None
        elif model.startswith("without_"):
            row["leave_one_property_out_status"] = "available"
            row["omitted_property"] = model.removeprefix("without_")
        else:
            continue
        for metric in metrics:
            row[f"delta_{metric}_vs_five_axis"] = float(row[metric] - full[metric])
        row["prediction_equivalence_smallest_auc_difference"] = smallest_auc_difference
        if model == "five_axis":
            row["prediction_equivalence_status"] = "reference_full_model"
            row["prediction_equivalence_unavailable_reason"] = None
            continue
        try:
            full_auroc = np.asarray(bootstrap_distributions["five_axis"]["auroc"], dtype=float)
            reduced_auroc = np.asarray(bootstrap_distributions[model]["auroc"], dtype=float)
            if len(full_auroc) != len(reduced_auroc):
                raise ValueError("paired AUROC bootstrap distributions do not align")
            interval = participant_bootstrap_tost_interval(
                reduced_auroc - full_auroc,
                estimate=float(row["delta_auroc_vs_five_axis"]),
                smallest_effect_size=smallest_auc_difference,
            )
            row.update(
                {
                    "prediction_equivalence_status": (
                        "equivalent" if interval.equivalent else "not_equivalent"
                    ),
                    "prediction_equivalence_unavailable_reason": None,
                    "prediction_equivalence_ci_low": interval.ci_low,
                    "prediction_equivalence_ci_high": interval.ci_high,
                    "prediction_equivalence_alpha": interval.alpha,
                    "prediction_equivalence_lower_test_p_value": (interval.lower_test_p_value),
                    "prediction_equivalence_upper_test_p_value": (interval.upper_test_p_value),
                    "prediction_equivalence_method": interval.method,
                }
            )
        except (KeyError, ValueError) as error:
            row["prediction_equivalence_status"] = (
                "unavailable_no_paired_bootstrap_difference_distribution"
            )
            row["prediction_equivalence_unavailable_reason"] = f"{type(error).__name__}: {error}"


def _estimand_metadata(
    frame: pd.DataFrame,
    *,
    estimand_id: str,
    estimand_role: str,
    sampling_basis: str,
) -> dict[str, Any]:
    return {
        "estimand_id": estimand_id,
        "estimand_role": estimand_role,
        "estimand_status": "available",
        "sampling_basis": sampling_basis,
        **overlap_output_fields(frame),
    }


def _unavailable_estimand(
    *,
    contrast_id: str,
    estimand_id: str,
    estimand_role: str,
    sampling_basis: str,
    component: str,
    error: Exception | str,
) -> dict[str, Any]:
    message = str(error) if isinstance(error, str) else f"{type(error).__name__}: {error}"
    return {
        "contrast_id": contrast_id,
        "estimand_id": estimand_id,
        "estimand_role": estimand_role,
        "sampling_basis": sampling_basis,
        "component": component,
        "status": "unavailable",
        "error": message,
    }


def run_models(
    *,
    profiles_path: str | Path,
    benchmarks_path: str | Path | None = None,
    matched_profiles_path: str | Path | None = None,
    contrasts_path: str | Path,
    output_root: str | Path,
    study: StudyConfig,
    repetitions: int | None = None,
) -> tuple[Path, Path, Path, Path]:
    """Run every configured contrast; failed contrasts remain in an issue ledger."""

    frame = pd.read_parquet(profiles_path)
    assert_no_direct_tms(frame, stage="general model input")
    required = {
        "participant_id",
        "dataset_id",
        "prediction_evaluation_eligible",
        *AXIS_NAMES,
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"profiles are missing {sorted(missing)}")
    frame = ensure_pretraining_overlap_columns(frame)
    benchmark_frame: pd.DataFrame | None = None
    if benchmarks_path is not None:
        benchmark_frame = pd.read_parquet(benchmarks_path)
        benchmark_required = {
            "participant_id",
            "dataset_id",
            "benchmark_status",
            *CONVENTIONAL_FEATURES,
        }
        benchmark_missing = benchmark_required.difference(benchmark_frame.columns)
        if benchmark_missing:
            raise ValueError(f"benchmarks are missing {sorted(benchmark_missing)}")
    matched_frame: pd.DataFrame | None = None
    if matched_profiles_path is not None:
        matched_frame = pd.read_parquet(matched_profiles_path)
    contrasts = _load_contrasts(contrasts_path)
    permutation_repeats = (
        study.statistics.permutation_repetitions if repetitions is None else repetitions
    )
    bootstrap_repeats = (
        study.statistics.participant_bootstrap_repetitions if repetitions is None else repetitions
    )
    if min(permutation_repeats, bootstrap_repeats) < 99:
        raise ValueError("at least 99 resamples are required")
    axes: list[dict[str, Any]] = []
    omnibus: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    estimands: list[dict[str, Any]] = []
    redundancy_audit: list[dict[str, Any]] = []
    prediction_bootstrap_distributions: dict[tuple[str, str, str], dict[str, list[float]]] = {}
    for index, contrast in enumerate(contrasts):
        contrast_id = str(contrast["id"])
        analysis_type = str(contrast.get("analysis_type", "binary"))
        seed = study.random_seeds[index % len(study.random_seeds)]
        try:
            profile_subset = _select_contrast(frame, contrast)
            if profile_subset.empty:
                raise ValueError("no rows match this contrast")
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
            primary_id = (
                CONTINUOUS_ALL_AVAILABLE_PRIMARY
                if analysis_type == "continuous"
                else REPEATED_EQUAL_WINDOW_PRIMARY
            )
            primary_basis = (
                "all_available_participant_condition_profiles"
                if analysis_type == "continuous"
                else "repeated_equal_window_profiles"
            )
            unavailable = _unavailable_estimand(
                contrast_id=contrast_id,
                estimand_id=primary_id,
                estimand_role="primary",
                sampling_basis=primary_basis,
                component="contrast_selection",
                error=error,
            )
            issues.append(unavailable)
            estimands.append(unavailable)
            continue

        blocks: list[tuple[pd.DataFrame, str, str, str]] = []
        if analysis_type == "continuous":
            blocks.append(
                (
                    profile_subset,
                    CONTINUOUS_ALL_AVAILABLE_PRIMARY,
                    "primary",
                    "all_available_participant_condition_profiles",
                )
            )
        else:
            if matched_frame is None:
                unavailable = _unavailable_estimand(
                    contrast_id=contrast_id,
                    estimand_id=REPEATED_EQUAL_WINDOW_PRIMARY,
                    estimand_role="primary",
                    sampling_basis="repeated_equal_window_profiles",
                    component="equal_window_primary",
                    error="matched equal-window profiles were not provided",
                )
                issues.append(unavailable)
                estimands.append(unavailable)
            else:
                try:
                    matched_subset = _matched_contrast_subset(
                        matched_frame,
                        contrast=contrast,
                        profile_subset=profile_subset,
                    )
                    blocks.append(
                        (
                            matched_subset,
                            REPEATED_EQUAL_WINDOW_PRIMARY,
                            "primary",
                            "repeated_equal_window_profiles",
                        )
                    )
                except ValueError as error:
                    unavailable = _unavailable_estimand(
                        contrast_id=contrast_id,
                        estimand_id=REPEATED_EQUAL_WINDOW_PRIMARY,
                        estimand_role="primary",
                        sampling_basis="repeated_equal_window_profiles",
                        component="equal_window_primary",
                        error=error,
                    )
                    issues.append(unavailable)
                    estimands.append(unavailable)
            blocks.append(
                (
                    profile_subset,
                    ALL_AVAILABLE_SENSITIVITY,
                    "secondary_sensitivity",
                    "all_available_participant_condition_profiles",
                )
            )

        for subset, estimand_id, estimand_role, sampling_basis in blocks:
            metadata = _estimand_metadata(
                profile_subset,
                estimand_id=estimand_id,
                estimand_role=estimand_role,
                sampling_basis=sampling_basis,
            )
            try:
                if analysis_type == "continuous":
                    axis_rows, omnibus_row = _continuous_inference(
                        subset,
                        contrast_id=contrast_id,
                        repetitions=permutation_repeats,
                        seed=seed,
                        bootstrap_repetitions=bootstrap_repeats,
                        equivalence_margin=study.statistics.continuous_smallest_effect,
                    )
                else:
                    axis_rows, omnibus_row = _contrast_inference(
                        subset,
                        contrast_id=contrast_id,
                        design=str(contrast["design"]),
                        repetitions=permutation_repeats,
                        seed=seed,
                        bootstrap_repetitions=bootstrap_repeats,
                        equivalence_margin=study.statistics.continuous_smallest_effect,
                    )
            except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
                unavailable = _unavailable_estimand(
                    contrast_id=contrast_id,
                    estimand_id=estimand_id,
                    estimand_role=estimand_role,
                    sampling_basis=sampling_basis,
                    component="inference",
                    error=error,
                )
                issues.append(unavailable)
                estimands.append(unavailable)
                continue
            mixed = _mixed_model_diagnostics(subset, analysis_type=analysis_type)
            for row in axis_rows:
                row.update(mixed[str(row["axis"])])
                row.update(metadata)
            redundancy = _axis_redundancy_diagnostics(subset)
            redundancy_record = {
                "contrast_id": contrast_id,
                "estimand_id": estimand_id,
                "estimand_role": estimand_role,
                "sampling_basis": sampling_basis,
                **redundancy,
            }
            redundancy_audit.append(redundancy_record)
            omnibus_row.update(
                {
                    "axis_redundancy_status": redundancy["status"],
                    "axis_redundancy_unavailable_reason": redundancy.get("reason"),
                    "axis_max_absolute_spearman": redundancy.get("max_absolute_spearman"),
                    "axis_condition_number": redundancy.get("condition_number"),
                    "axis_redundancy_json": json.dumps(
                        redundancy,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                }
            )
            omnibus_row.update(metadata)
            axes.extend(axis_rows)
            omnibus.append(omnibus_row)
            estimands.append(
                {
                    "contrast_id": contrast_id,
                    "estimand_id": estimand_id,
                    "estimand_role": estimand_role,
                    "sampling_basis": sampling_basis,
                    "component": "inference",
                    "status": "available",
                }
            )
            if analysis_type == "continuous":
                continue

            prediction_specs: list[tuple[str, Sequence[str]]] = [
                ("five_axis", AXIS_NAMES),
                *[
                    (f"without_{omitted}", [axis for axis in AXIS_NAMES if axis != omitted])
                    for omitted in AXIS_NAMES
                ],
            ]
            block_prediction_rows: list[dict[str, Any]] = []
            block_bootstrap_distributions: dict[str, dict[str, list[float]]] = {}
            for model_name, features in prediction_specs:
                try:
                    result = _prediction(
                        subset,
                        features=features,
                        folds=study.statistics.participant_stratified_folds,
                        seed=seed,
                        bootstrap_repetitions=bootstrap_repeats,
                    )
                except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
                    issues.append(
                        _unavailable_estimand(
                            contrast_id=contrast_id,
                            estimand_id=estimand_id,
                            estimand_role=estimand_role,
                            sampling_basis=sampling_basis,
                            component=f"{model_name}_prediction",
                            error=error,
                        )
                    )
                    continue
                distribution = result.pop("_participant_bootstrap_distributions")
                if not isinstance(distribution, dict):
                    raise TypeError("prediction bootstrap distribution has the wrong type")
                block_bootstrap_distributions[model_name] = distribution
                prediction_bootstrap_distributions[(contrast_id, estimand_id, model_name)] = (
                    distribution
                )
                block_prediction_rows.append(
                    {
                        "contrast_id": contrast_id,
                        "model": model_name,
                        **metadata,
                        **result,
                    }
                )
            _attach_leave_one_property_out_deltas(
                block_prediction_rows,
                bootstrap_distributions=block_bootstrap_distributions,
                smallest_auc_difference=(study.statistics.prediction_smallest_auc_difference),
            )
            predictions.extend(block_prediction_rows)
            try:
                leave_one_dataset_rows = _leave_one_dataset_out_predictions(
                    subset,
                    features=AXIS_NAMES,
                    seed=seed,
                    bootstrap_repetitions=bootstrap_repeats,
                )
                predictions.extend(
                    {
                        "contrast_id": contrast_id,
                        "model": "five_axis_leave_one_dataset_out",
                        **metadata,
                        **result,
                    }
                    for result in leave_one_dataset_rows
                )
            except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
                issues.append(
                    _unavailable_estimand(
                        contrast_id=contrast_id,
                        estimand_id=estimand_id,
                        estimand_role=estimand_role,
                        sampling_basis=sampling_basis,
                        component="leave_one_dataset_out_prediction",
                        error=error,
                    )
                )

        if analysis_type != "continuous" and benchmark_frame is not None:
            secondary_metadata = _estimand_metadata(
                profile_subset,
                estimand_id=ALL_AVAILABLE_SENSITIVITY,
                estimand_role="secondary_sensitivity",
                sampling_basis="all_available_participant_condition_profiles",
            )
            try:
                conventional_subset = _select_contrast(benchmark_frame, contrast)
                conventional_subset = conventional_subset[
                    conventional_subset["benchmark_status"].eq("computed")
                ].copy()
                conventional_subset[list(CONVENTIONAL_FEATURES)] = conventional_subset[
                    list(CONVENTIONAL_FEATURES)
                ].replace([np.inf, -np.inf], np.nan)
                valid_conventional = (
                    conventional_subset[list(CONVENTIONAL_FEATURES)].notna().all(axis=1)
                )
                conventional_subset = conventional_subset[valid_conventional].copy()
                cell_keys = ["participant_id", "dataset_id", "binary_target"]
                eligibility = profile_subset[
                    [
                        *cell_keys,
                        "prediction_evaluation_eligible",
                    ]
                ].drop_duplicates()
                if eligibility.duplicated(cell_keys, keep=False).any():
                    raise ValueError("profile rows disagree on prediction evaluation eligibility")
                profile_key_set = {
                    tuple(value)
                    for value in eligibility[cell_keys].itertuples(index=False, name=None)
                }
                conventional_key_set = {
                    tuple(value)
                    for value in conventional_subset[cell_keys]
                    .drop_duplicates()
                    .itertuples(index=False, name=None)
                }
                if conventional_key_set != profile_key_set:
                    missing_cells = len(profile_key_set - conventional_key_set)
                    extra_cells = len(conventional_key_set - profile_key_set)
                    raise ValueError(
                        "conventional baseline cell keys differ from the corresponding "
                        f"five-axis estimand (missing={missing_cells}, extra={extra_cells})"
                    )
                conventional_subset = conventional_subset.merge(
                    eligibility,
                    on=cell_keys,
                    how="inner",
                    validate="many_to_one",
                )
                corresponding_five_axis = next(
                    (
                        row
                        for row in predictions
                        if row.get("contrast_id") == contrast_id
                        and row.get("estimand_id") == ALL_AVAILABLE_SENSITIVITY
                        and row.get("model") == "five_axis"
                    ),
                    None,
                )
                if corresponding_five_axis is None:
                    raise ValueError(
                        "corresponding all-available five-axis prediction is unavailable"
                    )
                conventional = _prediction(
                    conventional_subset,
                    features=CONVENTIONAL_FEATURES,
                    folds=study.statistics.participant_stratified_folds,
                    seed=seed,
                    bootstrap_repetitions=bootstrap_repeats,
                )
                conventional_distribution = conventional.pop("_participant_bootstrap_distributions")
                if (
                    conventional["n_observations"] != corresponding_five_axis["n_observations"]
                    or conventional["n_evaluation_participants"]
                    != corresponding_five_axis["n_evaluation_participants"]
                    or conventional["n_fixed_transform_participants"]
                    != corresponding_five_axis["n_fixed_transform_participants"]
                ):
                    raise ValueError(
                        "conventional and five-axis predictions do not have identical "
                        "participant/sample counts"
                    )
                match_id = f"{contrast_id}:{ALL_AVAILABLE_SENSITIVITY}:exact_cells"
                cell_key_hash = hashlib.sha256(
                    json.dumps(
                        sorted([list(value) for value in profile_key_set]),
                        sort_keys=True,
                        separators=(",", ":"),
                        default=str,
                    ).encode()
                ).hexdigest()
                matched_fields = {
                    "baseline_match_id": match_id,
                    "baseline_match_status": "available_exact_participant_condition_cells",
                    "baseline_matched_cell_count": len(profile_key_set),
                    "baseline_matched_participant_count": len(
                        {value[0] for value in profile_key_set}
                    ),
                    "baseline_cell_keys_verified_identical": True,
                    "baseline_cell_key_sha256": cell_key_hash,
                }
                corresponding_five_axis.update(matched_fields)
                conventional_row = {
                    "contrast_id": contrast_id,
                    "model": "conventional_scalar",
                    **secondary_metadata,
                    **conventional,
                    **matched_fields,
                }
                for metric in ("auroc", "auprc", "balanced_accuracy", "brier", "ece"):
                    conventional_row[f"delta_{metric}_vs_matched_five_axis"] = float(
                        conventional_row[metric] - corresponding_five_axis[metric]
                    )
                    corresponding_five_axis[f"delta_{metric}_vs_matched_five_axis"] = 0.0
                conventional_row["prediction_equivalence_smallest_auc_difference"] = (
                    study.statistics.prediction_smallest_auc_difference
                )
                full_distribution = prediction_bootstrap_distributions[
                    (contrast_id, ALL_AVAILABLE_SENSITIVITY, "five_axis")
                ]
                full_auroc = np.asarray(full_distribution["auroc"], dtype=float)
                conventional_auroc = np.asarray(conventional_distribution["auroc"], dtype=float)
                if len(full_auroc) != len(conventional_auroc):
                    raise ValueError("paired conventional/five-axis AUROC bootstraps do not align")
                interval = participant_bootstrap_tost_interval(
                    conventional_auroc - full_auroc,
                    estimate=float(conventional_row["delta_auroc_vs_matched_five_axis"]),
                    smallest_effect_size=(study.statistics.prediction_smallest_auc_difference),
                )
                conventional_row.update(
                    {
                        "prediction_equivalence_status": (
                            "equivalent" if interval.equivalent else "not_equivalent"
                        ),
                        "prediction_equivalence_unavailable_reason": None,
                        "prediction_equivalence_ci_low": interval.ci_low,
                        "prediction_equivalence_ci_high": interval.ci_high,
                        "prediction_equivalence_alpha": interval.alpha,
                        "prediction_equivalence_lower_test_p_value": (interval.lower_test_p_value),
                        "prediction_equivalence_upper_test_p_value": (interval.upper_test_p_value),
                        "prediction_equivalence_method": interval.method,
                    }
                )
                predictions.append(conventional_row)
            except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
                issues.append(
                    _unavailable_estimand(
                        contrast_id=contrast_id,
                        estimand_id=ALL_AVAILABLE_SENSITIVITY,
                        estimand_role="secondary_sensitivity",
                        sampling_basis="all_available_participant_condition_profiles",
                        component="conventional_scalar_prediction",
                        error=error,
                    )
                )
    destination = Path(output_root)
    if not omnibus:
        audit_path = destination / "model-audit.json"
        atomic_write_json(
            audit_path,
            {
                "schema_version": 1,
                "profiles_sha256": sha256_file(profiles_path),
                "benchmarks_sha256": (
                    sha256_file(benchmarks_path) if benchmarks_path is not None else None
                ),
                "matched_profiles_sha256": (
                    sha256_file(matched_profiles_path)
                    if matched_profiles_path is not None
                    else None
                ),
                "contrasts_sha256": sha256_file(contrasts_path),
                "configured_contrasts": len(contrasts),
                "completed_inference": 0,
                "completed_primary_inference": 0,
                "permutation_repetitions": permutation_repeats,
                "participant_bootstrap_repetitions": bootstrap_repeats,
                "participant_bootstrap_unit": "participant",
                "equivalence_smallest_effect_size": (study.statistics.continuous_smallest_effect),
                "prediction_smallest_auc_difference": (
                    study.statistics.prediction_smallest_auc_difference
                ),
                "axis_redundancy": redundancy_audit,
                "estimands": estimands,
                "pretraining_overlap": summarize_pretraining_overlap(frame),
                "issues": issues,
                "technical_failure": "zero_valid_inference_blocks",
                "scientific_gate_applied": False,
            },
        )
        raise RuntimeError("model phase produced zero valid inference blocks; see model-audit.json")
    axis_path = _atomic_parquet(pd.DataFrame(axes), destination / "axis-contrasts.parquet")
    omnibus_path = _atomic_parquet(pd.DataFrame(omnibus), destination / "omnibus-contrasts.parquet")
    prediction_path = _atomic_parquet(
        pd.DataFrame(predictions), destination / "predictions.parquet"
    )
    audit_path = destination / "model-audit.json"
    atomic_write_json(
        audit_path,
        {
            "schema_version": 1,
            "profiles_sha256": sha256_file(profiles_path),
            "benchmarks_sha256": (
                sha256_file(benchmarks_path) if benchmarks_path is not None else None
            ),
            "matched_profiles_sha256": (
                sha256_file(matched_profiles_path) if matched_profiles_path is not None else None
            ),
            "contrasts_sha256": sha256_file(contrasts_path),
            "configured_contrasts": len(contrasts),
            "completed_inference": len(omnibus),
            "completed_primary_inference": sum(
                row.get("estimand_role") == "primary" for row in omnibus
            ),
            "completed_secondary_sensitivity_inference": sum(
                row.get("estimand_role") == "secondary_sensitivity" for row in omnibus
            ),
            "conventional_prediction_completed": sum(
                row.get("model") == "conventional_scalar" for row in predictions
            ),
            "conventional_baseline_exact_cell_match_required": True,
            "leave_one_dataset_out_prediction_completed": sum(
                row.get("model") == "five_axis_leave_one_dataset_out" for row in predictions
            ),
            "permutation_repetitions": permutation_repeats,
            "participant_bootstrap_repetitions": bootstrap_repeats,
            "participant_bootstrap_unit": "participant",
            "equivalence_smallest_effect_size": (study.statistics.continuous_smallest_effect),
            "equivalence_interval_level": 0.90,
            "equivalence_method": "participant_cluster_bootstrap_percentile_tost",
            "prediction_smallest_auc_difference": (
                study.statistics.prediction_smallest_auc_difference
            ),
            "axis_redundancy": redundancy_audit,
            "mixed_model_status_counts": {
                status: sum(row.get("mixed_model_status") == status for row in axes)
                for status in sorted(
                    {
                        str(row.get("mixed_model_status"))
                        for row in axes
                        if row.get("mixed_model_status") is not None
                    }
                )
            },
            "estimands": estimands,
            "pretraining_overlap": summarize_pretraining_overlap(frame),
            "issues": issues,
            "resampling_unit": "participant",
            "permutations_plus_one": True,
            "scientific_gate_applied": False,
        },
    )
    return axis_path, omnibus_path, prediction_path, audit_path
