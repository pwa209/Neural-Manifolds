"""Participant-level contrasts and leakage-safe predictive comparisons."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from neural_manifolds.config import StudyConfig
from neural_manifolds.manifold.profile import AXIS_NAMES
from neural_manifolds.provenance import atomic_write_json, sha256_file
from neural_manifolds.stages.benchmarks import CONVENTIONAL_FEATURES
from neural_manifolds.statistics.multivariate import benjamini_hochberg
from neural_manifolds.statistics.resampling import participant_folds


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
    if (pd.to_numeric(subset["successful_repeats"], errors="coerce") <= 0).any():
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
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
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
    rng = np.random.default_rng(seed)
    participants = collapsed["participant_id"].unique()
    bootstrap = []
    null = []
    by_participant = {name: collapsed[collapsed["participant_id"] == name] for name in participants}
    observed_participant_slopes = []
    for name in participants:
        group = by_participant[name]
        gx = group["continuous_target"] - group["continuous_target"].mean()
        gy = group[list(AXIS_NAMES)] - group[list(AXIS_NAMES)].mean()
        denom = float(np.sum(gx.to_numpy() ** 2))
        if denom > 0:
            observed_participant_slopes.append(
                (gx.to_numpy()[:, None] * gy.to_numpy()).sum(0) / denom
            )
    if len(observed_participant_slopes) != len(participants):
        raise ValueError("a participant has no within-participant covariate variation")
    # The observed estimate, cluster bootstrap, and permutation distribution all
    # use the same estimand: the equally weighted mean participant-specific slope.
    slopes = np.mean(observed_participant_slopes, axis=0)
    for _ in range(repetitions):
        sampled = rng.choice(participants, len(participants), replace=True)
        # Cluster bootstrap rows can duplicate participant IDs; compute each draw
        # directly as a mean of participant-specific slopes to retain equal weight.
        participant_slopes = []
        for name in sampled:
            group = by_participant[name]
            gx = group["continuous_target"] - group["continuous_target"].mean()
            gy = group[list(AXIS_NAMES)] - group[list(AXIS_NAMES)].mean()
            denom = float(np.sum(gx.to_numpy() ** 2))
            if denom > 0:
                participant_slopes.append((gx.to_numpy()[:, None] * gy.to_numpy()).sum(0) / denom)
        bootstrap.append(np.mean(participant_slopes, axis=0))
        permuted_parts = []
        for name in participants:
            group = by_participant[name]
            permuted = rng.permutation(group["continuous_target"].to_numpy())
            gx = permuted - permuted.mean()
            gy = group[list(AXIS_NAMES)].to_numpy(copy=True)
            gy -= gy.mean(0)
            denom = float(np.sum(gx**2))
            if denom > 0:
                permuted_parts.append((gx[:, None] * gy).sum(0) / denom)
        null.append(np.mean(permuted_parts, axis=0))
    bootstrap_values = np.stack(bootstrap)
    null_values = np.stack(null)
    p_values = (np.count_nonzero(np.abs(null_values) >= np.abs(slopes), axis=0) + 1) / (
        repetitions + 1
    )
    adjusted, rejected = benjamini_hochberg(p_values)
    rows = []
    for index, axis in enumerate(AXIS_NAMES):
        low, high = np.quantile(bootstrap_values[:, index], [0.025, 0.975])
        rows.append(
            {
                "contrast_id": contrast_id,
                "axis": axis,
                "effect": float(slopes[index]),
                "ci_low": float(low),
                "ci_high": float(high),
                "p_value": float(p_values[index]),
                "q_value": float(adjusted[index]),
                "fdr_reject": bool(rejected[index]),
                "n_participants": len(participants),
                "design": "within_participant_continuous",
            }
        )
    observed_norm = float(np.linalg.norm(slopes))
    null_norm = np.linalg.norm(null_values, axis=1)
    return rows, {
        "contrast_id": contrast_id,
        "profile_distance": observed_norm,
        "p_value": float((np.count_nonzero(null_norm >= observed_norm) + 1) / (repetitions + 1)),
        "n_participants": len(participants),
        "permutations": repetitions,
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
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = np.random.default_rng(seed)
    if design == "within_participant":
        grouped = frame.groupby(["participant_id", "binary_target"])[list(AXIS_NAMES)].mean()
        wide = grouped.unstack("binary_target").dropna()
        differences = (
            wide.xs(1, axis=1, level=1).to_numpy() - wide.xs(0, axis=1, level=1).to_numpy()
        )
        n_participants = len(differences)
        observed = differences.mean(axis=0)
        bootstrap = np.stack(
            [
                differences[rng.integers(0, n_participants, n_participants)].mean(axis=0)
                for _ in range(repetitions)
            ]
        )
        signs = rng.choice((-1.0, 1.0), size=(repetitions, n_participants, 1))
        null = (signs * differences[None]).mean(axis=1)
    else:
        collapsed = _participant_rows(frame)
        values = collapsed[list(AXIS_NAMES)].to_numpy()
        labels = collapsed["binary_target"].to_numpy(dtype=int)
        n_participants = len(collapsed)
        observed = values[labels == 1].mean(axis=0) - values[labels == 0].mean(axis=0)
        bootstrap_rows = []
        for _ in range(repetitions):
            group_values = []
            for label in (0, 1):
                indices = np.flatnonzero(labels == label)
                group_values.append(
                    values[rng.choice(indices, len(indices), replace=True)].mean(axis=0)
                )
            bootstrap_rows.append(group_values[1] - group_values[0])
        bootstrap = np.stack(bootstrap_rows)
        null_rows = []
        for _ in range(repetitions):
            shuffled = rng.permutation(labels)
            null_rows.append(values[shuffled == 1].mean(0) - values[shuffled == 0].mean(0))
        null = np.stack(null_rows)
    p_values = (np.count_nonzero(np.abs(null) >= np.abs(observed), axis=0) + 1) / (repetitions + 1)
    adjusted, rejected = benjamini_hochberg(p_values)
    axis_rows = []
    for index, axis in enumerate(AXIS_NAMES):
        low, high = np.quantile(bootstrap[:, index], [0.025, 0.975])
        axis_rows.append(
            {
                "contrast_id": contrast_id,
                "axis": axis,
                "effect": float(observed[index]),
                "ci_low": float(low),
                "ci_high": float(high),
                "p_value": float(p_values[index]),
                "q_value": float(adjusted[index]),
                "fdr_reject": bool(rejected[index]),
                "n_participants": n_participants,
                "design": design,
            }
        )
    observed_norm = float(np.linalg.norm(observed))
    null_norm = np.linalg.norm(null, axis=1)
    omnibus = {
        "contrast_id": contrast_id,
        "profile_distance": observed_norm,
        "p_value": float((np.count_nonzero(null_norm >= observed_norm) + 1) / (repetitions + 1)),
        "n_participants": n_participants,
        "permutations": repetitions,
    }
    return axis_rows, omnibus


def _expected_calibration_error(
    labels: np.ndarray, probabilities: np.ndarray, bins: int = 10
) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    value = 0.0
    for lower, upper in pairwise(edges):
        mask = (probabilities >= lower) & (
            probabilities <= upper if upper == 1.0 else probabilities < upper
        )
        if np.any(mask):
            value += mask.mean() * abs(labels[mask].mean() - probabilities[mask].mean())
    return float(value)


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


def _prediction(
    frame: pd.DataFrame,
    *,
    features: Sequence[str],
    folds: int,
    seed: int,
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
    n_splits = min(folds, len(evaluation_participants))
    scored_labels = labels[eligible]
    if n_splits < 3 or len(np.unique(scored_labels)) != 2:
        raise ValueError(
            "prediction requires two classes and at least three untouched evaluation participants"
        )
    probabilities = np.full(len(collapsed), np.nan)
    selected_c: list[float] = []
    # Split only untouched evaluation participants. Discovery/validation
    # participants remain fixed training rows and are never scored.
    for _, evaluation_test in participant_folds(
        evaluation_participants,
        n_splits=n_splits,
        seed=seed,
    ):
        test_participants = evaluation_participants[evaluation_test]
        test = np.flatnonzero(eligible & np.isin(participant, test_participants))
        train = np.flatnonzero(~np.isin(participant, test_participants))
        if set(participant[train]).intersection(participant[test]):
            raise AssertionError("participant leakage detected in outer prediction fold")
        if len(np.unique(labels[train])) != 2:
            raise ValueError("an outer training fold contains one class")
        best_score = -np.inf
        best_c = 1.0
        inner_candidates = np.unique(participant[train][eligible[train]])
        inner_splits = min(3, len(inner_candidates))
        for c_value in (0.01, 0.1, 1.0, 10.0):
            scores = []
            if inner_splits >= 2:
                for _, inner_test_indices in participant_folds(
                    inner_candidates,
                    n_splits=inner_splits,
                    seed=seed + 1,
                ):
                    inner_test_participants = inner_candidates[inner_test_indices]
                    inner_test = train[np.isin(participant[train], inner_test_participants)]
                    inner_train = train[~np.isin(participant[train], inner_test_participants)]
                    if set(participant[inner_train]).intersection(participant[inner_test]):
                        raise AssertionError(
                            "participant leakage detected in inner prediction fold"
                        )
                    if len(np.unique(labels[inner_train])) != 2:
                        continue
                    model = make_pipeline(
                        StandardScaler(),
                        LogisticRegression(
                            C=c_value,
                            l1_ratio=0.0,
                            class_weight="balanced",
                            max_iter=2000,
                            random_state=seed,
                        ),
                    )
                    model.fit(values[inner_train], labels[inner_train])
                    predicted = model.predict_proba(values[inner_test])[:, 1]
                    if len(np.unique(labels[inner_test])) == 2:
                        scores.append(roc_auc_score(labels[inner_test], predicted))
            score = float(np.mean(scores)) if scores else -np.inf
            if score > best_score:
                best_score, best_c = score, c_value
        selected_c.append(best_c)
        final = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=best_c,
                l1_ratio=0.0,
                class_weight="balanced",
                max_iter=2000,
                random_state=seed,
            ),
        )
        final.fit(values[train], labels[train])
        probabilities[test] = final.predict_proba(values[test])[:, 1]
    if not np.all(np.isfinite(probabilities[eligible])):
        raise RuntimeError("outer predictions are incomplete")
    scored_probabilities = probabilities[eligible]
    hard = scored_probabilities >= 0.5
    return {
        "features": list(features),
        "auroc": float(roc_auc_score(scored_labels, scored_probabilities)),
        "auprc": float(average_precision_score(scored_labels, scored_probabilities)),
        "balanced_accuracy": float(balanced_accuracy_score(scored_labels, hard)),
        "brier": float(brier_score_loss(scored_labels, scored_probabilities)),
        "ece": _expected_calibration_error(scored_labels, scored_probabilities),
        "n_participants": len(evaluation_participants),
        "n_evaluation_participants": len(evaluation_participants),
        "n_fixed_transform_participants": len(fixed_participants),
        "n_observations": int(eligible.sum()),
        "n_training_observations_total": len(collapsed),
        "selected_c": selected_c,
        "representation_heldout": True,
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
    required = {
        "participant_id",
        "dataset_id",
        "prediction_evaluation_eligible",
        *AXIS_NAMES,
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"profiles are missing {sorted(missing)}")
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
    repeats = repetitions or study.statistics.permutation_repetitions
    if repeats < 99:
        raise ValueError("at least 99 resamples are required")
    axes: list[dict[str, Any]] = []
    omnibus: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    for index, contrast in enumerate(contrasts):
        contrast_id = str(contrast["id"])
        try:
            profile_subset = _select_contrast(frame, contrast)
            if profile_subset.empty:
                raise ValueError("no rows match this contrast")
            subset = profile_subset
            sampling_basis = "all_available_participant_condition_profiles"
            if matched_frame is not None and contrast.get("analysis_type") != "continuous":
                try:
                    subset = _matched_contrast_subset(
                        matched_frame,
                        contrast=contrast,
                        profile_subset=profile_subset,
                    )
                    sampling_basis = "repeated_equal_window_profiles"
                except ValueError as error:
                    issues.append(
                        {
                            "contrast_id": contrast_id,
                            "component": "equal_window_primary",
                            "error": f"ValueError: {error}",
                            "fallback": sampling_basis,
                        }
                    )
            if contrast.get("analysis_type") == "continuous":
                axis_rows, omnibus_row = _continuous_inference(
                    subset,
                    contrast_id=contrast_id,
                    repetitions=repeats,
                    seed=study.random_seeds[index % len(study.random_seeds)],
                )
            else:
                axis_rows, omnibus_row = _contrast_inference(
                    subset,
                    contrast_id=contrast_id,
                    design=str(contrast["design"]),
                    repetitions=repeats,
                    seed=study.random_seeds[index % len(study.random_seeds)],
                )
            for row in axis_rows:
                row["sampling_basis"] = sampling_basis
            omnibus_row["sampling_basis"] = sampling_basis
            axes.extend(axis_rows)
            omnibus.append(omnibus_row)
            if contrast.get("analysis_type") == "continuous":
                continue
            full = _prediction(
                subset,
                features=AXIS_NAMES,
                folds=study.statistics.participant_stratified_folds,
                seed=study.random_seeds[index % len(study.random_seeds)],
            )
            predictions.append(
                {
                    "contrast_id": contrast_id,
                    "model": "five_axis",
                    "sampling_basis": sampling_basis,
                    **full,
                }
            )
            for omitted in AXIS_NAMES:
                features = [axis for axis in AXIS_NAMES if axis != omitted]
                result = _prediction(
                    subset,
                    features=features,
                    folds=study.statistics.participant_stratified_folds,
                    seed=study.random_seeds[index % len(study.random_seeds)],
                )
                predictions.append(
                    {
                        "contrast_id": contrast_id,
                        "model": f"without_{omitted}",
                        "sampling_basis": sampling_basis,
                        **result,
                    }
                )
            if benchmark_frame is not None:
                try:
                    conventional_subset = _select_contrast(benchmark_frame, contrast)
                    conventional_subset = conventional_subset[
                        conventional_subset["benchmark_status"].eq("computed")
                    ].copy()
                    conventional_subset[list(CONVENTIONAL_FEATURES)] = conventional_subset[
                        list(CONVENTIONAL_FEATURES)
                    ].replace([np.inf, -np.inf], np.nan)
                    conventional_subset = conventional_subset.dropna(
                        subset=list(CONVENTIONAL_FEATURES)
                    )
                    eligibility = subset[
                        [
                            "participant_id",
                            "dataset_id",
                            "binary_target",
                            "prediction_evaluation_eligible",
                        ]
                    ].drop_duplicates()
                    if eligibility.duplicated(
                        ["participant_id", "dataset_id", "binary_target"], keep=False
                    ).any():
                        raise ValueError(
                            "profile rows disagree on prediction evaluation eligibility"
                        )
                    conventional_subset = conventional_subset.merge(
                        eligibility,
                        on=["participant_id", "dataset_id", "binary_target"],
                        how="inner",
                        validate="many_to_one",
                    )
                    conventional = _prediction(
                        conventional_subset,
                        features=CONVENTIONAL_FEATURES,
                        folds=study.statistics.participant_stratified_folds,
                        seed=study.random_seeds[index % len(study.random_seeds)],
                    )
                    predictions.append(
                        {
                            "contrast_id": contrast_id,
                            "model": "conventional_scalar",
                            "sampling_basis": "all_available_participant_condition_profiles",
                            **conventional,
                        }
                    )
                except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
                    issues.append(
                        {
                            "contrast_id": contrast_id,
                            "component": "conventional_scalar_prediction",
                            "error": f"{type(error).__name__}: {error}",
                        }
                    )
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
            issues.append({"contrast_id": contrast_id, "error": f"{type(error).__name__}: {error}"})
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
            "conventional_prediction_completed": sum(
                row.get("model") == "conventional_scalar" for row in predictions
            ),
            "issues": issues,
            "resampling_unit": "participant",
            "permutations_plus_one": True,
            "scientific_gate_applied": False,
        },
    )
    return axis_path, omnibus_path, prediction_path, audit_path
