"""Nested study transfer with fold-contained trajectory fitting and paired losses."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from neural_manifolds.manifold.alignment import pairwise_module_alignment
from neural_manifolds.manifold.directionality import estimate_directionality
from neural_manifolds.manifold.metastability import estimate_metastability
from neural_manifolds.manifold.reachability import (
    fit_local_linear_dynamics,
    state_weighted_reachability,
)
from neural_manifolds.manifold.repertoire import estimate_repertoire
from neural_manifolds.provenance import atomic_write_json, sha256_file
from neural_manifolds.statistics.study_transfer import independent_weights, validate_units

SPECTRAL = ["delta", "theta", "alpha", "beta", "spectral_exponent"]
CONVENTIONAL = [*SPECTRAL, "lz", "permutation_entropy", "log_rms"]
DYNAMICS = [
    "repertoire",
    "log_dwell",
    "dwell_dispersion",
    "recurrence",
    "exit_entropy",
    "directionality",
    "alignment",
    "reachability",
]


class FoldDynamics:
    """Small shared dictionary learned only from a specified training partition."""

    def __init__(self, *, dimensions=2, states=3, seed=20260916):
        self.dimensions, self.states, self.seed = dimensions, states, seed

    def fit(self, frame: pd.DataFrame, arrays: dict):
        blocks = []
        # Equal participants and studies for unsupervised fitting as for prediction.
        weights = independent_weights(frame)
        sample_weights = []
        for weight, unit in zip(weights, frame.unit_id, strict=True):
            x = arrays[unit]["trajectory"]
            blocks.append(x)
            sample_weights.extend([weight / len(x)] * len(x))
        x = np.concatenate(blocks)
        w = np.asarray(sample_weights)
        if not np.isfinite(x).all() or len(x) < self.states * 5:
            raise ValueError("insufficient_finite_training_trajectory")
        self.center_ = np.average(x, axis=0, weights=w)
        # Weighted covariance PCA, rather than letting prolific participants dominate.
        centered = x - self.center_
        covariance = (centered.T * (w / w.sum())) @ centered
        eigenvalues, vectors = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues)[::-1]
        self.components_ = vectors[:, order[: min(self.dimensions, x.shape[1])]]
        self.scale_ = np.sqrt(np.maximum(eigenvalues[order[: self.components_.shape[1]]], 1e-12))
        projected = self.project(x)
        self.dictionary_ = KMeans(n_clusters=self.states, n_init=10, random_state=self.seed).fit(
            projected, sample_weight=w
        )
        self.training_ids_ = sorted(frame.unit_id.tolist())
        return self

    def project(self, x):
        return ((x - self.center_) @ self.components_) / self.scale_

    def transform_one(self, record: dict) -> tuple[dict, dict]:
        x, segments = record["trajectory"], record["segments"]
        z = self.project(x)
        states = self.dictionary_.predict(z)
        result, unavailable = {}, {}
        operations = {
            "repertoire": lambda: estimate_repertoire(x, shrinkage="oas").participation_ratio,
            "directionality": lambda: (
                estimate_directionality(
                    states, state_labels=np.arange(self.states), segment_ids=segments
                ).entropy_production
            ),
            "alignment": lambda: (
                pairwise_module_alignment(
                    record["regions"], rank=1, cv=2, ridge=0.1, segment_ids=segments
                ).mean_shared_predictive_variance
            ),
        }
        meta = estimate_metastability(states, segment_ids=segments)
        result.update(
            log_dwell=float(np.log(meta.median_dwell)),
            dwell_dispersion=meta.dwell_dispersion,
            recurrence=meta.recurrence_probability,
            exit_entropy=meta.exit_entropy,
        )
        for name, operation in operations.items():
            try:
                value = float(operation())
                if not np.isfinite(value):
                    raise ValueError("nonfinite_estimate")
                result[name] = value
            except ValueError as exc:
                result[name], unavailable[name] = np.nan, str(exc)
        try:
            # Pooled local linear model avoids unsupported per-state 32-D fits.
            model = fit_local_linear_dynamics(
                z, segment_ids=segments, ridge=0.1, min_transitions=max(8, 4 * (z.shape[1] + 1))
            )
            result["reachability"] = state_weighted_reachability(
                model.transition_matrices,
                model.innovation_covariances,
                model.occupancy,
                horizon=5,
                regularization=1e-6,
            ).log_determinant
        except ValueError as exc:
            result["reachability"], unavailable["reachability"] = np.nan, str(exc)
        return result, unavailable

    def transform(self, frame: pd.DataFrame, arrays: dict) -> tuple[pd.DataFrame, list]:
        rows, failures = [], []
        for unit in frame.unit_id:
            values, missing = self.transform_one(arrays[unit])
            rows.append(values)
            failures.append({"unit_id": unit, "unavailable": missing})
        return pd.DataFrame(rows, index=frame.index), failures


def classifier(family: str, strength: float, dimension: int | None):
    steps = [("impute", SimpleImputer(keep_empty_features=True)), ("scale", StandardScaler())]
    if family == "nonlinear_scalar":
        steps.append(("polynomial", PolynomialFeatures(3, include_bias=False)))
    if dimension is not None:
        steps.append(("pca", PCA(n_components=dimension, svd_solver="full")))
    return Pipeline([*steps, ("model", LogisticRegression(C=strength, max_iter=3000))])


def family_columns(family):
    if family.startswith("dimension_"):
        return DYNAMICS
    if family.startswith("without_"):
        omitted = {
            "repertoire": ["repertoire"],
            "metastability": ["log_dwell", "dwell_dispersion", "recurrence", "exit_entropy"],
            "directionality": ["directionality"],
            "alignment": ["alignment"],
            "reachability": ["reachability"],
        }[family.removeprefix("without_")]
        return [c for c in DYNAMICS if c not in omitted]
    return {
        "spectral_arousal": SPECTRAL,
        "nonlinear_scalar": ["lz"],
        "conventional_multivariate": CONVENTIONAL,
        "shared_dynamics": DYNAMICS,
    }[family]


def _fit_predict(fit, test, features, family, strength, dimension):
    columns = family_columns(family)
    measured = [c for c in columns if features.loc[fit.index, c].notna().any()]
    if not measured or (dimension is not None and dimension > min(len(measured), len(fit))):
        raise ValueError("candidate_dimension_unavailable")
    model = classifier(family, strength, dimension)
    model.fit(
        features.loc[fit.index, measured],
        fit.experience.astype(int),
        model__sample_weight=independent_weights(fit),
    )
    return model.predict_proba(features.loc[test.index, measured])[:, 1], measured


def nested_transfer(
    frame: pd.DataFrame,
    arrays: dict,
    *,
    outer="study_group",
    seed=20260916,
    families=None,
    strengths=(0.1, 1.0, 10.0),
    dimensions=(1, 2, 3, 4, 5),
) -> dict:
    validate_units(frame, "experience")
    if (
        outer == "participant_id"
        and frame.get("participant_alias_status", pd.Series(dtype=str))
        .eq("within_oslo_unresolved")
        .any()
    ):
        raise ValueError("participant_transfer_requires_resolved_aliases")
    if frame[outer].nunique() < 3:
        raise ValueError("nested_transfer_requires_three_independent_groups")
    if set(frame.experience.unique()) != {0, 1}:
        raise ValueError("binary_report_categories_required")
    families = families or [
        "spectral_arousal",
        "nonlinear_scalar",
        "conventional_multivariate",
        "shared_dynamics",
    ]
    predictions, tuning, failures = [], [], []
    frame = frame.reset_index(drop=True)
    conventional = pd.DataFrame(frame.conventional.tolist(), index=frame.index)
    for held in sorted(frame[outer].unique()):
        train, test = frame[frame[outer] != held], frame[frame[outer] == held]
        if train.experience.nunique() != 2:
            failures.append({"held_group": held, "reason": "outer_training_one_category"})
            continue
        inner = GroupKFold(n_splits=min(5, train[outer].nunique()))
        partitions = []
        for a, b in inner.split(train, groups=train[outer]):
            fit, validation = train.iloc[a], train.iloc[b]
            if fit.experience.nunique() != 2:
                continue
            transform = FoldDynamics(seed=seed).fit(fit, arrays)
            values, unavailable = transform.transform(pd.concat([fit, validation]), arrays)
            features = conventional.join(values)
            partitions.append((fit, validation, features))
        if len(partitions) < 2:
            failures.append(
                {"held_group": held, "reason": "insufficient_two_class_inner_training_groups"}
            )
            continue
        transform = FoldDynamics(seed=seed).fit(train, arrays)
        dynamics, unavailable = transform.transform(frame, arrays)
        final_features = conventional.join(dynamics)
        for family in families:
            # Equal count of tuning evaluations; dimensions vary only for the profile.
            grid = (
                [(c, d) for c in strengths for d in dimensions]
                if family == "shared_dynamics"
                else [
                    (float(c), None)
                    for c in np.geomspace(
                        min(strengths), max(strengths), len(strengths) * len(dimensions)
                    )
                ]
            )
            if family.startswith("dimension_"):
                grid = [(c, int(family.removeprefix("dimension_"))) for c, _ in grid]
            choices = []
            for c, d in grid:
                losses = []
                try:
                    for fit, validation, features in partitions:
                        probability, _ = _fit_predict(fit, validation, features, family, c, d)
                        losses.append(
                            log_loss(
                                validation.experience,
                                probability,
                                labels=[0, 1],
                                sample_weight=independent_weights(validation),
                            )
                        )
                    choices.append((float(np.mean(losses)), c, d))
                except ValueError:
                    continue
            if not choices:
                failures.append(
                    {"held_group": held, "model": family, "reason": "no_estimable_inner_candidate"}
                )
                continue
            _, c, d = min(choices, key=lambda value: value[0])
            probability, columns = _fit_predict(train, test, final_features, family, c, d)
            for (_, row), p in zip(test.iterrows(), probability, strict=True):
                predictions.append(
                    {k: row[k] for k in ["unit_id", "participant_id", "study_group", "experience"]}
                    | {"model": family, "probability": float(p), "held_group": held}
                )
            tuning.append(
                {
                    "held_group": held,
                    "model": family,
                    "strength": c,
                    "profile_dimension": d,
                    "features": columns,
                    "training_ids": transform.training_ids_,
                    "inner_losses": choices,
                    "tuning_trials_requested": len(grid),
                    "tuning_trials_estimable": len(choices),
                    "axis_availability": unavailable,
                }
            )
    return {
        "predictions": predictions,
        "tuning": tuning,
        "unavailable": failures,
        "outer_unit": outer,
        "all_fitted_transforms_inside_inner_and_outer_training": True,
        "metastability": "raw_persistence_revisability_descriptors_not_assumed_wake_optimum",
        "reachability": "regularized_pooled_local_linear_proxy_not_physical_control",
    }


def summarize_predictions(predictions: pd.DataFrame, *, repetitions=1000, seed=20260916) -> dict:
    summaries, comparisons = [], []
    if predictions.empty:
        return {"models": [], "paired_comparisons": [], "status": "no_estimable_predictions"}
    for (study, model), block in predictions.groupby(["study_group", "model"]):
        weights = independent_weights(block)
        y, p = block.experience.to_numpy(int), block.probability.to_numpy(float)
        summaries.append(
            {
                "study_group": study,
                "model": model,
                "participants": block.participant_id.nunique(),
                "log_loss": float(log_loss(y, p, labels=[0, 1], sample_weight=weights)),
                "brier": float(brier_score_loss(y, p, sample_weight=weights)),
                "auroc": float(roc_auc_score(y, p, sample_weight=weights))
                if len(set(y)) == 2
                else None,
                "average_precision": float(average_precision_score(y, p, sample_weight=weights))
                if len(set(y)) == 2
                else None,
                "mean_predicted": float(np.average(p, weights=weights)),
                "observed_rate": float(np.average(y, weights=weights)),
            }
        )
    p = predictions.copy()
    probability = np.clip(p.probability.to_numpy(float), 1e-12, 1 - 1e-12)
    p["loss"] = -p.experience.to_numpy() * np.log(probability) - (
        1 - p.experience.to_numpy()
    ) * np.log1p(-probability)
    pivot = p.pivot(
        index=["unit_id", "participant_id", "study_group"], columns="model", values="loss"
    )
    rng = np.random.default_rng(seed)
    if "shared_dynamics" in pivot:
        for baseline in sorted(set(pivot) - {"shared_dynamics"}):
            paired = pivot[[baseline, "shared_dynamics"]].dropna()
            delta = (paired["shared_dynamics"] - paired[baseline]).rename("delta").reset_index()
            if delta.empty:
                continue
            persons = delta.groupby(["study_group", "participant_id"]).delta.mean()
            groups = {g: v.to_numpy() for g, v in persons.groupby(level=0)}
            draws = []
            keys = list(groups)
            for _ in range(repetitions):
                selected = rng.choice(keys, len(keys), replace=True)
                draws.append(
                    np.mean(
                        [
                            rng.choice(groups[g], len(groups[g]), replace=True).mean()
                            for g in selected
                        ]
                    )
                )
            comparisons.append(
                {
                    "baseline": baseline,
                    "dynamic_minus_baseline_log_loss": float(
                        np.mean([v.mean() for v in groups.values()])
                    ),
                    "interval_95": np.quantile(draws, [0.025, 0.975]).tolist(),
                    "independent_studies": len(keys),
                    "paired_units": len(delta),
                    "uncertainty": "hierarchical_study_participant_bootstrap_conditional_on_fitted_predictions",
                    "few_studies_warning": len(keys) < 10,
                }
            )
    return {"models": summaries, "paired_comparisons": comparisons}


def load_measurements(measurement_paths, *, primary=True):
    return [
        row
        for p in measurement_paths
        for row in json.loads(p.read_text())["records"]
        if row["status"] == "measured" and (row["primary_eligible"] or not primary)
    ]


def measurement_arrays(selected, representation):
    arrays, admitted = {}, []
    for row in selected:
        if sha256_file(row["array_path"]) != row["array_sha256"]:
            raise ValueError("measurement_checksum_changed")
        with np.load(row["array_path"], allow_pickle=False) as data:
            if representation not in data:
                continue
            x = data[representation]
            regional = {
                k.split("__")[-1]: data[k]
                for k in data.files
                if k.startswith(f"regional__{representation}__")
            }
            if representation in {"sensor", "time_frequency"}:
                regional = {str(i): x[:, i : i + 1] for i in range(x.shape[1])}
            arrays[row["unit_id"]] = {
                "trajectory": x,
                "segments": data["segments"],
                "regions": regional,
            }
        admitted.append(row)
    return pd.DataFrame(admitted), arrays


def run_transfer(measurement_paths: list[Path], output: Path, policy: dict):
    records = load_measurements(measurement_paths)
    results = {}
    for track in sorted({row["track"] for row in records}):
        selected = [row for row in records if row["track"] == track]
        for representation in ("sensor", "encoder", "time_frequency"):
            arrays, admitted = {}, []
            for row in selected:
                if sha256_file(row["array_path"]) != row["array_sha256"]:
                    raise ValueError("measurement_checksum_changed")
                with np.load(row["array_path"], allow_pickle=False) as data:
                    if representation not in data:
                        continue
                    x = data[representation]
                    regional = {
                        k.split("__")[-1]: data[k]
                        for k in data.files
                        if k.startswith(f"regional__{representation}__")
                    }
                    if representation in {"sensor", "time_frequency"}:
                        regional = {str(i): x[:, i : i + 1] for i in range(x.shape[1])}
                    arrays[row["unit_id"]] = {
                        "trajectory": x,
                        "segments": data["segments"],
                        "regions": regional,
                    }
                admitted.append(row)
            key = f"{track}:{representation}"
            if not admitted:
                results[key] = {"status": "unavailable", "reason": "no_measurements"}
                continue
            frame = pd.DataFrame(admitted)
            try:
                families = [
                    "spectral_arousal",
                    "nonlinear_scalar",
                    "conventional_multivariate",
                    "shared_dynamics",
                    *[f"dimension_{d}" for d in policy["profile_dimensions"]],
                    *[
                        f"without_{a}"
                        for a in (
                            "repertoire",
                            "metastability",
                            "directionality",
                            "alignment",
                            "reachability",
                        )
                    ],
                ]
                result = nested_transfer(
                    frame,
                    arrays,
                    seed=policy["seed"],
                    families=families,
                    dimensions=policy["profile_dimensions"],
                    strengths=policy["regularization"],
                )
                result["summary"] = summarize_predictions(
                    pd.DataFrame(result["predictions"]),
                    repetitions=policy["bootstrap_repetitions"],
                    seed=policy["seed"],
                )
                result["status"] = "analysed" if result["predictions"] else "unavailable"
                results[key] = result
            except ValueError as exc:
                results[key] = {"status": "unavailable", "reason": str(exc)}
            # Dataset-specific fitting uses held-out participants, not a claimed
            # zero-shot model of a laboratory absent from its training sample.
            for group, within in frame.groupby("study_group"):
                within_key = f"{key}:within:{group}"
                try:
                    within_result = nested_transfer(
                        within,
                        arrays,
                        outer="participant_id",
                        seed=policy["seed"],
                        dimensions=policy["profile_dimensions"],
                        strengths=policy["regularization"],
                    )
                    within_result["summary"] = summarize_predictions(
                        pd.DataFrame(within_result["predictions"]),
                        repetitions=policy["bootstrap_repetitions"],
                        seed=policy["seed"],
                    )
                    within_result["scope"] = (
                        "dataset_specific_subject_transfer_not_zero_shot_study_transfer"
                    )
                    results[within_key] = within_result
                except ValueError as exc:
                    results[within_key] = {"status": "unavailable", "reason": str(exc)}
    result = {
        "analyses": results,
        "inputs": {str(p): sha256_file(p) for p in measurement_paths},
        "scientific_gates": False,
        "no_consciousness_probability": True,
    }
    atomic_write_json(output / "transfer.json", result)
    return result
