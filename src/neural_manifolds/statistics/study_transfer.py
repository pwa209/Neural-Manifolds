"""Study-held-out model comparison on audited, common observation tables.

The caller supplies verified independent study groups and globally disambiguated
participant IDs. No learning or imputation uses the held-out study. Labels are
reported experience, never an inferred binary consciousness state.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


@dataclass(frozen=True)
class Candidate:
    name: str
    features: tuple[str, ...]
    nonlinear_scalar: bool = False
    dimensions: int | None = None


def independent_weights(frame: pd.DataFrame) -> np.ndarray:
    """Equal study weight, equal participant weight within each study."""
    sizes = frame.groupby(["study_group", "participant_id"])["participant_id"].transform("size")
    people = frame.groupby("study_group")["participant_id"].transform("nunique")
    weights = 1.0 / (sizes.to_numpy() * people.to_numpy())
    return weights / weights.mean()


def _pipeline(candidate: Candidate, strength: float, *, regression: bool = False) -> Pipeline:
    steps = [
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scale", StandardScaler()),
    ]
    if candidate.nonlinear_scalar:
        if len(candidate.features) != 1:
            raise ValueError("Scalar comparator requires exactly one feature")
        steps.append(("nonlinear", PolynomialFeatures(degree=3, include_bias=False)))
    if candidate.dimensions is not None:
        steps.append(("dimension", PCA(n_components=candidate.dimensions, svd_solver="full")))
    model = Ridge(alpha=strength) if regression else LogisticRegression(C=strength, max_iter=2000)
    return Pipeline([*steps, ("model", model)])


def validate_units(frame: pd.DataFrame, target: str) -> None:
    required = {"unit_id", "study_group", "participant_id", target}
    if not required.issubset(frame.columns) or frame[list(required)].isna().any().any():
        raise ValueError("Missing independent-unit or target metadata")
    if frame["unit_id"].duplicated().any():
        raise ValueError("Duplicate observational units")
    if (frame.groupby("participant_id")["study_group"].nunique() > 1).any():
        raise ValueError("Overlapping participants must share one study_group")


def compare_studies(
    frame: pd.DataFrame,
    candidates: list[Candidate],
    *,
    target: str = "experience",
    strengths: tuple[float, ...] = (0.1, 1.0, 10.0),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    validate_units(frame, target)
    if len({c.name for c in candidates}) != len(candidates) or not candidates:
        raise ValueError("Unique candidate names required")
    if set(frame[target].unique()) != {0, 1}:
        raise ValueError("Binary DE versus NE labels required; keep DEWR separate")
    if frame["study_group"].nunique() < 3:
        raise ValueError("At least three independent studies required for nested study transfer")
    columns = set().union(*(set(c.features) for c in candidates))
    if not columns.issubset(frame.columns):
        raise ValueError("All candidates must use the same audited observation table")
    if np.isinf(frame[sorted(columns)].to_numpy(dtype=float)).any():
        raise ValueError("Infinite features are invalid")
    predictions, audit = [], []
    for held in sorted(frame["study_group"].unique()):
        train = frame[frame.study_group != held].reset_index(drop=True)
        test = frame[frame.study_group == held].reset_index(drop=True)
        folds = list(
            GroupKFold(n_splits=min(5, train.study_group.nunique())).split(
                train, groups=train.study_group
            )
        )
        if any(train.iloc[a][target].nunique() < 2 for a, _ in folds):
            raise ValueError("An inner training fold lacks both experience categories")
        for candidate in candidates:
            if train[list(candidate.features)].notna().sum().eq(0).any():
                raise ValueError(f"{candidate.name}: unmeasured feature in training studies")
            scores = []
            for strength in strengths:
                losses = []
                for a, b in folds:
                    fit, validation = train.iloc[a], train.iloc[b]
                    model = _pipeline(candidate, strength)
                    model.fit(
                        fit[list(candidate.features)],
                        fit[target],
                        model__sample_weight=independent_weights(fit),
                    )
                    probability = model.predict_proba(validation[list(candidate.features)])[:, 1]
                    losses.append(
                        log_loss(
                            validation[target],
                            probability,
                            labels=[0, 1],
                            sample_weight=independent_weights(validation),
                        )
                    )
                scores.append(float(np.mean(losses)))
            selected = strengths[int(np.argmin(scores))]
            model = _pipeline(candidate, selected)
            model.fit(
                train[list(candidate.features)],
                train[target],
                model__sample_weight=independent_weights(train),
            )
            probability = model.predict_proba(test[list(candidate.features)])[:, 1]
            out = test[["unit_id", "participant_id", "study_group", target]].copy()
            out["model"] = candidate.name
            out["probability"] = probability
            predictions.append(out)
            audit.append(
                {
                    "held_study": held,
                    "model": candidate.name,
                    "selected_strength": selected,
                    "inner_scores": scores,
                    "training_studies": sorted(train.study_group.unique().tolist()),
                    "n_training_participants": train.participant_id.nunique(),
                    "n_test_participants": test.participant_id.nunique(),
                    "test_log_loss": log_loss(
                        test[target],
                        probability,
                        labels=[0, 1],
                        sample_weight=independent_weights(test),
                    ),
                }
            )
    return pd.concat(predictions, ignore_index=True), pd.DataFrame(audit)


def predict_tms(
    frame: pd.DataFrame,
    *,
    baseline: tuple[str, ...],
    dynamics: tuple[str, ...],
    outcome: str,
    alphas: tuple[float, ...] = (0.1, 1.0, 10.0),
) -> pd.DataFrame:
    """Participant-held-out TMS regression, with inner participant tuning.

    Baseline must include explicit condition indicators and measured conventional
    EEG predictors. Input rows must already be linked to independent spontaneous
    sessions; this routine does not manufacture linkage or missing covariates.
    """
    validate_units(frame, outcome)
    if not baseline or not dynamics or set(baseline) & set(dynamics):
        raise ValueError("Disjoint baseline and incremental features required")
    if frame.participant_id.nunique() < 4:
        raise ValueError("At least four participants required")
    outputs = []
    for held in sorted(frame.participant_id.unique()):
        train = frame[frame.participant_id != held].reset_index(drop=True)
        test = frame[frame.participant_id == held].reset_index(drop=True)
        splits = list(
            GroupKFold(n_splits=min(5, train.participant_id.nunique())).split(
                train, groups=train.participant_id
            )
        )
        for name, features in [
            ("condition_conventional", baseline),
            ("plus_dynamics", baseline + dynamics),
        ]:
            candidate = Candidate(name, features)
            errors = []
            for alpha in alphas:
                losses = []
                for a, b in splits:
                    fit, validation = train.iloc[a], train.iloc[b]
                    model = _pipeline(candidate, alpha, regression=True)
                    model.fit(
                        fit[list(features)],
                        fit[outcome],
                        model__sample_weight=independent_weights(fit),
                    )
                    residual = validation[outcome].to_numpy() - model.predict(
                        validation[list(features)]
                    )
                    losses.append(np.average(residual**2, weights=independent_weights(validation)))
                errors.append(np.mean(losses))
            selected = alphas[int(np.argmin(errors))]
            model = _pipeline(candidate, selected, regression=True)
            model.fit(
                train[list(features)],
                train[outcome],
                model__sample_weight=independent_weights(train),
            )
            out = test[["unit_id", "participant_id", "study_group", outcome]].copy()
            out["model"] = name
            out["prediction"] = model.predict(test[list(features)])
            out["selected_alpha"] = selected
            outputs.append(out)
    return pd.concat(outputs, ignore_index=True)
