"""Out-of-fold binary prediction diagnostics and participant bootstraps."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    roc_auc_score,
)

from neural_manifolds.statistics.resampling import participant_bootstrap


@dataclass(frozen=True)
class CalibrationDiagnostic:
    status: str
    intercept: float
    slope: float
    reason: str | None


def expected_calibration_error(
    labels: np.ndarray, probabilities: np.ndarray, *, bins: int = 10
) -> float:
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    value = 0.0
    for lower, upper in pairwise(edges):
        mask = (probabilities >= lower) & (
            probabilities <= upper if upper == 1.0 else probabilities < upper
        )
        if np.any(mask):
            value += mask.mean() * abs(labels[mask].mean() - probabilities[mask].mean())
    return float(value)


def calibration_slope_intercept(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    maximum_iterations: int = 100,
) -> CalibrationDiagnostic:
    """Fit an unpenalized logistic calibration diagnostic by Newton updates."""

    labels = np.asarray(labels, dtype=np.float64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if labels.ndim != 1 or probabilities.ndim != 1 or len(labels) != len(probabilities):
        raise ValueError("labels and probabilities must be aligned one-dimensional arrays")
    if len(labels) < 4 or set(np.unique(labels)) != {0.0, 1.0}:
        return CalibrationDiagnostic(
            status="unavailable",
            intercept=np.nan,
            slope=np.nan,
            reason="calibration requires two classes and at least four observations",
        )
    clipped = np.clip(probabilities, 1e-6, 1.0 - 1e-6)
    logit = np.log(clipped / (1.0 - clipped))
    if not np.all(np.isfinite(logit)) or float(np.std(logit)) <= np.finfo(float).eps:
        return CalibrationDiagnostic(
            status="unavailable",
            intercept=np.nan,
            slope=np.nan,
            reason="predicted logits have no finite variation",
        )
    design = np.column_stack([np.ones(len(logit)), logit])
    parameters = np.asarray([0.0, 1.0], dtype=np.float64)
    converged = False
    try:
        for _ in range(maximum_iterations):
            linear = np.clip(design @ parameters, -30.0, 30.0)
            fitted = 1.0 / (1.0 + np.exp(-linear))
            weights = np.clip(fitted * (1.0 - fitted), 1e-9, None)
            information = design.T @ (weights[:, None] * design)
            score = design.T @ (labels - fitted)
            update = np.linalg.solve(information, score)
            parameters += update
            if not np.all(np.isfinite(parameters)) or np.linalg.norm(parameters) > 100:
                raise np.linalg.LinAlgError("calibration fit diverged")
            if float(np.max(np.abs(update))) < 1e-8:
                converged = True
                break
    except np.linalg.LinAlgError as error:
        return CalibrationDiagnostic(
            status="unavailable",
            intercept=np.nan,
            slope=np.nan,
            reason=f"{type(error).__name__}: {error}",
        )
    if not converged:
        return CalibrationDiagnostic(
            status="unavailable",
            intercept=np.nan,
            slope=np.nan,
            reason="calibration fit did not converge",
        )
    return CalibrationDiagnostic(
        status="available",
        intercept=float(parameters[0]),
        slope=float(parameters[1]),
        reason=None,
    )


def binary_prediction_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if labels.ndim != 1 or probabilities.ndim != 1 or len(labels) != len(probabilities):
        raise ValueError("labels and probabilities must be aligned one-dimensional arrays")
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("binary prediction metrics require both classes")
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("probabilities contain non-finite values")
    hard = probabilities >= 0.5
    calibration = calibration_slope_intercept(labels, probabilities)
    return {
        "auroc": float(roc_auc_score(labels, probabilities)),
        "auprc": float(average_precision_score(labels, probabilities)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, hard)),
        "brier": float(brier_score_loss(labels, probabilities)),
        "ece": expected_calibration_error(labels, probabilities),
        "calibration_intercept": calibration.intercept,
        "calibration_slope": calibration.slope,
        "calibration_status": calibration.status,
        "calibration_unavailable_reason": calibration.reason,
        "calibration_is_oof_diagnostic": True,
        "calibration_refit_applied": False,
    }


def participant_bootstrap_prediction_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    participant_ids: np.ndarray,
    *,
    repetitions: int,
    seed: int,
    retain_distributions: bool = False,
) -> dict[str, Any]:
    """Attach cluster-bootstrap intervals without resampling individual rows."""

    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    participants = np.asarray(participant_ids, dtype=str)
    if not (len(labels) == len(probabilities) == len(participants)):
        raise ValueError("labels, probabilities, and participant_ids must align")
    observed = binary_prediction_metrics(labels, probabilities)
    metric_names = (
        "auroc",
        "auprc",
        "balanced_accuracy",
        "brier",
        "ece",
        "calibration_intercept",
        "calibration_slope",
    )
    distributions: dict[str, list[float]] = {name: [] for name in metric_names}
    valid_binary_resamples = 0
    for indices in participant_bootstrap(
        participants,
        repetitions=repetitions,
        seed=seed,
    ):
        sampled_labels = labels[indices]
        if set(np.unique(sampled_labels)) != {0, 1}:
            continue
        valid_binary_resamples += 1
        sampled = binary_prediction_metrics(sampled_labels, probabilities[indices])
        for name in metric_names:
            value = sampled[name]
            if isinstance(value, (int, float)) and np.isfinite(value):
                distributions[name].append(float(value))
    minimum_successes = min(repetitions, max(20, repetitions // 2))
    output = {
        **observed,
        "participant_bootstrap_unit": "participant",
        "participant_bootstrap_repetitions": repetitions,
        "participant_bootstrap_successful_binary_resamples": valid_binary_resamples,
    }
    for name, values in distributions.items():
        sufficient = len(values) >= minimum_successes
        output[f"{name}_bootstrap_status"] = (
            "available" if sufficient else "unavailable_insufficient_valid_resamples"
        )
        output[f"{name}_bootstrap_successful_repetitions"] = len(values)
        if sufficient:
            low, high = np.quantile(values, [0.025, 0.975])
            output[f"{name}_ci_low"] = float(low)
            output[f"{name}_ci_high"] = float(high)
        else:
            output[f"{name}_ci_low"] = np.nan
            output[f"{name}_ci_high"] = np.nan
    if retain_distributions:
        output["_participant_bootstrap_distributions"] = distributions
    return output
