"""Multivariate profile statistics with finite-sample safeguards."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.spatial.distance import cdist


def _as_matrix(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    if x.ndim != 2 or x.shape[0] < 2 or not np.all(np.isfinite(x)):
        raise ValueError("values must be a finite observations x features matrix")
    return x


def profile_mahalanobis_distance(
    first: np.ndarray,
    second: np.ndarray,
    *,
    shrinkage: float = 0.1,
) -> float:
    a = _as_matrix(first)
    b = _as_matrix(second)
    if a.shape[1] != b.shape[1]:
        raise ValueError("feature dimensions differ")
    if not 0 <= shrinkage <= 1:
        raise ValueError("shrinkage must lie in [0, 1]")
    pooled = np.vstack([a - a.mean(0), b - b.mean(0)])
    covariance = np.cov(pooled, rowvar=False)
    covariance = np.atleast_2d(covariance)
    scale = np.trace(covariance) / covariance.shape[0]
    regularised = (1 - shrinkage) * covariance + shrinkage * scale * np.eye(covariance.shape[0])
    difference = a.mean(0) - b.mean(0)
    return float(np.sqrt(difference @ np.linalg.pinv(regularised) @ difference))


def energy_distance(first: np.ndarray, second: np.ndarray) -> float:
    a = _as_matrix(first)
    b = _as_matrix(second)
    if a.shape[1] != b.shape[1]:
        raise ValueError("feature dimensions differ")
    cross = 2.0 * np.mean(cdist(a, b))
    within_a = np.mean(cdist(a, a))
    within_b = np.mean(cdist(b, b))
    return float(max(0.0, cross - within_a - within_b))


def permutation_pvalue(
    observed: float,
    null_values: Sequence[float],
    *,
    alternative: str = "greater",
) -> float:
    null = np.asarray(null_values, dtype=float)
    if null.ndim != 1 or null.size == 0 or not np.all(np.isfinite(null)):
        raise ValueError("null_values must be a nonempty finite vector")
    if alternative == "greater":
        extreme = np.count_nonzero(null >= observed)
    elif alternative == "less":
        extreme = np.count_nonzero(null <= observed)
    elif alternative == "two-sided":
        centre = np.median(null)
        extreme = np.count_nonzero(np.abs(null - centre) >= abs(observed - centre))
    else:
        raise ValueError("alternative must be greater, less, or two-sided")
    return float((extreme + 1) / (null.size + 1))


def benjamini_hochberg(
    p_values: Sequence[float], *, alpha: float = 0.05
) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(p_values, dtype=float)
    if p.ndim != 1 or np.any((p < 0) | (p > 1)) or not 0 < alpha < 1:
        raise ValueError("invalid p-values or alpha")
    order = np.argsort(p)
    ranked = p[order]
    adjusted_ranked = np.minimum.accumulate((ranked * p.size / np.arange(1, p.size + 1))[::-1])[
        ::-1
    ]
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = np.clip(adjusted_ranked, 0, 1)
    reject = adjusted <= alpha
    return adjusted, reject
