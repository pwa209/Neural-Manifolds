"""Estimators of broad but structured latent-state repertoire."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, lgamma, log
from numbers import Real

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._base import EstimatorMixin, require_fitted
from ._validation import FloatArray, as_float_matrix


@dataclass(frozen=True)
class RepertoireSummary:
    """Participant-condition summary of covariance-spectrum repertoire.

    ``participation_ratio`` is the primary estimator. ``effective_rank`` is the
    exponentiated Shannon entropy of the same normalised spectrum and is supplied
    as a sensitivity measure.  Neither quantity is raw total variance.
    """

    participation_ratio: float
    effective_rank: float
    total_variance: float
    leading_variance_fraction: float
    shrinkage: float
    noise_variance: float
    n_samples: int
    n_features: int
    eigenvalues: NDArray[np.float64]


def _validated_noise_variance(noise_variance: float) -> float:
    if not isinstance(noise_variance, Real) or isinstance(noise_variance, bool):
        raise TypeError("noise_variance must be a non-negative real number")
    value = float(noise_variance)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError("noise_variance must be finite and non-negative")
    return value


def _oas_covariance(x: FloatArray) -> tuple[FloatArray, float]:
    """Oracle-approximating shrinkage covariance using maximum-likelihood scale."""

    n_samples, n_features = x.shape
    empirical = (x.T @ x) / float(n_samples)
    trace_mean = float(np.trace(empirical)) / n_features
    alpha = float(np.mean(empirical * empirical))
    numerator = alpha + trace_mean * trace_mean
    denominator = (n_samples + 1.0) * (alpha - trace_mean * trace_mean / n_features)
    if denominator <= np.finfo(np.float64).eps * max(numerator, 1.0):
        coefficient = 1.0
    else:
        coefficient = min(max(numerator / denominator, 0.0), 1.0)
    covariance = (1.0 - coefficient) * empirical
    covariance.flat[:: n_features + 1] += coefficient * trace_mean
    return (covariance + covariance.T) / 2.0, float(coefficient)


def _estimate_covariance(
    x: FloatArray,
    shrinkage: str | float | None,
) -> tuple[FloatArray, float]:
    centered = x - np.mean(x, axis=0, keepdims=True)
    n_samples, n_features = centered.shape
    empirical = (centered.T @ centered) / float(max(n_samples - 1, 1))
    empirical = (empirical + empirical.T) / 2.0
    if shrinkage is None or shrinkage == "none":
        return empirical, 0.0
    if shrinkage == "oas":
        # OAS is derived for the ML (1/n) covariance.  Rescaling back to the
        # unbiased convention does not affect dimension ratios but keeps total
        # variance directly comparable to np.cov.
        covariance, coefficient = _oas_covariance(centered)
        covariance *= n_samples / max(n_samples - 1, 1)
        return covariance, coefficient
    if isinstance(shrinkage, Real) and not isinstance(shrinkage, bool):
        coefficient = float(shrinkage)
        if not 0.0 <= coefficient <= 1.0:
            raise ValueError("numeric shrinkage must lie in [0, 1]")
        target_scale = float(np.trace(empirical)) / n_features
        covariance = (1.0 - coefficient) * empirical
        covariance.flat[:: n_features + 1] += coefficient * target_scale
        return covariance, coefficient
    raise ValueError("shrinkage must be 'oas', 'none', None, or a number in [0, 1]")


def covariance_spectrum(
    x: ArrayLike,
    *,
    shrinkage: str | float | None = "oas",
    noise_variance: float = 0.0,
) -> NDArray[np.float64]:
    """Return descending noise-corrected eigenvalues of the latent covariance.

    Parameters
    ----------
    x:
        Samples by features. Samples should already have been matched across
        conditions before this function is called.
    shrinkage:
        ``"oas"`` (default), no shrinkage, or a fixed isotropic shrinkage
        coefficient. OAS is useful when embedding dimension approaches sample
        count.
    noise_variance:
        Independently estimated isotropic measurement-noise variance to subtract
        from every covariance eigenvalue. Negative corrected eigenvalues are
        clipped to zero. It must not be estimated from condition labels.
    """

    trajectory = as_float_matrix(x, name="x", min_samples=2)
    noise_variance = _validated_noise_variance(noise_variance)
    covariance, _ = _estimate_covariance(trajectory, shrinkage)
    eigenvalues = np.linalg.eigvalsh(covariance)[::-1]
    scale = max(float(eigenvalues[0]), 1.0)
    eigenvalues[np.abs(eigenvalues) < 1e-12 * scale] = 0.0
    eigenvalues = np.maximum(eigenvalues - noise_variance, 0.0)
    return eigenvalues


def participation_ratio(
    values: ArrayLike,
    *,
    shrinkage: str | float | None = "oas",
    noise_variance: float = 0.0,
) -> float:
    """Compute effective dimension ``(sum lambda)^2 / sum(lambda^2)``.

    A one-dimensional input is interpreted as a covariance spectrum. A
    two-dimensional input is interpreted as samples by features and its spectrum
    is estimated using :func:`covariance_spectrum`.
    """

    noise_variance = _validated_noise_variance(noise_variance)
    array = np.asarray(values)
    if array.ndim == 2:
        spectrum = covariance_spectrum(array, shrinkage=shrinkage, noise_variance=noise_variance)
    elif array.ndim == 1:
        try:
            spectrum = np.asarray(array, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("spectrum must contain numeric values") from exc
        if spectrum.size == 0:
            raise ValueError("spectrum must not be empty")
        if not np.all(np.isfinite(spectrum)):
            raise ValueError("spectrum contains NaN or infinite values")
        tolerance = 1e-12 * max(float(np.max(np.abs(spectrum))), 1.0)
        if np.any(spectrum < -tolerance):
            raise ValueError("covariance eigenvalues must be non-negative")
        spectrum = np.maximum(spectrum - noise_variance, 0.0)
    else:
        raise ValueError("values must be a one-dimensional spectrum or 2D data")
    total = float(np.sum(spectrum))
    squared = float(spectrum @ spectrum)
    if total <= 0.0 or squared <= 0.0:
        raise ValueError("the noise-corrected covariance has no positive variance")
    value = total * total / squared
    # Cauchy-Schwarz bounds the ratio by the number of positive eigenvalues;
    # clamp only round-off excursions above that exact mathematical bound.
    return min(value, float(np.count_nonzero(spectrum > 0.0)))


def _nearest_distances(x: FloatArray, k: int, block_size: int = 512) -> FloatArray:
    """Return the first ``k`` non-self distances without an n-by-n allocation."""

    n_samples = x.shape[0]
    if not 2 <= k < n_samples:
        raise ValueError(f"k must satisfy 2 <= k < n_samples ({n_samples})")
    squared_norm = np.sum(x * x, axis=1)
    nearest = np.empty((n_samples, k), dtype=np.float64)
    for start in range(0, n_samples, block_size):
        stop = min(start + block_size, n_samples)
        distances_squared = (
            squared_norm[start:stop, None] + squared_norm[None, :] - 2.0 * (x[start:stop] @ x.T)
        )
        np.maximum(distances_squared, 0.0, out=distances_squared)
        row_indices = np.arange(stop - start)
        distances_squared[row_indices, np.arange(start, stop)] = np.inf
        selected = np.partition(distances_squared, kth=k - 1, axis=1)[:, :k]
        selected.sort(axis=1)
        nearest[start:stop] = np.sqrt(selected)
    if np.any(nearest[:, -1] <= np.finfo(np.float64).eps):
        raise ValueError(
            "k-nearest-neighbour estimates are undefined because too many samples "
            "are exact duplicates"
        )
    return nearest


def knn_intrinsic_dimension(x: ArrayLike, *, k: int = 10) -> float:
    """Levina-Bickel local maximum-likelihood intrinsic-dimension estimate."""

    trajectory = as_float_matrix(x, name="x", min_samples=k + 1)
    distances = _nearest_distances(trajectory, k)
    radius = distances[:, -1]
    ratios = np.log(radius[:, None] / np.maximum(distances[:, :-1], 1e-15))
    denominators = np.sum(ratios, axis=1)
    valid = denominators > 1e-12
    if np.count_nonzero(valid) < max(3, trajectory.shape[0] // 2):
        raise ValueError("too few non-degenerate neighbourhoods for dimension estimation")
    local_dimension = (k - 1.0) / denominators[valid]
    # The median is substantially less sensitive than the mean to individual
    # nearly duplicated points while preserving recovery on homogeneous manifolds.
    return float(np.median(local_dimension))


def _digamma_positive_integer(value: int) -> float:
    if value < 1:
        raise ValueError("digamma argument must be positive")
    euler_mascheroni = 0.5772156649015329
    if value == 1:
        return -euler_mascheroni
    return float(np.sum(1.0 / np.arange(1, value, dtype=np.float64))) - euler_mascheroni


def knn_differential_entropy(
    x: ArrayLike,
    *,
    k: int = 5,
    dimension: int | None = None,
) -> float:
    """Kozachenko-Leonenko differential entropy in nats.

    ``dimension`` defaults to ambient feature count. Pass a dimension chosen in
    training data (for example a rounded intrinsic-dimension estimate) when the
    observations lie on a lower-dimensional manifold.
    """

    trajectory = as_float_matrix(x, name="x", min_samples=k + 1)
    n_samples, n_features = trajectory.shape
    if dimension is None:
        dimension = n_features
    if not isinstance(dimension, (int, np.integer)) or not 1 <= dimension <= n_features:
        raise ValueError("dimension must be an integer in [1, n_features]")
    radii = _nearest_distances(trajectory, k)[:, -1]
    log_unit_ball_volume = (dimension / 2.0) * log(np.pi) - lgamma(dimension / 2.0 + 1.0)
    return float(
        _digamma_positive_integer(n_samples)
        - _digamma_positive_integer(k)
        + log_unit_ball_volume
        + dimension * np.mean(np.log(radii))
    )


def local_neighbourhood_anisotropy(x: ArrayLike, *, k: int = 20) -> float:
    """Mean local anisotropy, where zero is isotropic and one is line-like.

    For each sample this computes one minus the participation ratio of its local
    covariance divided by the maximum possible local rank. This makes the result
    comparable across embedding dimensions and sample counts.
    """

    trajectory = as_float_matrix(x, name="x", min_samples=k + 1)
    n_samples, n_features = trajectory.shape
    distances = _nearest_distances(trajectory, k)
    # Recover neighbour indices in blocks. The first distance computation is kept
    # separate so duplicate/degenerate validation is shared with the kNN metrics.
    del distances
    squared_norm = np.sum(trajectory * trajectory, axis=1)
    maximum_rank = min(k - 1, n_features)
    anisotropy = np.empty(n_samples, dtype=np.float64)
    for start in range(0, n_samples, 256):
        stop = min(start + 256, n_samples)
        distances_squared = (
            squared_norm[start:stop, None]
            + squared_norm[None, :]
            - 2.0 * (trajectory[start:stop] @ trajectory.T)
        )
        rows = np.arange(stop - start)
        distances_squared[rows, np.arange(start, stop)] = np.inf
        indices = np.argpartition(distances_squared, kth=k - 1, axis=1)[:, :k]
        for offset, neighbours in enumerate(indices):
            local = trajectory[neighbours]
            spectrum = covariance_spectrum(local, shrinkage="none")
            local_pr = participation_ratio(spectrum)
            anisotropy[start + offset] = 1.0 - local_pr / maximum_rank
    return float(np.mean(np.clip(anisotropy, 0.0, 1.0)))


def estimate_repertoire(
    x: ArrayLike,
    *,
    shrinkage: str | float | None = "oas",
    noise_variance: float = 0.0,
) -> RepertoireSummary:
    """Estimate the proposal's primary repertoire summary."""

    trajectory = as_float_matrix(x, name="x", min_samples=2)
    noise_variance = _validated_noise_variance(noise_variance)
    covariance, coefficient = _estimate_covariance(trajectory, shrinkage)
    eigenvalues = np.linalg.eigvalsh(covariance)[::-1]
    eigenvalues = np.maximum(eigenvalues - noise_variance, 0.0)
    total = float(np.sum(eigenvalues))
    if total <= 0.0:
        raise ValueError("the noise-corrected covariance has no positive variance")
    probabilities = eigenvalues[eigenvalues > 0.0] / total
    effective_rank = exp(-float(np.sum(probabilities * np.log(probabilities))))
    return RepertoireSummary(
        participation_ratio=participation_ratio(eigenvalues),
        effective_rank=float(effective_rank),
        total_variance=total,
        leading_variance_fraction=float(eigenvalues[0] / total),
        shrinkage=coefficient,
        noise_variance=float(noise_variance),
        n_samples=trajectory.shape[0],
        n_features=trajectory.shape[1],
        eigenvalues=eigenvalues.copy(),
    )


# Configuration-facing aliases use the exact locked names in configs/study.yaml.
covariance_participation_ratio = participation_ratio
knn_entropy = knn_differential_entropy
local_anisotropy = local_neighbourhood_anisotropy


class RepertoireEstimator(EstimatorMixin):
    """Fit covariance-spectrum repertoire and expose an eigenbasis transform."""

    def __init__(
        self,
        *,
        shrinkage: str | float | None = "oas",
        noise_variance: float = 0.0,
    ) -> None:
        self.shrinkage = shrinkage
        self.noise_variance = noise_variance

    def fit(self, x: ArrayLike, y: ArrayLike | None = None) -> RepertoireEstimator:
        del y
        trajectory = as_float_matrix(x, name="x", min_samples=2)
        noise_variance = _validated_noise_variance(self.noise_variance)
        self.mean_ = np.mean(trajectory, axis=0)
        covariance, coefficient = _estimate_covariance(trajectory, self.shrinkage)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues)[::-1]
        corrected = np.maximum(eigenvalues[order] - noise_variance, 0.0)
        total = float(np.sum(corrected))
        if total <= 0.0:
            raise ValueError("the noise-corrected covariance has no positive variance")
        probabilities = corrected[corrected > 0.0] / total
        self.components_ = eigenvectors[:, order].T
        self.eigenvalues_ = corrected
        self.n_features_in_ = trajectory.shape[1]
        self.summary_ = RepertoireSummary(
            participation_ratio=participation_ratio(corrected),
            effective_rank=float(exp(-np.sum(probabilities * np.log(probabilities)))),
            total_variance=total,
            leading_variance_fraction=float(corrected[0] / total),
            shrinkage=coefficient,
            noise_variance=noise_variance,
            n_samples=trajectory.shape[0],
            n_features=trajectory.shape[1],
            eigenvalues=corrected.copy(),
        )
        return self

    def transform(self, x: ArrayLike) -> FloatArray:
        require_fitted(self, "mean_", "components_")
        trajectory = as_float_matrix(x, name="x", min_samples=1)
        if trajectory.shape[1] != self.n_features_in_:
            raise ValueError(
                f"x has {trajectory.shape[1]} features; expected {self.n_features_in_}"
            )
        return (trajectory - self.mean_) @ self.components_.T

    def fit_transform(self, x: ArrayLike, y: ArrayLike | None = None) -> FloatArray:
        return self.fit(x, y).transform(x)

    def score(self, x: ArrayLike | None = None, y: ArrayLike | None = None) -> float:
        del x, y
        require_fitted(self, "summary_")
        return self.summary_.participation_ratio
