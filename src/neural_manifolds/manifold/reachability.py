"""Local linear dynamics and finite-horizon stochastic reachability."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._base import EstimatorMixin, require_fitted
from ._validation import (
    FloatArray,
    as_float_matrix,
    encode_states,
    validate_segment_ids,
    validate_square_matrix,
)


@dataclass(frozen=True)
class ReachabilitySummary:
    """Reachability spectrum for one local linear state."""

    gramian: NDArray[np.float64]
    eigenvalues: NDArray[np.float64]
    log_determinant: float
    log_determinant_per_dimension: float
    effective_rank: float
    minimum_eigenvalue: float
    trace: float
    condition_number: float
    spectral_radius: float
    horizon: int
    regularization: float


@dataclass(frozen=True)
class StateWeightedReachabilitySummary:
    """Occupancy-weighted passive reachability across local states."""

    log_determinant: float
    log_determinant_per_dimension: float
    effective_rank: float
    minimum_eigenvalue: float
    trace: float
    occupancy: NDArray[np.float64]
    state_summaries: tuple[ReachabilitySummary, ...]


@dataclass(frozen=True)
class LocalLinearDynamics:
    """State-conditional fit of ``z[t+1] = A_s z[t] + b_s + eta``."""

    state_labels: NDArray[Any]
    transition_matrices: NDArray[np.float64]
    innovation_covariances: NDArray[np.float64]
    intercepts: NDArray[np.float64]
    occupancy: NDArray[np.float64]
    n_transitions_by_state: NDArray[np.int64]
    spectral_radii: NDArray[np.float64]
    ridge: float
    innovation_regularization: float


def stochastic_reachability_gramian(
    transition_matrix: ArrayLike,
    innovation_covariance: ArrayLike,
    *,
    horizon: int,
) -> FloatArray:
    """Compute ``sum_(rho=0)^(H-1) A^rho Q (A.T)^rho``.

    No asymptotic-stability assumption is required because the horizon is finite.
    The result is symmetrised to remove floating-point skew.
    """

    a = validate_square_matrix(transition_matrix, name="transition_matrix")
    q = validate_square_matrix(
        innovation_covariance,
        name="innovation_covariance",
        symmetric=True,
        positive_semidefinite=True,
    )
    if a.shape != q.shape:
        raise ValueError("transition_matrix and innovation_covariance shapes differ")
    if not isinstance(horizon, Integral) or isinstance(horizon, bool) or horizon < 1:
        raise ValueError("horizon must be a positive integer")
    dimension = a.shape[0]
    power = np.eye(dimension, dtype=np.float64)
    gramian = np.zeros_like(a)
    for _ in range(int(horizon)):
        gramian += power @ q @ power.T
        power = a @ power
        if not np.all(np.isfinite(power)) or not np.all(np.isfinite(gramian)):
            raise FloatingPointError(
                "finite-horizon Gramian overflowed; rescale or shorten the horizon"
            )
    return (gramian + gramian.T) / 2.0


def summarize_reachability(
    transition_matrix: ArrayLike,
    innovation_covariance: ArrayLike,
    *,
    horizon: int,
    regularization: float | None = None,
) -> ReachabilitySummary:
    """Compute the proposal's log-determinant proxy and spectral sensitivities."""

    a = validate_square_matrix(transition_matrix, name="transition_matrix")
    gramian = stochastic_reachability_gramian(a, innovation_covariance, horizon=horizon)
    dimension = gramian.shape[0]
    raw_eigenvalues = np.linalg.eigvalsh(gramian)[::-1]
    scale = max(float(np.max(np.abs(raw_eigenvalues))), 1.0)
    raw_eigenvalues[np.abs(raw_eigenvalues) < 1e-12 * scale] = 0.0
    if np.any(raw_eigenvalues < -1e-10 * scale):
        raise FloatingPointError("computed reachability Gramian is not PSD")
    raw_eigenvalues = np.maximum(raw_eigenvalues, 0.0)
    if regularization is None:
        mean_variance = float(np.trace(gramian)) / dimension
        epsilon = max(mean_variance * 1e-9, np.finfo(np.float64).eps)
    else:
        if not isinstance(regularization, Real) or isinstance(regularization, bool):
            raise TypeError("regularization must be a non-negative real or None")
        epsilon = float(regularization)
        if not np.isfinite(epsilon) or epsilon < 0.0:
            raise ValueError("regularization must be finite and non-negative")
    regularized = raw_eigenvalues + epsilon
    if np.any(regularized <= 0.0):
        log_determinant = float("-inf")
    else:
        log_determinant = float(np.sum(np.log(regularized)))
    total = float(np.sum(raw_eigenvalues))
    if total > 0.0:
        probabilities = raw_eigenvalues[raw_eigenvalues > 0.0] / total
        effective_rank = float(np.exp(-np.sum(probabilities * np.log(probabilities))))
    else:
        effective_rank = 0.0
    minimum = float(raw_eigenvalues[-1])
    positive = raw_eigenvalues[raw_eigenvalues > 1e-12 * max(raw_eigenvalues[0], 1.0)]
    condition_number = float(positive[0] / positive[-1]) if positive.size else float("inf")
    spectral_radius = float(np.max(np.abs(np.linalg.eigvals(a))))
    return ReachabilitySummary(
        gramian=gramian,
        eigenvalues=raw_eigenvalues,
        log_determinant=log_determinant,
        log_determinant_per_dimension=log_determinant / dimension,
        effective_rank=effective_rank,
        minimum_eigenvalue=minimum,
        trace=total,
        condition_number=condition_number,
        spectral_radius=spectral_radius,
        horizon=int(horizon),
        regularization=epsilon,
    )


def reachability_energy(
    gramian: ArrayLike,
    targets: ArrayLike,
    *,
    origins: ArrayLike | None = None,
    regularization: float = 0.0,
) -> NDArray[np.float64]:
    """Return minimum quadratic energy for target displacements.

    With no regularisation, a target containing a component in the Gramian's null
    space receives infinite energy rather than the misleading zero produced by a
    naive pseudoinverse.
    """

    w = validate_square_matrix(gramian, name="gramian", symmetric=True, positive_semidefinite=True)
    target = np.asarray(targets, dtype=np.float64)
    if target.ndim == 1:
        target = target[None, :]
    if target.ndim != 2 or target.shape[1] != w.shape[0]:
        raise ValueError("targets must have shape (n_targets, gramian_dimension)")
    if not np.all(np.isfinite(target)):
        raise ValueError("targets contains NaN or infinite values")
    if origins is None:
        displacement = target
    else:
        origin = np.asarray(origins, dtype=np.float64)
        if origin.ndim == 1:
            origin = origin[None, :]
        if origin.shape not in {(1, w.shape[0]), target.shape}:
            raise ValueError("origins must be one vector or match targets")
        displacement = target - origin
    if not isinstance(regularization, Real) or isinstance(regularization, bool):
        raise TypeError("regularization must be a non-negative real number")
    regularization = float(regularization)
    if regularization < 0.0 or not np.isfinite(regularization):
        raise ValueError("regularization must be finite and non-negative")
    eigenvalues, eigenvectors = np.linalg.eigh(w)
    tolerance = 1e-12 * max(float(eigenvalues[-1]), 1.0)
    coordinates = displacement @ eigenvectors
    energies = np.zeros(displacement.shape[0], dtype=np.float64)
    if regularization > 0.0:
        energies = np.sum(
            coordinates * coordinates / (eigenvalues[None, :] + regularization),
            axis=1,
        )
        return energies
    reachable = eigenvalues > tolerance
    if np.any(reachable):
        energies = np.sum(coordinates[:, reachable] ** 2 / eigenvalues[None, reachable], axis=1)
    unreachable_component = np.sum(coordinates[:, ~reachable] ** 2, axis=1)
    energies[unreachable_component > tolerance] = np.inf
    return energies


def state_weighted_reachability(
    transition_matrices: ArrayLike | Sequence[ArrayLike],
    innovation_covariances: ArrayLike | Sequence[ArrayLike],
    occupancy: ArrayLike,
    *,
    horizon: int,
    regularization: float | None = None,
) -> StateWeightedReachabilitySummary:
    """Average local-state reachability summaries by empirical state occupancy."""

    matrices = np.asarray(transition_matrices, dtype=np.float64)
    covariances = np.asarray(innovation_covariances, dtype=np.float64)
    if matrices.ndim == 2:
        matrices = matrices[None, :, :]
    if covariances.ndim == 2:
        covariances = covariances[None, :, :]
    if matrices.ndim != 3 or covariances.ndim != 3 or matrices.shape != covariances.shape:
        raise ValueError(
            "transition_matrices and innovation_covariances must have matching "
            "shape (n_states, dimension, dimension)"
        )
    weights = np.asarray(occupancy, dtype=np.float64)
    if weights.ndim != 1 or weights.size != matrices.shape[0]:
        raise ValueError("occupancy must contain one weight per state")
    if np.any(weights < 0.0) or not np.all(np.isfinite(weights)):
        raise ValueError("occupancy must be finite and non-negative")
    if not np.isclose(np.sum(weights), 1.0):
        raise ValueError("occupancy must sum to one")
    summaries = tuple(
        summarize_reachability(a, q, horizon=horizon, regularization=regularization)
        for a, q in zip(matrices, covariances, strict=True)
    )
    log_determinants = np.asarray([item.log_determinant for item in summaries])
    log_per_dimension = np.asarray([item.log_determinant_per_dimension for item in summaries])
    effective_ranks = np.asarray([item.effective_rank for item in summaries])
    minima = np.asarray([item.minimum_eigenvalue for item in summaries])
    traces = np.asarray([item.trace for item in summaries])
    return StateWeightedReachabilitySummary(
        log_determinant=float(weights @ log_determinants),
        log_determinant_per_dimension=float(weights @ log_per_dimension),
        effective_rank=float(weights @ effective_ranks),
        minimum_eigenvalue=float(weights @ minima),
        trace=float(weights @ traces),
        occupancy=weights.copy(),
        state_summaries=summaries,
    )


stochastic_gramian_logdet = summarize_reachability


def fit_local_linear_dynamics(
    trajectory: ArrayLike,
    states: ArrayLike | None = None,
    *,
    segment_ids: ArrayLike | None = None,
    ridge: float = 1e-4,
    innovation_regularization: float = 1e-8,
    min_transitions: int = 5,
) -> LocalLinearDynamics:
    """Fit state-conditional affine dynamics and empirical innovation covariance."""

    x = as_float_matrix(trajectory, name="trajectory", min_samples=3)
    if states is None:
        labels = np.asarray([0])
        encoded = np.zeros(x.shape[0], dtype=np.int64)
    else:
        labels, encoded = encode_states(states, n_samples=x.shape[0])
    segments = validate_segment_ids(segment_ids, x.shape[0])
    if (
        not isinstance(ridge, Real)
        or isinstance(ridge, bool)
        or not np.isfinite(ridge)
        or ridge < 0.0
    ):
        raise ValueError("ridge must be a non-negative real number")
    if (
        not isinstance(innovation_regularization, Real)
        or isinstance(innovation_regularization, bool)
        or not np.isfinite(innovation_regularization)
        or innovation_regularization < 0.0
    ):
        raise ValueError("innovation_regularization must be non-negative")
    if not isinstance(min_transitions, Integral) or min_transitions < 2:
        raise ValueError("min_transitions must be an integer of at least two")
    valid_steps = segments[:-1] == segments[1:]
    dimension = x.shape[1]
    transition_matrices = np.empty((labels.size, dimension, dimension), dtype=np.float64)
    innovation_covariances = np.empty_like(transition_matrices)
    intercepts = np.empty((labels.size, dimension), dtype=np.float64)
    counts = np.empty(labels.size, dtype=np.int64)
    for state_index in range(labels.size):
        mask = valid_steps & (encoded[:-1] == state_index)
        current = x[:-1][mask]
        following = x[1:][mask]
        counts[state_index] = current.shape[0]
        if current.shape[0] < min_transitions:
            raise ValueError(
                f"state {labels[state_index]!r} has {current.shape[0]} valid "
                f"transitions; at least {min_transitions} are required"
            )
        current_mean = np.mean(current, axis=0)
        following_mean = np.mean(following, axis=0)
        current_centered = current - current_mean
        following_centered = following - following_mean
        gram = current_centered.T @ current_centered
        scale = float(np.trace(gram)) / dimension
        penalty = float(ridge) * max(scale, np.finfo(np.float64).eps)
        gram.flat[:: dimension + 1] += penalty
        coefficients = np.linalg.solve(gram, current_centered.T @ following_centered)
        a = coefficients.T
        intercept = following_mean - a @ current_mean
        residuals = following - (current @ a.T + intercept)
        q = (residuals.T @ residuals) / max(current.shape[0] - 1, 1)
        q = (q + q.T) / 2.0
        q_scale = float(np.trace(q)) / dimension
        q.flat[:: dimension + 1] += float(innovation_regularization) * max(
            q_scale, np.finfo(np.float64).eps
        )
        transition_matrices[state_index] = a
        innovation_covariances[state_index] = q
        intercepts[state_index] = intercept
    occupancy = np.bincount(encoded, minlength=labels.size).astype(np.float64)
    occupancy /= occupancy.sum()
    spectral_radii = np.asarray(
        [np.max(np.abs(np.linalg.eigvals(a))) for a in transition_matrices],
        dtype=np.float64,
    )
    return LocalLinearDynamics(
        state_labels=labels.copy(),
        transition_matrices=transition_matrices,
        innovation_covariances=innovation_covariances,
        intercepts=intercepts,
        occupancy=occupancy,
        n_transitions_by_state=counts,
        spectral_radii=spectral_radii,
        ridge=float(ridge),
        innovation_regularization=float(innovation_regularization),
    )


class ReachabilityEstimator(EstimatorMixin):
    """Fit local dynamics and score occupancy-weighted stochastic reachability."""

    def __init__(
        self,
        *,
        horizon: int = 10,
        ridge: float = 1e-4,
        innovation_regularization: float = 1e-8,
        gramian_regularization: float | None = None,
        min_transitions: int = 5,
    ) -> None:
        self.horizon = horizon
        self.ridge = ridge
        self.innovation_regularization = innovation_regularization
        self.gramian_regularization = gramian_regularization
        self.min_transitions = min_transitions

    def fit(
        self,
        trajectory: ArrayLike,
        y: ArrayLike | None = None,
        *,
        states: ArrayLike | None = None,
        segment_ids: ArrayLike | None = None,
    ) -> ReachabilityEstimator:
        del y
        self.dynamics_ = fit_local_linear_dynamics(
            trajectory,
            states,
            segment_ids=segment_ids,
            ridge=self.ridge,
            innovation_regularization=self.innovation_regularization,
            min_transitions=self.min_transitions,
        )
        self.summary_ = state_weighted_reachability(
            self.dynamics_.transition_matrices,
            self.dynamics_.innovation_covariances,
            self.dynamics_.occupancy,
            horizon=self.horizon,
            regularization=self.gramian_regularization,
        )
        return self

    def score(self, trajectory: ArrayLike | None = None, y: ArrayLike | None = None) -> float:
        del trajectory, y
        require_fitted(self, "summary_")
        return self.summary_.log_determinant
