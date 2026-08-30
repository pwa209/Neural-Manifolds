"""Directed transition architecture and broken-detailed-balance estimators."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._base import EstimatorMixin, require_fitted
from ._validation import encode_states, validate_segment_ids, validate_square_matrix


@dataclass(frozen=True)
class DirectionalitySummary:
    """Participant-condition directionality estimates from a discrete path."""

    state_labels: NDArray[Any]
    transition_counts: NDArray[np.int64]
    transition_matrix: NDArray[np.float64]
    stationary_distribution: NDArray[np.float64]
    entropy_production: float
    entropy_production_rate: float
    flux_asymmetry: float
    detailed_balance_residual: float
    transition_graph_asymmetry: float
    n_transitions: int
    pseudocount: float
    sample_interval: float


def _encode_with_dictionary(
    states: ArrayLike,
    state_labels: ArrayLike | None,
) -> tuple[NDArray[Any], NDArray[np.int64]]:
    if state_labels is None:
        return encode_states(states)
    sequence = np.asarray(states)
    if sequence.ndim != 1 or sequence.size == 0:
        raise ValueError("states must be a non-empty one-dimensional sequence")
    labels = np.asarray(state_labels)
    if labels.ndim != 1 or labels.size == 0:
        raise ValueError("state_labels must be a non-empty one-dimensional sequence")
    try:
        dictionary = {label: index for index, label in enumerate(labels.tolist())}
    except TypeError as exc:
        raise ValueError("state labels must be hashable") from exc
    if len(dictionary) != labels.size:
        raise ValueError("state_labels must not contain duplicates")
    encoded = np.empty(sequence.size, dtype=np.int64)
    for index, label in enumerate(sequence.tolist()):
        if label not in dictionary:
            raise ValueError(f"state label {label!r} is absent from state_labels")
        encoded[index] = dictionary[label]
    return labels.copy(), encoded


def _transition_components(
    states: ArrayLike,
    *,
    segment_ids: ArrayLike | None,
    pseudocount: float,
    state_labels: ArrayLike | None,
) -> tuple[NDArray[Any], NDArray[np.int64], NDArray[np.float64]]:
    labels, encoded = _encode_with_dictionary(states, state_labels)
    segments = validate_segment_ids(segment_ids, encoded.size)
    if not isinstance(pseudocount, Real) or isinstance(pseudocount, bool):
        raise TypeError("pseudocount must be a non-negative real number")
    pseudocount = float(pseudocount)
    if not np.isfinite(pseudocount) or pseudocount < 0.0:
        raise ValueError("pseudocount must be finite and non-negative")
    counts = np.zeros((labels.size, labels.size), dtype=np.int64)
    if encoded.size > 1:
        valid = segments[:-1] == segments[1:]
        np.add.at(counts, (encoded[:-1][valid], encoded[1:][valid]), 1)
    smoothed = counts.astype(np.float64) + pseudocount
    row_totals = np.sum(smoothed, axis=1)
    empty = row_totals == 0.0
    if np.any(empty):
        # With no prior and no observed exit, the least-informative path-consistent
        # convention is an absorbing row. This avoids NaNs while adding no edge.
        empty_indices = np.flatnonzero(empty)
        smoothed[empty_indices, empty_indices] = 1.0
        row_totals = np.sum(smoothed, axis=1)
    transition_matrix = smoothed / row_totals[:, None]
    return labels, counts, transition_matrix


def estimate_transition_matrix(
    states: ArrayLike,
    *,
    segment_ids: ArrayLike | None = None,
    pseudocount: float = 0.5,
    state_labels: ArrayLike | None = None,
    return_counts: bool = False,
) -> NDArray[np.float64] | tuple[NDArray[np.float64], NDArray[np.int64], NDArray[Any]]:
    """Estimate a row-stochastic transition matrix without crossing segments.

    A symmetric Jeffreys pseudocount of 0.5 is the default finite-sample
    regularisation. Set it to zero to retain structural zeros; in that case
    entropy production is mathematically infinite when reverse flux is absent.
    """

    labels, counts, matrix = _transition_components(
        states,
        segment_ids=segment_ids,
        pseudocount=pseudocount,
        state_labels=state_labels,
    )
    if return_counts:
        return matrix, counts, labels
    return matrix


def estimate_stationary_distribution(
    transition_matrix: ArrayLike,
    *,
    tolerance: float = 1e-12,
    max_iterations: int = 100_000,
) -> NDArray[np.float64]:
    """Estimate ``pi`` satisfying ``pi @ P = pi``.

    An augmented least-squares system handles periodic chains directly.  If a
    reducible chain makes that unconstrained solution numerically non-positive,
    Cesaro power iteration provides the stationary mixture induced by a uniform
    initial state. Empirical analyses should normally use a positive symmetric
    pseudocount, which makes the solution unique.
    """

    matrix = validate_square_matrix(transition_matrix, name="transition_matrix")
    if np.any(matrix < -tolerance):
        raise ValueError("transition_matrix contains negative probabilities")
    matrix = np.maximum(matrix, 0.0)
    row_sums = np.sum(matrix, axis=1)
    if not np.allclose(row_sums, 1.0, atol=tolerance, rtol=1e-10):
        raise ValueError("each transition_matrix row must sum to one")
    if not isinstance(max_iterations, (int, np.integer)) or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer")
    n_states = matrix.shape[0]
    system = np.vstack([matrix.T - np.eye(n_states), np.ones((1, n_states))])
    target = np.r_[np.zeros(n_states), 1.0]
    least_squares = np.linalg.lstsq(system, target, rcond=None)[0]
    if np.min(least_squares) >= -max(tolerance * 100.0, 1e-12):
        stationary = np.maximum(least_squares, 0.0)
        stationary /= np.sum(stationary)
        if np.max(np.abs(stationary @ matrix - stationary)) <= max(tolerance * 100.0, 1e-10):
            return stationary

    distribution = np.full(n_states, 1.0 / n_states, dtype=np.float64)
    average = np.zeros(n_states, dtype=np.float64)
    for iteration in range(1, int(max_iterations) + 1):
        updated = distribution @ matrix
        average += (updated - average) / iteration
        if np.max(np.abs(updated - distribution)) < tolerance:
            distribution = updated
            break
        distribution = updated
    else:
        # A periodic chain need not converge pointwise; its Cesaro mean does.
        distribution = average
    distribution = np.maximum(distribution, 0.0)
    total = float(np.sum(distribution))
    if total <= 0.0:
        raise RuntimeError("failed to recover a stationary distribution")
    stationary = distribution / total
    if np.max(np.abs(stationary @ matrix - stationary)) > max(1e-8, tolerance * 100):
        raise RuntimeError("stationary-distribution iteration did not converge")
    return stationary


def entropy_production(
    transition_matrix: ArrayLike,
    stationary_distribution: ArrayLike | None = None,
    *,
    zero_tolerance: float = 0.0,
) -> float:
    """Compute Markov-chain entropy production in nats per transition.

    This implements ``sum_ij pi_i P_ij log((pi_i P_ij)/(pi_j P_ji))``.
    Structural one-way flux yields ``inf``. Use a prespecified symmetric
    pseudocount during transition estimation for a finite empirical estimate.
    """

    matrix = validate_square_matrix(transition_matrix, name="transition_matrix")
    if np.any(matrix < 0.0) or not np.allclose(np.sum(matrix, axis=1), 1.0):
        raise ValueError("transition_matrix must be row-stochastic")
    if stationary_distribution is None:
        stationary = estimate_stationary_distribution(matrix)
    else:
        stationary = np.asarray(stationary_distribution, dtype=np.float64)
        if stationary.ndim != 1 or stationary.size != matrix.shape[0]:
            raise ValueError("stationary_distribution has incompatible shape")
        if np.any(stationary < 0.0) or not np.all(np.isfinite(stationary)):
            raise ValueError("stationary_distribution must be finite and non-negative")
        if not np.isclose(np.sum(stationary), 1.0):
            raise ValueError("stationary_distribution must sum to one")
        if not np.allclose(stationary @ matrix, stationary, atol=1e-8, rtol=1e-8):
            raise ValueError("stationary_distribution is not stationary for the matrix")
    flux = stationary[:, None] * matrix
    reverse = flux.T
    diagonal = np.eye(matrix.shape[0], dtype=bool)
    one_way = (~diagonal) & (flux > zero_tolerance) & (reverse <= zero_tolerance)
    if np.any(one_way):
        return float("inf")
    valid = (~diagonal) & (flux > zero_tolerance) & (reverse > zero_tolerance)
    if not np.any(valid):
        return 0.0
    value = float(np.sum(flux[valid] * np.log(flux[valid] / reverse[valid])))
    # Round-off can make an exactly reversible process microscopically negative.
    return max(value, 0.0)


def estimate_directionality(
    states: ArrayLike,
    *,
    segment_ids: ArrayLike | None = None,
    pseudocount: float = 0.5,
    state_labels: ArrayLike | None = None,
    sample_interval: float = 1.0,
) -> DirectionalitySummary:
    """Estimate entropy production and independent transition-asymmetry checks."""

    if not isinstance(sample_interval, Real) or isinstance(sample_interval, bool):
        raise TypeError("sample_interval must be a positive real number")
    sample_interval = float(sample_interval)
    if not np.isfinite(sample_interval) or sample_interval <= 0.0:
        raise ValueError("sample_interval must be finite and strictly positive")
    labels, counts, matrix = _transition_components(
        states,
        segment_ids=segment_ids,
        pseudocount=pseudocount,
        state_labels=state_labels,
    )
    stationary = estimate_stationary_distribution(matrix)
    production = entropy_production(matrix, stationary)
    flux = stationary[:, None] * matrix
    off_diagonal = ~np.eye(matrix.shape[0], dtype=bool)
    total_off_diagonal_flux = float(np.sum(flux[off_diagonal]))
    flux_asymmetry = (
        float(np.sum(np.abs(flux - flux.T)) / (2.0 * total_off_diagonal_flux))
        if total_off_diagonal_flux > 0.0
        else 0.0
    )
    balance_residual = float(np.linalg.norm(flux - flux.T, ord="fro"))
    denominator = float(np.sum(matrix[off_diagonal]))
    graph_asymmetry = (
        float(np.sum(np.abs(matrix - matrix.T)) / (2.0 * denominator)) if denominator > 0.0 else 0.0
    )
    return DirectionalitySummary(
        state_labels=labels,
        transition_counts=counts,
        transition_matrix=matrix,
        stationary_distribution=stationary,
        entropy_production=production,
        entropy_production_rate=production / sample_interval,
        flux_asymmetry=float(np.clip(flux_asymmetry, 0.0, 1.0)),
        detailed_balance_residual=balance_residual,
        transition_graph_asymmetry=float(np.clip(graph_asymmetry, 0.0, 1.0)),
        n_transitions=int(np.sum(counts)),
        pseudocount=float(pseudocount),
        sample_interval=sample_interval,
    )


class DirectionalityEstimator(EstimatorMixin):
    """Estimator wrapper whose score is entropy production per transition."""

    def __init__(self, *, pseudocount: float = 0.5, sample_interval: float = 1.0) -> None:
        self.pseudocount = pseudocount
        self.sample_interval = sample_interval

    def fit(
        self,
        states: ArrayLike,
        y: ArrayLike | None = None,
        *,
        segment_ids: ArrayLike | None = None,
        state_labels: ArrayLike | None = None,
    ) -> DirectionalityEstimator:
        del y
        self.summary_ = estimate_directionality(
            states,
            segment_ids=segment_ids,
            pseudocount=self.pseudocount,
            state_labels=state_labels,
            sample_interval=self.sample_interval,
        )
        self.transition_matrix_ = self.summary_.transition_matrix
        self.stationary_distribution_ = self.summary_.stationary_distribution
        return self

    def score(self, states: ArrayLike | None = None, y: ArrayLike | None = None) -> float:
        del states, y
        require_fitted(self, "summary_")
        return self.summary_.entropy_production
