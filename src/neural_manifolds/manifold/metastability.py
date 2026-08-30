"""Metastable persistence and revisability summaries for discrete state paths."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._base import EstimatorMixin, require_fitted
from ._validation import encode_states, validate_segment_ids


@dataclass(frozen=True)
class MetastabilitySummary:
    """Persistence-revisability summary from a shared state dictionary.

    Dwell statistics and switching rate quantify persistence. ``exit_entropy``
    and ``recurrence_probability`` quantify complementary aspects of revisability.
    The scalar consciousness-facing score is intentionally *not* stored here; it
    must be calculated relative to a healthy reference using
    :func:`metastability_score`.
    """

    state_labels: NDArray[Any]
    occupancy: NDArray[np.float64]
    transition_counts: NDArray[np.int64]
    median_dwell: float
    mean_dwell: float
    dwell_standard_deviation: float
    dwell_dispersion: float
    switching_rate: float
    recurrence_probability: float
    exit_entropy: float
    n_observations: int
    n_runs: int
    n_segments: int
    sample_interval: float
    median_dwell_by_state: NDArray[np.float64]

    def feature_vector(self) -> NDArray[np.float64]:
        """Return the reference-scored persistence-revisability feature vector."""

        return np.asarray(
            [
                np.log(max(self.median_dwell, np.finfo(np.float64).tiny)),
                np.log1p(self.dwell_dispersion),
                self.recurrence_probability,
                self.exit_entropy,
            ],
            dtype=np.float64,
        )

    @property
    def persistence(self) -> NDArray[np.float64]:
        """Two transparent persistence coordinates: dwell and switching rate."""

        return np.asarray([self.median_dwell, self.switching_rate], dtype=np.float64)

    @property
    def revisability(self) -> NDArray[np.float64]:
        """Two transparent revisability coordinates: recurrence and exit entropy."""

        return np.asarray([self.recurrence_probability, self.exit_entropy], dtype=np.float64)


def _run_encoding(
    encoded_states: NDArray[np.int64], segment_ids: NDArray[Any]
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[Any]]:
    boundaries = np.ones(encoded_states.size, dtype=bool)
    if encoded_states.size > 1:
        boundaries[1:] = (encoded_states[1:] != encoded_states[:-1]) | (
            segment_ids[1:] != segment_ids[:-1]
        )
    starts = np.flatnonzero(boundaries)
    stops = np.r_[starts[1:], encoded_states.size]
    return encoded_states[starts], stops - starts, segment_ids[starts]


def estimate_metastability(
    states: ArrayLike,
    *,
    segment_ids: ArrayLike | None = None,
    sample_interval: float = 1.0,
) -> MetastabilitySummary:
    """Summarise dwell, recurrence and exit diversity of a state sequence.

    Segment boundaries terminate dwell runs and are never counted as state
    transitions. ``sample_interval`` converts run lengths and switching rate into
    physical time units (for example seconds and switches per second).
    """

    if not isinstance(sample_interval, Real) or isinstance(sample_interval, bool):
        raise TypeError("sample_interval must be a positive real number")
    sample_interval = float(sample_interval)
    if not np.isfinite(sample_interval) or sample_interval <= 0.0:
        raise ValueError("sample_interval must be finite and strictly positive")
    labels, encoded = encode_states(states)
    segments = validate_segment_ids(segment_ids, encoded.size)
    run_states, run_lengths_samples, run_segments = _run_encoding(encoded, segments)
    dwell = run_lengths_samples.astype(np.float64) * sample_interval
    n_states = labels.size

    occupancy = np.bincount(encoded, minlength=n_states).astype(np.float64)
    occupancy /= occupancy.sum()
    transition_counts = np.zeros((n_states, n_states), dtype=np.int64)
    if run_states.size > 1:
        valid_run_transitions = run_segments[1:] == run_segments[:-1]
        np.add.at(
            transition_counts,
            (
                run_states[:-1][valid_run_transitions],
                run_states[1:][valid_run_transitions],
            ),
            1,
        )

    exits_by_state = np.sum(transition_counts, axis=1)
    total_exits = int(np.sum(exits_by_state))
    if n_states <= 1 or total_exits == 0:
        exit_entropy = 0.0
    else:
        state_entropies = np.zeros(n_states, dtype=np.float64)
        normalizer = np.log(max(n_states - 1, 1))
        for state_index in range(n_states):
            row = transition_counts[state_index].astype(np.float64)
            row[state_index] = 0.0
            row_total = float(np.sum(row))
            if row_total <= 0.0:
                continue
            probabilities = row[row > 0.0] / row_total
            entropy = -float(np.sum(probabilities * np.log(probabilities)))
            # With two states an exit has only one possible destination. Define
            # its conditional exit entropy as zero rather than divide by log(1).
            state_entropies[state_index] = entropy / normalizer if normalizer > 0.0 else 0.0
        exit_entropy = float(np.sum(state_entropies * exits_by_state) / max(total_exits, 1))

    returns = 0
    eligible_revisits = 0
    seen: set[int] = set()
    previous_segment: Any = object()
    for state, segment in zip(run_states.tolist(), run_segments.tolist(), strict=True):
        if segment != previous_segment:
            seen = {state}
            previous_segment = segment
            continue
        eligible_revisits += 1
        if state in seen:
            returns += 1
        seen.add(state)
    recurrence_probability = returns / eligible_revisits if eligible_revisits else 0.0

    unique_segments = 1 + int(np.count_nonzero(segments[1:] != segments[:-1]))
    possible_within_segment_steps = encoded.size - unique_segments
    switching_rate = (
        total_exits / (possible_within_segment_steps * sample_interval)
        if possible_within_segment_steps > 0
        else 0.0
    )
    median = float(np.median(dwell))
    lower, upper = np.percentile(dwell, [25.0, 75.0])
    dispersion = float((upper - lower) / median) if median > 0.0 else 0.0
    median_by_state = np.asarray(
        [
            np.median(dwell[run_states == state_index])
            if np.any(run_states == state_index)
            else np.nan
            for state_index in range(n_states)
        ],
        dtype=np.float64,
    )
    return MetastabilitySummary(
        state_labels=labels.copy(),
        occupancy=occupancy,
        transition_counts=transition_counts,
        median_dwell=median,
        mean_dwell=float(np.mean(dwell)),
        dwell_standard_deviation=float(np.std(dwell, ddof=1)) if dwell.size > 1 else 0.0,
        dwell_dispersion=dispersion,
        switching_rate=float(switching_rate),
        recurrence_probability=float(recurrence_probability),
        exit_entropy=float(np.clip(exit_entropy, 0.0, 1.0)),
        n_observations=encoded.size,
        n_runs=run_states.size,
        n_segments=unique_segments,
        sample_interval=sample_interval,
        median_dwell_by_state=median_by_state,
    )


class MetastabilityReference(EstimatorMixin):
    """Healthy-reference optimum used for the secondary scalar visualisation.

    The score is the negative Mahalanobis distance from the reference mean, as
    specified in the proposal. It is never inferred from the condition being
    scored. A value closer to zero indicates a more reference-like persistence-
    revisability balance; it is not a monotonic dwell-time measure.
    """

    def __init__(self, *, covariance_shrinkage: float = 0.1, ridge: float = 1e-8) -> None:
        self.covariance_shrinkage = covariance_shrinkage
        self.ridge = ridge

    def fit(
        self,
        summaries: list[MetastabilitySummary] | NDArray[np.float64],
        y: ArrayLike | None = None,
    ) -> MetastabilityReference:
        del y
        if isinstance(summaries, np.ndarray):
            features = np.asarray(summaries, dtype=np.float64)
        else:
            features = np.asarray(
                [summary.feature_vector() for summary in summaries], dtype=np.float64
            )
        if features.ndim != 2 or features.shape[1] != 4:
            raise ValueError("summaries must provide an n-by-4 feature matrix")
        if features.shape[0] < 2:
            raise ValueError("at least two healthy-reference summaries are required")
        if not np.all(np.isfinite(features)):
            raise ValueError("reference features contain NaN or infinite values")
        coefficient = float(self.covariance_shrinkage)
        if not 0.0 <= coefficient <= 1.0:
            raise ValueError("covariance_shrinkage must lie in [0, 1]")
        if not np.isfinite(self.ridge) or self.ridge <= 0.0:
            raise ValueError("ridge must be finite and strictly positive")
        covariance = np.cov(features, rowvar=False, ddof=1)
        target = float(np.trace(covariance)) / covariance.shape[0]
        covariance = (1.0 - coefficient) * covariance
        covariance.flat[:: covariance.shape[0] + 1] += coefficient * target
        ridge_scale = max(target, 1.0) * float(self.ridge)
        covariance.flat[:: covariance.shape[0] + 1] += ridge_scale
        self.mean_ = np.mean(features, axis=0)
        self.covariance_ = (covariance + covariance.T) / 2.0
        self.precision_ = np.linalg.pinv(self.covariance_, hermitian=True)
        self.n_reference_ = features.shape[0]
        return self

    def transform(
        self, summaries: list[MetastabilitySummary] | NDArray[np.float64]
    ) -> NDArray[np.float64]:
        require_fitted(self, "mean_", "precision_")
        if isinstance(summaries, np.ndarray):
            features = np.asarray(summaries, dtype=np.float64)
            if features.ndim == 1:
                features = features[None, :]
        else:
            features = np.asarray(
                [summary.feature_vector() for summary in summaries], dtype=np.float64
            )
        if features.ndim != 2 or features.shape[1] != 4:
            raise ValueError("summaries must provide an n-by-4 feature matrix")
        differences = features - self.mean_
        squared = np.einsum("ni,ij,nj->n", differences, self.precision_, differences, optimize=True)
        return -np.sqrt(np.maximum(squared, 0.0))

    def score(self, summary: MetastabilitySummary) -> float:
        return float(self.transform([summary])[0])


def metastability_score(
    summary: MetastabilitySummary,
    reference: MetastabilityReference,
) -> float:
    """Return negative Mahalanobis distance to a fitted healthy-wake optimum."""

    if not isinstance(reference, MetastabilityReference):
        raise TypeError("reference must be a fitted MetastabilityReference")
    return reference.score(summary)


persistence_revisability_profile = estimate_metastability


class MetastabilityEstimator(EstimatorMixin):
    """Scikit-learn-compatible wrapper around :func:`estimate_metastability`."""

    def __init__(self, *, sample_interval: float = 1.0) -> None:
        self.sample_interval = sample_interval

    def fit(
        self,
        states: ArrayLike,
        y: ArrayLike | None = None,
        *,
        segment_ids: ArrayLike | None = None,
    ) -> MetastabilityEstimator:
        del y
        self.summary_ = estimate_metastability(
            states,
            segment_ids=segment_ids,
            sample_interval=self.sample_interval,
        )
        self.n_features_in_ = 1
        return self

    def score(self, states: ArrayLike | None = None, y: ArrayLike | None = None) -> float:
        del states, y
        require_fitted(self, "summary_")
        # This transparent score is useful for estimator diagnostics only. It is
        # not substituted for the reference-relative M axis in the profile API.
        return self.summary_.exit_entropy
