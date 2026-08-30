"""Discovery-fitted state dictionary with an explicitly recorded robust fallback.

The dictionary is fitted only from signal trajectories selected before outcome
labels are joined. HMM instability is reported, but it does not become a
scientific go/no-go rule: a fixed-K clustering dictionary keeps the exploratory
workflow executable and the primary HMM result remains marked unavailable.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler

from .hmm import HMMSelection, fit_stable_gaussian_hmm


def _validate_sequences(sequences: Sequence[np.ndarray]) -> list[NDArray[np.float64]]:
    values = [np.asarray(sequence, dtype=np.float64) for sequence in sequences]
    if not values:
        raise ValueError("at least one trajectory is required")
    if any(value.ndim != 2 or value.shape[0] < 3 for value in values):
        raise ValueError("trajectories must be time x features with at least three rows")
    if len({value.shape[1] for value in values}) != 1:
        raise ValueError("trajectory feature dimensions differ")
    if not all(np.all(np.isfinite(value)) for value in values):
        raise ValueError("trajectories contain non-finite values")
    return values


def _even_subsample(sequence: NDArray[np.float64], maximum: int) -> NDArray[np.float64]:
    if maximum <= 0:
        raise ValueError("maximum samples per sequence must be positive")
    if sequence.shape[0] <= maximum:
        return sequence
    indices = np.linspace(0, sequence.shape[0] - 1, maximum, dtype=int)
    return sequence[indices]


@dataclass
class StateDictionary:
    """Serializable signal-only projection and state assignment."""

    projection: PCA
    method: str
    n_states: int
    hmm: HMMSelection | None = None
    fallback_scaler: RobustScaler | None = None
    fallback_model: MiniBatchKMeans | None = None
    primary_status: str = "available"
    primary_error: str | None = None

    def project(self, trajectory: np.ndarray) -> NDArray[np.float64]:
        value = _validate_sequences([trajectory])[0]
        return np.asarray(self.projection.transform(value), dtype=np.float64)

    def predict_projected(
        self, projected: np.ndarray, *, segment_ids: np.ndarray | None = None
    ) -> NDArray[np.int64]:
        value = _validate_sequences([projected])[0]
        if self.method == "gaussian_hmm":
            if self.hmm is None:
                raise RuntimeError("HMM state dictionary is incomplete")
            scaled = self.hmm.scaler.transform(value)
            lengths = None
            if segment_ids is not None:
                segments = np.asarray(segment_ids)
                if segments.ndim != 1 or len(segments) != len(value):
                    raise ValueError("segment_ids must align with projected rows")
                boundaries = np.r_[
                    0, np.flatnonzero(segments[1:] != segments[:-1]) + 1, len(segments)
                ]
                lengths = np.diff(boundaries).tolist()
            return np.asarray(self.hmm.model.predict(scaled, lengths=lengths), dtype=np.int64)
        if self.method == "fixed_k_robust_kmeans":
            if self.fallback_scaler is None or self.fallback_model is None:
                raise RuntimeError("fallback state dictionary is incomplete")
            scaled = self.fallback_scaler.transform(value)
            return np.asarray(self.fallback_model.predict(scaled), dtype=np.int64)
        raise RuntimeError(f"unknown state dictionary method: {self.method}")

    def predict(
        self, trajectory: np.ndarray, *, segment_ids: np.ndarray | None = None
    ) -> NDArray[np.int64]:
        return self.predict_projected(self.project(trajectory), segment_ids=segment_ids)

    def audit(self) -> dict[str, Any]:
        output: dict[str, Any] = {
            "method": self.method,
            "n_states": self.n_states,
            "primary_status": self.primary_status,
            "primary_error": self.primary_error,
            "projection_components": int(self.projection.n_components_),
            "projection_explained_variance": float(
                np.sum(self.projection.explained_variance_ratio_)
            ),
        }
        if self.hmm is not None:
            output.update(
                {
                    "hmm_stability_ami": self.hmm.stability_ami,
                    "hmm_heldout_log_likelihood_per_sample": (
                        self.hmm.heldout_log_likelihood_per_sample
                    ),
                    "hmm_candidates": list(self.hmm.candidate_table),
                    "hmm_seeds": list(self.hmm.seeds),
                }
            )
        return output


def fit_state_dictionary(
    discovery_sequences: Sequence[np.ndarray],
    validation_sequences: Sequence[np.ndarray],
    *,
    rank: int,
    state_counts: Sequence[int],
    seeds: Sequence[int],
    minimum_stability_ami: float,
    maximum_samples_per_sequence: int = 600,
    force_fallback: bool = False,
) -> StateDictionary:
    """Fit PCA and a stable HMM, retaining a fixed method if stability is absent."""

    discovery = _validate_sequences(discovery_sequences)
    validation = _validate_sequences(validation_sequences)
    if discovery[0].shape[1] != validation[0].shape[1]:
        raise ValueError("discovery and validation feature dimensions differ")
    discovery_fit = [_even_subsample(value, maximum_samples_per_sequence) for value in discovery]
    stacked = np.concatenate(discovery_fit, axis=0)
    n_components = min(int(rank), stacked.shape[1], stacked.shape[0] - 1)
    if n_components < 2:
        raise ValueError("insufficient rank for the state dictionary")
    projection = PCA(n_components=n_components, svd_solver="full").fit(stacked)
    projected_discovery = [projection.transform(value) for value in discovery_fit]
    projected_validation = [
        projection.transform(_even_subsample(value, maximum_samples_per_sequence))
        for value in validation
    ]

    primary_error: str | None = None
    if not force_fallback:
        try:
            hmm = fit_stable_gaussian_hmm(
                projected_discovery,
                projected_validation,
                state_counts=state_counts,
                seeds=seeds,
                minimum_stability_ami=minimum_stability_ami,
            )
            return StateDictionary(
                projection=projection,
                method="gaussian_hmm",
                n_states=hmm.n_states,
                hmm=hmm,
            )
        except RuntimeError as error:
            primary_error = f"{type(error).__name__}: {error}"
    else:
        primary_error = "forced fallback for a technical control"

    eligible = sorted({int(value) for value in state_counts if int(value) >= 2})
    if not eligible:
        raise ValueError("state_counts contains no valid state count")
    fixed_states = eligible[len(eligible) // 2]
    scaler = RobustScaler(quantile_range=(25.0, 75.0)).fit(
        np.concatenate(projected_discovery, axis=0)
    )
    scaled = scaler.transform(np.concatenate(projected_discovery, axis=0))
    model = MiniBatchKMeans(
        n_clusters=fixed_states,
        batch_size=min(4096, max(256, scaled.shape[0])),
        n_init=20,
        random_state=int(seeds[0]),
    ).fit(scaled)
    return StateDictionary(
        projection=projection,
        method="fixed_k_robust_kmeans",
        n_states=fixed_states,
        fallback_scaler=scaler,
        fallback_model=model,
        primary_status="unavailable",
        primary_error=primary_error,
    )
