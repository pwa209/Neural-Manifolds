"""Stable Gaussian-HMM selection on discovery trajectories.

All sequences share one state dictionary. Discontinuous trajectories are passed
through ``lengths`` and are therefore never joined by an artificial transition.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.preprocessing import RobustScaler


@dataclass(frozen=True)
class HMMSelection:
    model: Any
    scaler: RobustScaler
    n_states: int
    heldout_log_likelihood_per_sample: float
    stability_ami: float
    candidate_table: tuple[dict[str, float | int | bool], ...]
    seeds: tuple[int, ...]


def _validated_sequences(sequences: Sequence[np.ndarray]) -> list[np.ndarray]:
    output = [np.asarray(sequence, dtype=float) for sequence in sequences]
    if not output or any(x.ndim != 2 or x.shape[0] < 3 for x in output):
        raise ValueError("each trajectory must be a time x features matrix with >=3 rows")
    if len({x.shape[1] for x in output}) != 1 or not all(np.all(np.isfinite(x)) for x in output):
        raise ValueError("trajectory feature dimensions must agree and values must be finite")
    return output


def _concatenate(sequences: Sequence[np.ndarray]) -> tuple[np.ndarray, list[int]]:
    return np.concatenate(sequences, axis=0), [len(sequence) for sequence in sequences]


def _new_model(n_states: int, seed: int, *, iterations: int, tolerance: float) -> Any:
    try:
        from hmmlearn.hmm import GaussianHMM
    except ImportError as exc:  # pragma: no cover - dynamics environment
        raise RuntimeError("install neural-manifolds[dynamics] for HMM estimation") from exc
    return GaussianHMM(
        n_components=n_states,
        covariance_type="diag",
        min_covar=1e-6,
        n_iter=iterations,
        tol=tolerance,
        random_state=seed,
        verbose=False,
        implementation="scaling",
    )


def fit_stable_gaussian_hmm(
    discovery_sequences: Sequence[np.ndarray],
    validation_sequences: Sequence[np.ndarray],
    *,
    state_counts: Sequence[int] = tuple(range(6, 21)),
    seeds: Sequence[int] = (1701, 2903, 4099, 6211, 7919, 8627, 9721, 11003, 12347, 14009),
    minimum_stability_ami: float = 0.70,
    iterations: int = 300,
    tolerance: float = 1e-3,
) -> HMMSelection:
    train_sequences = _validated_sequences(discovery_sequences)
    heldout_sequences = _validated_sequences(validation_sequences)
    if train_sequences[0].shape[1] != heldout_sequences[0].shape[1]:
        raise ValueError("discovery and validation feature dimensions differ")
    if len(seeds) < 2:
        raise ValueError("at least two initialisations are required for stability")
    scaler = RobustScaler(quantile_range=(25.0, 75.0)).fit(np.concatenate(train_sequences))
    train_scaled = [scaler.transform(sequence) for sequence in train_sequences]
    heldout_scaled = [scaler.transform(sequence) for sequence in heldout_sequences]
    train, train_lengths = _concatenate(train_scaled)
    heldout, heldout_lengths = _concatenate(heldout_scaled)

    candidates: list[dict[str, float | int | bool]] = []
    fitted: dict[tuple[int, int], Any] = {}
    for n_states in state_counts:
        if n_states < 2 or n_states >= train.shape[0]:
            continue
        predictions: list[np.ndarray] = []
        scores: list[float] = []
        converged: list[bool] = []
        for seed in seeds:
            model = _new_model(n_states, int(seed), iterations=iterations, tolerance=tolerance)
            model.fit(train, lengths=train_lengths)
            prediction = model.predict(heldout, lengths=heldout_lengths)
            predictions.append(prediction)
            scores.append(float(model.score(heldout, lengths=heldout_lengths) / heldout.shape[0]))
            converged.append(bool(model.monitor_.converged))
            fitted[(int(n_states), int(seed))] = model
        pairwise = [
            adjusted_mutual_info_score(first, second)
            for first, second in combinations(predictions, 2)
        ]
        stability = float(np.mean(pairwise))
        candidates.append(
            {
                "n_states": int(n_states),
                "heldout_log_likelihood_per_sample": float(np.mean(scores)),
                "heldout_score_sd": float(np.std(scores, ddof=1)),
                "stability_ami": stability,
                "all_converged": bool(all(converged)),
            }
        )
    eligible = [
        row
        for row in candidates
        if row["stability_ami"] >= minimum_stability_ami and bool(row["all_converged"])
    ]
    if not eligible:
        raise RuntimeError(
            f"no HMM candidate reached AMI >= {minimum_stability_ami} with convergence"
        )
    # State count is selected with the one-standard-error rule: among stable,
    # converged candidates statistically indistinguishable from the best held-out
    # score, retain the smallest dictionary. This avoids a systematic preference
    # for splitting a genuine state into redundant components.
    best = max(eligible, key=lambda row: float(row["heldout_log_likelihood_per_sample"]))
    standard_error = float(best["heldout_score_sd"]) / np.sqrt(len(seeds))
    threshold = float(best["heldout_log_likelihood_per_sample"]) - standard_error
    competitive = [
        row for row in eligible if float(row["heldout_log_likelihood_per_sample"]) >= threshold
    ]
    selected = min(competitive, key=lambda row: int(row["n_states"]))
    n_states = int(selected["n_states"])
    best_seed = max(
        seeds,
        key=lambda seed: fitted[(n_states, int(seed))].score(heldout, lengths=heldout_lengths),
    )
    return HMMSelection(
        model=fitted[(n_states, int(best_seed))],
        scaler=scaler,
        n_states=n_states,
        heldout_log_likelihood_per_sample=float(selected["heldout_log_likelihood_per_sample"]),
        stability_ami=float(selected["stability_ami"]),
        candidate_table=tuple(candidates),
        seeds=tuple(int(seed) for seed in seeds),
    )


def predict_sequences(selection: HMMSelection, sequences: Sequence[np.ndarray]) -> list[np.ndarray]:
    validated = _validated_sequences(sequences)
    return [selection.model.predict(selection.scaler.transform(sequence)) for sequence in validated]
