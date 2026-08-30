"""Time-lagged communication-subspace alignment between regional trajectories."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from itertools import combinations
from numbers import Integral, Real

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._validation import FloatArray, as_float_matrix, lagged_pairs, validate_lags


@dataclass(frozen=True)
class CommunicationSubspaceModel:
    """Reduced-rank, ridge-regularised map from one module to another."""

    lag: int
    rank: int
    ridge: float
    source_mean: NDArray[np.float64]
    target_mean: NDArray[np.float64]
    coefficients: NDArray[np.float64]
    source_basis: NDArray[np.float64]
    target_basis: NDArray[np.float64]
    singular_values: NDArray[np.float64]
    n_pairs: int

    def predict(self, source: ArrayLike) -> FloatArray:
        """Predict target states from already lag-aligned source rows."""

        x = as_float_matrix(source, name="source", min_samples=1)
        if x.shape[1] != self.source_mean.size:
            raise ValueError(f"source has {x.shape[1]} features; expected {self.source_mean.size}")
        return (x - self.source_mean) @ self.coefficients + self.target_mean


@dataclass(frozen=True)
class AlignmentSummary:
    """Cross-validated shared predictive variance over prespecified lags."""

    lags: NDArray[np.int64]
    shared_predictive_variance_by_lag: NDArray[np.float64]
    shared_predictive_variance: float
    best_lag: int
    best_lag_shared_predictive_variance: float
    rank: int
    best_model: CommunicationSubspaceModel
    n_pairs_by_lag: NDArray[np.int64]


@dataclass(frozen=True)
class PairwiseAlignmentSummary:
    """Alignment aggregated across named module pairs and both directions."""

    mean_shared_predictive_variance: float
    pair_names: tuple[str, ...]
    pair_values: NDArray[np.float64]
    directional_summaries: tuple[AlignmentSummary, ...]


def _validate_rank(rank: int | None, n_source: int, n_target: int) -> int:
    maximum = min(n_source, n_target)
    if rank is None:
        return maximum
    if not isinstance(rank, Integral) or isinstance(rank, bool):
        raise TypeError("rank must be an integer or None")
    rank = int(rank)
    if not 1 <= rank <= maximum:
        raise ValueError(f"rank must lie in [1, {maximum}]")
    return rank


def _fit_paired_subspace(
    source: FloatArray,
    target: FloatArray,
    *,
    lag: int,
    rank: int,
    ridge: float,
) -> CommunicationSubspaceModel:
    if source.shape[0] != target.shape[0]:
        raise ValueError("paired source and target must have equal sample counts")
    source_mean = np.mean(source, axis=0)
    target_mean = np.mean(target, axis=0)
    x = source - source_mean
    y = target - target_mean
    gram = x.T @ x
    scale = float(np.trace(gram)) / max(gram.shape[0], 1)
    penalty = ridge * max(scale, np.finfo(np.float64).eps)
    regularised = gram.copy()
    regularised.flat[:: regularised.shape[0] + 1] += penalty
    cross_covariance = x.T @ y
    try:
        coefficients = np.linalg.solve(regularised, cross_covariance)
    except np.linalg.LinAlgError:
        # ``ridge=0`` is allowed for explicit OLS sensitivity analyses. Collinear
        # source features then require the minimum-norm least-squares solution.
        coefficients = np.linalg.lstsq(regularised, cross_covariance, rcond=None)[0]
    fitted = x @ coefficients
    _, _, right_transpose = np.linalg.svd(fitted, full_matrices=False)
    target_basis = right_transpose[:rank].T
    reduced_coefficients = coefficients @ target_basis @ target_basis.T
    source_basis_full, singular_values, target_basis_transpose = np.linalg.svd(
        reduced_coefficients, full_matrices=False
    )
    retained_rank = min(rank, singular_values.size)
    return CommunicationSubspaceModel(
        lag=lag,
        rank=retained_rank,
        ridge=ridge,
        source_mean=source_mean,
        target_mean=target_mean,
        coefficients=reduced_coefficients,
        source_basis=source_basis_full[:, :retained_rank],
        target_basis=target_basis_transpose[:retained_rank].T,
        singular_values=singular_values[:retained_rank],
        n_pairs=source.shape[0],
    )


def fit_communication_subspace(
    source: ArrayLike,
    target: ArrayLike,
    *,
    lag: int = 1,
    rank: int | None = None,
    ridge: float = 1e-6,
    segment_ids: ArrayLike | None = None,
) -> CommunicationSubspaceModel:
    """Fit a time-lagged reduced-rank regression communication subspace."""

    x = as_float_matrix(source, name="source", min_samples=3)
    y = as_float_matrix(target, name="target", min_samples=3)
    lag = validate_lags(lag, allow_zero=True)[0]
    if not isinstance(ridge, Real) or isinstance(ridge, bool):
        raise TypeError("ridge must be a non-negative real number")
    ridge = float(ridge)
    if not np.isfinite(ridge) or ridge < 0.0:
        raise ValueError("ridge must be finite and non-negative")
    paired_x, paired_y = lagged_pairs(x, y, lag, segment_ids)
    selected_rank = _validate_rank(rank, x.shape[1], y.shape[1])
    return _fit_paired_subspace(paired_x, paired_y, lag=lag, rank=selected_rank, ridge=ridge)


def _blocked_cross_validated_variance(
    source: FloatArray,
    target: FloatArray,
    *,
    lag: int,
    rank: int,
    ridge: float,
    n_splits: int,
) -> float:
    n_pairs = source.shape[0]
    if not isinstance(n_splits, Integral) or isinstance(n_splits, bool):
        raise TypeError("cv must be an integer")
    n_splits = int(n_splits)
    if not 2 <= n_splits <= n_pairs // 2:
        raise ValueError("cv must leave at least two observations in every test fold")
    folds = [fold for fold in np.array_split(np.arange(n_pairs), n_splits) if fold.size]
    squared_error = 0.0
    baseline_error = 0.0
    all_indices = np.arange(n_pairs)
    for test_indices in folds:
        train_mask = np.ones(n_pairs, dtype=bool)
        train_mask[test_indices] = False
        train_indices = all_indices[train_mask]
        model = _fit_paired_subspace(
            source[train_indices],
            target[train_indices],
            lag=lag,
            rank=rank,
            ridge=ridge,
        )
        prediction = model.predict(source[test_indices])
        residual = target[test_indices] - prediction
        baseline = target[test_indices] - model.target_mean
        squared_error += float(np.sum(residual * residual))
        baseline_error += float(np.sum(baseline * baseline))
    if baseline_error <= np.finfo(np.float64).eps:
        raise ValueError("target has no out-of-sample variance")
    return 1.0 - squared_error / baseline_error


def communication_subspace_alignment(
    source: ArrayLike,
    target: ArrayLike,
    *,
    lags: tuple[int, ...] = (1,),
    rank: int | None = None,
    ridge: float = 1e-6,
    cv: int = 5,
    segment_ids: ArrayLike | None = None,
) -> AlignmentSummary:
    """Estimate held-out target variance shared through a communication subspace.

    The returned primary value is the arithmetic mean across the prespecified lag
    grid. ``best_lag`` is diagnostic only; choosing the maximum after inspecting
    conditions would introduce an avoidable selection bias.
    """

    x = as_float_matrix(source, name="source", min_samples=4)
    y = as_float_matrix(target, name="target", min_samples=4)
    if x.shape[0] != y.shape[0]:
        raise ValueError("source and target must have the same number of samples")
    selected_lags = validate_lags(lags, allow_zero=True)
    selected_rank = _validate_rank(rank, x.shape[1], y.shape[1])
    if not isinstance(ridge, Real) or isinstance(ridge, bool):
        raise TypeError("ridge must be a non-negative real number")
    ridge = float(ridge)
    if not np.isfinite(ridge) or ridge < 0.0:
        raise ValueError("ridge must be finite and non-negative")
    values: list[float] = []
    models: list[CommunicationSubspaceModel] = []
    n_pairs: list[int] = []
    for lag in selected_lags:
        paired_x, paired_y = lagged_pairs(x, y, lag, segment_ids)
        value = _blocked_cross_validated_variance(
            paired_x,
            paired_y,
            lag=lag,
            rank=selected_rank,
            ridge=ridge,
            n_splits=cv,
        )
        model = _fit_paired_subspace(
            paired_x,
            paired_y,
            lag=lag,
            rank=selected_rank,
            ridge=ridge,
        )
        values.append(value)
        models.append(model)
        n_pairs.append(paired_x.shape[0])
    values_array = np.asarray(values, dtype=np.float64)
    best_index = int(np.argmax(values_array))
    return AlignmentSummary(
        lags=np.asarray(selected_lags, dtype=np.int64),
        shared_predictive_variance_by_lag=values_array,
        shared_predictive_variance=float(np.mean(values_array)),
        best_lag=selected_lags[best_index],
        best_lag_shared_predictive_variance=float(values_array[best_index]),
        rank=selected_rank,
        best_model=models[best_index],
        n_pairs_by_lag=np.asarray(n_pairs, dtype=np.int64),
    )


def principal_angles(
    basis_a: ArrayLike,
    basis_b: ArrayLike,
    *,
    degrees: bool = False,
) -> NDArray[np.float64]:
    """Return principal angles between column spaces in increasing order."""

    a = as_float_matrix(basis_a, name="basis_a", min_samples=1)
    b = as_float_matrix(basis_b, name="basis_b", min_samples=1)
    if a.shape[0] != b.shape[0]:
        raise ValueError("basis_a and basis_b must occupy the same ambient space")
    qa, ra = np.linalg.qr(a, mode="reduced")
    qb, rb = np.linalg.qr(b, mode="reduced")
    rank_a = int(np.linalg.matrix_rank(ra))
    rank_b = int(np.linalg.matrix_rank(rb))
    if rank_a == 0 or rank_b == 0:
        raise ValueError("both bases must span a non-zero subspace")
    singular_values = np.linalg.svd(qa[:, :rank_a].T @ qb[:, :rank_b], compute_uv=False)
    angles = np.arccos(np.clip(singular_values, -1.0, 1.0))
    if degrees:
        angles = np.degrees(angles)
    return np.sort(angles)


def pairwise_module_alignment(
    regional_trajectories: Mapping[str, ArrayLike],
    *,
    module_pairs: tuple[tuple[str, str], ...] | None = None,
    lags: tuple[int, ...] = (1,),
    rank: int | None = None,
    ridge: float = 1e-6,
    cv: int = 5,
    segment_ids: ArrayLike | None = None,
    bidirectional: bool = True,
) -> PairwiseAlignmentSummary:
    """Aggregate communication-subspace alignment across regional module pairs."""

    if not isinstance(regional_trajectories, Mapping) or len(regional_trajectories) < 2:
        raise ValueError("regional_trajectories must map at least two module names")
    modules = {
        str(name): as_float_matrix(values, name=f"regional_trajectories[{name!r}]", min_samples=4)
        for name, values in regional_trajectories.items()
    }
    sample_counts = {values.shape[0] for values in modules.values()}
    if len(sample_counts) != 1:
        raise ValueError("all regional trajectories must have equal sample counts")
    pairs = tuple(combinations(sorted(modules), 2)) if module_pairs is None else tuple(module_pairs)
    if not pairs:
        raise ValueError("module_pairs must contain at least one pair")
    pair_names: list[str] = []
    pair_values: list[float] = []
    summaries: list[AlignmentSummary] = []
    for source_name, target_name in pairs:
        if source_name not in modules or target_name not in modules:
            raise ValueError(f"unknown module pair ({source_name!r}, {target_name!r})")
        forward = communication_subspace_alignment(
            modules[source_name],
            modules[target_name],
            lags=lags,
            rank=rank,
            ridge=ridge,
            cv=cv,
            segment_ids=segment_ids,
        )
        directional = [forward]
        values = [forward.shared_predictive_variance]
        if bidirectional:
            reverse = communication_subspace_alignment(
                modules[target_name],
                modules[source_name],
                lags=lags,
                rank=rank,
                ridge=ridge,
                cv=cv,
                segment_ids=segment_ids,
            )
            directional.append(reverse)
            values.append(reverse.shared_predictive_variance)
        pair_names.append(
            f"{source_name}<->{target_name}" if bidirectional else f"{source_name}->{target_name}"
        )
        pair_values.append(float(np.mean(values)))
        summaries.extend(directional)
    values_array = np.asarray(pair_values, dtype=np.float64)
    return PairwiseAlignmentSummary(
        mean_shared_predictive_variance=float(np.mean(values_array)),
        pair_names=tuple(pair_names),
        pair_values=values_array,
        directional_summaries=tuple(summaries),
    )


lagged_shared_predictive_variance = communication_subspace_alignment
