"""Shared input validation and deterministic random-number handling."""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def as_float_matrix(
    x: ArrayLike,
    *,
    name: str = "X",
    min_samples: int = 2,
    min_features: int = 1,
) -> FloatArray:
    """Return a finite, real-valued, two-dimensional ``float64`` array."""

    array = np.asarray(x)
    if array.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional; got shape {array.shape}")
    if np.iscomplexobj(array):
        raise ValueError(f"{name} must be real-valued")
    try:
        array = np.asarray(array, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric values") from exc
    if array.shape[0] < min_samples:
        raise ValueError(
            f"{name} must contain at least {min_samples} samples; got {array.shape[0]}"
        )
    if array.shape[1] < min_features:
        raise ValueError(
            f"{name} must contain at least {min_features} feature(s); got {array.shape[1]}"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or infinite values")
    return np.ascontiguousarray(array)


def encode_states(
    states: ArrayLike,
    *,
    n_samples: int | None = None,
    name: str = "states",
) -> tuple[NDArray[Any], IntArray]:
    """Validate a one-dimensional state sequence and encode labels as integers."""

    array = np.asarray(states)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional; got shape {array.shape}")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if n_samples is not None and array.size != n_samples:
        raise ValueError(f"{name} has {array.size} observations but expected {n_samples}")
    if array.dtype.kind in "fc" and not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or infinite labels")
    if array.dtype.kind == "O":
        if any(value is None for value in array.tolist()):
            raise ValueError(f"{name} contains None")
        try:
            labels, inverse = np.unique(array, return_inverse=True)
        except TypeError as exc:
            raise ValueError(f"{name} labels must be mutually comparable") from exc
    else:
        labels, inverse = np.unique(array, return_inverse=True)
    return labels, np.asarray(inverse, dtype=np.int64)


def validate_segment_ids(
    segment_ids: ArrayLike | None,
    n_samples: int,
) -> NDArray[Any]:
    """Return segment identifiers, using one continuous segment by default."""

    if segment_ids is None:
        return np.zeros(n_samples, dtype=np.int64)
    segments = np.asarray(segment_ids)
    if segments.ndim != 1 or segments.size != n_samples:
        raise ValueError("segment_ids must be one-dimensional and match the number of samples")
    if segments.dtype.kind in "fc" and not np.all(np.isfinite(segments)):
        raise ValueError("segment_ids contains NaN or infinite values")
    if segments.dtype.kind == "O" and any(value is None for value in segments.tolist()):
        raise ValueError("segment_ids contains None")
    return segments


def check_random_generator(
    random_state: int | np.random.Generator | None,
) -> np.random.Generator:
    """Return a NumPy generator without mutating NumPy's global RNG state."""

    if isinstance(random_state, np.random.Generator):
        return random_state
    if random_state is None or isinstance(random_state, Integral):
        return np.random.default_rng(random_state)
    raise TypeError("random_state must be None, an integer, or numpy.random.Generator")


def validate_square_matrix(
    matrix: ArrayLike,
    *,
    name: str,
    symmetric: bool = False,
    positive_semidefinite: bool = False,
    tolerance: float = 1e-10,
) -> FloatArray:
    """Validate a finite square matrix and optional symmetry/PSD constraints."""

    array = as_float_matrix(matrix, name=name, min_samples=1, min_features=1)
    if array.shape[0] != array.shape[1]:
        raise ValueError(f"{name} must be square; got shape {array.shape}")
    scale = max(float(np.linalg.norm(array, ord=2)), 1.0)
    if symmetric and not np.allclose(array, array.T, atol=tolerance * scale, rtol=tolerance):
        raise ValueError(f"{name} must be symmetric")
    if positive_semidefinite:
        eigenvalues = np.linalg.eigvalsh((array + array.T) / 2.0)
        if eigenvalues[0] < -tolerance * scale:
            raise ValueError(
                f"{name} must be positive semidefinite; minimum eigenvalue is {eigenvalues[0]:.3g}"
            )
    return array


def validate_lags(lags: int | Sequence[int], *, allow_zero: bool = False) -> tuple[int, ...]:
    """Validate and canonicalise a collection of unique non-negative lags."""

    values = (int(lags),) if isinstance(lags, Integral) else tuple(lags)
    if not values:
        raise ValueError("lags must contain at least one value")
    if any(not isinstance(lag, Integral) for lag in values):
        raise TypeError("lags must contain integers")
    values = tuple(int(lag) for lag in values)
    minimum = 0 if allow_zero else 1
    if any(lag < minimum for lag in values):
        relation = "non-negative" if allow_zero else "strictly positive"
        raise ValueError(f"lags must be {relation}")
    if len(set(values)) != len(values):
        raise ValueError("lags must not contain duplicates")
    return values


def lagged_pairs(
    source: FloatArray,
    target: FloatArray,
    lag: int,
    segment_ids: ArrayLike | None = None,
) -> tuple[FloatArray, FloatArray]:
    """Pair ``source[t]`` with ``target[t + lag]`` within segment boundaries."""

    if source.shape[0] != target.shape[0]:
        raise ValueError("source and target must have the same number of samples")
    if lag < 0 or lag >= source.shape[0]:
        raise ValueError("lag must satisfy 0 <= lag < n_samples")
    segments = validate_segment_ids(segment_ids, source.shape[0])
    if lag == 0:
        return source.copy(), target.copy()
    mask = segments[:-lag] == segments[lag:]
    x = source[:-lag][mask]
    y = target[lag:][mask]
    if x.shape[0] < 2:
        raise ValueError(f"lag {lag} leaves fewer than two within-segment pairs")
    return x, y


def safe_scale(values: ArrayLike, *, floor: float = 1e-12) -> float:
    """Return a robust nonzero scale based on median absolute deviation."""

    array = np.asarray(values, dtype=np.float64)
    median = float(np.median(array))
    mad = 1.4826 * float(np.median(np.abs(array - median)))
    if not np.isfinite(mad) or mad < floor:
        standard_deviation = float(np.std(array, ddof=1)) if array.size > 1 else 0.0
        return max(standard_deviation, floor)
    return mad
