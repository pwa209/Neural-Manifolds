"""Deterministic window-count matching for participant-condition metrics."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._validation import check_random_generator


@dataclass(frozen=True)
class MatchedMetricResult:
    """Metric draws, means and sampling variances after count matching."""

    group_names: tuple[str, ...]
    n_samples: int
    n_repeats: int
    estimates: dict[str, NDArray[np.float64]]
    means: dict[str, NDArray[np.float64]]
    sampling_variances: dict[str, NDArray[np.float64]]
    indices: tuple[dict[str, NDArray[np.int64]], ...]


def segment_ids_from_indices(indices: ArrayLike) -> NDArray[np.int64]:
    """Mark contiguous runs after subsampling a temporal trajectory.

    Random window matching can create gaps. Passing the returned identifiers to
    temporal estimators prevents them from treating samples on opposite sides of
    a gap as adjacent observations.
    """

    selected = np.asarray(indices)
    if selected.ndim != 1 or selected.size == 0:
        raise ValueError("indices must be a non-empty one-dimensional sequence")
    if selected.dtype.kind not in "iu":
        if selected.dtype.kind == "f" and np.all(selected == np.floor(selected)):
            selected = selected.astype(np.int64)
        else:
            raise ValueError("indices must contain integers")
    selected = np.asarray(selected, dtype=np.int64)
    if np.any(selected < 0) or np.any(np.diff(selected) <= 0):
        raise ValueError("indices must be strictly increasing and non-negative")
    boundaries = np.r_[True, np.diff(selected) != 1]
    return np.cumsum(boundaries, dtype=np.int64) - 1


def _normalise_group_sizes(
    group_sizes: Mapping[str, int] | Sequence[int],
) -> tuple[tuple[str, ...], tuple[int, ...]]:
    if isinstance(group_sizes, Mapping):
        names = tuple(str(name) for name in group_sizes)
        sizes_raw = tuple(group_sizes.values())
    else:
        sizes_raw = tuple(group_sizes)
        names = tuple(str(index) for index in range(len(sizes_raw)))
    if not sizes_raw:
        raise ValueError("group_sizes must contain at least one group")
    sizes: list[int] = []
    for size in sizes_raw:
        if not isinstance(size, Integral) or isinstance(size, bool) or size < 1:
            raise ValueError("every group size must be a positive integer")
        sizes.append(int(size))
    if len(set(names)) != len(names):
        raise ValueError("group names must be unique after conversion to strings")
    return names, tuple(sizes)


def matched_sample_indices(
    group_sizes: Mapping[str, int] | Sequence[int],
    *,
    n_samples: int | None = None,
    n_repeats: int = 100,
    random_state: int | np.random.Generator | None = 0,
    replace: bool = False,
) -> tuple[dict[str, NDArray[np.int64]], ...]:
    """Draw equal-size, order-preserving subsets for every group.

    Indices are sorted after sampling so temporal metrics see each selected
    trajectory in its original order. The draws themselves remain random and
    reproducible. Sampling with replacement is disabled by default because it can
    create artificial self-transitions and zero-distance neighbours.
    """

    names, sizes = _normalise_group_sizes(group_sizes)
    if n_samples is None:
        selected_count = min(sizes)
    else:
        if not isinstance(n_samples, Integral) or isinstance(n_samples, bool):
            raise TypeError("n_samples must be a positive integer or None")
        selected_count = int(n_samples)
        if selected_count < 1:
            raise ValueError("n_samples must be positive")
    if not replace and any(selected_count > size for size in sizes):
        raise ValueError("n_samples exceeds at least one group size without replacement")
    if not isinstance(n_repeats, Integral) or isinstance(n_repeats, bool) or n_repeats < 1:
        raise ValueError("n_repeats must be a positive integer")
    generator = check_random_generator(random_state)
    draws: list[dict[str, NDArray[np.int64]]] = []
    for _ in range(int(n_repeats)):
        repeat: dict[str, NDArray[np.int64]] = {}
        for name, size in zip(names, sizes, strict=True):
            indices = generator.choice(size, size=selected_count, replace=replace)
            repeat[name] = np.sort(np.asarray(indices, dtype=np.int64))
        draws.append(repeat)
    return tuple(draws)


def sample_matched_metric(
    groups: Mapping[str, ArrayLike],
    metric: Callable[[NDArray[Any]], ArrayLike | float],
    *,
    n_samples: int | None = None,
    n_repeats: int = 100,
    random_state: int | np.random.Generator | None = 0,
    replace: bool = False,
) -> MatchedMetricResult:
    """Evaluate a scalar or fixed-shape metric over repeated matched subsets."""

    if not isinstance(groups, Mapping) or not groups:
        raise ValueError("groups must be a non-empty mapping")
    if not callable(metric):
        raise TypeError("metric must be callable")
    arrays: dict[str, NDArray[Any]] = {}
    for raw_name, values in groups.items():
        name = str(raw_name)
        array = np.asarray(values)
        if array.ndim < 1 or array.shape[0] < 1:
            raise ValueError(f"group {name!r} must contain at least one sample")
        arrays[name] = array
    if len(arrays) != len(groups):
        raise ValueError("group names must be unique after conversion to strings")
    sizes = {name: array.shape[0] for name, array in arrays.items()}
    indices = matched_sample_indices(
        sizes,
        n_samples=n_samples,
        n_repeats=n_repeats,
        random_state=random_state,
        replace=replace,
    )
    estimates_list: dict[str, list[NDArray[np.float64]]] = {name: [] for name in arrays}
    expected_shapes: dict[str, tuple[int, ...]] = {}
    for repeat in indices:
        for name, array in arrays.items():
            value = np.asarray(metric(array[repeat[name]]), dtype=np.float64)
            if not np.all(np.isfinite(value)):
                raise ValueError(f"metric returned a non-finite value for group {name!r}")
            shape = value.shape
            if name in expected_shapes and expected_shapes[name] != shape:
                raise ValueError("metric output shape changed between matched draws")
            expected_shapes[name] = shape
            estimates_list[name].append(value)
    estimates = {
        name: np.stack(group_estimates, axis=0) for name, group_estimates in estimates_list.items()
    }
    means = {name: np.mean(values, axis=0) for name, values in estimates.items()}
    variances = {
        name: (
            np.var(values, axis=0, ddof=1)
            if values.shape[0] > 1
            else np.zeros(values.shape[1:], dtype=np.float64)
        )
        for name, values in estimates.items()
    }
    selected_count = len(indices[0][next(iter(arrays))])
    return MatchedMetricResult(
        group_names=tuple(arrays),
        n_samples=selected_count,
        n_repeats=int(n_repeats),
        estimates=estimates,
        means=means,
        sampling_variances=variances,
        indices=indices,
    )
