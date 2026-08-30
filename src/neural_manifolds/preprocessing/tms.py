"""TMS-EEG pulse handling and millisecond-resolution perturbational trajectories."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from scipy import interpolate
from sklearn.decomposition import PCA


def interpolate_continuous_pulses(
    data: np.ndarray,
    pulse_samples: np.ndarray,
    sampling_hz: float,
    *,
    start_seconds: float = -0.005,
    stop_seconds: float = 0.015,
    support_seconds: float = 0.020,
) -> np.ndarray:
    """Interpolate TMS gaps on continuous channels before temporal filtering.

    Pulse samples are zero-based indices into ``data``.  A local cubic spline is
    fitted from clean support on both sides of each gap.  Overlapping gaps and
    edge-truncated support are rejected instead of silently changing the pulse
    contract.
    """

    values = np.asarray(data, dtype=float)
    samples = np.asarray(pulse_samples, dtype=np.int64)
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 2:
        raise ValueError("continuous data must be channels x samples")
    if samples.ndim != 1 or samples.size < 1:
        raise ValueError("at least one pulse sample is required")
    if not np.isfinite(sampling_hz) or sampling_hz <= 0:
        raise ValueError("sampling_hz must be finite and positive")
    if not start_seconds < stop_seconds or support_seconds <= 0:
        raise ValueError("pulse and support intervals are invalid")
    start_offset = int(np.floor(start_seconds * sampling_hz))
    stop_offset = int(np.ceil(stop_seconds * sampling_hz))
    support = max(2, int(np.ceil(support_seconds * sampling_hz)))
    intervals = np.column_stack((samples + start_offset, samples + stop_offset))
    order = np.argsort(intervals[:, 0])
    intervals = intervals[order]
    if np.any(intervals[1:, 0] <= intervals[:-1, 1]):
        raise ValueError("TMS interpolation intervals overlap")
    if intervals[0, 0] - support < 0 or intervals[-1, 1] + support >= values.shape[1]:
        raise ValueError("a TMS pulse lacks complete interpolation support")

    output = values.copy()
    for start, stop in intervals:
        removed = np.arange(start, stop + 1, dtype=np.int64)
        support_indices = np.r_[
            np.arange(start - support, start, dtype=np.int64),
            np.arange(stop + 1, stop + support + 1, dtype=np.int64),
        ]
        for channel in range(values.shape[0]):
            model = interpolate.CubicSpline(support_indices, output[channel, support_indices])
            output[channel, removed] = model(removed)
    return output


def interpolate_pulse_interval(
    epochs: np.ndarray,
    times_seconds: np.ndarray,
    *,
    start_seconds: float = -0.005,
    stop_seconds: float = 0.015,
) -> np.ndarray:
    """Cubic-interpolate the TMS pulse interval independently by epoch/channel."""

    x = np.asarray(epochs, dtype=float)
    times = np.asarray(times_seconds, dtype=float)
    if x.ndim != 3 or x.shape[-1] != times.size:
        raise ValueError("epochs must be epochs x channels x time")
    removed = (times >= start_seconds) & (times <= stop_seconds)
    support = ~removed
    if np.count_nonzero(support) < 4 or not np.any(removed):
        raise ValueError("pulse interval/support is invalid")
    output = x.copy()
    for epoch in range(x.shape[0]):
        for channel in range(x.shape[1]):
            model = interpolate.CubicSpline(times[support], x[epoch, channel, support])
            output[epoch, channel, removed] = model(times[removed])
    return output


@dataclass(frozen=True)
class PerturbationalTrajectory:
    mean_trajectory: np.ndarray
    trial_trajectories: np.ndarray
    explained_variance_ratio: np.ndarray
    baseline_centroid: np.ndarray


def fit_shared_perturbational_trajectories(
    conditions: Mapping[str, np.ndarray],
    times_seconds: np.ndarray,
    *,
    baseline: tuple[float, float] = (-0.5, -0.05),
    sample_step_seconds: float = 0.005,
    rank: int = 32,
    random_state: int = 0,
) -> dict[str, PerturbationalTrajectory]:
    """Fit one label-agnostic PCA basis and then summarize named conditions."""

    if len(conditions) < 2:
        raise ValueError("at least two conditions are required for a shared trajectory basis")
    times = np.asarray(times_seconds, dtype=float)
    arrays = {name: np.asarray(value, dtype=float) for name, value in conditions.items()}
    shapes = {(value.shape[1], value.shape[2]) for value in arrays.values() if value.ndim == 3}
    if len(shapes) != 1 or any(value.ndim != 3 for value in arrays.values()):
        raise ValueError("conditions must share channels and time samples")
    channels, samples = next(iter(shapes))
    if samples != times.size or any(value.shape[0] < 2 for value in arrays.values()):
        raise ValueError("each condition needs at least two trials aligned to times")
    baseline_mask = (times >= baseline[0]) & (times <= baseline[1])
    if np.count_nonzero(baseline_mask) < 2:
        raise ValueError("baseline interval contains too few samples")
    native_step = float(np.median(np.diff(times)))
    stride = max(1, round(sample_step_seconds / native_step))
    baseline_sampled = baseline_mask[::stride]
    centred: dict[str, np.ndarray] = {}
    flattened: list[np.ndarray] = []
    for name, value in arrays.items():
        baseline_mean = value[:, :, baseline_mask].mean(axis=2, keepdims=True)
        sampled = (value - baseline_mean)[:, :, ::stride]
        centred[name] = sampled
        flattened.append(sampled.transpose(0, 2, 1).reshape(-1, channels))
    pooled = np.concatenate(flattened, axis=0)
    fitted_rank = min(rank, channels, pooled.shape[0])
    pca = PCA(n_components=fitted_rank, svd_solver="full", random_state=random_state).fit(pooled)
    output: dict[str, PerturbationalTrajectory] = {}
    for name, sampled in centred.items():
        transformed = pca.transform(sampled.transpose(0, 2, 1).reshape(-1, channels)).reshape(
            sampled.shape[0], sampled.shape[2], fitted_rank
        )
        output[name] = PerturbationalTrajectory(
            mean_trajectory=transformed.mean(axis=0),
            trial_trajectories=transformed,
            explained_variance_ratio=pca.explained_variance_ratio_,
            baseline_centroid=transformed[:, baseline_sampled].mean(axis=(0, 1)),
        )
    return output


def fit_perturbational_trajectory(
    epochs: np.ndarray,
    times_seconds: np.ndarray,
    *,
    baseline: tuple[float, float] = (-0.5, -0.05),
    sample_step_seconds: float = 0.005,
    rank: int = 32,
    random_state: int = 0,
) -> PerturbationalTrajectory:
    """Fit a training-local PCA trajectory without using condition labels."""

    x = np.asarray(epochs, dtype=float)
    times = np.asarray(times_seconds, dtype=float)
    if x.ndim != 3 or x.shape[2] != times.size:
        raise ValueError("epochs must be epochs x channels x time")
    baseline_mask = (times >= baseline[0]) & (times <= baseline[1])
    if np.count_nonzero(baseline_mask) < 2:
        raise ValueError("baseline interval contains too few samples")
    baseline_mean = np.mean(x[:, :, baseline_mask], axis=2, keepdims=True)
    centred = x - baseline_mean
    native_step = float(np.median(np.diff(times)))
    stride = max(1, round(sample_step_seconds / native_step))
    sampled = centred[:, :, ::stride]
    rank = min(rank, sampled.shape[1], sampled.shape[0] * sampled.shape[2])
    pca = PCA(n_components=rank, svd_solver="full", random_state=random_state)
    flat = sampled.transpose(0, 2, 1).reshape(-1, sampled.shape[1])
    transformed = pca.fit_transform(flat).reshape(sampled.shape[0], sampled.shape[2], rank)
    baseline_sampled = baseline_mask[::stride]
    baseline_centroid = transformed[:, baseline_sampled].mean(axis=(0, 1))
    return PerturbationalTrajectory(
        mean_trajectory=transformed.mean(axis=0),
        trial_trajectories=transformed,
        explained_variance_ratio=pca.explained_variance_ratio_,
        baseline_centroid=baseline_centroid,
    )


def trajectory_outcomes(
    trajectory: PerturbationalTrajectory,
    times_seconds: np.ndarray,
    *,
    post_interval: tuple[float, float] = (0.015, 1.0),
) -> dict[str, float]:
    """Compute displacement, occupied volume, differentiation, and recovery."""

    mean = trajectory.mean_trajectory
    times = np.asarray(times_seconds)
    if times.size != mean.shape[0]:
        # Caller may provide native times; deterministically interpolate its endpoints.
        times = np.linspace(times[0], times[-1], mean.shape[0])
    post = (times >= post_interval[0]) & (times <= post_interval[1])
    if np.count_nonzero(post) < 2:
        raise ValueError("post-pulse interval contains too few samples")
    displacement = np.linalg.norm(mean - trajectory.baseline_centroid, axis=1)
    post_states = trajectory.trial_trajectories[:, post].reshape(-1, mean.shape[1])
    covariance = np.cov(post_states, rowvar=False)
    eigenvalues = np.clip(np.linalg.eigvalsh(np.atleast_2d(covariance)), 0.0, None)
    occupied_log_volume = float(np.sum(np.log(eigenvalues[eigenvalues > 0] + 1e-12)))
    trial_distances = np.linalg.norm(
        trajectory.trial_trajectories[:, post] - mean[post][None, :, :], axis=2
    )
    peak_index = np.flatnonzero(post)[int(np.argmax(displacement[post]))]
    peak = float(displacement[peak_index])
    half = peak / 2.0
    after = np.flatnonzero((np.arange(displacement.size) > peak_index) & (displacement <= half))
    recovery_half_time = float(times[after[0]] - times[peak_index]) if after.size else float("nan")
    return {
        "maximum_displacement": peak,
        "occupied_log_volume": occupied_log_volume,
        "spatial_differentiation": float(np.mean(trial_distances)),
        "recovery_half_time_seconds": recovery_half_time,
    }
