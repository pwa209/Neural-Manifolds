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


@dataclass(frozen=True)
class EarlyPostPulseBurden:
    """Trial-by-channel residual burden after the interpolated pulse gap."""

    baseline_rms_uv: np.ndarray
    early_rms_uv: np.ndarray
    early_to_baseline_rms_ratio: np.ndarray
    early_derivative_rms_uv_per_second: np.ndarray


def _validate_epoch_grid(
    epochs: np.ndarray,
    times_seconds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(epochs, dtype=np.float64)
    times = np.asarray(times_seconds, dtype=np.float64)
    if values.ndim != 3 or values.shape[2] != times.size or values.shape[0] < 1:
        raise ValueError("epochs must be non-empty trials x channels x time")
    if times.ndim != 1 or times.size < 3 or np.any(np.diff(times) <= 0):
        raise ValueError("epoch times must be a strictly increasing one-dimensional grid")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(times)):
        raise ValueError("TMS epochs and time grid must be finite")
    return values, times


def early_post_pulse_burden(
    epochs: np.ndarray,
    times_seconds: np.ndarray,
    *,
    baseline: tuple[float, float] = (-0.5, -0.05),
    early_interval: tuple[float, float] = (0.020, 0.050),
) -> EarlyPostPulseBurden:
    """Quantify residual early EEG amplitude/derivative burden without rejection.

    The interval begins at the end of the declared interpolation gap.  The
    derivative term is deliberately described as a muscle-or-artifact burden,
    not as proof of a physiological muscle source.  Values are returned for
    every epoched trial and channel so later rejection remains auditable.
    """

    values, times = _validate_epoch_grid(epochs, times_seconds)
    baseline_mask = (times >= baseline[0]) & (times <= baseline[1])
    early_mask = (times >= early_interval[0]) & (times <= early_interval[1])
    if np.count_nonzero(baseline_mask) < 2:
        raise ValueError("TMS burden baseline contains fewer than two samples")
    if np.count_nonzero(early_mask) < 2:
        raise ValueError("early post-pulse burden interval contains fewer than two samples")
    baseline_mean = np.mean(values[:, :, baseline_mask], axis=2, keepdims=True)
    centred = values - baseline_mean
    baseline_rms = np.sqrt(np.mean(np.square(centred[:, :, baseline_mask]), axis=2))
    early_rms = np.sqrt(np.mean(np.square(centred[:, :, early_mask]), axis=2))
    scale_floor = np.finfo(np.float64).eps
    ratio = early_rms / np.maximum(baseline_rms, scale_floor)
    derivative = np.diff(centred, axis=2) / np.diff(times)[None, None, :]
    derivative_times = (times[:-1] + times[1:]) / 2.0
    derivative_mask = (derivative_times >= early_interval[0]) & (
        derivative_times <= early_interval[1]
    )
    if np.count_nonzero(derivative_mask) < 1:
        raise ValueError("early post-pulse derivative interval contains no samples")
    early_derivative_rms = np.sqrt(np.mean(np.square(derivative[:, :, derivative_mask]), axis=2))
    return EarlyPostPulseBurden(
        baseline_rms_uv=baseline_rms * 1e6,
        early_rms_uv=early_rms * 1e6,
        early_to_baseline_rms_ratio=ratio,
        early_derivative_rms_uv_per_second=early_derivative_rms * 1e6,
    )


def conventional_tms_eeg_outcomes(
    epochs: np.ndarray,
    times_seconds: np.ndarray,
    *,
    baseline: tuple[float, float] = (-0.5, -0.05),
    post_interval: tuple[float, float] = (0.020, 0.300),
    activation_threshold_mad: float = 3.0,
) -> dict[str, float | str]:
    """Compute deterministic TEP/GFP and sensor-spread comparators.

    Global field power is the across-sensor population standard deviation of
    the trial-averaged, baseline-corrected TEP.  Sensor activation uses a
    channel-specific median + MAD threshold estimated only from the baseline.
    Latency outcomes describe sensor-level temporal spread and are not claimed
    as source-space or causal propagation.
    """

    values, times = _validate_epoch_grid(epochs, times_seconds)
    if activation_threshold_mad <= 0:
        raise ValueError("sensor activation threshold multiplier must be positive")
    baseline_mask = (times >= baseline[0]) & (times <= baseline[1])
    post_mask = (times >= post_interval[0]) & (times <= post_interval[1])
    if np.count_nonzero(baseline_mask) < 2 or np.count_nonzero(post_mask) < 2:
        raise ValueError("TEP baseline and post-pulse intervals each need at least two samples")
    baseline_mean = np.mean(values[:, :, baseline_mask], axis=2, keepdims=True)
    centred = values - baseline_mean
    mean_tep = np.mean(centred, axis=0)
    global_field_power = np.std(mean_tep, axis=0, ddof=0)
    post_times = times[post_mask]
    post_gfp = global_field_power[post_mask]

    baseline_absolute = np.abs(centred[:, :, baseline_mask])
    baseline_median = np.median(baseline_absolute, axis=(0, 2))
    baseline_mad = 1.4826 * np.median(
        np.abs(baseline_absolute - baseline_median[None, :, None]), axis=(0, 2)
    )
    baseline_fallback = np.std(baseline_absolute, axis=(0, 2), ddof=0)
    baseline_scale = np.where(
        baseline_mad > np.finfo(np.float64).eps,
        baseline_mad,
        baseline_fallback,
    )
    thresholds = baseline_median + activation_threshold_mad * np.maximum(
        baseline_scale, np.finfo(np.float64).eps
    )
    post_absolute_tep = np.abs(mean_tep[:, post_mask])
    crossings = post_absolute_tep > thresholds[:, None]
    active = np.any(crossings, axis=1)
    first_latency = np.full(mean_tep.shape[0], np.nan, dtype=np.float64)
    for channel_index in np.flatnonzero(active):
        first_latency[channel_index] = post_times[np.flatnonzero(crossings[channel_index])[0]]
    available_latencies = first_latency[np.isfinite(first_latency)]
    if available_latencies.size >= 2:
        latency_range = float(np.ptp(available_latencies))
        latency_iqr = float(
            np.percentile(available_latencies, 75) - np.percentile(available_latencies, 25)
        )
        propagation_status = "available_sensor_level_temporal_spread"
    else:
        latency_range = float("nan")
        latency_iqr = float("nan")
        propagation_status = "unavailable_fewer_than_two_threshold_crossing_sensors"
    peak_index = int(np.argmax(post_gfp))
    return {
        "tep_peak_global_field_power_uv": float(post_gfp[peak_index] * 1e6),
        "tep_peak_global_field_power_latency_seconds": float(post_times[peak_index]),
        "tep_mean_global_field_power_uv": float(np.mean(post_gfp) * 1e6),
        "tep_global_field_power_auc_uv_seconds": float(np.trapezoid(post_gfp, post_times) * 1e6),
        "tep_mean_absolute_amplitude_uv": float(np.mean(post_absolute_tep) * 1e6),
        "sensor_spread_fraction": float(np.mean(active)),
        "sensor_propagation_latency_range_seconds": latency_range,
        "sensor_propagation_latency_iqr_seconds": latency_iqr,
        "sensor_propagation_status": propagation_status,
    }


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
