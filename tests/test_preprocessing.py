import numpy as np
import pytest

from neural_manifolds.preprocessing.eeg import (
    canonicalize_channel_name,
    detect_artifact_windows,
    detect_bad_channels,
    make_windows,
)
from neural_manifolds.preprocessing.tms import (
    interpolate_continuous_pulses,
    interpolate_pulse_interval,
)


def test_channel_name_normalisation() -> None:
    assert canonicalize_channel_name("EEG FP1-REF") == "Fp1"
    assert canonicalize_channel_name("T3") == "T7"
    assert canonicalize_channel_name("EEG FZ") == "Fz"


def test_windows_are_deterministic() -> None:
    x = np.arange(2 * 1000).reshape(2, 1000)
    windows, starts = make_windows(x, 200, 2.0, 1.0)
    assert windows.shape == (4, 2, 400)
    assert starts.tolist() == [0, 200, 400, 600]


def test_flat_bad_channel_is_found() -> None:
    rng = np.random.default_rng(4)
    x = rng.normal(scale=1e-6, size=(8, 2000))
    x[3] = 0.0
    result = detect_bad_channels(x, 200)
    assert 3 in result.bad_indices


def test_extreme_window_is_rejected() -> None:
    rng = np.random.default_rng(7)
    windows = rng.normal(scale=1e-6, size=(30, 4, 400))
    windows[-1, 0, 200] = 0.5
    result = detect_artifact_windows(windows, 200)
    assert not result.keep[-1]


def test_tms_interpolation_removes_pulse() -> None:
    times = np.linspace(-0.1, 0.1, 401)
    base = np.sin(2 * np.pi * 10 * times)
    epochs = np.tile(base, (3, 2, 1))
    epochs[:, :, (times >= -0.005) & (times <= 0.015)] = 100.0
    clean = interpolate_pulse_interval(epochs, times)
    assert np.max(np.abs(clean)) < 2.0


def test_continuous_tms_interpolation_uses_clean_bilateral_support() -> None:
    sampling_hz = 1000.0
    times = np.arange(2000) / sampling_hz
    base = np.sin(2 * np.pi * 10 * times)
    data = np.stack([base, 0.5 * base])
    corrupted = data.copy()
    corrupted[:, 995:1016] = 100.0
    clean = interpolate_continuous_pulses(corrupted, np.array([1000]), sampling_hz)
    assert np.max(np.abs(clean[:, 995:1016])) < 2.0
    np.testing.assert_array_equal(clean[:, :975], corrupted[:, :975])
    np.testing.assert_array_equal(clean[:, 1036:], corrupted[:, 1036:])


def test_continuous_tms_interpolation_rejects_overlapping_pulses() -> None:
    with pytest.raises(ValueError, match="overlap"):
        interpolate_continuous_pulses(np.zeros((2, 2000)), np.array([1000, 1005]), 1000.0)
