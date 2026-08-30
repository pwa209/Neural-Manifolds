from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from neural_manifolds.preprocessing.tms import (
    conventional_tms_eeg_outcomes,
    early_post_pulse_burden,
)
from neural_manifolds.stages.tms import (
    _auxiliary_channel_inventory,
    _epoch_drop_reasons,
    _epoch_selection,
)


def _time_grid() -> np.ndarray:
    return np.linspace(-0.5, 1.0, 301, dtype=np.float64)


def test_early_post_pulse_burden_is_trial_and_channel_resolved() -> None:
    times = _time_grid()
    rng = np.random.default_rng(20260830)
    epochs = rng.normal(scale=0.1e-6, size=(4, 3, len(times)))
    early = (times >= 0.015) & (times <= 0.050)
    epochs[2, 1, early] += np.linspace(0.0, 20e-6, np.count_nonzero(early))

    burden = early_post_pulse_burden(epochs, times)

    assert burden.early_rms_uv.shape == (4, 3)
    assert burden.early_to_baseline_rms_ratio.shape == (4, 3)
    assert burden.early_derivative_rms_uv_per_second.shape == (4, 3)
    assert burden.early_rms_uv[2, 1] > 10 * np.median(burden.early_rms_uv)
    assert burden.early_derivative_rms_uv_per_second[2, 1] > 0


def test_conventional_tms_outcomes_include_gfp_spread_and_sensor_latency() -> None:
    times = _time_grid()
    epochs = np.zeros((5, 3, len(times)), dtype=np.float64)
    for channel, (onset, amplitude) in enumerate(((0.025, 4e-6), (0.040, 7e-6), (0.060, 10e-6))):
        response = (times >= onset) & (times <= onset + 0.020)
        epochs[:, channel, response] = amplitude

    first = conventional_tms_eeg_outcomes(epochs, times)
    second = conventional_tms_eeg_outcomes(epochs, times)

    assert first == second
    assert first["tep_peak_global_field_power_uv"] > 0
    assert first["tep_global_field_power_auc_uv_seconds"] > 0
    assert first["sensor_spread_fraction"] == pytest.approx(1.0)
    assert first["sensor_propagation_latency_range_seconds"] > 0
    assert first["sensor_propagation_latency_iqr_seconds"] > 0
    assert first["sensor_propagation_status"] == ("available_sensor_level_temporal_spread")


def test_auxiliary_and_epoch_selection_statuses_fail_closed() -> None:
    raw = SimpleNamespace(
        ch_names=["Cz", "VEOG", "EMG1", "ECG"],
        get_channel_types=lambda: ["eeg", "eog", "emg", "ecg"],
    )
    inventory = _auxiliary_channel_inventory(raw)
    assert inventory["auxiliary_channel_status"] == "available_auxiliary_channels_present"
    assert inventory["ica_auxiliary_support_status"] == "available_eog_or_ecg_reference"
    assert json.loads(inventory["auxiliary_channel_inventory_json"]) == {
        "ecg": ["ECG"],
        "emg": ["EMG1"],
        "eog": ["VEOG"],
    }

    class Epochs:
        selection = np.asarray([0, 2])
        drop_log = ((), ("BAD boundary",), ())

        def __len__(self) -> int:
            return 2

    assert _epoch_selection(Epochs(), event_count=3).tolist() == [0, 2]
    assert _epoch_drop_reasons(Epochs(), event_count=3) == [
        "retained_by_epoch_constructor",
        "BAD boundary",
        "retained_by_epoch_constructor",
    ]

    class DuplicateEpochs:
        selection = np.asarray([1, 1])

        def __len__(self) -> int:
            return 2

    with pytest.raises(ValueError, match="invalid or duplicate"):
        _epoch_selection(DuplicateEpochs(), event_count=3)
