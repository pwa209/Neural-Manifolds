import numpy as np
import pytest

from neural_manifolds.stages.tms import audited_pulse_events


def test_recoded_events_require_sample_matched_authoritative_sidecar(tmp_path):
    source = tmp_path / "sub-1_acq-tms_eeg.vhdr"
    source.write_text("header")
    sidecar = tmp_path / "sub-1_acq-tms_events.tsv"
    events = np.array([[0, 0, 1], [100, 0, 2], [200, 0, 2]])
    with pytest.raises(ValueError, match="no audited"):
        audited_pulse_events(source, events, {"Stimulus/S 2": 2}, 0, 100)
    sidecar.write_text("onset\tsample\ttrial_type\n1\t100\tResponse/R128\n2\t200\tResponse/R128\n")
    selected, authority = audited_pulse_events(source, events, {"Stimulus/S 2": 2}, 0, 100)
    np.testing.assert_array_equal(selected, events[1:])
    assert authority["pulse_events_sidecar_sha256"]
    with pytest.raises(ValueError, match="do not match"):
        audited_pulse_events(source, events[:2], {"Stimulus/S 2": 2}, 0, 100)
    with pytest.raises(ValueError, match="onset/sample"):
        audited_pulse_events(source, events, {"Stimulus/S 2": 2}, 0, 200)


def test_sustained_flatline_distinguishes_quantization_from_dead_channel():
    from neural_manifolds.preprocessing.eeg import detect_bad_channels

    rng = np.random.default_rng(17)
    data = np.repeat(rng.normal(0, 1e-5, (4, 2000)), 5, axis=1)
    data[3] = 0
    result = detect_bad_channels(data, 5000, flat_minimum_duration_seconds=0.1)
    np.testing.assert_allclose(result.flat_fraction[:3], 0)
    assert result.flat_fraction[3] == 1
    assert 3 in result.bad_indices
