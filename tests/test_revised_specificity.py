import io

import numpy as np
import pandas as pd
import pytest
from scipy.io import savemat

from neural_manifolds.revised.perturbation import paired_error_summary
from neural_manifolds.revised.specificity import (
    causal_filter_audit,
    confidence_associations,
    fast_features,
    matched_contrasts,
    read_official_osf_matrix,
)


def test_fast_track_no_future_samples_and_scale_invariance():
    rng = np.random.default_rng(3)
    times = np.arange(600) / 500 - 0.4
    x = rng.normal(size=(5, 600))
    first, audit = fast_features(x, times, 500, response_seconds=0.15)
    scaled, _ = fast_features(x * 1e6, times, 500, response_seconds=0.15)
    assert audit["future_sample_support"] == 0
    assert first[1]["pre_response"]
    assert not first[-1]["pre_response"]
    np.testing.assert_allclose(
        [v["normalized_rms"] for v in first], [v["normalized_rms"] for v in scaled]
    )
    altered = x.copy()
    altered[:, times >= 0.3] *= 10
    changed, _ = fast_features(altered, times, 500)
    assert changed[1]["normalized_rms"] == first[1]["normalized_rms"]
    assert causal_filter_audit(500)[1]["peak_delay_seconds"] > 0


def test_mat_requires_unique_official_matrix():
    stream = io.BytesIO()
    savemat(stream, {"eeg": np.ones((110, 600, 2)), "metadata": np.ones(4)})
    name, data = read_official_osf_matrix(stream.getvalue())
    assert name == "eeg" and data.shape == (110, 600, 2)
    stream = io.BytesIO()
    savemat(stream, {"eeg": np.ones((110, 600, 2)), "second": np.ones((110, 600, 3))})
    with pytest.raises(ValueError, match="not_unique"):
        read_official_osf_matrix(stream.getvalue())


def test_intensity_matching_uses_people_not_trials():
    rows = []
    for person in range(3):
        for condition, value in [("yes", 2), ("no", 1)]:
            for intensity in [1, 2]:
                for repeat in range(1 + person):
                    rows.append(
                        dict(
                            participant_id=str(person),
                            condition=condition,
                            intensity=intensity,
                            interval="early",
                            confidence=repeat % 2,
                            normalized_rms=value,
                            spatial_participation=value,
                            normalized_roughness=value,
                        )
                    )
    frame = pd.DataFrame(rows)
    results = matched_contrasts(frame, contrasts=[("yes-no", "yes", "no")], repetitions=20)
    assert all(r["participants"] == 3 and r["difference"] == 1 for r in results)
    assert all(0 <= r["p_holm"] <= 1 for r in results)
    assert confidence_associations(frame, repetitions=20)


def test_tms_change_diagnostics_preserve_pairing():
    rows = []
    for person in range(5):
        for condition, shift in [("awake", 0), ("propofol_sedation", 2)]:
            for model, error in [("condition_conventional", 1), ("plus_dynamics", 0.5)]:
                rows.append(
                    dict(
                        unit_id=f"{person}-{condition}",
                        participant_id=str(person),
                        condition=condition,
                        model=model,
                        observed=person + shift,
                        predicted=person + shift + error,
                    )
                )
    result = paired_error_summary(rows, repetitions=20)
    assert result["participants"] == 5
    assert result["incremental_squared_error"] == -0.75
    assert len(result["within_participant_changes"]) == 2
    assert all(r["change_mse"] == 0 for r in result["within_participant_changes"])
