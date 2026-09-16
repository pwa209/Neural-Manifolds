import json

import numpy as np
import pandas as pd
from scipy.io import savemat

from neural_manifolds.revised.boundary import fieldtrip, paired_context_summary
from neural_manifolds.revised.controls import exact_electrode_records, null_arrays
from neural_manifolds.revised.recovery import as_record
from neural_manifolds.revised.robustness import perturb
from neural_manifolds.revised.synthesis import synthesize


def test_signal_phase_preserves_per_window_channel_spectrum():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(20, 3, 200))
    y, names = perturb(x, ["Fp1", "C3", "O1"], "independent_phase", rng)
    np.testing.assert_allclose(
        abs(np.fft.rfft(x, axis=-1)), abs(np.fft.rfft(y, axis=-1)), atol=1e-12
    )
    z, kept = perturb(x, names, "channel_dropout", rng)
    assert z.shape == (20, 2, 200) and kept == ["Fp1", "C3"]


def test_exact_electrode_sensitivity_uses_preprocessing_provenance():
    rows = [
        {"primary_eligible": True, "preprocessing": {"position_equivalence": v}}
        for v in ("native_named_channel", "approximate_not_identical_to_10_20", None)
    ]
    assert exact_electrode_records(rows) == [rows[0]]


def test_temporal_null_never_moves_samples_between_segments():
    x = np.arange(60).reshape(20, 3).astype(float)
    record = as_record(x, np.repeat([0, 1], 10))
    changed = null_arrays({"u": record}, "temporal_permutation", np.random.default_rng(4))["u"]
    for start in [0, 10]:
        assert set(changed["trajectory"][start : start + 10, 0]) == set(x[start : start + 10, 0])
    np.testing.assert_array_equal(changed["segments"], record["segments"])


def test_official_fieldtrip_structure_requires_axes(tmp_path):
    path = tmp_path / "test.mat"
    savemat(
        path,
        {
            "data": {
                "label": np.array(["Fp1", "C3", "O1"], dtype=object),
                "fsample": 200.0,
                "trial": np.ones((3, 4000)),
                "time": np.arange(4000) / 200,
            }
        },
    )
    names, rate, trials, times = fieldtrip(path)
    assert names == ["Fp1", "C3", "O1"] and rate == 200
    assert trials[0].shape == (3, 4000) and len(times[0]) == 4000


def test_boundary_pairing_and_context_difference():
    rows = [
        dict(
            participant_id=str(person),
            context=context,
            session=session,
            feature="lz",
            value=person + int(session) * (2 if context == "music" else 1),
        )
        for person in range(4)
        for context in ["rest", "music"]
        for session in ["01", "02"]
    ]
    result = paired_context_summary(pd.DataFrame(rows), {"seed": 1, "bootstrap_repetitions": 20})
    assert result["context_interactions"][0]["difference_in_change"] == 1


def test_synthesis_never_calls_partial_inputs_complete(tmp_path):
    source = tmp_path / "transfer.json"
    source.write_text(
        json.dumps(
            {
                "analyses": {
                    "synthetic_test": {
                        "summary": {
                            "paired_comparisons": [
                                {
                                    "baseline": "synthetic baseline",
                                    "dynamic_minus_baseline_log_loss": -0.1,
                                    "interval_95": [-0.2, 0.05],
                                }
                            ]
                        }
                    }
                }
            }
        )
    )
    paths = synthesize([source], tmp_path / "output", expected_controls=2)
    assert len([p for p in paths if p.suffix == ".png"]) == 1
    result = json.loads(paths[0].read_text())
    assert result["full_study_complete"] is False
    assert result["null_replicates_completed"] == 0
