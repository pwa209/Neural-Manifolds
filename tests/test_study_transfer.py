import numpy as np
import pandas as pd
import pytest

from neural_manifolds.statistics.study_transfer import (
    Candidate,
    compare_studies,
    independent_weights,
    predict_tms,
)


def fixture_frame():
    rng = np.random.default_rng(42)
    rows = []
    for study in range(4):
        for person in range(4):
            for y in (0, 1):
                rows.append(
                    dict(
                        unit_id=f"{study}-{person}-{y}",
                        study_group=str(study),
                        participant_id=f"{study}-{person}",
                        experience=y,
                        spectrum=rng.normal(),
                        dynamic=2 * y + rng.normal(scale=0.2),
                    )
                )
    return pd.DataFrame(rows)


def test_outer_test_features_do_not_change_training_or_tuning():
    frame = fixture_frame()
    candidates = [Candidate("shared", ("spectrum", "dynamic"))]
    predictions, audit = compare_studies(frame, candidates)
    changed = frame.copy()
    changed.loc[changed.study_group == "0", "dynamic"] += 1000
    _, second = compare_studies(changed, candidates)
    a = audit[audit.held_study == "0"].iloc[0]
    b = second[second.held_study == "0"].iloc[0]
    assert a.inner_scores == b.inner_scores
    assert a.selected_strength == b.selected_strength
    assert len(predictions) == len(frame)
    assert "0" not in a.training_studies


def test_participant_overlap_is_rejected():
    frame = fixture_frame()
    frame.loc[8, "participant_id"] = frame.loc[0, "participant_id"]
    with pytest.raises(ValueError, match="Overlapping"):
        compare_studies(frame, [Candidate("x", ("spectrum",))])


def test_participant_weight_not_window_count():
    frame = fixture_frame()
    extra = frame.iloc[[0] * 50].copy()
    frame = pd.concat([frame, extra])
    frame["weight"] = independent_weights(frame)
    sums = frame.groupby(["study_group", "participant_id"]).weight.sum()
    assert np.allclose(sums, sums.iloc[0])


def test_tms_condition_adjusted_increment_has_held_out_predictions():
    frame = fixture_frame()
    frame["condition"] = frame.experience
    frame["tms"] = 2 * frame.dynamic + frame.spectrum
    result = predict_tms(
        frame, baseline=("condition", "spectrum"), dynamics=("dynamic",), outcome="tms"
    )
    assert len(result) == 2 * len(frame)
    assert set(result.model) == {"condition_conventional", "plus_dynamics"}
    assert np.isfinite(result.prediction).all()
