import numpy as np
import pandas as pd
import pytest
import yaml

from neural_manifolds.revised.cohort import build_cohort, link_record, normalized_channel
from neural_manifolds.revised.inference import (
    CONVENTIONAL,
    FoldDynamics,
    nested_transfer,
    summarize_predictions,
)
from neural_manifolds.revised.measurement import sensor_trajectory, window_qc
from neural_manifolds.revised.recovery import as_record, simulate


def policy():
    with open("configs/revised_execution.yaml") as stream:
        return yaml.safe_load(stream)


def test_exact_source_record_link_not_basename():
    record = {"Filename": "a/b.edf", "records_origin": "source.zip::folder/Records.csv"}
    item = {"member": "folder/Data/PSG/a/b.edf"}
    assert link_record(record, [item]) == item
    with pytest.raises(ValueError):
        link_record(record, [{"member": "other/b.edf"}])
    with pytest.raises(ValueError):
        link_record(record, [item, item])


def test_channel_normalization_preserves_bipolar_derivation():
    assert normalized_channel("EEG C3-Ref") == "C3"
    assert normalized_channel("Fp1-A2") == "FP1"
    assert normalized_channel("EEG Fp1-O1") == "FP1-O1"


def test_cohort_preserves_dewr_and_groups_related_studies():
    row = {
        "Filename": "a.edf",
        "records_origin": "source.zip::Records.csv",
        "Case ID": "1",
        "Duration": "60",
        "Experience": "1",
        "Last sleep stage": "2",
        "Subject healthy": "1",
        "Subject ID": "1",
        "Treatment group": "",
    }
    audit = {
        "release": "/synthetic",
        "completion_marker_sha256": "test",
        "records": [row],
        "files": [
            {"member": "Data/PSG/a.edf", "duration_seconds": 60, "channels": ["Fp1", "C3", "O1"]}
        ],
    }
    result = build_cohort(
        {"dream_set_17": audit, "dream_set_18": audit, "unreviewed": audit}, policy()
    )
    assert len(result["units"]) == 2
    assert len({u["study_group"] for u in result["units"]}) == 1
    assert all(u["experience"] is None and not u["primary_eligible"] for u in result["units"])
    assert result["exclusions"][0]["reason"] == "source_semantics_or_timing_not_adjudicated"


def test_window_qc_retains_original_second_indices():
    rng = np.random.default_rng(42)
    x = rng.normal(scale=1e-5, size=(3, 4000))
    x[:, 1000:1200] = 0
    windows, reasons = window_qc(x, 200, policy())
    assert windows.shape == (20, 3, 200)
    assert reasons[5] == "flat"
    assert sensor_trajectory(windows[np.array([not r for r in reasons])]).shape == (19, 3)


def synthetic_frame():
    rng = np.random.default_rng(1)
    rows, arrays = [], {}
    for study in range(3):
        for person in range(4):
            for label in (0, 1):
                unit = f"{study}-{person}-{label}"
                rows.append(
                    {
                        "unit_id": unit,
                        "participant_id": f"{study}-{person}",
                        "study_group": str(study),
                        "experience": label,
                        "conventional": dict(
                            zip(CONVENTIONAL, rng.normal(size=len(CONVENTIONAL)), strict=True)
                        ),
                    }
                )
                arrays[unit] = as_record(simulate(20, rng, coupling=label * 0.2))
    return pd.DataFrame(rows), arrays


def test_fold_transform_does_not_fit_held_out_arrays():
    frame, arrays = synthetic_frame()
    train = frame[frame.study_group != "0"]
    first = FoldDynamics().fit(train, arrays)
    altered = dict(arrays)
    for unit in frame[frame.study_group == "0"].unit_id:
        altered[unit] = as_record(np.ones((20, 3)) * 1e9)
    second = FoldDynamics().fit(train, altered)
    np.testing.assert_array_equal(first.center_, second.center_)
    np.testing.assert_array_equal(
        first.dictionary_.cluster_centers_, second.dictionary_.cluster_centers_
    )
    assert set(first.training_ids_).isdisjoint(frame[frame.study_group == "0"].unit_id)


def test_nested_transfer_real_predictions_and_paired_summary():
    frame, arrays = synthetic_frame()
    result = nested_transfer(
        frame,
        arrays,
        families=["spectral_arousal", "shared_dynamics"],
        strengths=(1.0,),
        dimensions=(1,),
    )
    p = pd.DataFrame(result["predictions"])
    assert len(p) == len(frame) * 2
    for row in result["tuning"]:
        held_ids = set(frame[frame.study_group == row["held_group"]].unit_id)
        assert held_ids.isdisjoint(row["training_ids"])
    summary = summarize_predictions(p, repetitions=20)
    assert summary["paired_comparisons"][0]["paired_units"] == len(frame)
    assert summary["paired_comparisons"][0]["few_studies_warning"] is True
