from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from neural_manifolds.stages.fmri_statistics import (
    AXES,
    FMRIInferenceError,
    collapse_runs_within_participant_condition,
    infer_paired_condition_contrasts,
)


def _condition_cell(
    participant: str,
    *,
    partition: str,
    inference_condition: str,
    value: float,
    wake: bool = False,
    concentration: float = 1.0,
) -> dict[str, Any]:
    timing_verified = inference_condition in {
        "post_lor_unresponsive",
        "post_ror_responsive",
    }
    return {
        "participant_id": participant,
        "dataset_id": "propofol_fmri",
        "partition": partition,
        "condition": inference_condition,
        "inference_condition": inference_condition,
        "metadata_status": "verified",
        "healthy_wake_reference": wake,
        "effect_site_concentration_max": concentration,
        "timing_label_verified": timing_verified,
        "timing_label_source": (
            "official_lor_ror_table_with_explicit_index_origin" if timing_verified else None
        ),
        **{axis: value + index * 0.1 for index, axis in enumerate(AXES)},
    }


def test_run_collapse_precedes_participant_condition_inference() -> None:
    rows = []
    for run_id, value in (("1", 1.0), ("2", 3.0)):
        rows.append(
            {
                "unit_id": f"wake-{run_id}",
                "participant_id": "sub-01",
                "dataset_id": "propofol_fmri",
                "partition": "validation",
                "condition": "no_propofol_rest",
                "task": "rest",
                "run_id": run_id,
                "metadata_status": "verified",
                "healthy_wake_reference": True,
                "effect_site_concentration_min": 0.0,
                "effect_site_concentration_mean": 0.0,
                "effect_site_concentration_max": 0.0,
                "metric": value,
            }
        )
    collapsed = collapse_runs_within_participant_condition(
        pd.DataFrame(rows), metric_columns=["metric"]
    )

    assert len(collapsed) == 1
    assert collapsed.loc[0, "metric"] == 2.0
    assert collapsed.loc[0, "run_count"] == 2
    assert collapsed.loc[0, "unit_count"] == 2
    assert collapsed.loc[0, "inference_condition"] == "no_propofol_rest"


def test_explicit_lor_ror_boundaries_define_pairing_conditions() -> None:
    base = {
        "participant_id": "sub-01",
        "dataset_id": "propofol_fmri",
        "partition": "validation",
        "task": "imagery",
        "metadata_status": "verified",
        "healthy_wake_reference": False,
        "timing_index_origin": 0,
        "effect_site_concentration_min": 1.0,
        "effect_site_concentration_mean": 1.0,
        "effect_site_concentration_max": 1.0,
    }
    rows = [
        {
            **base,
            "unit_id": "post-lor",
            "run_id": "2",
            "condition": "behaviorally_unresponsive",
            "volume_start": 40,
            "volume_stop": 80,
            "lor_volume": 40,
            "lor_tr_csv": 40,
            "metric": 2.0,
        },
        {
            **base,
            "unit_id": "post-ror",
            "run_id": "3",
            "condition": "responsive_recovery",
            "volume_start": 60,
            "volume_stop": 100,
            "ror_volume": 60,
            "ror_tr_csv": 60,
            "metric": 1.0,
        },
    ]
    collapsed = collapse_runs_within_participant_condition(
        pd.DataFrame(rows), metric_columns=["metric"]
    )
    assert set(collapsed["inference_condition"]) == {
        "post_lor_unresponsive",
        "post_ror_responsive",
    }
    assert collapsed["timing_label_verified"].all()
    assert set(collapsed["timing_label_source"]) == {
        "official_lor_ror_table_with_explicit_index_origin"
    }

    broken = pd.DataFrame(rows)
    broken.loc[0, "volume_start"] = 39
    with pytest.raises(FMRIInferenceError, match="does not start at the explicit LOR"):
        collapse_runs_within_participant_condition(broken, metric_columns=["metric"])


def _paired_cells(participants: int = 4) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for index in range(participants):
        participant = f"sub-{index + 1:02d}"
        baseline = float(index) / 10.0
        rows.extend(
            [
                _condition_cell(
                    participant,
                    partition="validation",
                    inference_condition="no_propofol_rest",
                    value=baseline,
                    wake=True,
                    concentration=0.0,
                ),
                _condition_cell(
                    participant,
                    partition="validation",
                    inference_condition="responsive_induction",
                    value=baseline + 0.5,
                    concentration=1.0,
                ),
                _condition_cell(
                    participant,
                    partition="validation",
                    inference_condition="post_lor_unresponsive",
                    value=baseline + 0.8,
                    concentration=1.0,
                ),
                _condition_cell(
                    participant,
                    partition="validation",
                    inference_condition="post_ror_responsive",
                    value=baseline + 0.2,
                    concentration=0.5,
                ),
            ]
        )
    return pd.DataFrame(rows)


def test_pairing_bootstrap_and_plus_one_permutation_are_participant_level() -> None:
    result = infer_paired_condition_contrasts(
        _paired_cells(),
        bootstrap_repetitions=250,
        permutation_repetitions=300,
        random_seed=17,
    )
    validation = result.estimates[
        result.estimates["partition"].eq("validation") & result.estimates["status"].eq("available")
    ]
    assert len(validation) == 8
    assert set(validation["paired_participants"]) == {4}
    assert validation["participant_mean_difference"].notna().all()
    assert validation["bootstrap_interval_low"].notna().all()
    assert validation["bootstrap_interval_high"].notna().all()
    assert validation["permutation_pvalue_two_sided_plus_one"].between(0, 1).all()
    for row in validation.to_dict(orient="records"):
        expected = (row["permutation_extreme_count"] + 1) / (row["permutation_repetitions"] + 1)
        assert row["permutation_pvalue_two_sided_plus_one"] == expected
    assert result.ledger["permutation"]["scheme"].startswith("within_participant")
    assert result.ledger["pooled_cross_partition_inference"] is False
    assert result.ledger["cross_modal_evidence"]["relationship"] == (
        "independent_cohort_triangulation"
    )
    assert (
        result.ledger["cross_modal_evidence"]["participant_level_eeg_fmri_correlation_performed"]
        is False
    )


def test_resampling_is_deterministic_and_row_order_invariant() -> None:
    cells = _paired_cells()
    first = infer_paired_condition_contrasts(
        cells,
        bootstrap_repetitions=100,
        permutation_repetitions=120,
        random_seed=23,
    )
    second = infer_paired_condition_contrasts(
        cells.sample(frac=1.0, random_state=999).reset_index(drop=True),
        bootstrap_repetitions=100,
        permutation_repetitions=120,
        random_seed=23,
    )
    pd.testing.assert_frame_equal(first.paired_differences, second.paired_differences)
    pd.testing.assert_frame_equal(first.estimates, second.estimates)
    assert first.ledger == second.ledger


def test_insufficient_pairing_is_audited_without_window_substitution() -> None:
    result = infer_paired_condition_contrasts(
        _paired_cells(participants=1),
        bootstrap_repetitions=50,
        permutation_repetitions=60,
        random_seed=3,
    )
    validation = result.estimates[result.estimates["partition"].eq("validation")]
    assert set(validation["status"]) == {"insufficient_pairs"}
    assert validation["permutation_pvalue_two_sided_plus_one"].isna().all()
    issue = next(
        item
        for item in result.ledger["issues"]
        if item["partition"] == "validation" and item["contrast_id"] == "propofol_vs_wake"
    )
    assert issue["observed_pairs"] == 1
    assert issue["technical_gate"] is False
    assert issue["scientific_gate"] is False
    assert "no run- or window-level substitution" in issue["message"]


def test_participant_cannot_cross_inference_partitions() -> None:
    cells = _paired_cells(participants=2)
    cells.loc[cells.index[0], "partition"] = "test"
    with pytest.raises(FMRIInferenceError, match="cannot pool participants across splits"):
        infer_paired_condition_contrasts(
            cells,
            bootstrap_repetitions=10,
            permutation_repetitions=10,
            random_seed=1,
        )
