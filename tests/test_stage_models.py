from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from neural_manifolds.config import load_study
from neural_manifolds.manifold.profile import AXIS_NAMES
from neural_manifolds.stages.models import (
    _axis_redundancy_diagnostics,
    _continuous_inference,
    _mixed_model_diagnostics,
    run_models,
)


def test_models_keep_participants_out_of_outer_folds(tmp_path: Path) -> None:
    rng = np.random.default_rng(4)
    rows = []
    for participant in range(12):
        for label, condition in enumerate(("low", "high")):
            values = rng.normal(size=5) + label * np.array([0.7, 0.4, 0.0, 0.2, 0.5])
            rows.append(
                {
                    "participant_id": f"sub-{participant:02d}",
                    "dataset_id": "synthetic",
                    "condition": condition,
                    "prediction_evaluation_eligible": participant >= 3,
                    **dict(zip(AXIS_NAMES, values, strict=True)),
                }
            )
    profile_frame = pd.DataFrame(rows)
    profiles = tmp_path / "profiles.parquet"
    profile_frame.to_parquet(profiles, index=False)
    matched_rows = []
    for row in rows:
        matched_rows.append(
            {
                "contrast_id": "high-versus-low",
                "participant_id": row["participant_id"],
                "dataset_id": row["dataset_id"],
                "contrast_arm": row["condition"],
                "successful_repeats": 10,
                **{f"{axis}_mean": row[axis] for axis in AXIS_NAMES},
            }
        )
    matched = tmp_path / "sampling-matched-profiles.parquet"
    pd.DataFrame(matched_rows).to_parquet(matched, index=False)
    contrasts = tmp_path / "contrasts.yaml"
    contrasts.write_text(
        """schema_version: 1
contrasts:
  - id: high-versus-low
    datasets: [synthetic]
    label_column: condition
    positive: high
    negative: low
    design: within_participant
""",
        encoding="utf-8",
    )
    outputs = run_models(
        profiles_path=profiles,
        matched_profiles_path=matched,
        contrasts_path=contrasts,
        output_root=tmp_path / "models",
        study=load_study(Path(__file__).parents[1] / "configs" / "study.yaml"),
        repetitions=99,
    )
    axes = pd.read_parquet(outputs[0])
    omnibus = pd.read_parquet(outputs[1])
    predictions = pd.read_parquet(outputs[2])
    assert set(axes["axis"]) == set(AXIS_NAMES)
    assert "five_axis" in set(predictions["model"])
    assert predictions["auroc"].between(0, 1).all()
    assert set(predictions["sampling_basis"]) == {
        "repeated_equal_window_profiles",
        "all_available_participant_condition_profiles",
    }
    primary = predictions[predictions["estimand_role"] == "primary"]
    sensitivity = predictions[predictions["estimand_role"] == "secondary_sensitivity"]
    assert set(primary["estimand_id"]) == {"repeated_equal_window_primary"}
    assert set(primary["sampling_basis"]) == {"repeated_equal_window_profiles"}
    assert set(sensitivity["estimand_id"]) == {"all_available_participant_condition_sensitivity"}
    assert set(sensitivity["sampling_basis"]) == {"all_available_participant_condition_profiles"}
    assert predictions["representation_heldout"].all()
    assert set(predictions["n_evaluation_participants"]) == {9}
    assert set(predictions["n_fixed_transform_participants"]) == {3}
    assert set(predictions["pretraining_overlap_status"]) == {"unresolved"}
    assert not predictions["zero_shot_verified"].any()
    assert set(axes["participant_bootstrap_unit"]) == {"participant"}
    assert set(axes["participant_bootstrap_repetitions"]) == {99}
    assert axes["participant_bootstrap_ci_low"].notna().all()
    assert axes["participant_bootstrap_ci_high"].notna().all()
    assert set(axes["equivalence_status"]).issubset({"equivalent", "not_equivalent"})
    assert set(axes["equivalence_smallest_effect_size"]) == {0.20}
    assert axes["mixed_model_status"].notna().all()
    assert set(axes["mixed_model_random_effect"]) == {"participant_intercept"}
    assert set(omnibus["axis_redundancy_status"]) == {"available"}
    assert set(predictions["participant_bootstrap_unit"]) == {"participant"}
    assert set(predictions["participant_bootstrap_repetitions"]) == {99}
    assert predictions["auroc_ci_low"].notna().all()
    assert predictions["auroc_ci_high"].notna().all()
    assert predictions["calibration_status"].isin({"available", "unavailable"}).all()
    ablations = predictions[predictions["model"].str.startswith("without_")]
    assert set(ablations["leave_one_property_out_status"]) == {"available"}
    assert ablations["delta_auroc_vs_five_axis"].notna().all()
    assert set(ablations["prediction_equivalence_status"]).issubset(
        {"equivalent", "not_equivalent"}
    )
    assert set(ablations["prediction_equivalence_smallest_auc_difference"]) == {0.05}
    assert ablations["prediction_equivalence_ci_low"].notna().all()
    assert ablations["prediction_equivalence_ci_high"].notna().all()

    audit = json.loads(outputs[-1].read_text(encoding="utf-8"))
    assert audit["completed_primary_inference"] == 1
    assert audit["completed_secondary_sensitivity_inference"] == 1
    assert audit["pretraining_overlap"]["zero_shot_verified"] is False
    assert audit["participant_bootstrap_unit"] == "participant"
    assert audit["participant_bootstrap_repetitions"] == 99
    assert audit["equivalence_smallest_effect_size"] == 0.20
    assert any(
        issue["component"] == "leave_one_dataset_out_prediction"
        and issue["status"] == "unavailable"
        for issue in audit["issues"]
    )


@pytest.mark.parametrize("matched_mode", ["missing", "invalid_schema", "zero_repeats"])
def test_missing_or_invalid_equal_window_profiles_never_become_primary(
    tmp_path: Path, matched_mode: str
) -> None:
    rng = np.random.default_rng(42)
    rows = []
    for participant in range(12):
        for label, condition in enumerate(("low", "high")):
            rows.append(
                {
                    "participant_id": f"sub-{participant:02d}",
                    "dataset_id": "synthetic",
                    "condition": condition,
                    "prediction_evaluation_eligible": participant >= 3,
                    **dict(
                        zip(
                            AXIS_NAMES,
                            rng.normal(size=5) + label * np.array([0.6, 0.3, 0.1, 0.2, 0.5]),
                            strict=True,
                        )
                    ),
                }
            )
    profiles = tmp_path / "profiles.parquet"
    pd.DataFrame(rows).to_parquet(profiles, index=False)
    matched: Path | None = None
    if matched_mode == "invalid_schema":
        matched = tmp_path / "invalid-matched.parquet"
        pd.DataFrame({"contrast_id": ["high-versus-low"]}).to_parquet(matched, index=False)
    elif matched_mode == "zero_repeats":
        matched = tmp_path / "zero-repeats.parquet"
        pd.DataFrame(
            [
                {
                    "contrast_id": "high-versus-low",
                    "participant_id": row["participant_id"],
                    "dataset_id": row["dataset_id"],
                    "contrast_arm": row["condition"],
                    "successful_repeats": 0,
                    **{f"{axis}_mean": row[axis] for axis in AXIS_NAMES},
                }
                for row in rows
            ]
        ).to_parquet(matched, index=False)
    contrasts = tmp_path / "contrasts.yaml"
    contrasts.write_text(
        """schema_version: 1
contrasts:
  - id: high-versus-low
    datasets: [synthetic]
    label_column: condition
    positive: high
    negative: low
    design: within_participant
""",
        encoding="utf-8",
    )

    outputs = run_models(
        profiles_path=profiles,
        matched_profiles_path=matched,
        contrasts_path=contrasts,
        output_root=tmp_path / "models",
        study=load_study(Path(__file__).parents[1] / "configs" / "study.yaml"),
        repetitions=99,
    )

    axes = pd.read_parquet(outputs[0])
    omnibus = pd.read_parquet(outputs[1])
    predictions = pd.read_parquet(outputs[2])
    for output in (axes, omnibus, predictions):
        assert set(output["estimand_role"]) == {"secondary_sensitivity"}
        assert set(output["estimand_id"]) == {"all_available_participant_condition_sensitivity"}
        assert set(output["sampling_basis"]) == {"all_available_participant_condition_profiles"}
    audit = json.loads(outputs[-1].read_text(encoding="utf-8"))
    assert audit["completed_primary_inference"] == 0
    assert audit["completed_secondary_sensitivity_inference"] == 1
    primary = [
        row for row in audit["estimands"] if row["estimand_id"] == "repeated_equal_window_primary"
    ]
    assert len(primary) == 1
    assert primary[0]["status"] == "unavailable"
    assert primary[0]["sampling_basis"] == "repeated_equal_window_profiles"
    assert not any("fallback" in issue for issue in audit["issues"])


def test_model_phase_refuses_technical_success_with_zero_inference(tmp_path: Path) -> None:
    rows = [
        {
            "participant_id": f"sub-{index:02d}",
            "dataset_id": "synthetic",
            "condition": "low",
            "prediction_evaluation_eligible": True,
            **{axis: float(index) for axis in AXIS_NAMES},
        }
        for index in range(4)
    ]
    profiles = tmp_path / "profiles.parquet"
    pd.DataFrame(rows).to_parquet(profiles, index=False)
    contrasts = tmp_path / "contrasts.yaml"
    contrasts.write_text(
        """schema_version: 1
contrasts:
  - id: absent
    datasets: [different-dataset]
    label_column: condition
    positive: high
    negative: low
    design: between_participant
""",
        encoding="utf-8",
    )
    output = tmp_path / "models"
    with pytest.raises(RuntimeError, match="zero valid inference"):
        run_models(
            profiles_path=profiles,
            contrasts_path=contrasts,
            output_root=output,
            study=load_study(Path(__file__).parents[1] / "configs" / "study.yaml"),
            repetitions=99,
        )
    audit = json.loads((output / "model-audit.json").read_text(encoding="utf-8"))
    assert audit["technical_failure"] == "zero_valid_inference_blocks"
    assert not (output / "axis-contrasts.parquet").exists()


def test_continuous_observed_and_resampling_estimand_equal_weight_participants() -> None:
    rows = []
    for participant, (slope, scale) in enumerate(
        ((1.0, 100.0), (1.0, 50.0), (1.0, 1.0), (10.0, 1.0))
    ):
        for target in (0.0, scale, 2.0 * scale):
            rows.append(
                {
                    "participant_id": f"sub-{participant}",
                    "continuous_target": target,
                    **{axis: slope * target for axis in AXIS_NAMES},
                }
            )
    axes, _ = _continuous_inference(
        pd.DataFrame(rows), contrast_id="continuous", repetitions=99, seed=17
    )
    # Equal participant weighting gives (1 + 1 + 1 + 10) / 4 = 3.25;
    # a denominator-pooled regression would be dominated by the first subject.
    assert all(row["effect"] == pytest.approx(3.25) for row in axes)


def test_leave_one_dataset_out_is_available_only_with_two_complete_datasets(
    tmp_path: Path,
) -> None:
    rng = np.random.default_rng(73)
    rows = []
    for dataset_index, dataset_id in enumerate(("dataset_a", "dataset_b")):
        for participant in range(12):
            for label, condition in enumerate(("low", "high")):
                rows.append(
                    {
                        "participant_id": f"{dataset_id}:sub-{participant:02d}",
                        "dataset_id": dataset_id,
                        "condition": condition,
                        "prediction_evaluation_eligible": participant >= 3,
                        **dict(
                            zip(
                                AXIS_NAMES,
                                rng.normal(size=5)
                                + label * np.asarray([0.5, 0.3, 0.2, 0.1, 0.4])
                                + dataset_index * 0.05,
                                strict=True,
                            )
                        ),
                    }
                )
    profiles = tmp_path / "profiles.parquet"
    pd.DataFrame(rows).to_parquet(profiles, index=False)
    contrasts = tmp_path / "contrasts.yaml"
    contrasts.write_text(
        """schema_version: 1
contrasts:
  - id: high-versus-low
    datasets: [dataset_a, dataset_b]
    label_column: condition
    positive: high
    negative: low
    design: within_participant
""",
        encoding="utf-8",
    )

    outputs = run_models(
        profiles_path=profiles,
        contrasts_path=contrasts,
        output_root=tmp_path / "models",
        study=load_study(Path(__file__).parents[1] / "configs" / "study.yaml"),
        repetitions=99,
    )
    predictions = pd.read_parquet(outputs[2])
    leave_one_dataset_out = predictions[predictions["model"] == "five_axis_leave_one_dataset_out"]

    assert set(leave_one_dataset_out["held_out_dataset_id"]) == {
        "dataset_a",
        "dataset_b",
    }
    assert set(leave_one_dataset_out["generalization_scheme"]) == {"leave_one_dataset_out"}
    assert set(leave_one_dataset_out["estimand_role"]) == {"secondary_sensitivity"}
    assert set(leave_one_dataset_out["n_evaluation_participants"]) == {9}
    assert set(leave_one_dataset_out["n_training_participants"]) == {12}
    assert leave_one_dataset_out["outer_fold_participant_separation_verified"].all()
    assert set(leave_one_dataset_out["participant_bootstrap_unit"]) == {"participant"}

    audit = json.loads(outputs[-1].read_text(encoding="utf-8"))
    assert audit["leave_one_dataset_out_prediction_completed"] == 2


def test_repeated_model_and_redundancy_unsupported_designs_are_explicit() -> None:
    frame = pd.DataFrame(
        [
            {
                "participant_id": f"sub-{index}",
                "dataset_id": "synthetic",
                "binary_target": index % 2,
                **{axis: 1.0 for axis in AXIS_NAMES},
            }
            for index in range(6)
        ]
    )

    mixed = _mixed_model_diagnostics(frame, analysis_type="binary")
    assert set(mixed) == set(AXIS_NAMES)
    assert {result["mixed_model_status"] for result in mixed.values()} == {
        "unavailable_insufficient_repeated_observations"
    }
    assert all(result["mixed_model_unavailable_reason"] for result in mixed.values())

    redundancy = _axis_redundancy_diagnostics(frame)
    assert redundancy["status"] == "unavailable_no_finite_correlations"
    assert redundancy["condition_number_status"] == "unavailable_constant_axis"
    assert redundancy["reason"]
