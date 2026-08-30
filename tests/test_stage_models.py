from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from neural_manifolds.config import load_study
from neural_manifolds.manifold.profile import AXIS_NAMES
from neural_manifolds.stages.models import _continuous_inference, run_models


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
    predictions = pd.read_parquet(outputs[2])
    assert set(axes["axis"]) == set(AXIS_NAMES)
    assert "five_axis" in set(predictions["model"])
    assert predictions["auroc"].between(0, 1).all()
    assert set(predictions["sampling_basis"]) == {"repeated_equal_window_profiles"}
    assert predictions["representation_heldout"].all()
    assert set(predictions["n_evaluation_participants"]) == {9}
    assert set(predictions["n_fixed_transform_participants"]) == {3}


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
