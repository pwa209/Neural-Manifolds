from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from neural_manifolds.figure_sources import (
    prepare_clinical_figure_source,
    prepare_figure_sources,
    prepare_fmri_figure_source,
)
from neural_manifolds.figures.io import load_source_bundle
from neural_manifolds.manifold.profile import AXIS_NAMES


def _write_contrast_inputs(root: Path) -> dict[str, Path]:
    profiles_rows = []
    null_rows = []
    for participant_index, participant_id in enumerate(("p1", "p2")):
        for condition, base in (("awake", 3.0), ("sedation", 1.0)):
            for replicate in range(2):
                unit_id = f"{participant_id}-{condition}-{replicate}"
                axes = {
                    axis: base + participant_index * 0.2 + axis_index * 0.1
                    for axis_index, axis in enumerate(AXIS_NAMES)
                }
                profiles_rows.append(
                    {
                        "unit_id": unit_id,
                        "participant_id": participant_id,
                        "dataset_id": "ds",
                        "condition": condition,
                        **axes,
                    }
                )
                for repeat in (0, 1):
                    null_rows.append(
                        {
                            "unit_id": unit_id,
                            "participant_id": participant_id,
                            "dataset_id": "ds",
                            "family": "phase_randomization",
                            "repeat": repeat,
                            "seed": 100 + repeat,
                            **{axis: value * 0.25 for axis, value in axes.items()},
                        }
                    )
    profiles = root / "profiles.parquet"
    nulls = root / "nulls.parquet"
    pd.DataFrame(profiles_rows).to_parquet(profiles, index=False)
    pd.DataFrame(null_rows).to_parquet(nulls, index=False)

    contrasts = root / "contrasts.yaml"
    contrasts.write_text(
        """schema_version: 1
datasets:
  ds:
    contrasts:
      - id: awake_vs_sedation
        conditions: [awake, sedation]
        match_within: participant_id
""",
        encoding="utf-8",
    )

    tms = root / "tms.parquet"
    pd.DataFrame(
        {
            "participant_id": ["p1", "p2"],
            "condition": ["awake", "sedation"],
            "reachability": [0.4, 0.2],
            "maximum_displacement": [0.5, 0.25],
        }
    ).to_parquet(tms, index=False)
    trajectory = root / "trajectory.parquet"
    pd.DataFrame(
        {
            "participant_id": ["p1", "p2"],
            "condition": ["awake", "sedation"],
            "time_ms": [0.0, 0.0],
            "trajectory_value": [1.0, 0.7],
        }
    ).to_parquet(trajectory, index=False)
    return {
        "profiles": profiles,
        "nulls": nulls,
        "contrasts": contrasts,
        "tms": tms,
        "trajectory": trajectory,
    }


def test_sources_are_true_participant_contrasts_and_effect_survival(tmp_path: Path) -> None:
    inputs = _write_contrast_inputs(tmp_path)
    bundles = prepare_figure_sources(
        profiles_path=inputs["profiles"],
        nulls_path=inputs["nulls"],
        contrasts_path=inputs["contrasts"],
        tms_outcomes_path=inputs["tms"],
        tms_trajectory_path=inputs["trajectory"],
        output_root=tmp_path / "bundles",
    )
    models = bundles[1]
    content = pd.read_parquet(models / "content_report.parquet")
    r_effects = content.loc[content["axis"].eq("R")]
    assert set(r_effects["contrast"]) == {"awake_vs_sedation"}
    assert r_effects["value"].tolist() == pytest.approx([2.0, 2.0])
    assert set(r_effects["positive_conditions"]) == {"awake"}
    assert set(r_effects["negative_conditions"]) == {"sedation"}
    assert set(r_effects["matched_strata"]) == {1}

    robustness = pd.read_parquet(models / "robustness.parquet")
    r_robustness = robustness.loc[robustness["metric"].eq("R")]
    assert set(r_robustness["family"]) == {"phase_randomization"}
    assert set(r_robustness["repeat"]) == {0, 1}
    assert r_robustness["observed_effect"].tolist() == pytest.approx([2.0] * 4)
    assert r_robustness["null_effect"].tolist() == pytest.approx([0.5] * 4)
    assert r_robustness["observed_minus_null"].tolist() == pytest.approx([1.5] * 4)
    assert r_robustness["signed_effect_survival"].tolist() == pytest.approx([1.5] * 4)
    assert set(r_robustness["positive_conditions"]) == {"awake"}
    assert set(r_robustness["negative_conditions"]) == {"sedation"}

    loaded = load_source_bundle("models", models)
    assert set(loaded.tables) == {"content_report", "contrast_status", "robustness"}


def test_clinical_builder_will_not_invent_locked_endpoint(tmp_path: Path) -> None:
    clinical = tmp_path / "clinical.parquet"
    pd.DataFrame(
        {
            "participant_id": ["p1"],
            "dataset_id": ["doc"],
            "diagnosis": [None],
            **{axis: [0.1] for axis in AXIS_NAMES},
        }
    ).to_parquet(clinical, index=False)
    with pytest.raises(ValueError, match="will not derive a replacement endpoint"):
        prepare_clinical_figure_source(
            clinical_profiles_path=clinical,
            output_root=tmp_path / "clinical_bundle",
        )


def test_fmri_builder_requires_explicitly_calibrated_axes(tmp_path: Path) -> None:
    fmri = tmp_path / "fmri.parquet"
    pd.DataFrame(
        {
            "participant_id": ["p1"],
            "dataset_id": ["fmri"],
            "condition": ["awake"],
            "repertoire": [0.1],
            "metastability": [0.2],
            "directionality": [0.3],
            "alignment": [0.4],
        }
    ).to_parquet(fmri, index=False)
    with pytest.raises(ValueError, match="explicitly calibrated R/M/D/A"):
        prepare_fmri_figure_source(
            fmri_profiles_path=fmri,
            output_root=tmp_path / "fmri_bundle",
        )
