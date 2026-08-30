from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from neural_manifolds.figures import (
    FigureInputError,
    figure_run_artifacts,
    run_clinical_figure_supplement,
    run_figures,
    run_fmri_figure_supplement,
)
from neural_manifolds.provenance import sha256_file

AXES = ("R", "M", "D", "A", "P")


def _write_inputs(root: Path) -> dict[str, Path]:
    profiles_rows = []
    for condition_index, condition in enumerate(("wake", "sedation")):
        for participant_index in range(4):
            profiles_rows.append(
                {
                    "participant_id": f"p{participant_index}",
                    "dataset_id": "dsA" if participant_index < 2 else "dsB",
                    "condition": condition,
                    **{
                        axis: (participant_index - 1.5) * 0.16
                        + condition_index * (-0.45 + axis_index * 0.12)
                        for axis_index, axis in enumerate(AXES)
                    },
                    "source_artifact_sha256": "1" * 64,
                }
            )
    profiles = root / "profiles.csv"
    pd.DataFrame(profiles_rows).to_csv(profiles, index=False)

    models = root / "models"
    models.mkdir()
    content_rows = []
    for contrast_index, contrast in enumerate(("seen-unseen", "report-no_report")):
        for axis_index, axis in enumerate(AXES):
            for participant_index in range(4):
                content_rows.append(
                    {
                        "participant_id": f"p{participant_index}",
                        "dataset_id": "dsA" if participant_index < 2 else "dsB",
                        "contrast": contrast,
                        "axis": axis,
                        "value": 0.08 * participant_index
                        + (contrast_index - 0.5) * (axis_index - 2) * 0.14,
                        "positive_conditions": "positive",
                        "negative_conditions": "reference",
                        "n_positive_units": 2,
                        "n_negative_units": 2,
                        "matched_strata": 1,
                        "source_artifact_sha256": "2" * 64,
                    }
                )
    pd.DataFrame(content_rows).to_csv(models / "content_report.csv", index=False)
    robustness_rows = []
    families = (
        "phase_randomization",
        "blockwise_temporal_permutation",
        "post_encoder_latent_rotation_control",
        "covariance_dwell_matched_state_space",
    )
    for family_index, family in enumerate(families):
        for axis_index, axis in enumerate(AXES):
            for repeat in range(2):
                for participant_index in range(4):
                    observed = 0.72 + 0.03 * participant_index + 0.01 * axis_index
                    null_effect = 0.11 + 0.01 * family_index + 0.005 * repeat
                    survival = observed - null_effect
                    robustness_rows.append(
                        {
                            "participant_id": f"p{participant_index}",
                            "dataset_id": "dsA" if participant_index < 2 else "dsB",
                            "contrast": "awake_vs_propofol_sedation",
                            "analysis": family,
                            "family": family,
                            "repeat": repeat,
                            "seed": 1000 + family_index * 100 + repeat,
                            "metric": axis,
                            "observed_effect": observed,
                            "null_effect": null_effect,
                            "observed_minus_null": survival,
                            "signed_effect_survival": survival,
                            "value": survival,
                            "positive_conditions": "awake",
                            "negative_conditions": "propofol_sedation",
                            "n_positive_units": 2,
                            "n_negative_units": 2,
                            "matched_strata": 1,
                            "source_artifact_sha256": "3" * 64,
                        }
                    )
    pd.DataFrame(robustness_rows).to_csv(models / "robustness.csv", index=False)

    tms = root / "tms"
    tms.mkdir()
    tms_rows = []
    trajectory_rows = []
    for condition_index, condition in enumerate(("awake", "propofol_sedation")):
        for participant_index in range(4):
            passive_delta = 0.12 + 0.03 * participant_index
            direct_delta = 0.18 + 0.04 * participant_index
            passive_sedation = 0.2 + 0.1 * participant_index
            direct_sedation = 0.25 + 0.08 * participant_index
            if condition == "awake":
                passive = passive_sedation + passive_delta
                direct = direct_sedation + direct_delta
            else:
                passive = passive_sedation
                direct = direct_sedation
            tms_rows.append(
                {
                    "participant_id": f"p{participant_index}",
                    "dataset_id": "dsA",
                    "condition": condition,
                    "passive_reachability": passive,
                    "direct_response": direct,
                    "passive_delta": passive_delta,
                    "direct_delta": direct_delta,
                    "tms_contrast": "awake_minus_propofol_sedation",
                    "source_artifact_sha256": "4" * 64,
                }
            )
            for time_ms in (-20, 0, 40, 80, 120):
                trajectory_rows.append(
                    {
                        "participant_id": f"p{participant_index}",
                        "dataset_id": "dsA",
                        "condition": condition,
                        "time_ms": time_ms,
                        "trajectory_value": (1 - condition_index * 0.35)
                        * np.exp(-max(time_ms, 0) / 110)
                        + participant_index * 0.025,
                        "source_artifact_sha256": "5" * 64,
                    }
                )
    pd.DataFrame(tms_rows).to_csv(tms / "participants.csv", index=False)
    pd.DataFrame(trajectory_rows).to_csv(tms / "trajectory.csv", index=False)

    clinical_rows = []
    for participant_index in range(6):
        diagnosis = "MCS" if participant_index < 3 else "UWS"
        clinical_rows.append(
            {
                "participant_id": f"c{participant_index}",
                "dataset_id": "docA" if participant_index % 2 == 0 else "docB",
                "diagnosis": diagnosis,
                "crs_r_total": 4 + participant_index * 2,
                "wake_regime_log_likelihood_ratio": -0.5 + participant_index * 0.22,
                "wake_regime_score_status": ("available_frozen_healthy_wake_vs_propofol_reference"),
                "crs_r_status": "available",
                **{
                    axis: -0.7 + participant_index * 0.18 + axis_index * 0.07
                    for axis_index, axis in enumerate(AXES)
                },
                "source_artifact_sha256": "6" * 64,
            }
        )
    clinical = root / "clinical.csv"
    pd.DataFrame(clinical_rows).to_csv(clinical, index=False)

    fmri_rows = []
    for condition_index, condition in enumerate(("awake", "deep_propofol")):
        for participant_index in range(4):
            fmri_rows.append(
                {
                    "participant_id": f"f{participant_index}",
                    "dataset_id": "fmriA",
                    "condition": condition,
                    **{
                        axis: 0.1 * participant_index + condition_index * (-0.4 + axis_index * 0.08)
                        for axis_index, axis in enumerate(AXES[:-1])
                    },
                    "source_artifact_sha256": "7" * 64,
                }
            )
    fmri = root / "fmri.csv"
    pd.DataFrame(fmri_rows).to_csv(fmri, index=False)
    return {
        "profiles": profiles,
        "models": models,
        "tms": tms,
        "clinical": clinical,
        "fmri": fmri,
    }


def test_run_figures_exports_auditable_submission_bundle(tmp_path: Path) -> None:
    inputs = _write_inputs(tmp_path)
    output = tmp_path / "figure_run"
    result = run_figures(
        inputs["profiles"],
        inputs["models"],
        inputs["tms"],
        inputs["clinical"],
        inputs["fmri"],
        output,
    )
    assert result.manifest_path.is_file()
    assert result.skipped == ()
    assert set(result.figure_paths) == {f"figure_{number}" for number in range(1, 7)}

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["backend"] == "python-matplotlib"
    assert len(manifest["figures"]) == 6
    assert all(item["status"] == "rendered" for item in manifest["figures"])
    assert all(item["contract"]["core_conclusion"] for item in manifest["figures"])

    for figure_id, paths in result.figure_paths.items():
        assert set(paths) == {"svg", "pdf", "tiff"}
        svg = paths["svg"]
        assert "<text" in svg.read_text(encoding="utf-8")
        assert (
            b"/CIDFontType2" in paths["pdf"].read_bytes()
            or b"/FontFile2" in paths["pdf"].read_bytes()
        )
        with Image.open(paths["tiff"]) as image:
            assert image.info["dpi"][0] == pytest.approx(600, abs=2)
            assert image.info["dpi"][1] == pytest.approx(600, abs=2)
        manifest_item = next(item for item in manifest["figures"] if item["figure_id"] == figure_id)
        for extension, path in paths.items():
            assert manifest_item["outputs"][extension]["sha256"] == sha256_file(path)
        assert manifest_item["qa"]["p_value_stars_drawn"] is False

    for panels in result.source_data_paths.values():
        for source_path in panels.values():
            source = pd.read_csv(source_path)
            assert "source_artifact_sha256" in source.columns
            assert "source_table_sha256" in source.columns

    tms_delta_source = pd.read_csv(result.source_data_paths["figure_4"]["a"])
    assert len(tms_delta_source) == 4
    assert set(tms_delta_source["association_test"]) == {
        "spearman_participant_level_condition_delta"
    }
    assert set(tms_delta_source["tms_contrast"]) == {"awake_minus_propofol_sedation"}
    assert tms_delta_source["spearman_rho"].iloc[0] == pytest.approx(1.0)

    clinical_association_source = pd.read_csv(result.source_data_paths["figure_6"]["b"])
    assert set(clinical_association_source["association_test"]) == {"spearman_participant_level"}
    assert clinical_association_source["spearman_rho"].iloc[0] == pytest.approx(1.0)
    assert set(clinical_association_source["n_association"]) == {6}

    manifest_hash = sha256_file(result.manifest_path)
    recovered = run_figures(
        inputs["profiles"],
        inputs["models"],
        inputs["tms"],
        inputs["clinical"],
        inputs["fmri"],
        output,
    )
    assert recovered.figure_paths == result.figure_paths
    assert sha256_file(recovered.manifest_path) == manifest_hash


def test_missing_required_schema_fails_without_partial_output(tmp_path: Path) -> None:
    inputs = _write_inputs(tmp_path)
    broken = pd.read_csv(inputs["profiles"]).drop(columns="P")
    broken.to_csv(inputs["profiles"], index=False)
    output = tmp_path / "failed_run"
    with pytest.raises(FigureInputError, match="missing required columns"):
        run_figures(
            inputs["profiles"],
            inputs["models"],
            inputs["tms"],
            inputs["clinical"],
            inputs["fmri"],
            output,
        )
    assert not output.exists()


def test_late_supplements_are_explicit_and_restart_safe(tmp_path: Path) -> None:
    inputs = _write_inputs(tmp_path)
    clinical = pd.read_csv(inputs["clinical"])
    clinical["crs_r_total"] = np.nan
    clinical.to_csv(inputs["clinical"], index=False)

    clinical_output = tmp_path / "clinical_supplement"
    clinical_result = run_clinical_figure_supplement(inputs["clinical"], clinical_output)
    clinical_manifest = json.loads(clinical_result.manifest_path.read_text(encoding="utf-8"))
    assert clinical_manifest["stage"] == "clinical_figure_supplement"
    assert set(clinical_result.figure_paths) == {"figure_6"}
    assert all(path.is_file() for path in figure_run_artifacts(clinical_result))
    source_b = pd.read_csv(clinical_result.source_data_paths["figure_6"]["b"])
    assert not source_b["crs_r_available"].astype(bool).any()
    recovered = run_clinical_figure_supplement(inputs["clinical"], clinical_output)
    assert recovered.figure_paths == clinical_result.figure_paths

    fmri_result = run_fmri_figure_supplement(
        inputs["models"], inputs["fmri"], tmp_path / "fmri_supplement"
    )
    fmri_manifest = json.loads(fmri_result.manifest_path.read_text(encoding="utf-8"))
    assert fmri_manifest["stage"] == "fmri_figure_supplement"
    assert set(fmri_result.figure_paths) == {"figure_5"}
    assert all(path.is_file() for path in figure_run_artifacts(fmri_result))
