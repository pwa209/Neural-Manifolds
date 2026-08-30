from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from neural_manifolds.cohort import build_cohort_manifest
from neural_manifolds.config import load_study
from neural_manifolds.stage_units import encode_analysis_units, preprocess_analysis_units
from neural_manifolds.stages.metrics import run_metrics
from neural_manifolds.stages.models import run_models
from neural_manifolds.stages.tms import select_direct_tms_units
from neural_manifolds.tms_separation import assert_no_direct_tms, direct_tms_mask


def _write_ds005620_release(raw_root: Path) -> Path:
    release = raw_root / "propofol_tms_eeg" / "1.0.0"
    (release / ".acquisition").mkdir(parents=True)
    (release / ".acquisition" / "COMPLETE.json").write_text("{}\n", encoding="utf-8")
    pd.DataFrame(
        [
            {
                "participant_id": "sub-SD_1010",
                "age": "31",
                "sex": "M",
                "awakenings": "2",
                "TMS": "True",
                "tms_count": "1",
                "excluded": "False",
                "bad_after_preprocessing": "False",
            }
        ]
    ).to_csv(release / "participants.tsv", sep="\t", index=False)
    eeg = release / "sub-1010" / "eeg"
    eeg.mkdir(parents=True)
    passive = eeg / "sub-1010_task-awake_acq-rest_eeg.vhdr"
    direct = eeg / "sub-1010_task-sed_acq-tms_run-1_eeg.vhdr"
    passive.write_text("passive lineage fixture\n", encoding="utf-8")
    direct.write_text("direct TMS lineage fixture\n", encoding="utf-8")
    return direct


def test_cohort_keeps_direct_tms_lineage_only_for_dedicated_stage(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    direct_source = _write_ds005620_release(raw_root)
    labels_path, encoder_path, issues_path = build_cohort_manifest(
        raw_root=raw_root,
        output_root=tmp_path / "cohort",
        dataset_ids=("propofol_tms_eeg",),
    )

    labels = pd.read_parquet(labels_path)
    encoder_inputs = pd.read_parquet(encoder_path)
    direct = select_direct_tms_units(labels)

    assert len(labels) == 2
    assert len(encoder_inputs) == 1
    assert not direct_tms_mask(encoder_inputs).any()
    assert set(encoder_inputs["modality"]) == {"eeg"}
    assert len(direct) == 1
    assert direct.iloc[0]["source_path"] == str(direct_source.resolve(strict=True))
    assert direct.iloc[0]["unit_id"] not in set(encoder_inputs["unit_id"])

    audit = json.loads(issues_path.read_text(encoding="utf-8"))
    separation = audit["direct_tms_separation"]
    assert separation["status"] == "omitted_from_general_encoder_inputs"
    assert separation["retained_in_cohort_labels"] is True
    assert separation["raw_lineage_retained"] is True
    assert separation["dedicated_tms_units"] == [
        {
            "dataset_id": "propofol_tms_eeg",
            "source_path": str(direct_source.resolve(strict=True)),
            "unit_id": direct.iloc[0]["unit_id"],
        }
    ]
    assert audit["scientific_gate_applied"] is False


def test_general_contract_rejects_modality_and_legacy_acquisition_rows() -> None:
    frame = pd.DataFrame(
        [
            {
                "unit_id": "modern",
                "dataset_id": "propofol_tms_eeg",
                "modality": "tms-eeg",
                "acquisition": "tms",
            },
            {
                "unit_id": "legacy",
                "dataset_id": "propofol_tms_eeg",
                "modality": "eeg",
                "acquisition": "tms",
            },
            {
                "unit_id": "passive",
                "dataset_id": "propofol_tms_eeg",
                "modality": "eeg",
                "acquisition": "rest",
            },
        ]
    )
    assert direct_tms_mask(frame).tolist() == [True, True, False]
    with pytest.raises(ValueError, match="dedicated pulse-interpolation TMS stage"):
        assert_no_direct_tms(frame, stage="general test stage")


def test_preprocess_encode_and_metrics_fail_before_touching_direct_tms_signal(
    tmp_path: Path,
) -> None:
    study = load_study(Path("configs/study.yaml"))
    source = tmp_path / "direct.vhdr"
    source.write_text("must not be opened by general preprocessing\n", encoding="utf-8")
    encoder_inputs = tmp_path / "encoder-inputs.parquet"
    pd.DataFrame(
        [
            {
                "unit_id": "direct",
                "source_path": str(source),
                "modality": "tms-eeg",
                "selector_json": '{"kind":"full_recording"}',
            }
        ]
    ).to_parquet(encoder_inputs, index=False)
    with pytest.raises(ValueError, match="general preprocessing input forbids"):
        preprocess_analysis_units(
            encoder_inputs=encoder_inputs,
            output_root=tmp_path / "preprocessed",
            study=study,
        )

    preprocessing_manifest = tmp_path / "preprocessing-manifest.parquet"
    pd.DataFrame(
        [
            {
                "unit_id": "direct",
                "modality": "tms-eeg",
                "eligible": True,
                "preprocessed_path": str(tmp_path / "must-not-exist.fif"),
                "selector_json": '{"kind":"full_recording"}',
            }
        ]
    ).to_parquet(preprocessing_manifest, index=False)
    with pytest.raises(ValueError, match="general encoder input forbids"):
        encode_analysis_units(
            preprocessing_manifest=preprocessing_manifest,
            labels_manifest=tmp_path / "must-not-be-read.parquet",
            output_root=tmp_path / "encoded",
            study=study,
        )

    encoding_manifest = tmp_path / "encoding-manifest.parquet"
    pd.DataFrame(
        [
            {
                "unit_id": "direct",
                "participant_id": "p1",
                "dataset_id": "propofol_tms_eeg",
                "modality": "tms-eeg",
                "acquisition": "tms",
                "trajectory_path": str(tmp_path / "must-not-be-read.npz"),
                "encoded": True,
            }
        ]
    ).to_parquet(encoding_manifest, index=False)
    with pytest.raises(ValueError, match="general metric input forbids"):
        run_metrics(
            encoding_manifest=encoding_manifest,
            output_root=tmp_path / "metrics",
            study=study,
        )


def test_general_models_fail_closed_before_using_direct_tms_profiles(tmp_path: Path) -> None:
    profiles = tmp_path / "profiles.parquet"
    pd.DataFrame(
        [
            {
                "unit_id": "direct",
                "participant_id": "p1",
                "dataset_id": "propofol_tms_eeg",
                "modality": "tms-eeg",
                "acquisition": "tms",
                "prediction_evaluation_eligible": True,
            }
        ]
    ).to_parquet(profiles, index=False)
    with pytest.raises(ValueError, match="general model input forbids 1 direct TMS-EEG"):
        run_models(
            profiles_path=profiles,
            contrasts_path=Path("configs/contrasts.yaml"),
            output_root=tmp_path / "models",
            study=load_study(Path("configs/study.yaml")),
            repetitions=99,
        )


def test_direct_tms_general_contrast_is_explicitly_omitted() -> None:
    document = yaml.safe_load(Path("configs/contrasts.yaml").read_text(encoding="utf-8"))
    dataset = document["datasets"]["propofol_tms_eeg"]
    active = {item["id"] for item in dataset["contrasts"]}
    omitted = {item["id"]: item for item in dataset["omitted_contrasts"]}

    assert "awake_vs_direct_tms_under_propofol" not in active
    assert omitted["awake_vs_direct_tms_under_propofol"]["status"] == "omitted"
    assert (
        "dedicated pulse-interpolation TMS stage"
        in omitted["awake_vs_direct_tms_under_propofol"]["reason"]
    )
