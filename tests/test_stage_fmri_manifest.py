from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from neural_manifolds.stages import fmri_manifest
from neural_manifolds.stages.fmri import DEFAULT_CONFOUNDS, _validate_manifest
from neural_manifolds.stages.fmri_manifest import (
    ATLAS_ENV,
    COORDINATES_ENV,
    TIMING_ORIGIN_ENV,
    DS006623ManifestError,
    prepare_ds006623_fmri_manifest,
)

RUN_LENGTHS = {
    ("rest", 1): 5,
    ("rest", 2): 5,
    ("imagery", 1): 6,
    ("imagery", 2): 8,
    ("imagery", 3): 10,
    ("imagery", 4): 6,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _write_inventory(release: Path) -> None:
    entries = []
    for path in sorted(release.rglob("*")):
        if not path.is_file() or ".acquisition" in path.parts:
            continue
        entries.append(
            {
                "path": path.relative_to(release).as_posix(),
                "size": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    acquisition = release / ".acquisition"
    acquisition.mkdir()
    (acquisition / "manifest.json").write_text(json.dumps({"files": entries}), encoding="utf-8")


def _fixture_release(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    release = tmp_path / "release"
    release.mkdir()
    (release / "dataset_description.json").write_text(
        json.dumps(
            {
                "Name": fmri_manifest.DATASET_NAME,
                "DatasetDOI": fmri_manifest.DATASET_DOI,
                "DatasetType": "raw",
                "License": "CC0",
            }
        ),
        encoding="utf-8",
    )
    derivative = release / "derivatives" / "fmriprep_output"
    derivative.mkdir(parents=True)
    (derivative / "dataset_description.json").write_text(
        json.dumps(
            {"GeneratedBy": [{"Name": "fMRIPrep", "Version": fmri_manifest.FMRIPREP_VERSION}]}
        ),
        encoding="utf-8",
    )
    timing_path = release / "derivatives" / "LOR_ROR_Timing.csv"
    timing_path.write_text(
        "Subject,LOR time (TR in task2),ROR time (TR in task3)\nsub-02,4,6\n",
        encoding="utf-8",
    )

    raw_func = release / "sub-02" / "func"
    derivative_func = derivative / "sub-02" / "func"
    infusion = release / "derivatives" / "Propofol_Infusion" / "sub-02"
    raw_func.mkdir(parents=True)
    derivative_func.mkdir(parents=True)
    infusion.mkdir(parents=True)
    confound_frame_paths: dict[tuple[str, int], Path] = {}
    esc_paths: dict[tuple[str, int], Path] = {}
    for (task, run), n_volumes in RUN_LENGTHS.items():
        stem = f"sub-02_task-{task}_run-{run}"
        (raw_func / f"{stem}_bold.nii.gz").write_bytes(b"raw-bold-placeholder")
        preprocessed_stem = (
            f"{stem}_space-{fmri_manifest.MNI_SPACE}_"
            f"res-{fmri_manifest.MNI_RESOLUTION}_desc-preproc_bold"
        )
        (derivative_func / f"{preprocessed_stem}.nii.gz").write_bytes(
            b"preprocessed-bold-placeholder"
        )
        (derivative_func / f"{preprocessed_stem}.json").write_text(
            json.dumps({"RepetitionTime": fmri_manifest.OFFICIAL_TR_SECONDS}),
            encoding="utf-8",
        )
        confounds = derivative_func / f"{stem}_desc-confounds_timeseries.tsv"
        pd.DataFrame(
            {column: np.linspace(0.0, 1.0, n_volumes) for column in DEFAULT_CONFOUNDS}
        ).to_csv(confounds, sep="\t", index=False)
        confound_frame_paths[(task, run)] = confounds
        esc_label = f"rest{run}" if task == "rest" else f"task{run}"
        esc = infusion / f"sub-02_{esc_label}_ESC.1D"
        values = (
            np.zeros(n_volumes)
            if (task, run) in {("rest", 1), ("imagery", 1)}
            else np.linspace(0.1, 1.0, n_volumes)
        )
        esc.write_text("\n".join(str(value) for value in values) + "\n", encoding="utf-8")
        esc_paths[(task, run)] = esc
    _write_inventory(release)

    atlas = tmp_path / "UKB_424_atlas.nii.gz"
    atlas.write_bytes(b"atlas-placeholder")
    coordinates = tmp_path / "A424_coordinates.npy"
    index = np.arange(424, dtype=float)
    np.save(coordinates, np.column_stack((index, np.sin(index), np.cos(index))))

    monkeypatch.setattr(
        fmri_manifest,
        "validate_release",
        lambda *args, **kwargs: {
            "dataset_id": fmri_manifest.DATASET_ID,
            "release_version": fmri_manifest.RELEASE_VERSION,
            "valid": True,
        },
    )
    monkeypatch.setattr(fmri_manifest, "_validate_atlas_contract", lambda path: None)

    def fake_nifti_run_info(path: Path) -> tuple[int, float]:
        match = fmri_manifest._PREPROC_BOLD_RE.fullmatch(path.name)
        assert match is not None
        key = match.group("task"), int(match.group("run"))
        return RUN_LENGTHS[key], fmri_manifest.OFFICIAL_TR_SECONDS

    monkeypatch.setattr(fmri_manifest, "_nifti_run_info", fake_nifti_run_info)
    return {
        "release": release,
        "atlas": atlas,
        "coordinates": coordinates,
        "timing": timing_path,
        "confounds_imagery_2": confound_frame_paths[("imagery", 2)],
        "esc_imagery_2": esc_paths[("imagery", 2)],
    }


def test_prepares_exact_stage_manifest_with_global_subject_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture_release(tmp_path, monkeypatch)
    artifacts = prepare_ds006623_fmri_manifest(
        release_root=fixture["release"],
        output_root=tmp_path / "out",
        timing_index_origin=0,
        atlas_path=fixture["atlas"],
        coordinates_path=fixture["coordinates"],
    )

    frame = pd.read_parquet(artifacts.manifest_path)
    _validate_manifest(frame)
    assert artifacts.analysis_units == 8
    assert artifacts.participants == 1
    assert set(frame["participant_id"]) == {"propofol_fmri:sub-02"}
    assert set(frame["native_participant_id"]) == {"sub-02"}
    assert set(frame["dataset_id"]) == {"propofol_fmri"}
    assert set(frame["parcellation"]) == {"UKB_424"}
    assert set(frame["normalization"]) == {"unscaled_denoised"}
    assert set(frame["timeseries_scope"]) == {"run"}
    assert frame["bold_path"].str.contains("desc-preproc_bold.nii.gz", regex=False).all()

    induction = frame[(frame["task"] == "imagery") & (frame["run_id"] == "2")]
    assert list(induction["volume_start"]) == [0, 4]
    assert list(induction["volume_stop"]) == [4, 8]
    assert list(induction["condition"]) == [
        "responsive_induction",
        "behaviorally_unresponsive",
    ]
    recovery = frame[(frame["task"] == "imagery") & (frame["run_id"] == "3")]
    assert list(recovery["volume_start"]) == [0, 6]
    assert list(recovery["volume_stop"]) == [6, 10]
    assert artifacts.audit_path.is_file()
    audit = json.loads(artifacts.audit_path.read_text(encoding="utf-8"))
    assert audit["timing"]["index_origin"] == 0
    assert audit["manifest"]["runs"] == 6


def test_missing_ror_remains_explicitly_unresolved_and_assets_can_use_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture_release(tmp_path, monkeypatch)
    fixture["timing"].write_text(
        "Subject,LOR time (TR in task2),ROR time (TR in task3)\nsub-02,4,N/A\n",
        encoding="utf-8",
    )
    monkeypatch.setenv(ATLAS_ENV, str(fixture["atlas"]))
    monkeypatch.setenv(COORDINATES_ENV, str(fixture["coordinates"]))
    monkeypatch.setenv(TIMING_ORIGIN_ENV, "0")

    artifacts = prepare_ds006623_fmri_manifest(
        release_root=fixture["release"], output_root=tmp_path / "out"
    )
    frame = pd.read_parquet(artifacts.manifest_path)
    recovery = frame[(frame["task"] == "imagery") & (frame["run_id"] == "3")]
    assert len(recovery) == 1
    assert recovery.iloc[0]["condition"] == "recovery_status_unresolved"
    assert recovery.iloc[0]["metadata_status"] == "unresolved"
    assert int(recovery.iloc[0]["volume_start"]) == 0
    assert int(recovery.iloc[0]["volume_stop"]) == RUN_LENGTHS[("imagery", 3)]


def test_refuses_to_guess_timing_origin_or_missing_a424_assets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture_release(tmp_path, monkeypatch)
    monkeypatch.delenv(TIMING_ORIGIN_ENV, raising=False)
    with pytest.raises(DS006623ManifestError, match="does not document"):
        prepare_ds006623_fmri_manifest(
            release_root=fixture["release"], output_root=tmp_path / "out"
        )

    monkeypatch.delenv(ATLAS_ENV, raising=False)
    monkeypatch.delenv(COORDINATES_ENV, raising=False)
    with pytest.raises(DS006623ManifestError, match="not distributed in ds006623"):
        prepare_ds006623_fmri_manifest(
            release_root=fixture["release"],
            output_root=tmp_path / "out",
            timing_index_origin=0,
        )


@pytest.mark.parametrize("array", ["esc", "confounds"])
def test_rejects_non_full_run_arrays(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, array: str
) -> None:
    fixture = _fixture_release(tmp_path, monkeypatch)
    if array == "esc":
        fixture["esc_imagery_2"].write_text("0\n1\n", encoding="utf-8")
        expected = "ESC values"
    else:
        pd.DataFrame({column: np.zeros(2, dtype=float) for column in DEFAULT_CONFOUNDS}).to_csv(
            fixture["confounds_imagery_2"], sep="\t", index=False
        )
        expected = "confounds rows"
    with pytest.raises(DS006623ManifestError, match=expected):
        prepare_ds006623_fmri_manifest(
            release_root=fixture["release"],
            output_root=tmp_path / "out",
            timing_index_origin=0,
            atlas_path=fixture["atlas"],
            coordinates_path=fixture["coordinates"],
        )


def test_rejects_undocumented_timing_headers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture_release(tmp_path, monkeypatch)
    fixture["timing"].write_text("participant,lor,ror\nsub-02,4,6\n", encoding="utf-8")
    with pytest.raises(DS006623ManifestError, match="columns must be exactly"):
        prepare_ds006623_fmri_manifest(
            release_root=fixture["release"],
            output_root=tmp_path / "out",
            timing_index_origin=0,
            atlas_path=fixture["atlas"],
            coordinates_path=fixture["coordinates"],
        )
