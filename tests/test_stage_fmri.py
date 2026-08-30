import json
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd
import pytest

from neural_manifolds.config import load_study
from neural_manifolds.foundation.brainlm import BrainLMEncoding
from neural_manifolds.stages.fmri import FMRIManifestError, run_fmri_triangulation


class FakeBrainLM:
    metadata: ClassVar[dict[str, object]] = {
        "encoder": "fake_test_backend",
        "source_revision": "e" * 40,
        "checkpoint_config_sha256": "a" * 64,
        "checkpoint_weights_sha256": "b" * 64,
        "weights_frozen": True,
        "label_free": True,
        "parcellation": "UKB_424",
        "usage_license": "CC-BY-NC-ND-4.0",
        "commercial_use": False,
        "derivative_redistribution": False,
    }

    def __init__(self) -> None:
        self.metadata_fields: list[tuple[str, ...]] = []

    def encode(
        self,
        parcel_timeseries: np.ndarray,
        coordinates: np.ndarray,
        *,
        metadata_fields: tuple[str, ...] = (),
    ) -> BrainLMEncoding:
        self.metadata_fields.append(metadata_fields)
        assert coordinates.shape == (424, 3)
        starts = np.arange(0, 12 * 4, 4, dtype=np.int64)
        grid = np.linspace(0.0, 2.0 * np.pi, starts.size, endpoint=False)
        offset = float(np.mean(parcel_timeseries))
        states = np.column_stack(
            [
                np.sin(grid + 0.2 * feature) + 0.3 * np.cos((feature + 1) * grid / 3.0) + offset
                for feature in range(8)
            ]
        )
        return BrainLMEncoding(
            global_states=states,
            window_starts=starts,
            window_stops=starts + 200,
            metadata=dict(self.metadata),
        )


def _coordinates(tmp_path: Path) -> Path:
    index = np.arange(424)
    coordinates = np.column_stack(
        (
            np.linspace(-50.0, 50.0, 424),
            np.sin(index / 19.0) * 70.0 + index * 1e-4,
            np.cos(index / 23.0) * 40.0 + index * 1e-4,
        )
    )
    path = tmp_path / "coordinates.npy"
    np.save(path, coordinates)
    return path


def _manifest(tmp_path: Path, *, bad_parcellation: bool = False) -> Path:
    rng = np.random.default_rng(42)
    rows = []
    partitions = ["discovery", "discovery", "validation", "test"]
    conditions = ["responsive", "unresponsive", "responsive", "unresponsive"]
    for participant_index in range(4):
        innovations = rng.normal(size=(280, 424))
        values = np.empty_like(innovations)
        values[0] = innovations[0]
        for time_index in range(1, values.shape[0]):
            values[time_index] = 0.65 * values[time_index - 1] + innovations[time_index]
        path = tmp_path / f"sub-{participant_index + 1:02d}_run.npy"
        np.save(path, values.astype(np.float32))
        rows.append(
            {
                "unit_id": f"unit-{participant_index + 1}",
                "participant_id": f"sub-{participant_index + 1:02d}",
                "dataset_id": "propofol_fmri",
                "parcellation": "wrong" if bad_parcellation else "UKB_424",
                "tr_seconds": 1.0,
                "timeseries_path": str(path),
                "preprocessed": True,
                "timeseries_scope": "run",
                "normalization": "unscaled_denoised",
                "partition": partitions[participant_index],
                "condition": conditions[participant_index],
                "volume_start": 0,
                "volume_stop": 280,
            }
        )
    manifest = tmp_path / "fmri-manifest.parquet"
    pd.DataFrame(rows).to_parquet(manifest, index=False)
    return manifest


def test_fmri_stage_is_participant_safe_and_has_no_reachability_claim(tmp_path: Path) -> None:
    encoder = FakeBrainLM()
    encoded, units, participants, audit = run_fmri_triangulation(
        manifest_path=_manifest(tmp_path),
        coordinates_path=_coordinates(tmp_path),
        output_root=tmp_path / "out",
        encoder=encoder,
        study=load_study("configs/study.yaml"),
    )
    assert encoded.is_file() and units.is_file() and participants.is_file() and audit.is_file()
    unit_frame = pd.read_parquet(units)
    participant_frame = pd.read_parquet(participants)
    assert len(unit_frame) == 4
    assert participant_frame[["R", "M", "D", "A"]].apply(np.isfinite).all().all()
    assert not any("reachability" in column or "controllability" in column for column in unit_frame)
    assert all(fields == () for fields in encoder.metadata_fields)
    payload = json.loads(audit.read_text(encoding="utf-8"))
    assert payload["normalization_fit_partition"] == "discovery"
    assert payload["normalization_fit_participants"] == ["sub-01", "sub-02"]
    assert not any(payload["partition_overlaps"].values())
    assert payload["secondary_axes"]["fit_partition"] == "discovery"
    assert payload["secondary_axes"]["excluded"] == ["reachability"]
    assert payload["direct_perturbational_inference"] is False
    assert payload["scientific_gate_applied"] is False


def test_fmri_stage_rejects_participant_partition_overlap(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    frame = pd.read_parquet(manifest)
    duplicated = frame.iloc[[0]].copy()
    duplicated["unit_id"] = "unit-overlap"
    duplicated["partition"] = "test"
    frame = pd.concat([frame, duplicated], ignore_index=True)
    frame.to_parquet(manifest, index=False)
    with pytest.raises(FMRIManifestError, match="cross fMRI partitions"):
        run_fmri_triangulation(
            manifest_path=manifest,
            coordinates_path=_coordinates(tmp_path),
            output_root=tmp_path / "out",
            encoder=FakeBrainLM(),
            study=load_study("configs/study.yaml"),
        )


def test_fmri_stage_rejects_non_ukb424_manifest(tmp_path: Path) -> None:
    with pytest.raises(FMRIManifestError, match="UKB_424"):
        run_fmri_triangulation(
            manifest_path=_manifest(tmp_path, bad_parcellation=True),
            coordinates_path=_coordinates(tmp_path),
            output_root=tmp_path / "out",
            encoder=FakeBrainLM(),
            study=load_study("configs/study.yaml"),
        )
