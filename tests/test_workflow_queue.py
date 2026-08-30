from __future__ import annotations

import json
from pathlib import Path

import pytest

from workflow.phases import PHASES
from workflow.queue import (
    build_phase_command,
    establish_run_contract,
    prepare_clinical_lock,
    safe_repo_file,
    verified_fmri_input_fingerprints,
    verified_model_fingerprints,
    verify_server_config,
)
from workflow.state import ServerRoots, atomic_write_json, sha256_file


def test_phase_command_uses_dispatcher_contract(tmp_path: Path) -> None:
    phase = next(item for item in PHASES if item.name == "locked-clinical")
    command = build_phase_command(
        executable="/env/bin/neural-manifolds",
        phase=phase,
        study=tmp_path / "configs/study.yaml",
        datasets=tmp_path / "configs/datasets.yaml",
        server=tmp_path / "configs/server.yaml",
        run_id="main-001",
    )
    assert command[:4] == [
        "/env/bin/neural-manifolds",
        "run-phase",
        "--phase",
        "clinical",
    ]
    assert command[-2:] == ["--run-id", "main-001"]


def test_repo_files_cannot_escape_deployed_release(tmp_path: Path) -> None:
    repo = tmp_path / "release"
    repo.mkdir()
    assert safe_repo_file(repo, "configs/study.yaml", label="study") == (
        repo / "configs/study.yaml"
    )
    with pytest.raises(ValueError, match="inside"):
        safe_repo_file(repo, "../secret.yaml", label="study")


def test_server_config_must_resolve_and_match_all_roots(tmp_path: Path) -> None:
    pytest.importorskip("yaml")
    roots = ServerRoots(
        canonical=tmp_path / "canonical",
        work=tmp_path / "work",
        checkpoint=tmp_path / "checkpoint",
    )
    server = tmp_path / "server.yaml"
    server.write_text(
        """
scientific_gates: false
storage:
  canonical_root: null
  work_root: null
  checkpoint_root: null
  raw_data_location: canonical_only
scheduler:
  type: tmux
""".lstrip(),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unresolved"):
        verify_server_config(server, roots)

    server.write_text(
        f"""
scientific_gates: false
storage:
  canonical_root: {roots.canonical.as_posix()}
  work_root: {roots.work.as_posix()}
  checkpoint_root: {roots.checkpoint.as_posix()}
  raw_data_location: canonical_only
scheduler:
  type: tmux
""".lstrip(),
        encoding="utf-8",
    )
    verify_server_config(server, roots)


def test_clinical_lock_is_technical_non_preregistration_and_idempotent(
    tmp_path: Path,
) -> None:
    state_root = tmp_path / "state"
    names = [phase.name for phase in PHASES]
    healthy_names = names[: names.index("locked-clinical")]
    for name in healthy_names:
        artifact = tmp_path / "artifacts" / f"{name}.json"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text(f'{{"phase":"{name}"}}\n', encoding="utf-8")
        marker = state_root / "phases" / name / "success.json"
        atomic_write_json(
            marker,
            {
                "schema_version": 1,
                "phase": name,
                "status": "succeeded",
                "phase_hash": f"hash-{name}",
                "artifacts": [
                    {
                        "path": str(artifact),
                        "sha256": sha256_file(artifact),
                        "size": artifact.stat().st_size,
                    }
                ],
            },
        )

    first = prepare_clinical_lock(
        state_root=state_root,
        source_manifest_sha256="a" * 64,
        config_fingerprints={"study": {"sha256": "b" * 64, "size": 10}},
    )
    first_hash = sha256_file(first)
    second = prepare_clinical_lock(
        state_root=state_root,
        source_manifest_sha256="a" * 64,
        config_fingerprints={"study": {"sha256": "b" * 64, "size": 10}},
    )
    assert second == first
    assert sha256_file(second) == first_hash
    payload = json.loads(first.read_text(encoding="utf-8"))
    assert payload["project_status"] == "exploratory_non_preregistered"
    assert payload["scientific_gate"] is False
    assert set(payload["healthy_validated_artifacts"]) == set(healthy_names)
    assert healthy_names[-1] == "tms"
    assert "locked-clinical" not in payload["healthy_validated_artifacts"]
    assert "fmri" not in payload["healthy_validated_artifacts"]
    assert "figures" not in payload["healthy_validated_artifacts"]
    assert "not a registration or preregistration" in payload["notice"]


def test_run_id_is_bound_to_source_config_and_roots(tmp_path: Path) -> None:
    roots = ServerRoots(
        canonical=tmp_path / "canonical",
        work=tmp_path / "work",
        checkpoint=tmp_path / "checkpoint",
    )
    state_root = tmp_path / "state"
    common = {
        "state_root": state_root,
        "run_id": "main-001",
        "repo_root": tmp_path / "release",
        "roots": roots,
        "config_fingerprints": {"study": {"sha256": "b" * 64, "size": 10}},
    }
    first = establish_run_contract(source_manifest_sha256="a" * 64, **common)
    assert establish_run_contract(source_manifest_sha256="a" * 64, **common) == first
    with pytest.raises(RuntimeError, match="different source"):
        establish_run_contract(source_manifest_sha256="c" * 64, **common)


def test_model_phase_rehashes_verified_manifest_and_defers_brainlm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = tmp_path / "models"
    labram_source = cache / "sources/labram/revision"
    brain_source = cache / "sources/brain/revision"
    labram_checkpoint = cache / "checkpoints/labram.pth"
    for path, content in (
        (labram_source / "SOURCE_MANIFEST.json", "labram-source\n"),
        (brain_source / "SOURCE_MANIFEST.json", "brain-source\n"),
        (labram_checkpoint, "checkpoint\n"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    def declared(path: Path, *, size: bool = False) -> dict[str, object]:
        item: dict[str, object] = {"path": str(path), "sha256": sha256_file(path)}
        if size:
            item["size"] = path.stat().st_size
        return item

    manifest = cache / "MODEL_MANIFEST.json"
    atomic_write_json(
        manifest,
        {
            "schema_version": 1,
            "stage": "core",
            "models": {
                "labram_base": {
                    "trainable": False,
                    "source": declared(labram_source / "SOURCE_MANIFEST.json"),
                    "checkpoint": declared(labram_checkpoint, size=True),
                },
                "brainlm": {
                    "trainable": False,
                    "source": declared(brain_source / "SOURCE_MANIFEST.json"),
                    "checkpoint_status": "deferred_until_fmri",
                    "usage_license": "CC-BY-NC-ND-4.0",
                },
            },
        },
    )
    monkeypatch.setenv("NEURAL_MANIFOLDS_MODEL_MANIFEST", str(manifest))
    monkeypatch.setenv("NEURAL_MANIFOLDS_LABRAM_SOURCE", str(labram_source))
    monkeypatch.setenv("NEURAL_MANIFOLDS_LABRAM_CHECKPOINT", str(labram_checkpoint))
    encode = next(phase for phase in PHASES if phase.name == "encode")
    assert str(manifest) in verified_model_fingerprints(encode)

    fmri = next(phase for phase in PHASES if phase.name == "fmri")
    with pytest.raises(ValueError, match="only by the fMRI bootstrap"):
        verified_model_fingerprints(fmri)


def test_fmri_assets_and_timing_origin_are_phase_specific_fingerprints(
    tmp_path: Path,
) -> None:
    encode = next(phase for phase in PHASES if phase.name == "encode")
    assert verified_fmri_input_fingerprints(encode, {}) == ({}, {})

    fmri = next(phase for phase in PHASES if phase.name == "fmri")
    with pytest.raises(ValueError, match=r"requires configs/server\.yaml fmri_inputs"):
        verified_fmri_input_fingerprints(fmri, {})

    atlas = tmp_path / "UKB_424_atlas.nii.gz"
    coordinates = tmp_path / "A424_ordered_coordinates.npy"
    atlas.write_bytes(b"atlas")
    coordinates.write_bytes(b"ordered-coordinates")
    config = {
        "fmri_inputs": {
            "ukb424_atlas_path": str(atlas),
            "ukb424_coordinates_path": str(coordinates),
            "ds006623_timing_index_origin": 1,
        }
    }
    fingerprints, environment = verified_fmri_input_fingerprints(fmri, config)
    assert fingerprints["fmri_input:ukb424_atlas_path"]["sha256"] == sha256_file(atlas)
    assert fingerprints["fmri_input:ukb424_coordinates_path"]["sha256"] == sha256_file(coordinates)
    assert fingerprints["fmri_input:ds006623_timing_index_origin"]["value"] == 1
    assert environment == {
        "NEURAL_MANIFOLDS_UKB424_ATLAS": str(atlas),
        "NEURAL_MANIFOLDS_UKB424_COORDINATES": str(coordinates),
        "NEURAL_MANIFOLDS_DS006623_TIMING_INDEX_ORIGIN": "1",
    }

    origin_zero = {
        "fmri_inputs": {
            **config["fmri_inputs"],
            "ds006623_timing_index_origin": 0,
        }
    }
    zero_fingerprints, _ = verified_fmri_input_fingerprints(fmri, origin_zero)
    assert (
        zero_fingerprints["fmri_input:ds006623_timing_index_origin"]["sha256"]
        != fingerprints["fmri_input:ds006623_timing_index_origin"]["sha256"]
    )
