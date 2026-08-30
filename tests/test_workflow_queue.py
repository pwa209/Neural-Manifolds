from __future__ import annotations

import json
from pathlib import Path

import pytest

import workflow.queue as queue_module
from workflow.phases import PHASES
from workflow.queue import (
    bind_fmri_late_inputs,
    build_phase_command,
    establish_run_contract,
    load_fmri_input_manifest,
    prepare_clinical_lock,
    safe_repo_file,
    verified_fmri_input_fingerprints,
    verified_model_fingerprints,
    verify_deployed_source,
    verify_server_config,
)
from workflow.state import ServerRoots, atomic_write_json, sha256_file


def _write_source_manifest(path: Path, text: str) -> None:
    path.write_bytes(text.encode("utf-8"))


def test_phase_command_uses_dispatcher_contract(tmp_path: Path) -> None:
    phase = next(item for item in PHASES if item.name == "locked-clinical")
    command = build_phase_command(
        cli_prefix=("/env/bin/python", "-s", "-P", "-m", "neural_manifolds.cli"),
        phase=phase,
        study=tmp_path / "configs/study.yaml",
        datasets=tmp_path / "configs/datasets.yaml",
        server=tmp_path / "configs/server.yaml",
        run_id="main-001",
    )
    assert command[:8] == [
        "/env/bin/python",
        "-s",
        "-P",
        "-m",
        "neural_manifolds.cli",
        "run-phase",
        "--phase",
        "clinical",
    ]
    assert command[-2:] == ["--run-id", "main-001"]


def test_remote_launcher_pins_imports_to_selected_release() -> None:
    launcher = Path("scripts/remote/launch_queue.sh").read_text(encoding="utf-8")
    assert 'export PYTHONPATH="$repo_root/src:$repo_root"' in launcher
    assert '"$python_bin" -s -P -m workflow.queue' in launcher
    assert "import neural_manifolds.cli" in launcher
    assert "import workflow.queue" in launcher
    assert "cli_bin=" not in launcher


def test_check_only_runs_selected_phase_specific_checks_without_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "release"
    for relative in (
        "configs/study.yaml",
        "configs/datasets.yaml",
        "configs/models.yaml",
        "configs/server.yaml",
        "SOURCE_MANIFEST.sha256",
        "src/neural_manifolds/cli.py",
        "workflow/queue.py",
    ):
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("placeholder\n", encoding="utf-8")
    roots = ServerRoots(
        canonical=tmp_path / "canonical",
        work=tmp_path / "work",
        checkpoint=tmp_path / "checkpoint",
    )
    manifest = tmp_path / "reviewed-fmri.yaml"
    calls: list[tuple[str, str, Path | None]] = []
    source_checks: list[tuple[Path, Path]] = []

    monkeypatch.setattr(queue_module, "validate_roots", lambda **_kwargs: roots)
    monkeypatch.setattr(queue_module, "verify_remote_identity", lambda: None)
    monkeypatch.setattr(queue_module, "ensure_existing_roots", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(queue_module, "verify_server_config", lambda *_args: {})
    monkeypatch.setattr(
        queue_module,
        "verify_deployed_source",
        lambda repo_root, source_manifest: (
            source_checks.append((repo_root, source_manifest))
            or {"files": 7, "manifest_sha256": "a" * 64}
        ),
    )
    monkeypatch.setattr(
        queue_module.subprocess,
        "run",
        lambda *_args, **_kwargs: type("Result", (), {"returncode": 0})(),
    )

    def verify_fmri(phase: object, _config: object, fmri_input_manifest: Path | None = None):
        calls.append(("fmri", phase.name, fmri_input_manifest))  # type: ignore[attr-defined]
        return {}, {}

    def verify_models(phase: object):
        calls.append(("models", phase.name, None))  # type: ignore[attr-defined]
        return {}

    monkeypatch.setattr(queue_module, "verified_fmri_input_fingerprints", verify_fmri)
    monkeypatch.setattr(queue_module, "verified_model_fingerprints", verify_models)

    assert (
        queue_module.main(
            [
                "--repo-root",
                str(repo),
                "--canonical-root",
                "/private_nas/wangpeng/neural-manifolds",
                "--work-root",
                "/data1/wangpeng/neural-manifolds-work",
                "--checkpoint-root",
                "/data2/wangpeng/neural-manifolds-checkpoints",
                "--run-id",
                "main-001",
                "--only-phase",
                "fmri",
                "--fmri-input-manifest",
                str(manifest),
                "--check-only",
            ]
        )
        == 0
    )
    assert calls == [("fmri", "fmri", manifest), ("models", "fmri", None)]
    assert source_checks == [(repo, repo / "SOURCE_MANIFEST.sha256")]
    assert not roots.state_root("main-001").exists()


def _source_release(
    tmp_path: Path,
    *,
    repository: str = "https://github.com/pwa209/Neural-Manifolds.git",
    commit: str = "a" * 40,
) -> tuple[Path, Path]:
    release = tmp_path / commit
    release.mkdir()
    provenance = release / "SOURCE_PROVENANCE.json"
    provenance.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "repository": repository,
                "commit": commit,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    source = release / "src/example.py"
    source.parent.mkdir()
    source.write_text("VALUE = 1\n", encoding="utf-8")
    manifest = release / "SOURCE_MANIFEST.sha256"
    _write_source_manifest(
        manifest,
        "".join(
            (
                f"{sha256_file(provenance)}  SOURCE_PROVENANCE.json\n",
                f"{sha256_file(source)}  src/example.py\n",
            )
        ),
    )
    return release, manifest


def test_deployed_source_manifest_rehashes_files_and_validates_provenance(
    tmp_path: Path,
) -> None:
    release, manifest = _source_release(tmp_path)
    result = verify_deployed_source(release, manifest)
    assert result == {
        "repository": "https://github.com/pwa209/Neural-Manifolds.git",
        "commit": "a" * 40,
        "files": 2,
        "manifest_sha256": sha256_file(manifest),
    }


@pytest.mark.parametrize(
    "unsafe_line",
    [
        "malformed",
        f"{'0' * 64}  /absolute.py",
        f"{'0' * 64}  C:/absolute.py",
        f"{'0' * 64}  ../escape.py",
        f"{'0' * 64}  nested/../escape.py",
        f"{'0' * 64}  nested\\escape.py",
        f"{'0' * 64}  nested//normalised.py",
        f"{'0' * 64}  .",
    ],
)
def test_deployed_source_manifest_rejects_malformed_or_unsafe_paths(
    tmp_path: Path, unsafe_line: str
) -> None:
    release, manifest = _source_release(tmp_path)
    _write_source_manifest(manifest, f"{unsafe_line}\n")
    with pytest.raises(ValueError, match=r"malformed|unsafe"):
        verify_deployed_source(release, manifest)


def test_deployed_source_manifest_rejects_duplicates_missing_special_and_changed_files(
    tmp_path: Path,
) -> None:
    release, manifest = _source_release(tmp_path)
    provenance_hash = sha256_file(release / "SOURCE_PROVENANCE.json")
    _write_source_manifest(
        manifest,
        f"{provenance_hash}  SOURCE_PROVENANCE.json\n{provenance_hash}  SOURCE_PROVENANCE.json\n",
    )
    with pytest.raises(ValueError, match="duplicate"):
        verify_deployed_source(release, manifest)

    _write_source_manifest(
        manifest,
        f"{provenance_hash}  SOURCE_PROVENANCE.json\n{'0' * 64}  missing.py\n",
    )
    with pytest.raises(FileNotFoundError, match="missing"):
        verify_deployed_source(release, manifest)

    special = release / "directory"
    special.mkdir()
    _write_source_manifest(
        manifest,
        f"{provenance_hash}  SOURCE_PROVENANCE.json\n{'0' * 64}  directory\n",
    )
    with pytest.raises(ValueError, match="invalid file type"):
        verify_deployed_source(release, manifest)

    source = release / "src/example.py"
    _write_source_manifest(
        manifest,
        f"{provenance_hash}  SOURCE_PROVENANCE.json\n{'0' * 64}  src/example.py\n",
    )
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_deployed_source(release, manifest)
    assert source.is_file()


def test_deployed_source_manifest_rejects_unlisted_or_invalid_provenance(tmp_path: Path) -> None:
    release, manifest = _source_release(tmp_path)
    source = release / "src/example.py"
    _write_source_manifest(manifest, f"{sha256_file(source)}  src/example.py\n")
    with pytest.raises(ValueError, match="must include SOURCE_PROVENANCE"):
        verify_deployed_source(release, manifest)

    invalid_root = tmp_path / "invalid"
    invalid_root.mkdir()
    invalid_provenance = invalid_root / "SOURCE_PROVENANCE.json"
    invalid_provenance.write_text(
        '{"schema_version":1,"repository":"https://example.invalid/repo",'
        f'"commit":"{"b" * 40}"}}\n',
        encoding="utf-8",
    )
    invalid_manifest = invalid_root / "SOURCE_MANIFEST.sha256"
    _write_source_manifest(
        invalid_manifest,
        f"{sha256_file(invalid_provenance)}  SOURCE_PROVENANCE.json\n",
    )
    with pytest.raises(ValueError, match="unapproved repository"):
        verify_deployed_source(invalid_root, invalid_manifest)

    uppercase_root, uppercase_manifest = _source_release(tmp_path, commit="B" * 40)
    with pytest.raises(ValueError, match="exact lowercase object id"):
        verify_deployed_source(uppercase_root, uppercase_manifest)

    valid_root, valid_manifest = _source_release(tmp_path, commit="c" * 40)
    mismatched_root = tmp_path / "release-with-wrong-name"
    valid_root.rename(mismatched_root)
    with pytest.raises(ValueError, match="does not match the release directory"):
        verify_deployed_source(mismatched_root, mismatched_root / valid_manifest.name)


def test_deployed_source_manifest_rejects_symlink_entries(tmp_path: Path) -> None:
    release, manifest = _source_release(tmp_path)
    target = release / "src/example.py"
    link = release / "linked.py"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("local platform does not permit symlink creation")
    provenance = release / "SOURCE_PROVENANCE.json"
    _write_source_manifest(
        manifest,
        f"{sha256_file(provenance)}  SOURCE_PROVENANCE.json\n{sha256_file(target)}  linked.py\n",
    )
    with pytest.raises(ValueError, match="symlink"):
        verify_deployed_source(release, manifest)


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
    base_payload = json.loads(first.read_text(encoding="utf-8"))
    assert "fmri_inputs" not in base_payload
    assert "late_inputs" not in base_payload
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
    with pytest.raises(ValueError, match=r"requires --fmri-input-manifest or configs/server"):
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


def test_external_fmri_manifest_is_strict_hashed_and_unrelated_phases_ignore_it(
    tmp_path: Path,
) -> None:
    atlas = tmp_path / "UKB_424_atlas.nii.gz"
    coordinates = tmp_path / "A424_ordered_coordinates.npy"
    atlas.write_bytes(b"atlas")
    coordinates.write_bytes(b"ordered-coordinates")
    manifest = tmp_path / "reviewed-fmri-inputs.yaml"
    manifest.write_text(
        f"""
schema_version: 1
ukb424_atlas_path: {atlas.as_posix()}
ukb424_coordinates_path: {coordinates.as_posix()}
ds006623_timing_index_origin: 0
""".lstrip(),
        encoding="utf-8",
    )
    server_config = {
        "fmri_inputs": {
            "ukb424_atlas_path": None,
            "ukb424_coordinates_path": None,
            "ds006623_timing_index_origin": None,
        }
    }

    encode = next(phase for phase in PHASES if phase.name == "encode")
    missing = tmp_path / "not-yet-reviewed.yaml"
    assert verified_fmri_input_fingerprints(encode, server_config, missing) == ({}, {})

    fmri = next(phase for phase in PHASES if phase.name == "fmri")
    fingerprints, environment = verified_fmri_input_fingerprints(fmri, server_config, manifest)
    assert fingerprints["fmri_input:manifest"] == {
        "path": str(manifest),
        "sha256": sha256_file(manifest),
        "size": manifest.stat().st_size,
    }
    assert fingerprints["fmri_input:ukb424_atlas_path"]["sha256"] == sha256_file(atlas)
    assert fingerprints["fmri_input:ukb424_coordinates_path"]["sha256"] == sha256_file(coordinates)
    assert environment["NEURAL_MANIFOLDS_DS006623_TIMING_INDEX_ORIGIN"] == "0"

    ambiguous = {
        "fmri_inputs": {
            **server_config["fmri_inputs"],
            "ds006623_timing_index_origin": 0,
        }
    }
    with pytest.raises(ValueError, match="ambiguous fMRI inputs"):
        verified_fmri_input_fingerprints(fmri, ambiguous, manifest)


def test_fmri_manifest_rejects_relative_unknown_and_duplicate_keys(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="absolute path"):
        load_fmri_input_manifest(Path("relative.yaml"))

    unknown = tmp_path / "unknown.json"
    unknown.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "ukb424_atlas_path": "/atlas",
                "ukb424_coordinates_path": "/coordinates",
                "ds006623_timing_index_origin": 0,
                "typo": True,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"unknown=\['typo'\]"):
        load_fmri_input_manifest(unknown)

    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(
        '{"schema_version":1,"schema_version":1,"ukb424_atlas_path":"/atlas",'
        '"ukb424_coordinates_path":"/coordinates",'
        '"ds006623_timing_index_origin":0}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate key"):
        load_fmri_input_manifest(duplicate)


def test_fmri_late_input_contract_is_atomic_idempotent_and_run_bound(tmp_path: Path) -> None:
    fingerprints = {
        "fmri_input:manifest": {
            "path": "/checkpoint/metadata/fmri.yaml",
            "sha256": "a" * 64,
            "size": 100,
        },
        "fmri_input:ukb424_atlas_path": {
            "path": "/canonical/metadata/atlas.nii.gz",
            "sha256": "b" * 64,
            "size": 200,
        },
    }
    environment = {
        "NEURAL_MANIFOLDS_UKB424_ATLAS": "/canonical/metadata/atlas.nii.gz",
        "NEURAL_MANIFOLDS_DS006623_TIMING_INDEX_ORIGIN": "1",
    }
    contract = bind_fmri_late_inputs(
        state_root=tmp_path,
        run_id="main-001",
        fingerprints=fingerprints,
        environment=environment,
    )
    assert contract == tmp_path / "late-inputs/fmri.json"
    first_hash = sha256_file(contract)
    assert (
        bind_fmri_late_inputs(
            state_root=tmp_path,
            run_id="main-001",
            fingerprints=fingerprints,
            environment=environment,
        )
        == contract
    )
    assert sha256_file(contract) == first_hash

    changed = {
        **fingerprints,
        "fmri_input:manifest": {**fingerprints["fmri_input:manifest"], "sha256": "c" * 64},
    }
    with pytest.raises(RuntimeError, match="use a new run id"):
        bind_fmri_late_inputs(
            state_root=tmp_path,
            run_id="main-001",
            fingerprints=changed,
            environment=environment,
        )

    payload = json.loads(contract.read_text(encoding="utf-8"))
    payload["environment"]["NEURAL_MANIFOLDS_DS006623_TIMING_INDEX_ORIGIN"] = "0"
    contract.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="invalid or changed"):
        bind_fmri_late_inputs(
            state_root=tmp_path,
            run_id="main-001",
            fingerprints=fingerprints,
            environment=environment,
        )
