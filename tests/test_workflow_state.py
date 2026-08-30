from __future__ import annotations

import json
from pathlib import Path

import pytest

from workflow.state import (
    ServerRoots,
    atomic_write_json,
    phase_hash,
    sha256_file,
    validate_receipt,
    validate_roots,
    validate_run_id,
    validate_success_marker,
)


def test_explicit_project_roots_are_required() -> None:
    roots = validate_roots(
        canonical_root="/private_nas/wangpeng/neural-manifolds",
        work_root="/data1/wangpeng/neural-manifolds-work",
        checkpoint_root="/data2/wangpeng/neural-manifolds-checkpoint",
    )
    assert roots.raw.as_posix() == "/private_nas/wangpeng/neural-manifolds/raw"

    bad_values = (
        {
            "canonical_root": "/private_nas/wangpeng",
            "work_root": "/data1/wangpeng/neural-manifolds-work",
            "checkpoint_root": "/data2/wangpeng/neural-manifolds-checkpoint",
        },
        {
            "canonical_root": "/private_nas/wangpeng/guessed/nested",
            "work_root": "/data1/wangpeng/neural-manifolds-work",
            "checkpoint_root": "/data2/wangpeng/neural-manifolds-checkpoint",
        },
        {
            "canonical_root": "relative/path",
            "work_root": "/data1/wangpeng/neural-manifolds-work",
            "checkpoint_root": "/data2/wangpeng/neural-manifolds-checkpoint",
        },
    )
    for values in bad_values:
        with pytest.raises(ValueError):
            validate_roots(**values)


def test_run_id_rejects_shell_metacharacters() -> None:
    assert validate_run_id("main-2026-08-30") == "main-2026-08-30"
    with pytest.raises(ValueError):
        validate_run_id("run;touch-pwned")


def _temporary_roots(tmp_path: Path) -> ServerRoots:
    canonical = tmp_path / "canonical"
    work = tmp_path / "work"
    checkpoint = tmp_path / "checkpoint"
    for path in (canonical / "raw", work, checkpoint):
        path.mkdir(parents=True)
    return ServerRoots(canonical=canonical, work=work, checkpoint=checkpoint)


def test_receipt_rehashes_artifacts_and_enforces_storage(tmp_path: Path) -> None:
    roots = _temporary_roots(tmp_path)
    artifact = roots.work / "run" / "metrics.manifest.json"
    artifact.parent.mkdir()
    artifact.write_text('{"rows": 7}\n', encoding="utf-8")
    receipt_path = roots.checkpoint / "receipt.json"
    atomic_write_json(
        receipt_path,
        {
            "schema_version": 1,
            "phase": "metrics",
            "run_id": "main-001",
            "artifacts": [
                {
                    "path": str(artifact),
                    "sha256": sha256_file(artifact),
                    "size": artifact.stat().st_size,
                }
            ],
        },
    )
    receipt = validate_receipt(
        receipt_path,
        expected_cli_phase="metrics",
        expected_run_id="main-001",
        workflow_phase="metrics",
        roots=roots,
    )
    assert receipt["artifacts"][0]["path"] == str(artifact)

    artifact.write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="size/hash"):
        validate_receipt(
            receipt_path,
            expected_cli_phase="metrics",
            expected_run_id="main-001",
            workflow_phase="metrics",
            roots=roots,
        )


def test_acquisition_receipt_cannot_point_to_fast_work(tmp_path: Path) -> None:
    roots = _temporary_roots(tmp_path)
    misplaced = roots.work / "raw-file.manifest"
    misplaced.write_text("bad location\n", encoding="utf-8")
    receipt_path = roots.checkpoint / "acquire-receipt.json"
    atomic_write_json(
        receipt_path,
        {
            "schema_version": 1,
            "phase": "acquire",
            "run_id": "main-001",
            "artifacts": [
                {
                    "path": str(misplaced),
                    "sha256": sha256_file(misplaced),
                    "size": misplaced.stat().st_size,
                }
            ],
        },
    )
    with pytest.raises(ValueError, match="outside canonical"):
        validate_receipt(
            receipt_path,
            expected_cli_phase="acquire",
            expected_run_id="main-001",
            workflow_phase="acquire",
            roots=roots,
        )


def test_phase_hash_is_stable_and_config_sensitive(tmp_path: Path) -> None:
    roots = _temporary_roots(tmp_path)
    kwargs = {
        "phase_name": "encode",
        "command": ["neural-manifolds", "run-phase", "--phase", "encode"],
        "source_manifest_sha256": "a" * 64,
        "config_fingerprints": {"study": {"sha256": "b" * 64, "size": 1}},
        "dependency_marker_sha256": {"preprocess": "c" * 64},
        "roots": roots,
    }
    first = phase_hash(**kwargs)
    assert first == phase_hash(**kwargs)
    kwargs["config_fingerprints"] = {"study": {"sha256": "d" * 64, "size": 1}}
    assert phase_hash(**kwargs) != first


def test_success_marker_detects_changed_artifact(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.json"
    artifact.write_text("{}\n", encoding="utf-8")
    marker = {
        "schema_version": 1,
        "status": "succeeded",
        "phase_hash": "e" * 64,
        "artifacts": [
            {
                "path": str(artifact),
                "sha256": sha256_file(artifact),
                "size": artifact.stat().st_size,
            }
        ],
    }
    validate_success_marker(marker, expected_phase_hash="e" * 64)
    artifact.write_text(json.dumps({"changed": True}), encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        validate_success_marker(marker, expected_phase_hash="e" * 64)
