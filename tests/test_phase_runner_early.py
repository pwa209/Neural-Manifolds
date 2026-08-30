from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

import neural_manifolds.phase_runner as phase_runner
from neural_manifolds.data.acquisition import AcquisitionResult
from neural_manifolds.data.providers import AccessBlocked, ProviderError
from neural_manifolds.phase_runner import PhaseContext, run_acquire, run_phase
from neural_manifolds.provenance import sha256_file

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _context(
    tmp_path: Path,
    *,
    phase: str,
    study_path: Path | None = None,
    datasets_path: Path | None = None,
) -> PhaseContext:
    canonical = tmp_path / "canonical"
    raw = canonical / "raw"
    work = tmp_path / "work"
    checkpoint = tmp_path / "checkpoint"
    run_root = work / "runs" / "test-run"
    state_root = checkpoint / "runs" / "test-run"
    for path in (canonical, raw, work, checkpoint, run_root, state_root):
        path.mkdir(parents=True, exist_ok=True)
    server_path = tmp_path / "server.yaml"
    server_path.write_text(
        yaml.safe_dump(
            {
                "storage": {
                    "canonical_root": str(canonical),
                    "work_root": str(work),
                    "checkpoint_root": str(checkpoint),
                }
            }
        ),
        encoding="utf-8",
    )
    return PhaseContext(
        phase=phase,
        run_id="test-run",
        study_path=study_path or REPOSITORY_ROOT / "configs" / "study.yaml",
        datasets_path=datasets_path or REPOSITORY_ROOT / "configs" / "datasets.yaml",
        server_path=server_path,
        canonical_root=canonical,
        raw_root=raw,
        work_root=work,
        checkpoint_root=checkpoint,
        run_root=run_root,
        state_root=state_root,
        receipt_path=state_root / f"{phase}-receipt.json",
    )


def _archive_source_release(root: Path) -> str:
    commit = "b" * 40
    provenance = root / "SOURCE_PROVENANCE.json"
    provenance.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "repository": "https://github.com/pwa209/Neural-Manifolds.git",
                "commit": commit,
                "transport": {
                    "type": "server_local_git_archive",
                    "archive_path": "/tmp/source.tar",
                    "archive_sha256": "c" * 64,
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "SOURCE_MANIFEST.sha256").write_text(
        f"{sha256_file(provenance)}  SOURCE_PROVENANCE.json\n",
        encoding="utf-8",
    )
    return commit


def test_audit_publishes_explicit_tables_issue_ledger_and_archive_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    commit = _archive_source_release(source)
    monkeypatch.chdir(source)
    context = _context(tmp_path, phase="audit")

    artifacts = run_phase(context)

    assert {path.name for path in artifacts} == {
        "audit.json",
        "dataset-release-audit.json",
        "model-audit.json",
        "issue-ledger.json",
    }
    audit = json.loads((context.run_root / "audit" / "audit.json").read_text(encoding="utf-8"))
    assert audit["deployed_source"]["commit"] == commit
    assert audit["source_revision"] == commit
    assert audit["deployed_source"]["transport"]["type"] == "server_local_git_archive"
    assert set(audit["config_fingerprints"]) == {"study", "datasets", "models", "server"}
    dataset_table = json.loads(
        (context.run_root / "audit" / "dataset-release-audit.json").read_text(encoding="utf-8")
    )
    assert dataset_table["table"] == "dataset_release_licence_access"
    cogitate = next(row for row in dataset_table["rows"] if row["dataset_id"] == "cogitate_meeg")
    assert cogitate["release"]["version"] == "2024-doi-bundle"
    assert cogitate["licence"]["spdx"] == "CC-BY-4.0"
    assert cogitate["access"]["mode"] == "account_required"
    model_table = json.loads(
        (context.run_root / "audit" / "model-audit.json").read_text(encoding="utf-8")
    )
    assert model_table["table"] == "model_revision_weight_licence_pretraining_overlap"
    assert all(row["source_revision"] for row in model_table["rows"])
    assert all("weights" in row and "licence" in row for row in model_table["rows"])
    assert all(
        row["pretraining_overlap_audit"]["status"] == "unresolved" for row in model_table["rows"]
    )
    issues = json.loads(
        (context.run_root / "audit" / "issue-ledger.json").read_text(encoding="utf-8")
    )
    assert any(item["status"] == "access_blocked" for item in issues["issues"])
    assert any(item["category"] == "pretraining_overlap" for item in issues["issues"])
    receipt = json.loads(context.receipt_path.read_text(encoding="utf-8"))
    assert len(receipt["artifacts"]) == 4


def _dataset(dataset_id: str, *, access_mode: str = "open") -> Any:
    return SimpleNamespace(
        id=dataset_id,
        source=SimpleNamespace(
            version="1",
            landing_url=f"https://example.test/{dataset_id}",
        ),
        access=SimpleNamespace(
            mode=access_mode,
            instructions=("Obtain official access." if access_mode != "open" else None),
        ),
    )


def _write_release_records(raw_root: Path, dataset: Any) -> Path:
    release = raw_root / dataset.id / dataset.source.version
    acquisition = release / ".acquisition"
    acquisition.mkdir(parents=True, exist_ok=True)
    for relative, content in (
        (".acquisition/manifest.json", '{"manifest":true}\n'),
        (".acquisition/MANIFEST.sha256", "abc  3  raw.bin\n"),
        (".acquisition/provenance.json", '{"source":"test"}\n'),
        (".acquisition/COMPLETE.json", '{"immutable":true}\n'),
    ):
        (release / relative).write_text(content, encoding="utf-8")
    return release


def _remove_release_write_bits(release: Path) -> None:
    for path in sorted(release.rglob("*"), reverse=True):
        if not path.is_symlink():
            path.chmod(path.stat().st_mode & ~0o222)
    release.chmod(release.stat().st_mode & ~0o222)


def _restore_release_owner_write(release: Path) -> None:
    for path in sorted(release.rglob("*"), reverse=True):
        if not path.is_symlink():
            path.chmod(path.stat().st_mode | 0o200)
    release.chmod(release.stat().st_mode | 0o200)


def test_acquire_attempts_every_open_dataset_and_keeps_scoped_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    datasets = [
        _dataset("success"),
        _dataset("upstream_down"),
        _dataset("integrity_failure"),
        _dataset("new_access_block"),
        _dataset("restricted", access_mode="account_required"),
    ]
    registry = SimpleNamespace(model=SimpleNamespace(datasets=datasets))
    calls: list[str] = []

    class FakeManager:
        def __init__(self, _registry: Any) -> None:
            pass

        def acquire(self, dataset: Any, raw_root: Path) -> AcquisitionResult:
            calls.append(dataset.id)
            if dataset.id == "upstream_down":
                raise ProviderError("endpoint unavailable")
            if dataset.id == "integrity_failure":
                raise RuntimeError("checksum mismatch")
            if dataset.id == "new_access_block":
                raise AccessBlocked("official route now requires approval")
            release = _write_release_records(raw_root, dataset)
            return AcquisitionResult(dataset.id, "1", "published", str(release), {})

    monkeypatch.setattr(phase_runner, "load_dataset_registry", lambda _path: registry)
    monkeypatch.setattr(phase_runner, "AcquisitionManager", FakeManager)
    context = _context(tmp_path, phase="acquire")

    with pytest.raises(RuntimeError, match="one or more open acquisitions failed"):
        run_acquire(context)

    assert calls == ["success", "upstream_down", "integrity_failure", "new_access_block"]
    summary_path = context.canonical_root / "provenance" / "runs" / "test-run"
    summary = json.loads((summary_path / "acquisition-summary.json").read_text(encoding="utf-8"))
    by_id = {result["dataset_id"]: result for result in summary["results"]}
    assert by_id["success"]["status"] == "published"
    assert by_id["upstream_down"]["status"] == "unavailable"
    assert by_id["integrity_failure"]["status"] == "failure"
    assert by_id["new_access_block"]["status"] == "access_blocked"
    assert by_id["restricted"]["status"] == "access_blocked"
    assert {failure["dataset_id"] for failure in summary["failures"]} == {
        "upstream_down",
        "integrity_failure",
    }
    assert len(by_id["success"]["release_records"]) == 4


def test_acquisition_receipt_fingerprints_every_existing_release_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    datasets = [_dataset("success"), _dataset("restricted", access_mode="account_required")]
    registry = SimpleNamespace(model=SimpleNamespace(datasets=datasets))

    class FakeManager:
        def __init__(self, _registry: Any) -> None:
            pass

        def acquire(self, dataset: Any, raw_root: Path) -> AcquisitionResult:
            release = _write_release_records(raw_root, dataset)
            _remove_release_write_bits(release)
            return AcquisitionResult(dataset.id, "1", "published", str(release), {})

    monkeypatch.setattr(phase_runner, "load_dataset_registry", lambda _path: registry)
    monkeypatch.setattr(phase_runner, "AcquisitionManager", FakeManager)
    context = _context(tmp_path, phase="acquire")

    artifacts = run_phase(context)

    assert len(artifacts) == 5
    summary = json.loads(artifacts[0].read_text(encoding="utf-8"))
    assert summary["status"] == "complete_with_access_blocks"
    assert {record["record_type"] for record in summary["release_records"]} == {
        "manifest",
        "checksum_inventory",
        "provenance",
        "completion",
    }
    receipt = json.loads(context.receipt_path.read_text(encoding="utf-8"))
    assert len(receipt["artifacts"]) == 5
    receipt_by_path = {item["path"]: item for item in receipt["artifacts"]}
    for record in summary["release_records"]:
        assert receipt_by_path[record["path"]]["sha256"] == record["sha256"]
        assert receipt_by_path[record["path"]]["size"] == record["size"]
        assert Path(record["path"]).stat().st_mode & 0o222 == 0
    _restore_release_owner_write(context.raw_root / "success" / "1")


def test_successful_acquisition_missing_a_release_record_is_a_scoped_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = _dataset("raced_release")
    registry = SimpleNamespace(model=SimpleNamespace(datasets=[dataset]))

    class FakeManager:
        def __init__(self, _registry: Any) -> None:
            pass

        def acquire(self, item: Any, raw_root: Path) -> AcquisitionResult:
            release = raw_root / item.id / item.source.version
            acquisition = release / ".acquisition"
            acquisition.mkdir(parents=True)
            (acquisition / "manifest.json").write_text("{}\n", encoding="utf-8")
            return AcquisitionResult(item.id, "1", "published", str(release), {})

    monkeypatch.setattr(phase_runner, "load_dataset_registry", lambda _path: registry)
    monkeypatch.setattr(phase_runner, "AcquisitionManager", FakeManager)
    context = _context(tmp_path, phase="acquire")

    with pytest.raises(RuntimeError, match="one or more open acquisitions failed"):
        run_acquire(context)

    summary = json.loads(
        (
            context.canonical_root / "provenance" / "runs" / "test-run" / "acquisition-summary.json"
        ).read_text(encoding="utf-8")
    )
    result = summary["results"][0]
    assert result["status"] == "failure"
    assert result["status_before_release_record_audit"] == "published"
    assert {issue["record_type"] for issue in result["release_record_issues"]} == {
        "checksum_inventory",
        "provenance",
        "completion",
    }
    assert summary["failures"][0]["dataset_id"] == "raced_release"
