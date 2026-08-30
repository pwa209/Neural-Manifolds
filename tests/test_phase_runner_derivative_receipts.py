from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from neural_manifolds import phase_runner
from neural_manifolds.config import load_study
from neural_manifolds.phase_runner import PhaseContext, run_phase
from neural_manifolds.provenance import sha256_file
from workflow.state import ServerRoots, validate_receipt

ROOT = Path(__file__).resolve().parents[1]


def _context(tmp_path: Path, phase: str) -> tuple[PhaseContext, ServerRoots]:
    canonical = tmp_path / "canonical"
    work = tmp_path / "work"
    checkpoint = tmp_path / "checkpoint"
    raw = canonical / "raw"
    run_root = work / "run-001"
    state_root = checkpoint / "run-001"
    for path in (raw, run_root, state_root):
        path.mkdir(parents=True)
    config = tmp_path / "config.yaml"
    config.write_text("schema_version: 1\n", encoding="utf-8")
    context = PhaseContext(
        phase=phase,
        run_id="run-001",
        study_path=config,
        datasets_path=config,
        server_path=config,
        canonical_root=canonical,
        raw_root=raw,
        work_root=work,
        checkpoint_root=checkpoint,
        run_root=run_root,
        state_root=state_root,
        receipt_path=state_root / f"{phase}-receipt.json",
    )
    return context, ServerRoots(canonical=canonical, work=work, checkpoint=checkpoint)


def _write(path: Path, payload: bytes = b"artifact") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _install_phase_fakes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(phase_runner, "load_study", lambda path: object())

    def cohort(*, output_root: str | Path, **kwargs: Any) -> tuple[Path, Path, Path]:
        del kwargs
        output = Path(output_root)
        output.mkdir(parents=True, exist_ok=True)
        labels = output / "cohort-labels.parquet"
        pd.DataFrame([{"unit_id": "unit-1", "dataset_id": "doc_polysomnography"}]).to_parquet(
            labels, index=False
        )
        inputs = output / "encoder-inputs.parquet"
        pd.DataFrame([{"unit_id": "unit-1"}]).to_parquet(inputs, index=False)
        issues = _write(output / "cohort-issues.json", b"{}\n")
        return labels, inputs, issues

    def preprocess(*, output_root: str | Path, **kwargs: Any) -> tuple[Path, Path]:
        del kwargs
        output = Path(output_root)
        manifest = _write(output / "preprocessing-manifest.parquet")
        flow = _write(output / "preprocessing-flow.json", b"{}\n")
        _write(output / "source-recordings" / "source-raw.fif")
        _write(output / "source-recordings" / "source-raw.provenance.json", b"{}\n")
        _write(output / "units" / "unit-1-raw.fif")
        _write(output / "provenance" / "unit-1.json", b"{}\n")
        return manifest, flow

    def encode(*, output_root: str | Path, **kwargs: Any) -> tuple[Path, Path]:
        del kwargs
        output = Path(output_root)
        manifest = _write(output / "encoding-manifest.parquet")
        flow = _write(output / "encoding-flow.json", b"{}\n")
        _write(output / "trajectories" / "unit-1.npz")
        _write(output / "provenance" / "unit-1.encoding.json", b"{}\n")
        return manifest, flow

    def clinical_transfer(*, output_root: str | Path, **kwargs: Any) -> tuple[Path]:
        del kwargs
        return (_write(Path(output_root) / "clinical-transfer.json", b"{}\n"),)

    def inventory_subset(
        _source: str | Path,
        destination: str | Path,
        **_kwargs: Any,
    ) -> Path:
        output = Path(destination)
        output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [{"recording_id": "clinical-1", "dataset_id": "doc_polysomnography"}]
        ).to_parquet(output, index=False)
        return output

    def signal_qc(*, output_root: str | Path, **_kwargs: Any) -> tuple[Path, Path, Path]:
        output = Path(output_root)
        return (
            _write(output / "recording-flow.parquet"),
            _write(output / "channel-qc.parquet"),
            _write(output / "signal-qc-audit.json", b"{}\n"),
        )

    monkeypatch.setattr(phase_runner, "build_cohort_manifest", cohort)
    monkeypatch.setattr(phase_runner, "preprocess_analysis_units", preprocess)
    monkeypatch.setattr(phase_runner, "encode_analysis_units", encode)
    monkeypatch.setattr(phase_runner, "run_clinical_transfer", clinical_transfer)
    monkeypatch.setattr(phase_runner, "_write_inventory_subset", inventory_subset)
    monkeypatch.setattr(phase_runner, "run_signal_qc", signal_qc)
    monkeypatch.setattr(phase_runner, "validate_clinical_lock", lambda _path: {})


@pytest.mark.parametrize(
    ("phase", "mutated_relative", "manifest_relative"),
    [
        (
            "preprocess",
            Path("preprocess/units/unit-1-raw.fif"),
            Path("preprocess/preprocessing-manifest.parquet"),
        ),
        (
            "encode",
            Path("encode/trajectories/unit-1.npz"),
            Path("encode/encoding-manifest.parquet"),
        ),
        (
            "clinical",
            Path("clinical/encode/trajectories/unit-1.npz"),
            Path("clinical/encode/encoding-manifest.parquet"),
        ),
    ],
)
def test_phase_receipt_rehashes_nested_derivatives_when_manifest_is_unchanged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
    mutated_relative: Path,
    manifest_relative: Path,
) -> None:
    context, roots = _context(tmp_path, phase)
    _install_phase_fakes(monkeypatch)
    if phase == "encode":
        _write(context.run_root / "preprocess" / "preprocessing-manifest.parquet")
        _write(context.run_root / "preprocess" / "cohort-labels.parquet")
    if phase == "clinical":
        lock = _write(tmp_path / "clinical.lock", b"{}\n")
        _write(context.run_root / "qc" / "recordings.parquet")
        monkeypatch.setenv("NEURAL_MANIFOLDS_CLINICAL_LOCK", str(lock))

    run_phase(context)
    receipt = json.loads(context.receipt_path.read_text(encoding="utf-8"))
    artifact_paths = {item["path"] for item in receipt["artifacts"]}
    mutated = context.run_root / mutated_relative
    manifest = context.run_root / manifest_relative
    assert str(mutated.resolve(strict=True)) in artifact_paths
    manifest_hash = sha256_file(manifest)

    mutated.write_bytes(mutated.read_bytes() + b"-tampered")
    assert sha256_file(manifest) == manifest_hash
    with pytest.raises(ValueError, match="size/hash validation"):
        validate_receipt(
            context.receipt_path,
            expected_cli_phase=phase,
            expected_run_id=context.run_id,
            workflow_phase=phase,
            roots=roots,
        )


@pytest.mark.parametrize(
    ("configured_status", "issue_status"),
    [
        ("unresolved", "unresolved"),
        ("confirmed_overlap", "confirmed_overlap"),
        ("verified_no_overlap", None),
    ],
)
def test_explicit_model_overlap_status_remains_visible_as_an_analysis_control(
    configured_status: str,
    issue_status: str | None,
) -> None:
    rows, issues = phase_runner._model_audit_table(
        {
            "models": {
                "frozen-model": {
                    "trainable": False,
                    "pretraining_overlap_audit": {
                        "status": configured_status,
                        "target_dataset_ids": ["target"],
                        "evidence": None,
                        "control": "frozen_weights_no_study_finetuning",
                        "limitation": "Corpus membership cannot establish a zero-shot claim.",
                    },
                }
            }
        },
        study=load_study(ROOT / "configs" / "study.yaml"),
        dataset_ids=["target"],
    )
    assert rows[0]["pretraining_overlap_audit"]["status"] == configured_status
    overlap_issues = [item for item in issues if item["category"] == "pretraining_overlap"]
    if issue_status is None:
        assert overlap_issues == []
    else:
        assert len(overlap_issues) == 1
        assert overlap_issues[0]["status"] == issue_status
        assert overlap_issues[0]["technical_gate"] is False
        assert overlap_issues[0]["control"] == "frozen_weights_no_study_finetuning"
        assert overlap_issues[0]["limitation"].startswith("Corpus membership")
