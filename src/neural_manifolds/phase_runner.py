"""Scientific phase entry points consumed by the durable server queue."""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from neural_manifolds.cohort import build_cohort_manifest
from neural_manifolds.config import config_sha256, load_study, load_yaml
from neural_manifolds.data.acquisition import AcquisitionManager
from neural_manifolds.data.manifest import (
    COMPLETION_MARKER,
    MANIFEST_JSON,
    MANIFEST_SHA256,
    PROVENANCE_JSON,
)
from neural_manifolds.data.providers import AccessBlocked, ProviderError
from neural_manifolds.data.registry import load_dataset_registry
from neural_manifolds.inventory import scan_recordings, write_inventory
from neural_manifolds.provenance import atomic_write_json, sha256_file
from neural_manifolds.stage_units import encode_analysis_units, preprocess_analysis_units
from neural_manifolds.stages.benchmarks import run_benchmarks
from neural_manifolds.stages.channel_permutation import (
    combine_null_profile_tables,
    run_preencoder_channel_permutation_control,
)
from neural_manifolds.stages.clinical import run_clinical_transfer
from neural_manifolds.stages.fmri_driver import run_ds006623_brainlm_stage
from neural_manifolds.stages.metrics import run_metrics
from neural_manifolds.stages.models import run_models
from neural_manifolds.stages.sampling import run_sampling_sensitivity
from neural_manifolds.stages.tms import build_tms_epoch_manifest, run_tms_validation

PHASES = {
    "audit",
    "acquire",
    "qc",
    "preprocess",
    "encode",
    "metrics",
    "models",
    "tms",
    "clinical",
    "fmri",
    "figures",
}

HEALTHY_DATASETS = (
    "propofol_tms_eeg",
    "dream_tononi_serial_awakenings",
    "tactile_detection",
    "somatosensory_report_task",
    "cogitate_meeg",
    "psiconnect",
)
CLINICAL_DATASETS = ("doc_resting_eeg", "doc_polysomnography")


@dataclass(frozen=True)
class PhaseContext:
    phase: str
    run_id: str
    study_path: Path
    datasets_path: Path
    server_path: Path
    canonical_root: Path
    raw_root: Path
    work_root: Path
    checkpoint_root: Path
    run_root: Path
    state_root: Path
    receipt_path: Path

    @classmethod
    def from_environment(
        cls,
        *,
        phase: str,
        run_id: str,
        study_path: str | Path,
        datasets_path: str | Path,
        server_path: str | Path,
    ) -> PhaseContext:
        if phase not in PHASES:
            raise ValueError(f"unknown phase {phase!r}")

        def required_path(name: str) -> Path:
            value = os.environ.get(name)
            if not value:
                raise RuntimeError(f"server queue did not provide {name}")
            path = Path(value)
            if not path.is_absolute():
                raise ValueError(f"{name} must be absolute")
            return path

        return cls(
            phase=phase,
            run_id=run_id,
            study_path=Path(study_path).resolve(strict=True),
            datasets_path=Path(datasets_path).resolve(strict=True),
            server_path=Path(server_path).resolve(strict=True),
            canonical_root=required_path("NEURAL_MANIFOLDS_CANONICAL_ROOT"),
            raw_root=required_path("NEURAL_MANIFOLDS_RAW_ROOT"),
            work_root=required_path("NEURAL_MANIFOLDS_WORK_ROOT"),
            checkpoint_root=required_path("NEURAL_MANIFOLDS_CHECKPOINT_ROOT"),
            run_root=required_path("NEURAL_MANIFOLDS_RUN_ROOT"),
            state_root=required_path("NEURAL_MANIFOLDS_STATE_ROOT"),
            receipt_path=required_path("NEURAL_MANIFOLDS_PHASE_RECEIPT"),
        )

    def output_directory(self) -> Path:
        if self.phase == "acquire":
            return self.canonical_root / "provenance" / "runs" / self.run_id
        return self.run_root / self.phase


def _command_output(argv: list[str]) -> dict[str, Any]:
    try:
        result = subprocess.run(argv, check=False, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"argv": argv, "error": str(error)}
    return {
        "argv": argv,
        "exit_code": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def _artifact(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"phase artifact is not a file: {resolved}")
    return {"path": str(resolved), "sha256": sha256_file(resolved), "size": resolved.stat().st_size}


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read valid {label} JSON: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return payload


def _source_deployment(repo_root: Path) -> dict[str, Any]:
    provenance_path = repo_root / "SOURCE_PROVENANCE.json"
    manifest_path = repo_root / "SOURCE_MANIFEST.sha256"
    for path in (provenance_path, manifest_path):
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"deployed source metadata is missing or unsafe: {path}")
    provenance = _load_json_object(provenance_path, label="source provenance")
    commit = provenance.get("commit")
    if not isinstance(commit, str) or re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", commit) is None:
        raise ValueError("SOURCE_PROVENANCE.json lacks an exact lowercase deployed commit")
    repository = provenance.get("repository")
    if not isinstance(repository, str) or not repository.startswith("https://github.com/"):
        raise ValueError("SOURCE_PROVENANCE.json lacks an HTTPS GitHub repository")
    provenance_sha256 = sha256_file(provenance_path)
    expected_binding = f"{provenance_sha256}  SOURCE_PROVENANCE.json"
    try:
        manifest_lines = manifest_path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ValueError(f"cannot read deployed source manifest: {manifest_path}") from error
    matching = [line for line in manifest_lines if line.endswith("  SOURCE_PROVENANCE.json")]
    if matching != [expected_binding]:
        raise ValueError("source manifest does not exactly bind SOURCE_PROVENANCE.json")
    transport = provenance.get("transport")
    if transport is not None and not isinstance(transport, dict):
        raise ValueError("source provenance transport must be an object")
    if isinstance(transport, dict) and transport.get("type") == "server_local_git_archive":
        archive_sha256 = transport.get("archive_sha256")
        if (
            not isinstance(archive_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", archive_sha256) is None
        ):
            raise ValueError("archive deployment provenance lacks an exact archive SHA-256")
    return {
        "repository": repository,
        "commit": commit,
        "transport": transport or {"type": "git_fetch"},
        "source_provenance": _artifact(provenance_path),
        "source_manifest": _artifact(manifest_path),
        "source_manifest_entry_count": len(manifest_lines),
    }


def _dataset_audit_table(registry: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    for dataset in registry.model.datasets:
        rows.append(
            {
                "dataset_id": dataset.id,
                "title": dataset.title,
                "role": dataset.role,
                "modalities": list(dataset.modalities),
                "release": {
                    "provider": dataset.source.provider,
                    "accession": dataset.source.accession,
                    "version": dataset.source.version,
                    "revision": dataset.source.revision,
                    "doi": dataset.source.doi,
                    "landing_url": dataset.source.landing_url,
                    "repository_url": dataset.source.repository_url,
                    "api_url": dataset.source.api_url,
                    "mutable_upstream": dataset.source.mutable_upstream,
                },
                "licence": {
                    "spdx": dataset.license.spdx,
                    "status": dataset.license.status,
                    "source_url": dataset.license.source_url,
                    "note": dataset.license.note,
                },
                "access": {
                    "mode": dataset.access.mode,
                    "terms_url": dataset.access.terms_url,
                    "instructions": dataset.access.instructions,
                },
                "validation_contract": {
                    "required_paths": list(dataset.validation.required_paths),
                    "required_globs": list(dataset.validation.required_globs),
                    "minimum_files": dataset.validation.minimum_files,
                    "minimum_bytes": dataset.validation.minimum_bytes,
                },
                "official_sources": list(dataset.official_sources),
            }
        )
        if dataset.access.mode != "open":
            issues.append(
                {
                    "issue_id": f"dataset:{dataset.id}:access",
                    "scope": "dataset",
                    "subject_id": dataset.id,
                    "category": "access",
                    "status": "access_blocked",
                    "severity": "dataset_blocker",
                    "technical_gate": False,
                    "message": "Official access requires account-mediated or manual action.",
                    "required_action": dataset.access.instructions,
                }
            )
        if dataset.license.status != "verified":
            issues.append(
                {
                    "issue_id": f"dataset:{dataset.id}:licence",
                    "scope": "dataset",
                    "subject_id": dataset.id,
                    "category": "licence",
                    "status": dataset.license.status,
                    "severity": "redistribution_blocker",
                    "technical_gate": False,
                    "message": dataset.license.note
                    or "Dataset-level reuse terms are not fully resolved.",
                    "required_action": "Retain locally; do not redistribute until clarified.",
                }
            )
        if dataset.source.mutable_upstream:
            issues.append(
                {
                    "issue_id": f"dataset:{dataset.id}:mutable-upstream",
                    "scope": "dataset",
                    "subject_id": dataset.id,
                    "category": "release_identity",
                    "status": "freeze_at_first_acquisition",
                    "severity": "warning",
                    "technical_gate": False,
                    "message": "The upstream record is mutable despite the local release label.",
                    "required_action": "Freeze and hash the complete first acquired inventory.",
                }
            )
    return rows, issues


def _model_audit_table(
    models_config: Mapping[str, Any],
    *,
    study: Any,
    dataset_ids: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    models = models_config.get("models")
    if not isinstance(models, dict) or not models:
        raise ValueError("models configuration must contain a nonempty models mapping")
    rows: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    for model_id, raw_spec in sorted(models.items()):
        if not isinstance(model_id, str) or not isinstance(raw_spec, dict):
            raise ValueError("each model audit entry must be a named mapping")
        checkpoint_files = raw_spec.get("checkpoint_files")
        weights: list[dict[str, Any]] = []
        if isinstance(checkpoint_files, list):
            for item in checkpoint_files:
                if not isinstance(item, dict):
                    raise ValueError(f"model {model_id} has a malformed checkpoint file")
                weights.append(
                    {
                        "name": item.get("name"),
                        "url": item.get("url"),
                        "sha256": item.get("sha256"),
                        "size": item.get("size"),
                    }
                )
        elif raw_spec.get("checkpoint_url"):
            weights.append(
                {
                    "name": Path(str(raw_spec["checkpoint_url"])).name,
                    "url": raw_spec.get("checkpoint_url"),
                    "sha256": raw_spec.get("checkpoint_sha256"),
                    "git_blob_sha1": raw_spec.get("checkpoint_git_blob_sha1"),
                    "size": raw_spec.get("checkpoint_size"),
                }
            )
        overlap = raw_spec.get("pretraining_overlap_audit")
        if isinstance(overlap, dict):
            overlap_record = dict(overlap)
        else:
            overlap_record = {
                "status": "unresolved",
                "target_dataset_ids": dataset_ids,
                "evidence": None,
            }
            issues.append(
                {
                    "issue_id": f"model:{model_id}:pretraining-overlap",
                    "scope": "model",
                    "subject_id": model_id,
                    "category": "pretraining_overlap",
                    "status": "unresolved",
                    "severity": "analysis_control_required",
                    "technical_gate": False,
                    "message": "Target-dataset overlap with the published pretraining corpus is unresolved.",
                    "required_action": "Preserve as an explicit limitation and representation-control item.",
                }
            )
        if weights and any(not item.get("sha256") for item in weights):
            issues.append(
                {
                    "issue_id": f"model:{model_id}:weight-sha256",
                    "scope": "model",
                    "subject_id": model_id,
                    "category": "weight_identity",
                    "status": "pending_materialisation",
                    "severity": "technical_pending",
                    "technical_gate": True,
                    "message": "A configured model weight lacks its final local SHA-256.",
                    "required_action": "Verify the pinned upstream identity and record SHA-256 before inference.",
                }
            )
        usage_license = raw_spec.get("usage_license")
        if usage_license and usage_license != raw_spec.get("source_license"):
            issues.append(
                {
                    "issue_id": f"model:{model_id}:usage-licence",
                    "scope": "model",
                    "subject_id": model_id,
                    "category": "licence",
                    "status": "usage_restricted",
                    "severity": "warning",
                    "technical_gate": True,
                    "message": f"Checkpoint use is governed by {usage_license}.",
                    "required_action": "Retain licence receipts and comply with use/redistribution terms.",
                }
            )
        rows.append(
            {
                "model_id": model_id,
                "role": raw_spec.get("role"),
                "repository": raw_spec.get("repository"),
                "source_revision": raw_spec.get("revision"),
                "checkpoint_revision": raw_spec.get("checkpoint_revision"),
                "checkpoint_variant": raw_spec.get("checkpoint_variant"),
                "weights": weights,
                "licence": {
                    "source": raw_spec.get("source_license"),
                    "usage": usage_license,
                    "commercial_use": raw_spec.get("commercial_use"),
                    "derivative_redistribution": raw_spec.get("derivative_redistribution"),
                },
                "architecture": {
                    "factory": raw_spec.get("factory"),
                    "parameters_reported": raw_spec.get("parameters_reported"),
                    "parcellation": raw_spec.get("parcellation"),
                },
                "representation": {
                    "weights_frozen": raw_spec.get("trainable") is False,
                    "layer": study.representation.layer,
                    "pooling": study.representation.pooling,
                    "secondary_pooling": study.representation.secondary_pooling,
                },
                "pretraining_corpus": raw_spec.get("pretraining_corpus"),
                "pretraining_overlap_audit": overlap_record,
            }
        )
    return rows, issues


_RELEASE_RECORDS = (
    ("manifest", MANIFEST_JSON),
    ("checksum_inventory", MANIFEST_SHA256),
    ("provenance", PROVENANCE_JSON),
    ("completion", COMPLETION_MARKER),
)
_COMPLETE_ACQUISITION_STATUSES = {
    "published",
    "published_recovered_stage",
    "already_complete",
}


def _release_record_fingerprints(
    dataset: Any, raw_root: Path, *, require_complete: bool = False
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    release = raw_root / dataset.id / dataset.source.version
    records: list[dict[str, Any]] = []
    issues: list[dict[str, str]] = []
    if release.exists() and (not release.is_dir() or release.is_symlink()):
        issues.append(
            {
                "record_type": "release_root",
                "path": str(release),
                "error": "release path is not a regular directory",
            }
        )
        return records, issues
    for record_type, relative in _RELEASE_RECORDS:
        path = release / relative
        if path.is_symlink():
            issues.append(
                {
                    "record_type": record_type,
                    "path": str(path),
                    "error": "release metadata symlinks are forbidden",
                }
            )
        elif path.exists() and not path.is_file():
            issues.append(
                {
                    "record_type": record_type,
                    "path": str(path),
                    "error": "release metadata record is not a regular file",
                }
            )
        elif path.is_file():
            records.append(
                {
                    "dataset_id": dataset.id,
                    "release_version": dataset.source.version,
                    "record_type": record_type,
                    **_artifact(path),
                }
            )
        elif require_complete:
            issues.append(
                {
                    "record_type": record_type,
                    "path": str(path),
                    "error": "successful acquisition is missing required release metadata",
                }
            )
    return records, issues


def write_receipt(context: PhaseContext, artifacts: list[Path]) -> None:
    if not artifacts:
        raise ValueError("a phase must publish at least one artifact")
    atomic_write_json(
        context.receipt_path,
        {
            "schema_version": 1,
            "phase": context.phase,
            "run_id": context.run_id,
            "created_at": datetime.now(UTC).isoformat(),
            "artifacts": [_artifact(path) for path in artifacts],
        },
    )


def run_audit(context: PhaseContext) -> list[Path]:
    study = load_study(context.study_path)
    registry = load_dataset_registry(context.datasets_path)
    server = load_yaml(context.server_path)
    models_path = context.study_path.parent / "models.yaml"
    models = load_yaml(models_path)
    storage = server.get("storage", {})
    expected_roots = {
        "canonical_root": str(context.canonical_root),
        "work_root": str(context.work_root),
        "checkpoint_root": str(context.checkpoint_root),
    }
    if any(storage.get(key) != value for key, value in expected_roots.items()):
        raise ValueError("server configuration roots differ from queue-provided roots")
    output_directory = context.output_directory()
    output_directory.mkdir(parents=True, exist_ok=True)
    dataset_table_path = output_directory / "dataset-release-audit.json"
    model_table_path = output_directory / "model-audit.json"
    issue_ledger_path = output_directory / "issue-ledger.json"
    output = output_directory / "audit.json"
    filesystem = {}
    for name, path in {
        "canonical": context.canonical_root,
        "work": context.work_root,
        "checkpoint": context.checkpoint_root,
    }.items():
        usage = shutil.disk_usage(path)
        filesystem[name] = {
            "path": str(path.resolve(strict=True)),
            "available_bytes": usage.free,
            "total_bytes": usage.total,
        }
    dataset_rows, dataset_issues = _dataset_audit_table(registry)
    model_rows, model_issues = _model_audit_table(
        models,
        study=study,
        dataset_ids=[dataset.id for dataset in registry.model.datasets],
    )
    issues = dataset_issues + model_issues
    issue_counts: dict[str, int] = {}
    for issue in issues:
        status = str(issue["status"])
        issue_counts[status] = issue_counts.get(status, 0) + 1
    atomic_write_json(
        dataset_table_path,
        {
            "schema_version": 1,
            "table": "dataset_release_licence_access",
            "registry_sha256": registry.sha256,
            "rows": dataset_rows,
        },
    )
    atomic_write_json(
        model_table_path,
        {
            "schema_version": 1,
            "table": "model_revision_weight_licence_pretraining_overlap",
            "models_config_sha256": sha256_file(models_path),
            "rows": model_rows,
        },
    )
    atomic_write_json(
        issue_ledger_path,
        {
            "schema_version": 1,
            "scientific_gate_applied": False,
            "counts_by_status": issue_counts,
            "issues": issues,
        },
    )
    source = _source_deployment(Path.cwd().resolve(strict=True))
    config_fingerprints = {
        name: _artifact(path)
        for name, path in {
            "study": context.study_path,
            "datasets": context.datasets_path,
            "models": models_path,
            "server": context.server_path,
        }.items()
    }
    audit_artifacts = {
        "dataset_table": _artifact(dataset_table_path),
        "model_table": _artifact(model_table_path),
        "issue_ledger": _artifact(issue_ledger_path),
    }
    payload = {
        "schema_version": 2,
        "phase": "audit",
        "project_status": study.status,
        "scientific_gates": study.scientific_gates,
        "study_config_sha256": config_sha256(study),
        "dataset_registry_sha256": registry.sha256,
        "config_fingerprints": config_fingerprints,
        "deployed_source": source,
        "dataset_count": len(registry.model.datasets),
        "open_dataset_count": sum(
            dataset.access.mode == "open" for dataset in registry.model.datasets
        ),
        "access_restricted": [
            dataset.id for dataset in registry.model.datasets if dataset.access.mode != "open"
        ],
        "host": platform.node(),
        "python": platform.python_version(),
        "filesystem": filesystem,
        "gpu": _command_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,driver_version",
                "--format=csv,noheader",
            ]
        ),
        "source_revision": source["commit"],
        "git_revision_probe": _command_output(["git", "rev-parse", "HEAD"]),
        "audit_artifacts": audit_artifacts,
        "issue_counts_by_status": issue_counts,
        "created_at": datetime.now(UTC).isoformat(),
    }
    atomic_write_json(output, payload)
    return [dataset_table_path, model_table_path, issue_ledger_path, output]


def run_acquire(context: PhaseContext) -> list[Path]:
    registry = load_dataset_registry(context.datasets_path)
    manager = AcquisitionManager(registry)
    results: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    release_records: list[dict[str, Any]] = []
    for dataset in registry.model.datasets:
        outcome: dict[str, Any]
        if dataset.access.mode != "open":
            outcome = {
                "dataset_id": dataset.id,
                "release_version": dataset.source.version,
                "status": "access_blocked",
                "release_path": str(context.raw_root / dataset.id / dataset.source.version),
                "details": {
                    "access_mode": dataset.access.mode,
                    "instructions": dataset.access.instructions,
                    "landing_url": dataset.source.landing_url,
                },
            }
        else:
            try:
                outcome = asdict(manager.acquire(dataset, context.raw_root))
            except AccessBlocked as error:
                outcome = {
                    "dataset_id": dataset.id,
                    "release_version": dataset.source.version,
                    "status": "access_blocked",
                    "release_path": str(context.raw_root / dataset.id / dataset.source.version),
                    "details": {
                        "access_mode": dataset.access.mode,
                        "error_type": type(error).__name__,
                        "error": str(error),
                    },
                }
            except ProviderError as error:
                outcome = {
                    "dataset_id": dataset.id,
                    "release_version": dataset.source.version,
                    "status": "unavailable",
                    "release_path": str(context.raw_root / dataset.id / dataset.source.version),
                    "details": {
                        "error_type": type(error).__name__,
                        "error": str(error),
                    },
                }
                failures.append(
                    {
                        "dataset_id": dataset.id,
                        "status": "unavailable",
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                )
            except Exception as error:  # retain every dataset outcome before raising
                outcome = {
                    "dataset_id": dataset.id,
                    "release_version": dataset.source.version,
                    "status": "failure",
                    "release_path": str(context.raw_root / dataset.id / dataset.source.version),
                    "details": {
                        "error_type": type(error).__name__,
                        "error": str(error),
                    },
                }
                failures.append(
                    {
                        "dataset_id": dataset.id,
                        "status": "failure",
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                )
        records, record_issues = _release_record_fingerprints(
            dataset,
            context.raw_root,
            require_complete=outcome["status"] in _COMPLETE_ACQUISITION_STATUSES,
        )
        release_records.extend(records)
        outcome["release_records"] = records
        if record_issues:
            outcome["status_before_release_record_audit"] = outcome["status"]
            outcome["status"] = "failure"
            outcome["release_record_issues"] = record_issues
            failures.append(
                {
                    "dataset_id": dataset.id,
                    "status": "failure",
                    "error_type": "ReleaseMetadataError",
                    "error": repr(record_issues),
                }
            )
        results.append(outcome)
    summary = context.output_directory() / "acquisition-summary.json"
    summary.parent.mkdir(parents=True, exist_ok=True)
    blocked_count = sum(result["status"] == "access_blocked" for result in results)
    atomic_write_json(
        summary,
        {
            "schema_version": 2,
            "phase": "acquire",
            "run_id": context.run_id,
            "raw_root": str(context.raw_root),
            "status": (
                "failed"
                if failures
                else "complete_with_access_blocks"
                if blocked_count
                else "complete"
            ),
            "configured_dataset_count": len(registry.model.datasets),
            "attempted_open_dataset_count": sum(
                dataset.access.mode == "open" for dataset in registry.model.datasets
            ),
            "access_blocked_dataset_count": blocked_count,
            "results": results,
            "release_records": release_records,
            "failures": failures,
            "created_at": datetime.now(UTC).isoformat(),
        },
    )
    if failures:
        raise RuntimeError("one or more open acquisitions failed; inspect " + str(summary))
    record_paths = list(dict.fromkeys(Path(record["path"]) for record in release_records))
    return [summary, *record_paths]


def run_qc(context: PhaseContext) -> list[Path]:
    registry = load_dataset_registry(context.datasets_path)
    manager = AcquisitionManager(registry)
    validations: list[dict[str, Any]] = []
    for dataset in registry.model.datasets:
        if dataset.access.mode != "open":
            validations.append({"dataset_id": dataset.id, "status": "access_blocked"})
            continue
        validations.append(asdict(manager.validate(dataset, context.raw_root)))
    output_dir = context.output_directory()
    output_dir.mkdir(parents=True, exist_ok=True)
    validation_path = output_dir / "raw-validation.json"
    atomic_write_json(
        validation_path,
        {
            "schema_version": 1,
            "phase": "qc",
            "validations": validations,
            "created_at": datetime.now(UTC).isoformat(),
        },
    )
    inventory_path = write_inventory(
        scan_recordings(context.raw_root), output_dir / "recordings.parquet"
    )
    return [validation_path, inventory_path]


def run_preprocess(context: PhaseContext) -> list[Path]:
    study = load_study(context.study_path)
    labels, encoder_inputs, issues = build_cohort_manifest(
        raw_root=context.raw_root,
        output_root=context.output_directory(),
        dataset_ids=HEALTHY_DATASETS,
    )
    manifest, flow = preprocess_analysis_units(
        encoder_inputs=encoder_inputs,
        output_root=context.output_directory(),
        study=study,
    )
    return [labels, encoder_inputs, issues, manifest, flow]


def run_encode(context: PhaseContext) -> list[Path]:
    study = load_study(context.study_path)
    manifest = context.run_root / "preprocess" / "preprocessing-manifest.parquet"
    if not manifest.is_file():
        raise FileNotFoundError(f"preprocessing manifest is missing: {manifest}")
    labels = context.run_root / "preprocess" / "cohort-labels.parquet"
    if not labels.is_file():
        raise FileNotFoundError(f"cohort label manifest is missing: {labels}")
    encoded_manifest, flow = encode_analysis_units(
        preprocessing_manifest=manifest,
        labels_manifest=labels,
        output_root=context.output_directory(),
        study=study,
    )
    return [encoded_manifest, flow]


def run_metric_stage(context: PhaseContext) -> list[Path]:
    study = load_study(context.study_path)
    manifest = context.run_root / "encode" / "encoding-manifest.parquet"
    if not manifest.is_file():
        raise FileNotFoundError(f"encoding manifest is missing: {manifest}")
    metric_artifacts = list(
        run_metrics(
            encoding_manifest=manifest,
            output_root=context.output_directory(),
            study=study,
        )
    )
    contrasts = context.study_path.parent / "contrasts.yaml"
    preprocessing_manifest = context.run_root / "preprocess" / "preprocessing-manifest.parquet"
    labels_manifest = context.run_root / "preprocess" / "cohort-labels.parquet"
    if (
        not contrasts.is_file()
        or not preprocessing_manifest.is_file()
        or not labels_manifest.is_file()
    ):
        raise FileNotFoundError(
            "contrast, label-free preprocessing, or post-encoding label manifest is missing"
        )
    sampling_artifacts = list(
        run_sampling_sensitivity(
            encoding_manifest=manifest,
            state_dictionary_path=metric_artifacts[2],
            profile_estimator_path=metric_artifacts[3],
            contrasts_path=contrasts,
            output_root=context.output_directory() / "sampling",
            study=study,
        )
    )
    channel_control = run_preencoder_channel_permutation_control(
        preprocessing_manifest=preprocessing_manifest,
        labels_manifest=labels_manifest,
        state_dictionary_path=metric_artifacts[2],
        profile_estimator_path=metric_artifacts[3],
        output_root=context.output_directory() / "preencoder-channel-permutation",
        study=study,
        repeats=study.sampling.repeats,
        modalities=("eeg",),
    )
    combined_nulls = combine_null_profile_tables(
        metric_artifacts[1],
        channel_control.profiles_path,
        context.output_directory() / "all-null-profiles.parquet",
    )
    benchmark_artifacts = list(
        run_benchmarks(
            encoding_manifest=manifest,
            output_root=context.output_directory(),
        )
    )
    return [
        *metric_artifacts,
        *sampling_artifacts,
        channel_control.profiles_path,
        channel_control.audit_path,
        combined_nulls,
        *benchmark_artifacts,
    ]


def run_model_stage(context: PhaseContext) -> list[Path]:
    study = load_study(context.study_path)
    profiles = context.run_root / "metrics" / "profiles.parquet"
    benchmarks = context.run_root / "metrics" / "benchmarks.parquet"
    matched_profiles = (
        context.run_root / "metrics" / "sampling" / "sampling-matched-profiles.parquet"
    )
    contrasts = context.study_path.parent / "contrasts.yaml"
    if (
        not profiles.is_file()
        or not benchmarks.is_file()
        or not matched_profiles.is_file()
        or not contrasts.is_file()
    ):
        raise FileNotFoundError(
            "metric profiles, matched-window profiles, conventional benchmarks, or contrast "
            "configuration is missing"
        )
    return list(
        run_models(
            profiles_path=profiles,
            benchmarks_path=benchmarks,
            matched_profiles_path=matched_profiles,
            contrasts_path=contrasts,
            output_root=context.output_directory(),
            study=study,
        )
    )


def run_tms_stage(context: PhaseContext) -> list[Path]:
    study = load_study(context.study_path)
    labels = context.run_root / "preprocess" / "cohort-labels.parquet"
    profiles = context.run_root / "metrics" / "profiles.parquet"
    if not labels.is_file() or not profiles.is_file():
        raise FileNotFoundError("cohort labels or metric profiles are missing for TMS")
    epoch_manifest, epoch_audit = build_tms_epoch_manifest(
        cohort_labels=labels,
        output_root=context.output_directory(),
        study=study,
    )
    return [
        epoch_manifest,
        epoch_audit,
        *run_tms_validation(
            tms_manifest=epoch_manifest,
            profiles_path=profiles,
            output_root=context.output_directory(),
            study=study,
        ),
    ]


def run_figure_stage(context: PhaseContext) -> list[Path]:
    from neural_manifolds.figure_sources import (
        prepare_clinical_figure_source,
        prepare_figure_sources,
        prepare_fmri_figure_source,
    )
    from neural_manifolds.figures import figure_run_artifacts, run_figures

    profiles = context.run_root / "metrics" / "profiles.parquet"
    nulls = context.run_root / "metrics" / "all-null-profiles.parquet"
    contrasts = context.study_path.parent / "contrasts.yaml"
    tms_outcomes = context.run_root / "tms" / "tms-outcomes.parquet"
    tms_trajectory = context.run_root / "tms" / "tms-trajectory.parquet"
    bundles = prepare_figure_sources(
        profiles_path=profiles,
        nulls_path=nulls,
        contrasts_path=contrasts,
        tms_outcomes_path=tms_outcomes,
        tms_trajectory_path=tms_trajectory,
        output_root=context.run_root / "figure-source-bundles",
    )
    clinical_bundle = prepare_clinical_figure_source(
        clinical_profiles_path=(
            context.run_root / "clinical" / "transfer" / "clinical-profiles.parquet"
        ),
        output_root=context.run_root / "figure-source-bundles" / "clinical",
    )
    fmri_bundle = prepare_fmri_figure_source(
        fmri_profiles_path=(
            context.run_root / "fmri" / "analysis" / "fmri-participant-summaries.parquet"
        ),
        output_root=context.run_root / "figure-source-bundles" / "fmri",
    )
    all_bundles = (*bundles, clinical_bundle, fmri_bundle)
    result = run_figures(*all_bundles, context.output_directory())
    bundle_artifacts = [
        path for bundle in all_bundles for path in sorted(bundle.rglob("*")) if path.is_file()
    ]
    return [*bundle_artifacts, *figure_run_artifacts(result)]


def run_clinical_stage(context: PhaseContext) -> list[Path]:
    lock_value = os.environ.get("NEURAL_MANIFOLDS_CLINICAL_LOCK")
    if not lock_value:
        raise RuntimeError("queue did not provide the technical clinical lock")
    study = load_study(context.study_path)
    destination = context.output_directory()
    labels, encoder_inputs, issues = build_cohort_manifest(
        raw_root=context.raw_root,
        output_root=destination / "cohort",
        dataset_ids=CLINICAL_DATASETS,
    )
    preprocessing_manifest, preprocessing_flow = preprocess_analysis_units(
        encoder_inputs=encoder_inputs,
        output_root=destination / "preprocess",
        study=study,
    )
    encoding_manifest, encoding_flow = encode_analysis_units(
        preprocessing_manifest=preprocessing_manifest,
        labels_manifest=labels,
        output_root=destination / "encode",
        study=study,
    )
    transfer = run_clinical_transfer(
        encoding_manifest=encoding_manifest,
        state_dictionary_path=context.run_root / "metrics" / "state-dictionary.joblib",
        profile_estimator_path=context.run_root / "metrics" / "profile-estimator.joblib",
        clinical_lock_path=lock_value,
        output_root=destination / "transfer",
        study=study,
    )
    return [
        labels,
        encoder_inputs,
        issues,
        preprocessing_manifest,
        preprocessing_flow,
        encoding_manifest,
        encoding_flow,
        *transfer,
    ]


def run_fmri_stage(context: PhaseContext) -> list[Path]:
    study = load_study(context.study_path)
    release = context.raw_root / "propofol_fmri" / "1.0.0"
    models = context.study_path.parent / "models.yaml"
    if not release.is_dir() or not models.is_file():
        raise FileNotFoundError("pinned ds006623 release or model configuration is missing")
    return list(
        run_ds006623_brainlm_stage(
            release_root=release,
            models_path=models,
            output_root=context.output_directory(),
            study=study,
        )
    )


def _unavailable_phase(context: PhaseContext) -> list[Path]:
    raise RuntimeError(
        f"phase {context.phase!r} requires its dataset-specific stage driver; "
        "the queue must not mark a planning artifact as scientific completion"
    )


HANDLERS: dict[str, Callable[[PhaseContext], list[Path]]] = {
    "audit": run_audit,
    "acquire": run_acquire,
    "qc": run_qc,
    "preprocess": run_preprocess,
    "encode": run_encode,
    "metrics": run_metric_stage,
    "models": run_model_stage,
    "tms": run_tms_stage,
    "clinical": run_clinical_stage,
    "fmri": run_fmri_stage,
    "figures": run_figure_stage,
}


def run_phase(context: PhaseContext) -> list[Path]:
    artifacts = HANDLERS[context.phase](context)
    write_receipt(context, artifacts)
    return artifacts
