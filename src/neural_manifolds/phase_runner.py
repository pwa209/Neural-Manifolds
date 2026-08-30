"""Scientific phase entry points consumed by the durable server queue."""

from __future__ import annotations

import os
import platform
import subprocess
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from neural_manifolds.cohort import build_cohort_manifest
from neural_manifolds.config import config_sha256, load_study, load_yaml
from neural_manifolds.data.acquisition import AcquisitionManager
from neural_manifolds.data.providers import AccessBlocked
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
    storage = server.get("storage", {})
    expected_roots = {
        "canonical_root": str(context.canonical_root),
        "work_root": str(context.work_root),
        "checkpoint_root": str(context.checkpoint_root),
    }
    if any(storage.get(key) != value for key, value in expected_roots.items()):
        raise ValueError("server configuration roots differ from queue-provided roots")
    output = context.output_directory() / "audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    filesystem = {}
    for name, path in {
        "canonical": context.canonical_root,
        "work": context.work_root,
        "checkpoint": context.checkpoint_root,
    }.items():
        usage = os.statvfs(path)
        filesystem[name] = {
            "path": str(path.resolve(strict=True)),
            "available_bytes": usage.f_bavail * usage.f_frsize,
            "total_bytes": usage.f_blocks * usage.f_frsize,
        }
    payload = {
        "schema_version": 1,
        "phase": "audit",
        "project_status": study.status,
        "scientific_gates": study.scientific_gates,
        "study_config_sha256": config_sha256(study),
        "dataset_registry_sha256": registry.sha256,
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
        "source_revision": _command_output(["git", "rev-parse", "HEAD"]),
        "created_at": datetime.now(UTC).isoformat(),
    }
    atomic_write_json(output, payload)
    return [output]


def run_acquire(context: PhaseContext) -> list[Path]:
    registry = load_dataset_registry(context.datasets_path)
    manager = AcquisitionManager(registry)
    results: list[dict[str, Any]] = []
    failures: list[str] = []
    for dataset in registry.model.datasets:
        if dataset.access.mode != "open":
            results.append(
                {
                    "dataset_id": dataset.id,
                    "status": "access_blocked",
                    "instructions": dataset.access.instructions,
                    "landing_url": dataset.source.landing_url,
                }
            )
            continue
        try:
            result = manager.acquire(dataset, context.raw_root)
            results.append(asdict(result))
        except AccessBlocked as error:
            results.append(
                {"dataset_id": dataset.id, "status": "access_blocked", "error": str(error)}
            )
        except Exception as error:  # retain complete failure inventory before raising
            failures.append(f"{dataset.id}: {type(error).__name__}: {error}")
    summary = context.output_directory() / "acquisition-summary.json"
    summary.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        summary,
        {
            "schema_version": 1,
            "phase": "acquire",
            "run_id": context.run_id,
            "raw_root": str(context.raw_root),
            "results": results,
            "failures": failures,
            "created_at": datetime.now(UTC).isoformat(),
        },
    )
    if failures:
        raise RuntimeError("one or more open acquisitions failed; inspect " + str(summary))
    return [summary]


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
