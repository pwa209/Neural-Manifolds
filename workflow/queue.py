"""Run the study phase graph safely inside a persistent server session.

This module is the scheduler for the target host, which has ``tmux`` but no
Slurm.  Scientific commands never run through a shell.  A successful exit code
is necessary but insufficient: each command must atomically publish a receipt
whose listed artifacts pass size and SHA-256 validation.
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import shutil
import socket
import subprocess
import sys
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .phases import PHASE_BY_NAME, PHASES, PhaseSpec, select_phases
from .state import (
    EXPECTED_HOSTNAME,
    EXPECTED_USER,
    ServerRoots,
    atomic_write_json,
    canonical_json,
    ensure_existing_roots,
    file_fingerprints,
    load_json,
    phase_hash,
    sha256_file,
    sha256_json,
    validate_receipt,
    validate_roots,
    validate_run_id,
    validate_success_marker,
)


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def safe_repo_file(repo_root: Path, value: str, *, label: str) -> Path:
    candidate = Path(value)
    path = candidate if candidate.is_absolute() else repo_root / candidate
    path = path.resolve(strict=False)
    try:
        path.relative_to(repo_root)
    except ValueError as exc:
        raise ValueError(f"{label} must reside inside the deployed repository: {path}") from exc
    return path


def verify_remote_identity() -> None:
    hostname = socket.gethostname()
    user = getpass.getuser()
    if hostname != EXPECTED_HOSTNAME or user != EXPECTED_USER:
        raise RuntimeError(
            f"refusing server workflow on {user}@{hostname}; expected "
            f"{EXPECTED_USER}@{EXPECTED_HOSTNAME}"
        )


def verify_server_config(path: Path, roots: ServerRoots) -> dict[str, Any]:
    try:
        import yaml
    except ModuleNotFoundError as exc:  # pragma: no cover - declared project dependency
        raise RuntimeError("PyYAML is required to validate configs/server.yaml") from exc
    with path.open("r", encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    if not isinstance(document, dict):
        raise ValueError("server configuration must be a mapping")
    storage = document.get("storage")
    if not isinstance(storage, dict):
        raise ValueError("server configuration has no storage mapping")
    expected = {
        "canonical_root": roots.canonical,
        "work_root": roots.work,
        "checkpoint_root": roots.checkpoint,
    }
    for key, explicit_path in expected.items():
        configured = storage.get(key)
        if configured is None or configured == "":
            raise ValueError(
                f"configs/server.yaml leaves storage.{key} unresolved; obtain and record "
                "the approved project root before deployment"
            )
        if not isinstance(configured, str):
            raise ValueError(f"storage.{key} must be an absolute path string")
        configured_path = Path(configured)
        if configured_path != explicit_path:
            raise ValueError(
                f"explicit {key} ({explicit_path}) does not match configs/server.yaml "
                f"({configured})"
            )
    if storage.get("raw_data_location") != "canonical_only":
        raise ValueError("server configuration must enforce raw_data_location=canonical_only")
    scheduler = document.get("scheduler")
    if not isinstance(scheduler, dict) or scheduler.get("type") != "tmux":
        raise ValueError("server configuration must select the tmux scheduler")
    if document.get("scientific_gates") is not False:
        raise ValueError("server configuration must not introduce scientific gates")
    fmri_inputs = document.get("fmri_inputs")
    if fmri_inputs is not None:
        if not isinstance(fmri_inputs, dict):
            raise ValueError("server configuration fmri_inputs must be a mapping")
        for key in ("ukb424_atlas_path", "ukb424_coordinates_path"):
            value = fmri_inputs.get(key)
            if value is not None and (
                not isinstance(value, str) or not value or not Path(value).is_absolute()
            ):
                raise ValueError(f"fmri_inputs.{key} must be null or an absolute path")
        origin = fmri_inputs.get("ds006623_timing_index_origin")
        if origin is not None and (
            not isinstance(origin, int) or isinstance(origin, bool) or origin not in {0, 1}
        ):
            raise ValueError("fmri_inputs.ds006623_timing_index_origin must be null, 0, or 1")
    return document


def verified_fmri_input_fingerprints(
    phase: PhaseSpec, server_config: dict[str, Any]
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    """Hash fMRI-only external assets and bind the explicit timing convention."""

    if phase.name != "fmri":
        return {}, {}
    configured = server_config.get("fmri_inputs")
    if not isinstance(configured, dict):
        raise ValueError(
            "the fMRI phase requires configs/server.yaml fmri_inputs; earlier phases do not"
        )
    asset_contract = {
        "ukb424_atlas_path": "NEURAL_MANIFOLDS_UKB424_ATLAS",
        "ukb424_coordinates_path": "NEURAL_MANIFOLDS_UKB424_COORDINATES",
    }
    fingerprints: dict[str, dict[str, Any]] = {}
    environment: dict[str, str] = {}
    for key, environment_name in asset_contract.items():
        value = configured.get(key)
        if not isinstance(value, str) or not value or not Path(value).is_absolute():
            raise ValueError(
                f"the fMRI phase requires fmri_inputs.{key} as an explicit absolute path"
            )
        source = Path(value).resolve(strict=True)
        if not source.is_file():
            raise ValueError(f"configured fMRI input is not a regular file: {source}")
        fingerprints[f"fmri_input:{key}"] = {
            "path": str(source),
            "sha256": sha256_file(source),
            "size": source.stat().st_size,
        }
        environment[environment_name] = str(source)
    origin = configured.get("ds006623_timing_index_origin")
    if not isinstance(origin, int) or isinstance(origin, bool) or origin not in {0, 1}:
        raise ValueError("the fMRI phase requires fmri_inputs.ds006623_timing_index_origin=0 or 1")
    fingerprints["fmri_input:ds006623_timing_index_origin"] = {
        "value": origin,
        "sha256": sha256_json({"ds006623_timing_index_origin": origin}),
        "size": 1,
    }
    environment["NEURAL_MANIFOLDS_DS006623_TIMING_INDEX_ORIGIN"] = str(origin)
    return fingerprints, environment


def verified_model_fingerprints(phase: PhaseSpec) -> dict[str, dict[str, int | str]]:
    """Rehash the model manifest and every model file needed by a phase."""

    model_phases = {
        "encode",
        "metrics",
        "models",
        "tms",
        "figures",
        "locked-clinical",
        "fmri",
    }
    if phase.name not in model_phases:
        return {}
    manifest_value = os.environ.get("NEURAL_MANIFOLDS_MODEL_MANIFEST")
    if not manifest_value:
        raise RuntimeError(
            f"phase {phase.name} requires a verified model cache; run bootstrap_models.sh"
        )
    manifest_path = Path(manifest_value).resolve(strict=True)
    manifest = load_json(manifest_path)
    if manifest.get("schema_version") != 1 or not isinstance(manifest.get("models"), dict):
        raise ValueError("invalid model manifest")
    models = manifest["models"]
    files = [manifest_path]

    def validate_declared_file(item: Any, *, label: str) -> Path:
        if not isinstance(item, dict):
            raise ValueError(f"model manifest has invalid {label}")
        path_value = item.get("path")
        expected_hash = item.get("sha256")
        if not isinstance(path_value, str) or not isinstance(expected_hash, str):
            raise ValueError(f"model manifest has incomplete {label}")
        path = Path(path_value).resolve(strict=True)
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise ValueError(f"model file changed after bootstrap: {path}")
        expected_size = item.get("size")
        if expected_size is not None and path.stat().st_size != expected_size:
            raise ValueError(f"model file size changed after bootstrap: {path}")
        files.append(path)
        return path

    labram = models.get("labram_base")
    if not isinstance(labram, dict) or labram.get("trainable") is not False:
        raise ValueError("LaBraM model manifest is missing or not frozen")
    labram_source_manifest = validate_declared_file(labram.get("source"), label="LaBraM source")
    labram_checkpoint = validate_declared_file(labram.get("checkpoint"), label="LaBraM checkpoint")
    if os.environ.get("NEURAL_MANIFOLDS_LABRAM_SOURCE") != str(labram_source_manifest.parent):
        raise ValueError("LaBraM source environment path does not match model manifest")
    if os.environ.get("NEURAL_MANIFOLDS_LABRAM_CHECKPOINT") != str(labram_checkpoint):
        raise ValueError("LaBraM checkpoint environment path does not match model manifest")

    if phase.name == "fmri":
        brainlm = models.get("brainlm")
        if not isinstance(brainlm, dict) or brainlm.get("trainable") is not False:
            raise ValueError("BrainLM model manifest is missing or not frozen")
        if brainlm.get("checkpoint_status") != "verified_for_fmri":
            raise ValueError("BrainLM weights may be materialised only by the fMRI bootstrap")
        if brainlm.get("usage_license") != "CC-BY-NC-ND-4.0":
            raise ValueError("BrainLM licence receipt is missing or unexpected")
        brainlm_source_manifest = validate_declared_file(
            brainlm.get("source"), label="BrainLM source"
        )
        checkpoint_files = brainlm.get("checkpoint_files")
        if not isinstance(checkpoint_files, list) or not checkpoint_files:
            raise ValueError("BrainLM fMRI manifest lists no checkpoint files")
        validated_checkpoints = [
            validate_declared_file(item, label="BrainLM checkpoint") for item in checkpoint_files
        ]
        if os.environ.get("NEURAL_MANIFOLDS_BRAINLM_SOURCE") != str(brainlm_source_manifest.parent):
            raise ValueError("BrainLM source environment path does not match model manifest")
        checkpoint_dir = os.environ.get("NEURAL_MANIFOLDS_BRAINLM_CHECKPOINT_DIR")
        if not checkpoint_dir:
            raise ValueError("BrainLM checkpoint environment path is missing")
        resolved_checkpoint_dir = Path(checkpoint_dir).resolve(strict=True)
        if not all(path.is_relative_to(resolved_checkpoint_dir) for path in validated_checkpoints):
            raise ValueError("BrainLM checkpoint file escapes the verified checkpoint directory")
        if os.environ.get("NEURAL_MANIFOLDS_BRAINLM_LICENSE") != "CC-BY-NC-ND-4.0":
            raise ValueError("BrainLM licence environment value is missing or unexpected")
    return file_fingerprints(files)


def build_phase_command(
    *,
    executable: str,
    phase: PhaseSpec,
    study: Path,
    datasets: Path,
    server: Path,
    run_id: str,
) -> list[str]:
    return [
        executable,
        "run-phase",
        "--phase",
        phase.cli_phase,
        "--study",
        str(study),
        "--datasets",
        str(datasets),
        "--server",
        str(server),
        "--run-id",
        run_id,
    ]


def build_validate_command(
    *, executable: str, study: Path, datasets: Path, server: Path
) -> list[str]:
    return [
        executable,
        "validate-config",
        "--study",
        str(study),
        "--datasets",
        str(datasets),
        "--server",
        str(server),
    ]


def process_token(pid: int) -> dict[str, str | int | None]:
    token: dict[str, str | int | None] = {"pid": pid, "boot_id": None, "start_ticks": None}
    boot_id = Path("/proc/sys/kernel/random/boot_id")
    stat = Path(f"/proc/{pid}/stat")
    try:
        token["boot_id"] = boot_id.read_text(encoding="utf-8").strip()
        fields = stat.read_text(encoding="utf-8").split()
        token["start_ticks"] = fields[21]
    except (FileNotFoundError, IndexError, OSError):
        pass
    return token


def process_is_current(token: Any) -> bool:
    if not isinstance(token, dict) or not isinstance(token.get("pid"), int):
        return False
    pid = token["pid"]
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError, OSError):
        return False
    current = process_token(pid)
    for key in ("boot_id", "start_ticks"):
        expected = token.get(key)
        if expected is not None and current.get(key) != expected:
            return False
    return True


class QueueLock(AbstractContextManager["QueueLock"]):
    """Advisory lock held for one run's entire phase loop."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._stream: Any = None

    def __enter__(self) -> QueueLock:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self.path.open("a+", encoding="utf-8")
        try:
            import fcntl

            fcntl.flock(self._stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self._stream.close()
            raise RuntimeError(f"another queue process holds {self.path}") from exc
        self._stream.seek(0)
        self._stream.truncate()
        self._stream.write(canonical_json({"pid": os.getpid(), "started_at": utc_now()}))
        self._stream.write("\n")
        self._stream.flush()
        os.fsync(self._stream.fileno())
        return self

    def __exit__(self, *exc_info: Any) -> None:
        if self._stream is not None:
            try:
                import fcntl

                fcntl.flock(self._stream.fileno(), fcntl.LOCK_UN)
            finally:
                self._stream.close()


def append_event(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(canonical_json(payload))
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def run_logged(
    command: Sequence[str],
    *,
    cwd: Path,
    env: dict[str, str],
    log_path: Path,
    on_start: Callable[[dict[str, str | int | None]], None] | None = None,
) -> tuple[int, dict[str, str | int | None]]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", newline="\n") as log:
        log.write(f"[{utc_now()}] argv={canonical_json(list(command))}\n")
        log.flush()
        process = subprocess.Popen(
            list(command),
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        token = process_token(process.pid)
        if on_start is not None:
            on_start(token)
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
            log.flush()
        return_code = process.wait()
        log.write(f"[{utc_now()}] exit_code={return_code}\n")
        log.flush()
        os.fsync(log.fileno())
    return return_code, token


def dependency_markers(
    *, phase: PhaseSpec, state_root: Path, phase_hashes: dict[str, str]
) -> dict[str, str]:
    marker_hashes: dict[str, str] = {}
    for dependency in phase.dependencies:
        marker = state_root / "phases" / dependency / "success.json"
        if dependency in phase_hashes:
            expected_hash = phase_hashes[dependency]
        elif marker.is_file():
            expected_hash = str(load_json(marker).get("phase_hash", ""))
        else:
            raise RuntimeError(
                f"phase {phase.name} requires completed technical dependency {dependency}"
            )
        validate_success_marker(load_json(marker), expected_phase_hash=expected_hash)
        marker_hashes[dependency] = sha256_file(marker)
    return marker_hashes


def prepare_clinical_lock(
    *,
    state_root: Path,
    source_manifest_sha256: str,
    config_fingerprints: dict[str, Any],
) -> Path:
    """Freeze the healthy implementation state before held-out clinical transfer.

    This is a provenance snapshot, not a registration, preregistration, or
    result-based decision gate.
    """

    clinical_index = next(
        index for index, phase in enumerate(PHASES) if phase.name == "locked-clinical"
    )
    healthy_names = [phase.name for phase in PHASES[:clinical_index]]
    if not healthy_names or healthy_names[-1] != "tms":
        raise RuntimeError("clinical lock boundary must freeze the healthy workflow through TMS")
    markers: dict[str, str] = {}
    validated_artifacts: dict[str, dict[str, str]] = {}
    for name in healthy_names:
        marker = state_root / "phases" / name / "success.json"
        if not marker.is_file():
            raise RuntimeError(f"cannot create clinical lock; missing {name} success marker")
        payload = load_json(marker)
        expected_phase_hash = str(payload.get("phase_hash", ""))
        validate_success_marker(payload, expected_phase_hash=expected_phase_hash)
        markers[name] = sha256_file(marker)
        validated_artifacts[name] = {
            str(item["path"]): str(item["sha256"]) for item in payload["artifacts"]
        }
    basis = {
        "schema_version": 1,
        "kind": "technical_clinical_transfer_snapshot",
        "project_status": "exploratory_non_preregistered",
        "scientific_gate": False,
        "source_manifest_sha256": source_manifest_sha256,
        "config_fingerprints": config_fingerprints,
        "healthy_success_markers": markers,
        "healthy_validated_artifacts": validated_artifacts,
    }
    basis_hash = sha256_json(basis)
    lock_path = state_root / "clinical_lock.json"
    if lock_path.is_file():
        existing = load_json(lock_path)
        if existing.get("basis_sha256") != basis_hash:
            raise RuntimeError(
                "clinical lock differs from current healthy implementation; use a new run id"
            )
        return lock_path
    atomic_write_json(
        lock_path,
        {
            **basis,
            "basis_sha256": basis_hash,
            "created_at": utc_now(),
            "notice": "Technical provenance only; this is not a registration or preregistration.",
        },
    )
    return lock_path


def establish_run_contract(
    *,
    state_root: Path,
    run_id: str,
    repo_root: Path,
    roots: ServerRoots,
    source_manifest_sha256: str,
    config_fingerprints: dict[str, Any],
) -> Path:
    """Create or validate the immutable source/config identity for a run id."""

    basis = {
        "schema_version": 1,
        "run_id": run_id,
        "repo_root": str(repo_root),
        "roots": {
            "canonical": str(roots.canonical),
            "work": str(roots.work),
            "checkpoint": str(roots.checkpoint),
        },
        "source_manifest_sha256": source_manifest_sha256,
        "config_fingerprints": config_fingerprints,
        "project_status": "exploratory_non_preregistered",
        "scientific_gates": False,
    }
    basis_sha256 = sha256_json(basis)
    contract_path = state_root / "run_contract.json"
    if contract_path.is_file():
        existing = load_json(contract_path)
        if existing.get("basis_sha256") != basis_sha256:
            raise RuntimeError(
                "run id is already bound to different source, configuration, roots, or release; "
                "use a new run id"
            )
        return contract_path
    atomic_write_json(
        contract_path,
        {**basis, "basis_sha256": basis_sha256, "created_at": utc_now()},
    )
    return contract_path


def status_rows(state_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for phase in PHASES:
        phase_root = state_root / "phases" / phase.name
        success = phase_root / "success.json"
        current = phase_root / "current.json"
        if success.is_file():
            payload = load_json(success)
            state = "succeeded"
        elif current.is_file():
            payload = load_json(current)
            state = str(payload.get("status", "unknown"))
        else:
            payload = {}
            state = "pending"
        rows.append(
            {
                "phase": phase.name,
                "state": state,
                "attempt": payload.get("attempt"),
                "updated_at": payload.get("finished_at", payload.get("started_at")),
            }
        )
    return rows


def print_status(state_root: Path, *, as_json: bool) -> None:
    rows = status_rows(state_root)
    if as_json:
        print(json.dumps({"state_root": str(state_root), "phases": rows}, indent=2))
        return
    width = max(len(row["phase"]) for row in rows)
    for row in rows:
        suffix = f" attempt={row['attempt']}" if row["attempt"] is not None else ""
        print(f"{row['phase']:<{width}}  {row['state']}{suffix}")


def next_attempt_number(phase_root: Path) -> int:
    attempts = phase_root / "attempts"
    if not attempts.is_dir():
        return 1
    numbers = [
        int(path.name) for path in attempts.iterdir() if path.is_dir() and path.name.isdigit()
    ]
    return max(numbers, default=0) + 1


def execute_phase(
    *,
    phase: PhaseSpec,
    command: list[str],
    phase_digest: str,
    roots: ServerRoots,
    repo_root: Path,
    state_root: Path,
    run_id: str,
    base_env: dict[str, str],
    events_path: Path,
) -> None:
    phase_root = state_root / "phases" / phase.name
    success_path = phase_root / "success.json"
    if success_path.is_file():
        validate_success_marker(load_json(success_path), expected_phase_hash=phase_digest)
        append_event(events_path, {"at": utc_now(), "event": "phase_skipped", "phase": phase.name})
        print(f"[skip] {phase.name}: validated success marker and artifacts")
        return

    current_path = phase_root / "current.json"
    if current_path.is_file():
        current = load_json(current_path)
        if current.get("phase_hash") != phase_digest:
            raise RuntimeError(
                f"phase {phase.name} has state from different source/configuration; use a new run id"
            )
        if current.get("status") == "running" and process_is_current(current.get("process")):
            raise RuntimeError(
                f"phase {phase.name} still has a live process; refusing a duplicate launch"
            )
        previous_attempt = current.get("attempt")
        if isinstance(previous_attempt, int) and previous_attempt > 0:
            previous_receipt = phase_root / "attempts" / f"{previous_attempt:04d}" / "receipt.json"
            if current.get("status") in {"starting", "running"} and previous_receipt.is_file():
                receipt = validate_receipt(
                    previous_receipt,
                    expected_cli_phase=phase.cli_phase,
                    expected_run_id=run_id,
                    workflow_phase=phase.name,
                    roots=roots,
                )
                recovered = {
                    **current,
                    "status": "succeeded",
                    "finished_at": utc_now(),
                    "recovered_after_queue_interruption": True,
                    "receipt_sha256": sha256_file(previous_receipt),
                    "artifacts": receipt["artifacts"],
                }
                atomic_write_json(
                    phase_root / "attempts" / f"{previous_attempt:04d}" / "result.json",
                    recovered,
                )
                atomic_write_json(success_path, recovered)
                atomic_write_json(current_path, recovered)
                append_event(
                    events_path,
                    {
                        "at": utc_now(),
                        "event": "phase_recovered_from_receipt",
                        "phase": phase.name,
                        "attempt": previous_attempt,
                    },
                )
                print(f"[recover] {phase.name}: validated receipt from interrupted queue")
                return

    attempt = next_attempt_number(phase_root)
    attempt_root = phase_root / "attempts" / f"{attempt:04d}"
    attempt_root.mkdir(parents=True, exist_ok=False)
    receipt_path = attempt_root / "receipt.json"
    log_path = state_root / "logs" / f"{phase.name}.attempt-{attempt:04d}.log"
    started = {
        "schema_version": 1,
        "status": "starting",
        "phase": phase.name,
        "cli_phase": phase.cli_phase,
        "attempt": attempt,
        "phase_hash": phase_digest,
        "started_at": utc_now(),
        "command": command,
        "log": str(log_path),
    }
    atomic_write_json(attempt_root / "started.json", started)
    atomic_write_json(current_path, started)
    append_event(
        events_path,
        {"at": utc_now(), "event": "phase_started", "phase": phase.name, "attempt": attempt},
    )

    env = dict(base_env)
    env.update(
        {
            "NEURAL_MANIFOLDS_WORKFLOW_PHASE": phase.name,
            "NEURAL_MANIFOLDS_CLI_PHASE": phase.cli_phase,
            "NEURAL_MANIFOLDS_PHASE_RECEIPT": str(receipt_path),
        }
    )
    try:

        def record_running(token: dict[str, str | int | None]) -> None:
            running = {**started, "status": "running", "process": token}
            atomic_write_json(attempt_root / "process.json", running)
            atomic_write_json(current_path, running)

        exit_code, _token = run_logged(
            command,
            cwd=repo_root,
            env=env,
            log_path=log_path,
            on_start=record_running,
        )
        if exit_code != 0:
            raise RuntimeError(f"phase command exited with status {exit_code}")
        if not receipt_path.is_file():
            raise RuntimeError(
                "phase command returned success but did not atomically publish "
                f"NEURAL_MANIFOLDS_PHASE_RECEIPT={receipt_path}"
            )
        receipt = validate_receipt(
            receipt_path,
            expected_cli_phase=phase.cli_phase,
            expected_run_id=run_id,
            workflow_phase=phase.name,
            roots=roots,
        )
        finished = {
            **started,
            "status": "succeeded",
            "finished_at": utc_now(),
            "receipt_sha256": sha256_file(receipt_path),
            "artifacts": receipt["artifacts"],
        }
        atomic_write_json(attempt_root / "result.json", finished)
        atomic_write_json(success_path, finished)
        atomic_write_json(current_path, finished)
        append_event(
            events_path,
            {"at": utc_now(), "event": "phase_succeeded", "phase": phase.name, "attempt": attempt},
        )
        print(f"[done] {phase.name}: {len(receipt['artifacts'])} validated artifact(s)")
    except BaseException as exc:
        failed = {
            **started,
            "status": "failed",
            "finished_at": utc_now(),
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        atomic_write_json(attempt_root / "result.json", failed)
        atomic_write_json(current_path, failed)
        append_event(
            events_path,
            {
                "at": utc_now(),
                "event": "phase_failed",
                "phase": phase.name,
                "attempt": attempt,
                "error": str(exc),
            },
        )
        raise


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--repo-root", required=True, help="absolute deployed source release")
    value.add_argument("--canonical-root", required=True)
    value.add_argument("--work-root", required=True)
    value.add_argument("--checkpoint-root", required=True)
    value.add_argument("--run-id", required=True)
    value.add_argument("--study", default="configs/study.yaml")
    value.add_argument("--datasets", default="configs/datasets.yaml")
    value.add_argument("--models", default="configs/models.yaml")
    value.add_argument("--server", default="configs/server.yaml")
    value.add_argument("--source-manifest", default="SOURCE_MANIFEST.sha256")
    value.add_argument("--cli", default="neural-manifolds")
    value.add_argument("--from-phase", choices=tuple(PHASE_BY_NAME))
    value.add_argument("--through-phase", choices=tuple(PHASE_BY_NAME))
    value.add_argument("--only-phase", choices=tuple(PHASE_BY_NAME))
    mode = value.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="print resolved plan; perform no writes or subprocesses",
    )
    mode.add_argument(
        "--check-only",
        action="store_true",
        help="run read-only identity, root, source, and config checks",
    )
    mode.add_argument("--status", action="store_true", help="read queue markers only")
    value.add_argument("--json", action="store_true", help="machine-readable status/dry-run output")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    roots = validate_roots(
        canonical_root=args.canonical_root,
        work_root=args.work_root,
        checkpoint_root=args.checkpoint_root,
    )
    run_id = validate_run_id(args.run_id)
    repo_root = Path(args.repo_root).resolve(strict=False)
    if not repo_root.is_absolute():
        raise ValueError("--repo-root must be absolute")
    study = safe_repo_file(repo_root, args.study, label="study config")
    datasets = safe_repo_file(repo_root, args.datasets, label="dataset config")
    models = safe_repo_file(repo_root, args.models, label="model config")
    server = safe_repo_file(repo_root, args.server, label="server config")
    source_manifest = safe_repo_file(repo_root, args.source_manifest, label="source manifest")
    selected = select_phases(
        from_phase=args.from_phase,
        through_phase=args.through_phase,
        only_phase=args.only_phase,
    )
    state_root = roots.state_root(run_id)

    if args.status:
        ensure_existing_roots(roots, require_writable=False)
        print_status(state_root, as_json=args.json)
        return 0

    config_paths = (study, datasets, models, server)
    for path in config_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    server_config = verify_server_config(server, roots)
    commands = [
        build_phase_command(
            executable=args.cli,
            phase=phase,
            study=study,
            datasets=datasets,
            server=server,
            run_id=run_id,
        )
        for phase in selected
    ]

    if args.dry_run:
        plan = {
            "mode": "dry-run",
            "run_id": run_id,
            "roots": {
                "canonical": str(roots.canonical),
                "raw": str(roots.raw),
                "work": str(roots.work),
                "checkpoint": str(roots.checkpoint),
            },
            "state_root": str(state_root),
            "phases": [
                {
                    "name": phase.name,
                    "cli_phase": phase.cli_phase,
                    "dependencies": phase.dependencies,
                    "command": command,
                }
                for phase, command in zip(selected, commands, strict=True)
            ],
        }
        if args.json:
            print(json.dumps(plan, indent=2))
        else:
            print(json.dumps(plan, indent=2))
        return 0

    verify_remote_identity()
    ensure_existing_roots(roots, require_writable=not args.check_only)
    if not repo_root.is_dir():
        raise FileNotFoundError(f"deployed repository does not exist: {repo_root}")
    if not source_manifest.is_file():
        raise FileNotFoundError(
            f"deployment source manifest is required before execution: {source_manifest}"
        )
    if shutil.which(args.cli) is None:
        raise FileNotFoundError(f"workflow CLI is not on PATH: {args.cli}")

    validation = build_validate_command(
        executable=args.cli, study=study, datasets=datasets, server=server
    )
    validation_result = subprocess.run(validation, cwd=repo_root, check=False)
    if validation_result.returncode != 0:
        raise RuntimeError("configuration validation failed")
    if args.check_only:
        print("check-only passed: identity, roots, deployment manifest, CLI, and configuration")
        return 0

    config_fingerprints = file_fingerprints(config_paths)
    source_manifest_sha256 = sha256_file(source_manifest)
    base_env = dict(os.environ)
    base_env.update(
        {
            "PYTHONUNBUFFERED": "1",
            "NEURAL_MANIFOLDS_CANONICAL_ROOT": str(roots.canonical),
            "NEURAL_MANIFOLDS_RAW_ROOT": str(roots.raw),
            "NEURAL_MANIFOLDS_WORK_ROOT": str(roots.work),
            "NEURAL_MANIFOLDS_CHECKPOINT_ROOT": str(roots.checkpoint),
            "NEURAL_MANIFOLDS_RUN_ROOT": str(roots.run_root(run_id)),
            "NEURAL_MANIFOLDS_STATE_ROOT": str(state_root),
            "NEURAL_MANIFOLDS_SOURCE_MANIFEST": str(source_manifest),
        }
    )
    state_root.mkdir(parents=True, exist_ok=True)
    roots.run_root(run_id).mkdir(parents=True, exist_ok=True)
    events_path = state_root / "events.jsonl"
    phase_hashes: dict[str, str] = {}

    with QueueLock(state_root / "queue.lock"):
        establish_run_contract(
            state_root=state_root,
            run_id=run_id,
            repo_root=repo_root,
            roots=roots,
            source_manifest_sha256=source_manifest_sha256,
            config_fingerprints=config_fingerprints,
        )
        for phase, command in zip(selected, commands, strict=True):
            fmri_input_fingerprints, fmri_environment = verified_fmri_input_fingerprints(
                phase, server_config
            )
            phase_config_fingerprints = {
                **config_fingerprints,
                **verified_model_fingerprints(phase),
                **fmri_input_fingerprints,
            }
            dependency_hashes = dependency_markers(
                phase=phase, state_root=state_root, phase_hashes=phase_hashes
            )
            digest = phase_hash(
                phase_name=phase.name,
                command=command,
                source_manifest_sha256=source_manifest_sha256,
                config_fingerprints=phase_config_fingerprints,
                dependency_marker_sha256=dependency_hashes,
                roots=roots,
            )
            phase_hashes[phase.name] = digest
            if phase.name == "locked-clinical":
                clinical_lock = prepare_clinical_lock(
                    state_root=state_root,
                    source_manifest_sha256=source_manifest_sha256,
                    config_fingerprints=phase_config_fingerprints,
                )
                base_env["NEURAL_MANIFOLDS_CLINICAL_LOCK"] = str(clinical_lock)
            phase_environment = {**base_env, **fmri_environment}
            execute_phase(
                phase=phase,
                command=command,
                phase_digest=digest,
                roots=roots,
                repo_root=repo_root,
                state_root=state_root,
                run_id=run_id,
                base_env=phase_environment,
                events_path=events_path,
            )
    print_status(state_root, as_json=False)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
