"""Bounded, label-blind signal QC over completed archive inventories."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import zipfile
from contextlib import contextmanager
from pathlib import Path

from neural_manifolds.config import load_study
from neural_manifolds.continuous.audit import safe_member
from neural_manifolds.provenance import atomic_write_json, sha256_file
from neural_manifolds.stages.qc import _inspect_recording


@contextmanager
def materialize_recording(release: Path, item: dict, scratch: Path):
    """Copy one ZIP member to a generated scratch name; raw release stays sealed."""
    member = item["member"]
    if not safe_member(member):
        raise ValueError("Unsafe recording member")
    if item.get("container") is None:
        source = (release / member).resolve(strict=True)
        if not source.is_relative_to(release):
            raise ValueError("Recording resolves outside its release")
        yield source
        return
    container = (release / item["container"]).resolve(strict=True)
    if not container.is_relative_to(release) or container.suffix.lower() != ".zip":
        raise ValueError("Unsupported recording container")
    expected = int(item["bytes"])
    if expected <= 0 or expected > 8 * 1024**3:
        raise ValueError("Single recording exceeds bounded scratch allowance")
    scratch.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(scratch).free < expected + 2 * 1024**3:
        raise ValueError("Insufficient scratch headroom")
    with tempfile.TemporaryDirectory(prefix="nm-qc-", dir=scratch) as temporary:
        destination = Path(temporary) / ("recording" + Path(member).suffix.lower())
        with zipfile.ZipFile(container) as archive:
            metadata = archive.getinfo(member)
            if metadata.file_size != expected:
                raise ValueError("Archive entry size differs from inventory")
            written = 0
            with archive.open(metadata) as source, destination.open("xb") as target:
                while chunk := source.read(1024 * 1024):
                    written += len(chunk)
                    if written > expected:
                        raise ValueError("Archive entry exceeded declared size")
                    target.write(chunk)
        if written != expected:
            raise ValueError("Truncated recording extraction")
        destination.chmod(0o444)
        try:
            yield destination
        finally:
            destination.chmod(0o600)


def run(audit_path: Path, output: Path) -> dict:
    audit = json.loads(audit_path.read_text())
    identity = sha256_file(audit_path)
    release = Path(audit["release"]).resolve(strict=True)
    if sha256_file(release / ".acquisition/COMPLETE.json") != audit["completion_marker_sha256"]:
        raise ValueError("Raw completion receipt changed")
    study = load_study(Path("configs/study.yaml"))
    scratch = Path(os.environ["NM_SCRATCH_ROOT"]) / "continuous-qc"
    output.mkdir(parents=True, exist_ok=True)
    rows, channels, failures = [], [], []
    for item in audit["files"]:
        if Path(item["member"]).suffix.lower() not in {".edf", ".bdf", ".vhdr", ".set", ".fif"}:
            continue
        # Multi-file formats cannot be flattened out of ZIP archives.
        if item.get("container") and Path(item["member"]).suffix.lower() not in {
            ".edf",
            ".bdf",
            ".fif",
        }:
            failures.append(
                {"member": item["member"], "reason": "multi_file_archive_materialization_required"}
            )
            continue
        key = hashlib.sha256(json.dumps(item, sort_keys=True).encode()).hexdigest()[:24]
        checkpoint = output / "recordings" / f"{key}.json"
        if checkpoint.exists():
            record = json.loads(checkpoint.read_text())
            if record["audit_sha256"] != identity:
                raise ValueError("Signal QC checkpoint belongs to a different inventory")
        else:
            record = {
                "audit_sha256": identity,
                "container": item.get("container"),
                "member": item["member"],
                "labels_consumed": [],
            }
            try:
                with materialize_recording(release, item, scratch) as source:
                    row, channel_rows = _inspect_recording(
                        {
                            "recording_id": key,
                            "dataset_id": release.parent.name,
                            "source_path": str(source),
                            "events_path": None,
                            "channels_path": None,
                        },
                        study=study,
                    )
                    # Temporary paths are not durable data references.
                    row.pop("source_path", None)
                    row["materialization"] = (
                        "temporary_verified_copy" if item.get("container") else "sealed_raw_release"
                    )
                    record.update(status="inspected", recording=row, channels=channel_rows)
            except Exception as exc:
                record.update(status="unavailable", error=type(exc).__name__, detail=str(exc))
            atomic_write_json(checkpoint, record)
        if record["status"] == "inspected":
            rows.append(record["recording"])
            channels.extend(record["channels"])
        else:
            failures.append({"member": record["member"], "reason": record["detail"]})
    result = {
        "scope": "label_blind_sampled_signal_qc_not_cohort_admission",
        "audit_sha256": identity,
        "recordings": rows,
        "channels": channels,
        "unavailable": failures,
        "archive_issues": audit["issues"],
        "labels_consumed": [],
        "analysis_ready": False,
    }
    atomic_write_json(output / "signal_qc.json", result)
    return result
