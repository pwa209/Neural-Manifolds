"""Label-preserving archive inventory; never extracts untrusted archive paths."""

from __future__ import annotations

import csv
import io
import math
import stat
import zipfile
from pathlib import Path, PurePosixPath

from neural_manifolds.provenance import atomic_write_json, sha256_file


def safe_member(name: str) -> bool:
    path = PurePosixPath(name.replace("\\", "/"))
    return not path.is_absolute() and ".." not in path.parts and ":" not in name


def edf_header(stream) -> dict:
    """Read only EDF header fields, without interpreting subject identifiers."""
    fixed = stream.read(256)
    if len(fixed) != 256:
        raise ValueError("Truncated EDF header")
    size, records = int(fixed[184:192]), int(fixed[236:244])
    duration, channels = float(fixed[244:252]), int(fixed[252:256])
    if not 1 <= channels <= 4096 or size != 256 * (channels + 1):
        raise ValueError("Unsupported EDF header size")
    rest = stream.read(size - 256)
    if len(rest) != size - 256 or duration <= 0:
        raise ValueError("Invalid EDF channel header")
    labels = [
        rest[i * 16 : (i + 1) * 16].decode("ascii", errors="replace").strip()
        for i in range(channels)
    ]
    offset = channels * 216
    samples = [int(rest[offset + i * 8 : offset + (i + 1) * 8]) for i in range(channels)]
    return {
        "channels": labels,
        "sample_rates": [n / duration for n in samples],
        "duration_seconds": records * duration if records >= 0 else None,
        "header_status": "read",
        "signal_qc_status": "not_run",
    }


def read_records(payload: bytes, origin: str) -> list[dict]:
    text = payload.decode("utf-8-sig")
    rows = list(csv.DictReader(io.StringIO(text)))
    required = {"Filename", "Case ID", "Subject ID", "Experience", "Last sleep stage", "Duration"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Unrecognized DREAM schema: {origin}")
    for row in rows:
        row["records_origin"] = origin
        # Preserve blank and ambiguous codes; they are not absence of experience.
        row["primary_report_stage_candidate"] = (
            row["Experience"].strip() in {"0", "2"} and row["Last sleep stage"].strip() == "2"
        )
        row["cohort_independence"] = "not_yet_audited"
        row["analysis_ready"] = False
    return rows


def inventory(release: Path, output: Path) -> dict:
    release = release.resolve(strict=True)
    marker = release / ".acquisition/COMPLETE.json"
    if not marker.is_file():
        raise ValueError("Only completed immutable acquisitions can be inventoried")
    files, records, issues = [], [], []
    for path in sorted(release.rglob("*")):
        if (
            not path.is_file()
            or ".acquisition" in path.relative_to(release).parts
            or ".git" in path.relative_to(release).parts
        ):
            continue
        relative = path.relative_to(release).as_posix()
        if not path.resolve().is_relative_to(release):
            issues.append({"path": relative, "reason": "external_symlink"})
            continue
        if path.suffix.lower() == ".zip":
            try:
                with zipfile.ZipFile(path) as archive:
                    for member in archive.infolist():
                        if member.is_dir():
                            continue
                        if not safe_member(member.filename) or stat.S_ISLNK(
                            member.external_attr >> 16
                        ):
                            issues.append(
                                {
                                    "path": relative,
                                    "reason": "unsafe_archive_member",
                                    "member": member.filename,
                                }
                            )
                            continue
                        row = {
                            "container": relative,
                            "member": member.filename,
                            "bytes": member.file_size,
                        }
                        try:
                            if member.filename.lower().endswith(".edf"):
                                with archive.open(member) as stream:
                                    row.update(edf_header(stream))
                            if PurePosixPath(member.filename).name.lower() == "records.csv":
                                if member.file_size > 16 * 1024**2:
                                    raise ValueError("Records file exceeds metadata size limit")
                                records.extend(
                                    read_records(
                                        archive.read(member), f"{relative}::{member.filename}"
                                    )
                                )
                        except Exception as exc:
                            row["error"] = str(exc)
                            issues.append(
                                {"path": relative, "member": member.filename, "reason": str(exc)}
                            )
                        files.append(row)
            except (OSError, zipfile.BadZipFile) as exc:
                issues.append({"path": relative, "reason": str(exc)})
        elif path.suffix.lower() == ".rar":
            from neural_manifolds.continuous.rar_archive import members, metadata_member

            try:
                for item in members(path):
                    files.append({"container": relative, **item, "header_status": "deferred_to_qc"})
                    if PurePosixPath(item["member"]).name.lower() == "records.csv":
                        records.extend(
                            read_records(
                                metadata_member(path, item), f"{relative}::{item['member']}"
                            )
                        )
            except (OSError, ValueError, RuntimeError) as exc:
                issues.append({"path": relative, "reason": "rar_adapter_error: " + str(exc)})
        else:
            row = {"container": None, "member": relative, "bytes": path.stat().st_size}
            try:
                if path.suffix.lower() == ".edf":
                    with path.open("rb") as stream:
                        row.update(edf_header(stream))
                if path.name.lower() == "records.csv":
                    if path.stat().st_size > 16 * 1024**2:
                        raise ValueError("Records file exceeds metadata size limit")
                    records.extend(read_records(path.read_bytes(), relative))
                if path.suffix.lower() == ".7z":
                    issues.append({"path": relative, "reason": "archive_adapter_required"})
            except Exception as exc:
                row["error"] = str(exc)
                issues.append({"path": relative, "reason": str(exc)})
            files.append(row)
    result = {
        "schema_version": 1,
        "release": str(release),
        "completion_marker_sha256": sha256_file(marker),
        "files": files,
        "records": records,
        "issues": issues,
        "analysis_ready": False,
        "scope": "archive_metadata_and_edf_headers_not_signal_qc",
    }
    atomic_write_json(output, result)
    return result


def observed_lengths(audit: dict) -> list[int]:
    lengths = set()
    for row in audit["files"]:
        duration = row.get("duration_seconds")
        if duration is not None and math.isfinite(duration) and duration >= 2:
            lengths.add(min(20, int(duration)))
    return sorted(lengths)
