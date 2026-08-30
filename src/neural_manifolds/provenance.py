"""Atomic provenance records used by every workflow phase."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import tempfile
from collections.abc import Iterable
from contextlib import suppress
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision(repo: str | Path = ".") -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


@dataclass(frozen=True)
class PhaseRecord:
    phase: str
    run_id: str
    started_at: str
    completed_at: str
    source_revision: str | None
    config_sha256: str
    inputs: dict[str, str]
    outputs: dict[str, str]
    command: list[str]
    hostname: str
    python: str
    metadata: dict[str, Any]


def build_phase_record(
    *,
    phase: str,
    run_id: str,
    started_at: datetime,
    config_sha256: str,
    input_paths: Iterable[str | Path],
    output_paths: Iterable[str | Path],
    command: list[str],
    repo: str | Path = ".",
    metadata: dict[str, Any] | None = None,
) -> PhaseRecord:
    completed = datetime.now(UTC)
    return PhaseRecord(
        phase=phase,
        run_id=run_id,
        started_at=started_at.astimezone(UTC).isoformat(),
        completed_at=completed.isoformat(),
        source_revision=git_revision(repo),
        config_sha256=config_sha256,
        inputs={str(Path(p)): sha256_file(p) for p in input_paths},
        outputs={str(Path(p)): sha256_file(p) for p in output_paths},
        command=command,
        hostname=platform.node(),
        python=platform.python_version(),
        metadata=metadata or {},
    )


def atomic_write_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    except BaseException:
        with suppress(FileNotFoundError):
            os.unlink(temporary)
        raise


def write_phase_record(path: str | Path, record: PhaseRecord) -> None:
    atomic_write_json(path, asdict(record))
