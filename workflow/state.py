"""Safety checks, hashing, receipts, and atomic queue state."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

ROOT_KINDS = ("canonical", "work", "checkpoint")

RUN_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{2,79}$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class ServerRoots:
    """Explicit, project-specific server roots."""

    canonical: Path
    work: Path
    checkpoint: Path

    @property
    def raw(self) -> Path:
        return self.canonical / "raw"

    def state_root(self, run_id: str) -> Path:
        return self.checkpoint / "queue" / run_id

    def run_root(self, run_id: str) -> Path:
        return self.work / "runs" / run_id


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: Any) -> None:
    """Write JSON durably and publish it with one atomic rename."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, ensure_ascii=True, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def validate_run_id(run_id: str) -> str:
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError(
            "run id must be 3-80 lowercase letters, digits, dots, underscores, or hyphens"
        )
    return run_id


def validate_root_bases(
    values: Mapping[str, str | os.PathLike[str]],
) -> dict[str, PurePosixPath]:
    """Validate the server-only parent mounts used to constrain project roots."""

    missing = set(ROOT_KINDS).difference(values)
    unknown = set(values).difference(ROOT_KINDS)
    if missing or unknown:
        raise ValueError(
            "allowed parent mounts must define exactly canonical, work, and checkpoint; "
            f"missing={sorted(missing)}, unknown={sorted(unknown)}"
        )
    bases: dict[str, PurePosixPath] = {}
    for kind in ROOT_KINDS:
        raw = str(values[kind])
        pure = PurePosixPath(raw)
        if not pure.is_absolute():
            raise ValueError(f"{kind} parent mount must be an absolute POSIX path")
        if ".." in pure.parts:
            raise ValueError(f"{kind} parent mount cannot contain '..'")
        if pure == PurePosixPath("/"):
            raise ValueError(f"refusing broad {kind} parent mount: {pure}")
        bases[kind] = pure
    if len({str(path) for path in bases.values()}) != len(ROOT_KINDS):
        raise ValueError("canonical, work, and checkpoint parent mounts must be distinct")
    return bases


def _validate_project_root(
    kind: str,
    value: str | os.PathLike[str],
    *,
    root_bases: Mapping[str, str | os.PathLike[str]],
) -> Path:
    if kind not in root_bases:
        raise ValueError(f"unknown root kind: {kind}")
    raw = str(value)
    pure = PurePosixPath(raw)
    raw_base = root_bases[kind]
    base = raw_base if isinstance(raw_base, PurePosixPath) else PurePosixPath(str(raw_base))
    if not pure.is_absolute():
        raise ValueError(f"{kind} root must be an absolute POSIX path")
    if ".." in pure.parts:
        raise ValueError(f"{kind} root cannot contain '..'")
    if pure == base or base not in pure.parents:
        raise ValueError(f"{kind} root must be a project-specific child of {base}")
    if len(pure.parts) != len(base.parts) + 1:
        raise ValueError(
            f"{kind} root must be one direct project directory below {base}; got {pure}"
        )
    project_component = pure.name
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{2,79}", project_component):
        raise ValueError(f"unsafe project directory name in {kind} root: {project_component!r}")
    return Path(str(pure))


def validate_roots(
    *,
    canonical_root: str,
    work_root: str,
    checkpoint_root: str,
    root_bases: Mapping[str, str | os.PathLike[str]],
) -> ServerRoots:
    """Validate explicit roots without deriving one root from another."""

    bases = validate_root_bases(root_bases)
    roots = ServerRoots(
        canonical=_validate_project_root("canonical", canonical_root, root_bases=bases),
        work=_validate_project_root("work", work_root, root_bases=bases),
        checkpoint=_validate_project_root("checkpoint", checkpoint_root, root_bases=bases),
    )
    if len({str(roots.canonical), str(roots.work), str(roots.checkpoint)}) != 3:
        raise ValueError("canonical, work, and checkpoint roots must be distinct")
    return roots


def ensure_existing_roots(
    roots: ServerRoots,
    *,
    root_bases: Mapping[str, str | os.PathLike[str]],
    require_writable: bool,
) -> None:
    """Check roots after bootstrap; never create them here."""

    bases = validate_root_bases(root_bases)
    for kind, path in (
        ("canonical", roots.canonical),
        ("work", roots.work),
        ("checkpoint", roots.checkpoint),
    ):
        if not path.is_dir():
            raise FileNotFoundError(f"{kind} root does not exist or is not a directory: {path}")
        resolved = path.resolve(strict=True)
        resolved_base = Path(str(bases[kind])).resolve(strict=True)
        try:
            resolved.relative_to(resolved_base)
        except ValueError as exc:
            raise ValueError(f"resolved {kind} root escapes {resolved_base}: {resolved}") from exc
        if resolved == resolved_base:
            raise ValueError(f"refusing broad {kind} root: {resolved}")
        if require_writable and not os.access(resolved, os.W_OK | os.X_OK):
            raise PermissionError(f"{kind} root is not writable/searchable: {resolved}")


def file_fingerprints(paths: Sequence[Path]) -> dict[str, dict[str, int | str]]:
    fingerprints: dict[str, dict[str, int | str]] = {}
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"required file is missing: {path}")
        fingerprints[str(path)] = {"sha256": sha256_file(path), "size": path.stat().st_size}
    return fingerprints


def is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _validate_artifact_storage(*, phase: str, path: Path, roots: ServerRoots) -> None:
    resolved = path.resolve(strict=True)
    canonical = roots.canonical.resolve(strict=True)
    work = roots.work.resolve(strict=True)
    checkpoint = roots.checkpoint.resolve(strict=True)
    if phase == "acquire":
        if not is_relative_to(resolved, canonical):
            raise ValueError(f"acquisition artifact is outside canonical NAS storage: {resolved}")
        return
    allowed = (canonical, work, checkpoint) if phase == "audit" else (work, checkpoint)
    if not any(is_relative_to(resolved, root) for root in allowed):
        raise ValueError(f"{phase} artifact is outside its allowed storage roots: {resolved}")
    if phase not in {"audit", "acquire"} and is_relative_to(resolved, roots.raw.resolve()):
        raise ValueError(f"{phase} cannot write into immutable raw storage: {resolved}")


def validate_receipt(
    receipt_path: Path,
    *,
    expected_cli_phase: str,
    expected_run_id: str,
    workflow_phase: str,
    roots: ServerRoots,
) -> dict[str, Any]:
    """Validate the scientific command's atomic output receipt.

    Receipts list small, checksum-bearing artifact or manifest files.  Directory
    outputs such as Zarr stores must be represented by a validated inventory file.
    """

    receipt = load_json(receipt_path)
    if receipt.get("schema_version") != 1:
        raise ValueError("phase receipt schema_version must equal 1")
    if receipt.get("phase") != expected_cli_phase:
        raise ValueError(
            f"phase receipt names {receipt.get('phase')!r}; expected {expected_cli_phase!r}"
        )
    if receipt.get("run_id") != expected_run_id:
        raise ValueError("phase receipt run_id does not match the queue run")
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("phase receipt must contain at least one artifact")

    seen: set[str] = set()
    validated: list[dict[str, Any]] = []
    for index, item in enumerate(artifacts):
        if not isinstance(item, dict):
            raise ValueError(f"receipt artifact {index} must be an object")
        path_value = item.get("path")
        expected_hash = item.get("sha256")
        expected_size = item.get("size")
        if not isinstance(path_value, str) or not Path(path_value).is_absolute():
            raise ValueError(f"receipt artifact {index} path must be absolute")
        if path_value in seen:
            raise ValueError(f"duplicate receipt artifact: {path_value}")
        seen.add(path_value)
        if not isinstance(expected_hash, str) or not SHA256_PATTERN.fullmatch(expected_hash):
            raise ValueError(f"receipt artifact {index} has invalid SHA-256")
        if not isinstance(expected_size, int) or expected_size < 0:
            raise ValueError(f"receipt artifact {index} has invalid size")
        artifact_path = Path(path_value)
        if not artifact_path.is_file():
            raise FileNotFoundError(f"receipt artifact is not a regular file: {artifact_path}")
        _validate_artifact_storage(phase=workflow_phase, path=artifact_path, roots=roots)
        actual_size = artifact_path.stat().st_size
        actual_hash = sha256_file(artifact_path)
        if actual_size != expected_size or actual_hash != expected_hash:
            raise ValueError(f"receipt artifact failed size/hash validation: {artifact_path}")
        validated.append({"path": path_value, "sha256": actual_hash, "size": actual_size})
    receipt["artifacts"] = validated
    return receipt


def validate_success_marker(marker: Mapping[str, Any], *, expected_phase_hash: str) -> None:
    if marker.get("schema_version") != 1 or marker.get("status") != "succeeded":
        raise ValueError("invalid success marker")
    if marker.get("phase_hash") != expected_phase_hash:
        raise ValueError(
            "existing success marker was produced by a different source/configuration; "
            "use a new run id"
        )
    artifacts = marker.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("success marker contains no validated artifacts")
    for item in artifacts:
        if not isinstance(item, dict):
            raise ValueError("invalid artifact in success marker")
        path = Path(str(item.get("path", "")))
        expected_size = item.get("size")
        expected_hash = item.get("sha256")
        if not path.is_file() or path.stat().st_size != expected_size:
            raise ValueError(f"completed artifact is missing or changed: {path}")
        if sha256_file(path) != expected_hash:
            raise ValueError(f"completed artifact hash changed: {path}")


def phase_hash(
    *,
    phase_name: str,
    command: Sequence[str],
    source_manifest_sha256: str,
    config_fingerprints: Mapping[str, Any],
    dependency_marker_sha256: Mapping[str, str],
    roots: ServerRoots,
) -> str:
    return sha256_json(
        {
            "schema_version": 1,
            "phase": phase_name,
            "command": list(command),
            "source_manifest_sha256": source_manifest_sha256,
            "config_fingerprints": config_fingerprints,
            "dependency_marker_sha256": dependency_marker_sha256,
            "roots": {
                "canonical": str(roots.canonical),
                "work": str(roots.work),
                "checkpoint": str(roots.checkpoint),
            },
        }
    )
