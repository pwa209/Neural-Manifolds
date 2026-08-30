"""Content manifests and immutable-release validation."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterator
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from neural_manifolds.provenance import sha256_file

MANIFEST_DIRECTORY = ".acquisition"
MANIFEST_JSON = f"{MANIFEST_DIRECTORY}/manifest.json"
MANIFEST_SHA256 = f"{MANIFEST_DIRECTORY}/MANIFEST.sha256"
PROVENANCE_JSON = f"{MANIFEST_DIRECTORY}/provenance.json"
COMPLETION_MARKER = f"{MANIFEST_DIRECTORY}/COMPLETE.json"
EXCLUDED_ROOTS = {".git", MANIFEST_DIRECTORY}


class ManifestError(RuntimeError):
    """Raised when a raw release differs from its committed manifest."""


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        with suppress(FileNotFoundError):
            os.unlink(temporary_name)
        raise


def _safe_relative_file(root: Path, path: Path) -> str:
    relative = path.relative_to(root).as_posix()
    if path.is_symlink():
        try:
            resolved = path.resolve(strict=True)
        except FileNotFoundError as error:
            raise ManifestError(f"broken data symlink: {relative}") from error
        try:
            resolved.relative_to(root.resolve())
        except ValueError as error:
            raise ManifestError(f"data symlink escapes release root: {relative}") from error
    return relative


def iter_raw_files(root: str | Path) -> Iterator[tuple[str, Path]]:
    """Yield logical raw files, excluding acquisition metadata and DataLad internals."""

    release = Path(root).resolve()
    if not release.is_dir():
        raise ManifestError(f"release is not a directory: {release}")
    for current, directory_names, file_names in os.walk(release, topdown=True, followlinks=False):
        current_path = Path(current)
        if current_path == release:
            directory_names[:] = sorted(
                name for name in directory_names if name not in EXCLUDED_ROOTS
            )
        else:
            directory_names.sort()
        for directory_name in tuple(directory_names):
            directory_path = current_path / directory_name
            if directory_path.is_symlink():
                relative = directory_path.relative_to(release).as_posix()
                raise ManifestError(f"directory symlinks are forbidden in raw releases: {relative}")
        for file_name in sorted(file_names):
            path = current_path / file_name
            if path.name.endswith(".part"):
                raise ManifestError(f"unfinished partial file present: {path.relative_to(release)}")
            if path.is_symlink() and not path.is_file():
                raise ManifestError(f"non-file data symlink present: {path.relative_to(release)}")
            if path.is_file():
                yield _safe_relative_file(release, path), path


def build_manifest(
    root: str | Path,
    *,
    dataset_id: str,
    release_version: str,
    registry_sha256: str,
    source: dict[str, Any],
    retrieved_at: str | None = None,
) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    total_bytes = 0
    for relative, path in iter_raw_files(root):
        size = path.stat().st_size
        files.append({"path": relative, "size": size, "sha256": sha256_file(path)})
        total_bytes += size
    if not files:
        raise ManifestError("refusing to publish an empty raw release")
    return {
        "schema_version": 1,
        "dataset_id": dataset_id,
        "release_version": release_version,
        "retrieved_at": retrieved_at or datetime.now(UTC).isoformat(),
        "registry_sha256": registry_sha256,
        "source": source,
        "file_count": len(files),
        "total_bytes": total_bytes,
        "files": files,
    }


def write_manifest(root: str | Path, manifest: dict[str, Any]) -> dict[str, str]:
    release = Path(root)
    json_path = release / MANIFEST_JSON
    checksum_path = release / MANIFEST_SHA256
    manifest_text = json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    checksum_text = "".join(
        f"{entry['sha256']}  {entry['size']}  {entry['path']}\n" for entry in manifest["files"]
    )
    _atomic_write_text(json_path, manifest_text)
    _atomic_write_text(checksum_path, checksum_text)
    return {
        "manifest_json_sha256": sha256_file(json_path),
        "manifest_list_sha256": sha256_file(checksum_path),
    }


def write_completion_marker(
    root: str | Path,
    *,
    dataset_id: str,
    release_version: str,
    manifest_hashes: dict[str, str],
) -> None:
    marker = {
        "schema_version": 1,
        "dataset_id": dataset_id,
        "release_version": release_version,
        "completed_at": datetime.now(UTC).isoformat(),
        "immutable": True,
        **manifest_hashes,
    }
    _atomic_write_text(
        Path(root) / COMPLETION_MARKER,
        json.dumps(marker, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
    )


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ManifestError(f"cannot read valid JSON from {path}") from error
    if not isinstance(value, dict):
        raise ManifestError(f"JSON object expected in {path}")
    return value


def validate_release(
    root: str | Path,
    *,
    expected_dataset_id: str | None = None,
    expected_release_version: str | None = None,
) -> dict[str, Any]:
    """Rehash every logical raw file and reject missing, extra, or altered content."""

    release = Path(root).resolve()
    marker_path = release / COMPLETION_MARKER
    manifest_path = release / MANIFEST_JSON
    checksum_path = release / MANIFEST_SHA256
    provenance_path = release / PROVENANCE_JSON
    for required in (marker_path, manifest_path, checksum_path, provenance_path):
        if not required.is_file():
            raise ManifestError(f"missing acquisition metadata: {required}")
    marker = _load_json(marker_path)
    manifest = _load_json(manifest_path)
    if marker.get("manifest_json_sha256") != sha256_file(manifest_path):
        raise ManifestError("manifest.json hash does not match completion marker")
    if marker.get("manifest_list_sha256") != sha256_file(checksum_path):
        raise ManifestError("MANIFEST.sha256 hash does not match completion marker")
    dataset_id = manifest.get("dataset_id")
    release_version = manifest.get("release_version")
    if marker.get("immutable") is not True:
        raise ManifestError("completion marker does not declare an immutable release")
    if marker.get("dataset_id") != dataset_id:
        raise ManifestError("dataset id differs between manifest and completion marker")
    if marker.get("release_version") != release_version:
        raise ManifestError("release version differs between manifest and completion marker")
    source = manifest.get("source")
    if not isinstance(source, dict) or not isinstance(source.get("provenance_sha256"), str):
        raise ManifestError("manifest does not bind acquisition provenance")
    if source["provenance_sha256"] != sha256_file(provenance_path):
        raise ManifestError("provenance.json hash does not match manifest")
    if expected_dataset_id is not None and dataset_id != expected_dataset_id:
        raise ManifestError(f"dataset id mismatch: {dataset_id!r} != {expected_dataset_id!r}")
    if expected_release_version is not None and release_version != expected_release_version:
        raise ManifestError(
            f"release version mismatch: {release_version!r} != {expected_release_version!r}"
        )
    entries = manifest.get("files")
    if not isinstance(entries, list):
        raise ManifestError("manifest files must be a list")
    expected: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise ManifestError("malformed file entry in manifest")
        relative = entry["path"]
        if relative in expected:
            raise ManifestError(f"duplicate manifest path: {relative}")
        target = (release / relative).resolve()
        try:
            target.relative_to(release)
        except ValueError as error:
            raise ManifestError(f"manifest path escapes release root: {relative}") from error
        expected[relative] = entry
    expected_checksum_text = "".join(
        f"{entry.get('sha256')}  {entry.get('size')}  {relative}\n"
        for relative, entry in expected.items()
    )
    try:
        observed_checksum_text = checksum_path.read_text(encoding="utf-8")
    except OSError as error:
        raise ManifestError(f"cannot read checksum inventory: {checksum_path}") from error
    if observed_checksum_text != expected_checksum_text:
        raise ManifestError("MANIFEST.sha256 does not correspond to manifest.json")
    actual = dict(iter_raw_files(release))
    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    if missing or extra:
        raise ManifestError(f"release inventory drift; missing={missing}, extra={extra}")
    total_bytes = 0
    for relative, entry in expected.items():
        path = actual[relative]
        size = path.stat().st_size
        total_bytes += size
        if size != entry.get("size"):
            raise ManifestError(f"size mismatch for {relative}: {size} != {entry.get('size')}")
        digest = sha256_file(path)
        if digest != entry.get("sha256"):
            raise ManifestError(f"SHA-256 mismatch for {relative}")
    if len(expected) != manifest.get("file_count") or total_bytes != manifest.get("total_bytes"):
        raise ManifestError("manifest aggregate counts do not match file entries")
    return {
        "dataset_id": dataset_id,
        "release_version": release_version,
        "file_count": len(expected),
        "total_bytes": total_bytes,
        "manifest_json_sha256": sha256_file(manifest_path),
        "valid": True,
    }
