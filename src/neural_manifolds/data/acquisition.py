"""Transactional publication of immutable raw dataset releases."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from filelock import FileLock
from filelock import Timeout as FileLockTimeout

from neural_manifolds.provenance import atomic_write_json, sha256_file

from .http import HttpClient
from .manifest import (
    COMPLETION_MARKER,
    MANIFEST_DIRECTORY,
    build_manifest,
    iter_raw_files,
    validate_release,
    write_completion_marker,
    write_manifest,
)
from .models import DatasetSpec
from .providers import AccessBlocked, Provider, make_provider
from .registry import DatasetRegistry


class AcquisitionError(RuntimeError):
    """A dataset could not be safely published."""


class ImmutableReleaseError(AcquisitionError):
    """An existing raw release cannot be changed in place."""


@contextmanager
def _dataset_lock(path: Path):
    try:
        with FileLock(str(path), timeout=1):
            yield
    except FileLockTimeout as error:
        raise AcquisitionError(f"another acquisition holds the dataset lock: {path}") from error


@dataclass(frozen=True)
class AcquisitionResult:
    dataset_id: str
    release_version: str
    status: str
    release_path: str
    details: dict[str, Any]


def _absolute_safe_root(root: str | Path) -> Path:
    candidate = Path(root).expanduser()
    if not candidate.is_absolute():
        raise AcquisitionError("raw root must be an absolute path on the target storage")
    resolved = candidate.resolve()
    anchor = Path(resolved.anchor)
    if resolved == anchor:
        raise AcquisitionError("refusing to use a filesystem root as the raw-data root")
    return resolved


def _json_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _normalise_license(value: str) -> str:
    return "".join(character for character in value.lower() if character.isalnum())


def _licenses_compatible(expected: str, observed: str) -> bool:
    expected_value = _normalise_license(expected)
    observed_value = _normalise_license(observed)
    if expected_value.startswith("cc0") and observed_value.startswith("cc0"):
        return True
    return expected_value in observed_value or observed_value in expected_value


def _remove_write_permissions(root: Path) -> dict[str, Any]:
    """Remove every write bit from a completely validated release tree.

    This is deliberately called only after ``validate_release`` has rehashed the
    completed staging tree.  Symlinks are skipped because their in-tree targets
    are visited independently and symlink permission bits are not portable.
    """

    release = root.resolve(strict=True)
    if not release.is_dir() or release.is_symlink():
        raise ImmutableReleaseError(f"release is not a regular directory: {release}")
    regular_files = 0
    directories = 0
    for current, _directory_names, file_names in os.walk(release, topdown=False, followlinks=False):
        current_path = Path(current)
        for file_name in file_names:
            path = current_path / file_name
            if path.is_symlink():
                continue
            metadata = path.stat()
            if not stat.S_ISREG(metadata.st_mode):
                raise ImmutableReleaseError(
                    f"special files are forbidden in immutable releases: {path}"
                )
            os.chmod(path, stat.S_IMODE(metadata.st_mode) & ~0o222)
            if stat.S_IMODE(path.stat().st_mode) & 0o222:
                raise ImmutableReleaseError(f"could not remove write permissions from {path}")
            regular_files += 1
        metadata = current_path.stat()
        os.chmod(current_path, stat.S_IMODE(metadata.st_mode) & ~0o222)
        if stat.S_IMODE(current_path.stat().st_mode) & 0o222:
            raise ImmutableReleaseError(
                f"could not remove directory write permissions from {current_path}"
            )
        directories += 1
    return {
        "policy": "all_write_bits_removed_after_complete_manifest_validation",
        "regular_files": regular_files,
        "directories": directories,
        "read_only": True,
    }


def _validate_read_only_permissions(root: Path) -> dict[str, Any]:
    release = root.resolve(strict=True)
    writable: list[str] = []
    regular_files = 0
    directories = 0
    for current, _directory_names, file_names in os.walk(release, topdown=True, followlinks=False):
        current_path = Path(current)
        for name in file_names:
            path = current_path / name
            if path.is_symlink():
                continue
            mode = stat.S_IMODE(path.stat().st_mode)
            if mode & 0o222:
                writable.append(path.relative_to(release).as_posix())
            if path.is_file():
                regular_files += 1
        if stat.S_IMODE(current_path.stat().st_mode) & 0o222:
            relative = current_path.relative_to(release).as_posix() or "."
            writable.append(relative)
        directories += 1
    if writable:
        raise ImmutableReleaseError(
            "published release has writable entries; examples=" + repr(writable[:10])
        )
    return {
        "policy": "all_write_bits_removed_after_complete_manifest_validation",
        "regular_files": regular_files,
        "directories": directories,
        "read_only": True,
    }


class AcquisitionManager:
    def __init__(
        self,
        registry: DatasetRegistry,
        *,
        client: HttpClient | None = None,
        provider_factory: Any = make_provider,
    ) -> None:
        self.registry = registry
        defaults = registry.model.defaults
        self.client = client or HttpClient(
            maximum_attempts=defaults.maximum_attempts,
            connect_timeout_seconds=defaults.connect_timeout_seconds,
            read_timeout_seconds=defaults.read_timeout_seconds,
        )
        self.provider_factory = provider_factory

    def provider(self, dataset: DatasetSpec) -> Provider:
        return self.provider_factory(dataset, self.client)

    def plan(self, dataset: DatasetSpec, root: str | Path) -> AcquisitionResult:
        raw_root = _absolute_safe_root(root)
        release = raw_root / dataset.id / dataset.source.version
        return AcquisitionResult(
            dataset_id=dataset.id,
            release_version=dataset.source.version,
            status="planned" if dataset.access.mode == "open" else "access_blocked",
            release_path=str(release),
            details={
                "provider": dataset.source.provider,
                "accession": dataset.source.accession,
                "doi": dataset.source.doi,
                "license": asdict_like(dataset.license),
                "access": asdict_like(dataset.access),
                "mutable_upstream": dataset.source.mutable_upstream,
                "immutable_local_release": True,
            },
        )

    def check(self, dataset: DatasetSpec) -> AcquisitionResult:
        check = self.provider(dataset).check()
        status = str(check.get("status", "ready"))
        return AcquisitionResult(
            dataset_id=dataset.id,
            release_version=dataset.source.version,
            status=status,
            release_path="",
            details=check,
        )

    def acquire(
        self,
        dataset: DatasetSpec,
        root: str | Path,
        *,
        dry_run: bool = False,
    ) -> AcquisitionResult:
        if dry_run:
            return self.plan(dataset, root)
        if dataset.access.mode != "open":
            raise AccessBlocked(
                f"{dataset.id} requires account-mediated access. {dataset.access.instructions}"
            )
        raw_root = _absolute_safe_root(root)
        release = raw_root / dataset.id / dataset.source.version
        staging = raw_root / ".staging" / dataset.id / dataset.source.version
        lock_path = raw_root / ".locks" / f"{dataset.id}--{dataset.source.version}.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with _dataset_lock(lock_path):
            if release.exists():
                if not (release / COMPLETION_MARKER).is_file():
                    raise ImmutableReleaseError(
                        f"release path exists without a completion marker; inspect manually: {release}"
                    )
                validation = validate_release(
                    release,
                    expected_dataset_id=dataset.id,
                    expected_release_version=dataset.source.version,
                )
                permissions = _remove_write_permissions(release)
                return AcquisitionResult(
                    dataset_id=dataset.id,
                    release_version=dataset.source.version,
                    status="already_complete",
                    release_path=str(release),
                    details={**validation, "permissions": permissions},
                )
            release.parent.mkdir(parents=True, exist_ok=True)
            staging.parent.mkdir(parents=True, exist_ok=True)
            if (staging / COMPLETION_MARKER).is_file():
                validation = validate_release(
                    staging,
                    expected_dataset_id=dataset.id,
                    expected_release_version=dataset.source.version,
                )
                permissions = _remove_write_permissions(staging)
                os.replace(staging, release)
                return AcquisitionResult(
                    dataset_id=dataset.id,
                    release_version=dataset.source.version,
                    status="published_recovered_stage",
                    release_path=str(release),
                    details={**validation, "permissions": permissions},
                )
            started_at = datetime.now(UTC).isoformat()
            provider_metadata = self.provider(dataset).materialize(staging)
            content_validation = self._validate_content(dataset, staging)
            licence_validation = self._validate_dataset_license(dataset, staging, provider_metadata)
            acquisition_record = {
                "schema_version": 1,
                "dataset_id": dataset.id,
                "release_version": dataset.source.version,
                "started_at": started_at,
                "materialised_at": datetime.now(UTC).isoformat(),
                "registry_path": str(self.registry.source_path),
                "registry_sha256": self.registry.sha256,
                "source": asdict_like(dataset.source),
                "license": asdict_like(dataset.license),
                "access": asdict_like(dataset.access),
                "provider": provider_metadata,
                "content_validation": content_validation,
                "license_validation": licence_validation,
                "upstream_mutable": dataset.source.mutable_upstream,
                "local_release_immutable": True,
            }
            acquisition_dir = staging / MANIFEST_DIRECTORY
            acquisition_dir.mkdir(parents=True, exist_ok=True)
            provenance_path = acquisition_dir / "provenance.json"
            atomic_write_json(provenance_path, acquisition_record)
            source_summary = {
                "provider": dataset.source.provider,
                "accession": dataset.source.accession,
                "version": dataset.source.version,
                "doi": dataset.source.doi,
                "landing_url": dataset.source.landing_url,
                "mutable_upstream": dataset.source.mutable_upstream,
                "provider_metadata_sha256": _json_sha256(provider_metadata),
                "provenance_sha256": sha256_file(provenance_path),
                "revision": dataset.source.revision,
                "license": dataset.license.spdx,
                "license_status": dataset.license.status,
            }
            manifest = build_manifest(
                staging,
                dataset_id=dataset.id,
                release_version=dataset.source.version,
                registry_sha256=self.registry.sha256,
                source=source_summary,
            )
            manifest_hashes = write_manifest(staging, manifest)
            write_completion_marker(
                staging,
                dataset_id=dataset.id,
                release_version=dataset.source.version,
                manifest_hashes=manifest_hashes,
            )
            validation = validate_release(
                staging,
                expected_dataset_id=dataset.id,
                expected_release_version=dataset.source.version,
            )
            permissions = _remove_write_permissions(staging)
            os.replace(staging, release)
            return AcquisitionResult(
                dataset_id=dataset.id,
                release_version=dataset.source.version,
                status="published",
                release_path=str(release),
                details={
                    **validation,
                    "content_validation": content_validation,
                    "permissions": permissions,
                },
            )

    @staticmethod
    def _validate_content(dataset: DatasetSpec, staging: Path) -> dict[str, Any]:
        missing_paths = [
            path for path in dataset.validation.required_paths if not (staging / path).exists()
        ]
        missing_globs = [
            pattern
            for pattern in dataset.validation.required_globs
            if not any(staging.glob(pattern))
        ]
        if missing_paths or missing_globs:
            raise AcquisitionError(
                f"content validation failed for {dataset.id}; "
                f"missing_paths={missing_paths}, missing_globs={missing_globs}"
            )
        inventory = list(iter_raw_files(staging))
        file_count = len(inventory)
        total_bytes = sum(path.stat().st_size for _, path in inventory)
        if file_count < dataset.validation.minimum_files:
            raise AcquisitionError(
                f"too few files for {dataset.id}: {file_count} < {dataset.validation.minimum_files}"
            )
        if total_bytes < dataset.validation.minimum_bytes:
            raise AcquisitionError(
                f"too few bytes for {dataset.id}: {total_bytes} < "
                f"{dataset.validation.minimum_bytes}"
            )
        return {"file_count": file_count, "total_bytes": total_bytes, "valid": True}

    @staticmethod
    def _validate_dataset_license(
        dataset: DatasetSpec, staging: Path, provider_metadata: dict[str, Any]
    ) -> dict[str, Any]:
        observed: str | None = None
        description = staging / "dataset_description.json"
        if description.is_file():
            try:
                payload = json.loads(description.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise AcquisitionError("invalid BIDS dataset_description.json") from error
            if isinstance(payload, dict) and isinstance(payload.get("License"), str):
                observed = payload["License"]
        metadata = provider_metadata.get("metadata")
        if (
            observed is None
            and isinstance(metadata, dict)
            and isinstance(metadata.get("license"), str)
        ):
            observed = metadata["license"]
        expected = dataset.license.spdx
        if observed and expected != "NOASSERTION" and not _licenses_compatible(expected, observed):
            raise AcquisitionError(
                f"dataset licence differs from registry: expected={expected!r}, observed={observed!r}"
            )
        return {
            "expected": expected,
            "observed": observed,
            "status": "unresolved" if expected == "NOASSERTION" else "compatible",
        }

    def validate(self, dataset: DatasetSpec, root: str | Path) -> AcquisitionResult:
        raw_root = _absolute_safe_root(root)
        release = raw_root / dataset.id / dataset.source.version
        details = validate_release(
            release,
            expected_dataset_id=dataset.id,
            expected_release_version=dataset.source.version,
        )
        permissions = _validate_read_only_permissions(release)
        return AcquisitionResult(
            dataset_id=dataset.id,
            release_version=dataset.source.version,
            status="valid",
            release_path=str(release),
            details={**details, "permissions": permissions},
        )


def asdict_like(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        result = value.model_dump(mode="json")
        if isinstance(result, dict):
            return result
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    raise TypeError(f"cannot convert {type(value).__name__} to a mapping")
