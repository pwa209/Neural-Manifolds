"""Content-addressed foundation-model cache used by bootstrap_models.sh.

This helper performs no authentication and never resolves a moving branch/tag for
execution. Repository checkouts use exact Git object ids from configs/models.yaml.
BrainLM weights are deliberately unavailable in the core stage and require exact
Hugging Face revision and file hashes before the fMRI stage can materialise them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shlex
import stat
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Any

import yaml
from filelock import FileLock
from filelock import Timeout as FileLockTimeout

REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
SHA1_PATTERN = re.compile(r"^[0-9a-f]{40}$")

APPROVED_REPOSITORIES = {
    "labram_base": "https://github.com/935963004/LaBraM.git",
    "brainlm": "https://github.com/vandijklab/BrainLM.git",
}
LABRAM_URL_PREFIX = "https://raw.githubusercontent.com/935963004/LaBraM/"
BRAINLM_URL_PREFIX = "https://huggingface.co/vandijklab/brainlm/resolve/"
BRAINLM_LICENSE = "CC-BY-NC-ND-4.0"
SOURCE_MANIFEST_NAME = "SOURCE_MANIFEST.json"
MODEL_CACHE_LOCK_NAME = ".model_cache.lock"
DEFAULT_LOCK_TIMEOUT_SECONDS = 30.0
LOCAL_GIT_TIMEOUT_SECONDS = 60
NETWORK_GIT_TIMEOUT_SECONDS = 300


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


@contextmanager
def exclusive_model_cache_lock(
    cache: Path, *, timeout: float = DEFAULT_LOCK_TIMEOUT_SECONDS
) -> Iterator[Path]:
    """Serialise all mutations of the shared, content-addressed model cache.

    ``filelock`` uses an operating-system lock, so an abandoned lock file does
    not block recovery after a process exits.  The bounded timeout also prevents
    a second bootstrap from waiting indefinitely behind a long or wedged download.
    """

    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("model-cache lock timeout must be finite and positive")
    cache.mkdir(parents=True, exist_ok=True)
    lock_path = cache / MODEL_CACHE_LOCK_NAME
    lock = FileLock(str(lock_path), timeout=timeout)
    try:
        lock.acquire()
    except FileLockTimeout as exc:
        raise RuntimeError(
            f"model cache is locked by another bootstrap after {timeout:g} seconds: {lock_path}"
        ) from exc
    try:
        yield lock_path
    finally:
        lock.release()


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def git_blob_sha1(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    size = path.stat().st_size
    digest = hashlib.sha1(usedforsecurity=False)
    digest.update(f"blob {size}\0".encode())
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def run_git(
    git: str,
    *arguments: str,
    cwd: Path | None = None,
    timeout: int = LOCAL_GIT_TIMEOUT_SECONDS,
) -> str:
    try:
        result = subprocess.run(
            [git, *arguments],
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"git {' '.join(arguments)} exceeded the bounded {timeout}-second timeout"
        ) from exc
    if result.returncode:
        stderr = result.stderr.strip()
        raise RuntimeError(f"git {' '.join(arguments)} failed: {stderr}")
    return result.stdout.strip()


def require_clean_pinned_checkout(git: str, checkout: Path, revision: str) -> None:
    result = subprocess.run(
        [git, "diff", "--no-ext-diff", "--quiet", revision, "--"],
        cwd=checkout,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=LOCAL_GIT_TIMEOUT_SECONDS,
    )
    if result.returncode == 1:
        raise RuntimeError(f"cached source checkout has tracked changes: {checkout}")
    if result.returncode:
        raise RuntimeError(f"could not verify cached source checkout: {result.stderr.strip()}")


def load_models(path: Path) -> dict[str, dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    if not isinstance(document, dict) or document.get("schema_version") != 1:
        raise ValueError("models configuration must use schema_version 1")
    models = document.get("models")
    if not isinstance(models, dict):
        raise ValueError("models configuration has no models mapping")
    required = {"labram_base", "brainlm"}
    if not required <= set(models):
        raise ValueError(f"models configuration is missing: {sorted(required - set(models))}")
    validated: dict[str, dict[str, Any]] = {}
    for name in required:
        item = models[name]
        if not isinstance(item, dict):
            raise ValueError(f"model {name} must be a mapping")
        repository = item.get("repository")
        revision = item.get("revision")
        if repository != APPROVED_REPOSITORIES[name]:
            raise ValueError(f"model {name} repository is not the approved upstream")
        if not isinstance(revision, str) or not REVISION_PATTERN.fullmatch(revision):
            raise ValueError(f"model {name} revision must be an exact Git object id")
        if item.get("trainable") is not False:
            raise ValueError(f"model {name} must remain frozen")
        validated[name] = dict(item)

    labram = validated["labram_base"]
    checkpoint_url = labram.get("checkpoint_url")
    blob_sha1 = labram.get("checkpoint_git_blob_sha1")
    revision = str(labram["revision"])
    if not isinstance(checkpoint_url, str) or not checkpoint_url.startswith(
        f"{LABRAM_URL_PREFIX}{revision}/"
    ):
        raise ValueError("LaBraM checkpoint URL must include the exact configured revision")
    if not isinstance(blob_sha1, str) or not SHA1_PATTERN.fullmatch(blob_sha1):
        raise ValueError("LaBraM checkpoint Git-blob SHA-1 is required")
    return validated


def tracked_names(git: str, checkout: Path) -> list[str]:
    output = subprocess.run(
        [git, "ls-files", "-z"],
        cwd=checkout,
        check=True,
        stdout=subprocess.PIPE,
        timeout=LOCAL_GIT_TIMEOUT_SECONDS,
    ).stdout
    names = sorted(item.decode("utf-8") for item in output.split(b"\0") if item)
    if SOURCE_MANIFEST_NAME in names:
        raise RuntimeError(
            f"upstream source tracks reserved cache inventory name: {SOURCE_MANIFEST_NAME}"
        )
    return names


def safe_tracked_path(checkout: Path, name: str) -> Path:
    relative = PurePosixPath(name)
    if (
        relative.is_absolute()
        or not relative.parts
        or relative == PurePosixPath(".")
        or ".." in relative.parts
        or ".git" in relative.parts
    ):
        raise ValueError(f"unsafe tracked source path: {name!r}")
    path = checkout.joinpath(*relative.parts)
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"tracked source path is not a regular file: {path}")
    checkout_root = checkout.resolve(strict=True)
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(checkout_root):
        raise RuntimeError(f"tracked source path escapes its checkout: {path}")
    return path


def verify_source_tree_coverage(
    checkout: Path,
    tracked: list[str],
    *,
    manifest_expected: bool | None,
) -> None:
    """Reject every physical checkout entry not covered by the source inventory.

    Git status is insufficient here because ignored files are deliberately omitted,
    yet an ignored Python module could still affect imports from the cached source.
    Only Git's own top-level ``.git`` directory and our generated manifest are exempt.
    ``manifest_expected=None`` permits a resumable pre-publication checkout with or
    without a manifest left by an interrupted prior attempt.
    """

    if checkout.is_symlink() or not checkout.is_dir():
        raise RuntimeError(f"source checkout must be a regular directory: {checkout}")
    physical_files: list[str] = []
    manifest_seen = False
    git_directory_seen = False

    def visit(directory: Path, relative_parts: tuple[str, ...]) -> None:
        nonlocal git_directory_seen, manifest_seen
        try:
            with os.scandir(directory) as iterator:
                entries = sorted(iterator, key=lambda item: item.name)
        except OSError as exc:
            raise RuntimeError(f"could not inspect cached source directory: {directory}") from exc
        for entry in entries:
            relative = PurePosixPath(*relative_parts, entry.name).as_posix()
            try:
                mode = entry.stat(follow_symlinks=False).st_mode
            except OSError as exc:
                raise RuntimeError(f"could not inspect cached source entry: {relative}") from exc
            if not relative_parts and entry.name == ".git":
                if entry.is_symlink() or not stat.S_ISDIR(mode):
                    raise RuntimeError(
                        f"cached source .git entry is not a regular directory: {checkout}"
                    )
                git_directory_seen = True
                continue
            if entry.is_symlink() or stat.S_ISLNK(mode):
                raise RuntimeError(f"cached source contains a symbolic link: {relative}")
            if stat.S_ISDIR(mode):
                visit(Path(entry.path), (*relative_parts, entry.name))
                continue
            if not stat.S_ISREG(mode):
                raise RuntimeError(f"cached source contains a special file: {relative}")
            if relative == SOURCE_MANIFEST_NAME:
                manifest_seen = True
                continue
            physical_files.append(relative)

    visit(checkout, ())
    if not git_directory_seen:
        raise RuntimeError(f"cached source checkout has no regular .git directory: {checkout}")
    if manifest_expected is True and not manifest_seen:
        raise RuntimeError(
            f"cached source has no regular inventory: {checkout / SOURCE_MANIFEST_NAME}"
        )
    if manifest_expected is False and manifest_seen:
        raise RuntimeError(f"cached source unexpectedly contains an inventory: {checkout}")

    tracked_set = set(tracked)
    physical_set = set(physical_files)
    unexpected = sorted(physical_set - tracked_set)
    if unexpected:
        raise RuntimeError(f"cached source contains untracked or ignored files: {unexpected}")
    missing = sorted(tracked_set - physical_set)
    if missing:
        raise RuntimeError(f"cached source is missing tracked files: {missing}")


def tracked_inventory(git: str, checkout: Path) -> dict[str, Any]:
    names = tracked_names(git, checkout)
    verify_source_tree_coverage(checkout, names, manifest_expected=None)
    files: list[dict[str, Any]] = []
    for name in names:
        path = safe_tracked_path(checkout, name)
        files.append({"path": name, "sha256": sha256_file(path), "size": path.stat().st_size})
    payload = {"schema_version": 1, "files": files}
    manifest = checkout / SOURCE_MANIFEST_NAME
    atomic_write(manifest, json.dumps(payload, indent=2) + "\n")
    verify_source_tree_coverage(checkout, names, manifest_expected=True)
    return {
        "path": str(manifest),
        "sha256": sha256_file(manifest),
        "file_count": len(files),
    }


def verify_tracked_inventory(git: str, checkout: Path, manifest: Path) -> dict[str, Any]:
    """Rehash every pinned source file against its recorded cache inventory."""

    if manifest.is_symlink() or not manifest.is_file():
        raise RuntimeError(f"cached source has no regular inventory: {manifest}")
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cached source inventory is unreadable: {manifest}") from exc
    if not isinstance(payload, dict) or set(payload) != {"schema_version", "files"}:
        raise RuntimeError(f"cached source inventory has an invalid document shape: {manifest}")
    if payload["schema_version"] != 1 or not isinstance(payload["files"], list):
        raise RuntimeError(f"cached source inventory must use schema_version 1: {manifest}")

    entries: dict[str, tuple[str, int]] = {}
    recorded_order: list[str] = []
    for item in payload["files"]:
        if not isinstance(item, dict) or set(item) != {"path", "sha256", "size"}:
            raise RuntimeError(f"cached source inventory contains an invalid entry: {manifest}")
        name, digest, size = item["path"], item["sha256"], item["size"]
        if not isinstance(name, str) or not name:
            raise RuntimeError(f"cached source inventory contains an invalid path: {manifest}")
        if not isinstance(digest, str) or not SHA256_PATTERN.fullmatch(digest):
            raise RuntimeError(f"cached source inventory contains an invalid SHA-256: {manifest}")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise RuntimeError(f"cached source inventory contains an invalid size: {manifest}")
        if name in entries:
            raise RuntimeError(f"cached source inventory contains duplicate path {name!r}")
        entries[name] = (digest, size)
        recorded_order.append(name)

    expected_names = tracked_names(git, checkout)
    if recorded_order != expected_names:
        raise RuntimeError(
            "cached source inventory paths do not exactly match the pinned Git checkout: "
            f"{manifest}"
        )
    verify_source_tree_coverage(checkout, expected_names, manifest_expected=True)
    for name in expected_names:
        path = safe_tracked_path(checkout, name)
        expected_digest, expected_size = entries[name]
        actual_size = path.stat().st_size
        if actual_size != expected_size:
            raise RuntimeError(
                f"cached source size mismatch for {name!r}: {actual_size} != {expected_size}"
            )
        actual_digest = sha256_file(path)
        if actual_digest != expected_digest:
            raise RuntimeError(f"cached source SHA-256 mismatch for {name!r}: {path}")
    return {
        "path": str(manifest),
        "sha256": sha256_file(manifest),
        "file_count": len(expected_names),
    }


def ensure_checkout(
    *, git: str, name: str, repository: str, revision: str, target: Path, apply: bool
) -> dict[str, Any]:
    if target.is_symlink():
        raise RuntimeError(f"cached {name} checkout must not be a symbolic link: {target}")
    if target.is_dir():
        configured_remote = run_git(git, "remote", "get-url", "origin", cwd=target)
        if configured_remote != repository:
            raise RuntimeError(f"cached {name} checkout has unexpected remote: {configured_remote}")
        actual = run_git(git, "rev-parse", "HEAD", cwd=target)
        if actual != revision:
            raise RuntimeError(f"cached {name} checkout is {actual}, expected {revision}")
        require_clean_pinned_checkout(git, target, revision)
        manifest = target / SOURCE_MANIFEST_NAME
        if not manifest.is_file():
            raise RuntimeError(f"cached {name} source has no inventory: {manifest}")
        return verify_tracked_inventory(git, target, manifest)
    if target.exists():
        raise RuntimeError(f"cached {name} checkout is not a directory: {target}")
    if not apply:
        return {"path": str(target / SOURCE_MANIFEST_NAME), "status": "would_create"}

    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.parent / f".{revision}.partial"
    if partial.is_symlink():
        raise RuntimeError(f"partial checkout must not be a symbolic link: {partial}")
    if partial.exists() and not (partial / ".git").is_dir():
        raise RuntimeError(f"partial checkout is not resumable Git state: {partial}")
    partial.mkdir(parents=True, exist_ok=True)
    if not (partial / ".git").is_dir():
        run_git(git, "init", "--quiet", cwd=partial)
        run_git(git, "remote", "add", "origin", repository, cwd=partial)
    else:
        configured_remote = run_git(git, "remote", "get-url", "origin", cwd=partial)
        if configured_remote != repository:
            raise RuntimeError(f"partial checkout has unexpected remote: {configured_remote}")
    run_git(
        git,
        "fetch",
        "--quiet",
        "--depth",
        "1",
        "origin",
        revision,
        cwd=partial,
        timeout=NETWORK_GIT_TIMEOUT_SECONDS,
    )
    run_git(git, "checkout", "--quiet", "--force", "--detach", "FETCH_HEAD", cwd=partial)
    actual = run_git(git, "rev-parse", "HEAD", cwd=partial)
    if actual != revision:
        raise RuntimeError(f"fetched {name} revision {actual}, expected {revision}")
    inventory = tracked_inventory(git, partial)
    partial.rename(target)
    inventory["path"] = str(target / SOURCE_MANIFEST_NAME)
    return inventory


def _download_once(url: str, partial: Path, *, timeout: int = 120) -> None:
    existing = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": "neural-manifolds-model-bootstrap/1"}
    if existing:
        headers["Range"] = f"bytes={existing}-"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        status = getattr(response, "status", response.getcode())
        if status not in {200, 206}:
            raise urllib.error.URLError(f"unexpected HTTP status {status} for {url}")
        if status == 206:
            content_range = response.headers.get("Content-Range", "")
            match = re.fullmatch(r"bytes (\d+)-(\d+)/(?:\d+|\*)", content_range)
            expected_start = existing if existing else 0
            if match is None or int(match.group(1)) != expected_start:
                raise ValueError(
                    f"server returned an invalid Content-Range for resume: {content_range!r}"
                )
        mode = "ab" if existing and status == 206 else "wb"
        with partial.open(mode) as stream:
            while chunk := response.read(1024 * 1024):
                stream.write(chunk)
            stream.flush()
            os.fsync(stream.fileno())


def download_verified(
    *,
    url: str,
    target: Path,
    expected_sha256: str | None,
    expected_git_blob_sha1: str | None,
    minimum_bytes: int,
    apply: bool,
) -> dict[str, Any]:
    if expected_sha256 is None and expected_git_blob_sha1 is None:
        raise ValueError("a pinned artifact digest is required before download")
    if expected_sha256 is not None and not SHA256_PATTERN.fullmatch(expected_sha256):
        raise ValueError("expected artifact SHA-256 is invalid")
    if expected_git_blob_sha1 is not None and not SHA1_PATTERN.fullmatch(expected_git_blob_sha1):
        raise ValueError("expected artifact Git-blob SHA-1 is invalid")
    if minimum_bytes < 0:
        raise ValueError("minimum artifact size must not be negative")

    def validate(path: Path) -> tuple[str, int]:
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"download is not a regular file: {path}")
        size = path.stat().st_size
        if size < minimum_bytes:
            raise ValueError(f"download is unexpectedly small ({size} bytes): {path}")
        actual_sha256 = sha256_file(path)
        if expected_sha256 is not None and actual_sha256 != expected_sha256:
            raise ValueError(f"download SHA-256 mismatch: {path}")
        if expected_git_blob_sha1 is not None:
            actual_blob = git_blob_sha1(path)
            if actual_blob != expected_git_blob_sha1:
                raise ValueError(f"download Git-blob SHA-1 mismatch: {path}")
        return actual_sha256, size

    if target.is_symlink():
        raise RuntimeError(f"download target must not be a symbolic link: {target}")
    if target.is_file():
        digest, size = validate(target)
        return {"path": str(target), "sha256": digest, "size": size}
    if target.exists():
        raise RuntimeError(f"download target is not a regular file: {target}")
    if not apply:
        return {"path": str(target), "status": "would_download", "url": url}

    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(f".{target.name}.partial")
    if partial.is_symlink() or (partial.exists() and not partial.is_file()):
        raise RuntimeError(f"download partial must be a regular file: {partial}")

    # A previous process can be interrupted after receiving and fsyncing the last
    # byte but before the atomic rename.  Verify that state before sending a Range
    # request; otherwise an exact-length partial provokes a repeatable HTTP 416.
    if partial.is_file():
        try:
            digest, size = validate(partial)
        except ValueError:
            pass
        else:
            os.replace(partial, target)
            return {"path": str(target), "sha256": digest, "size": size}

    last_error: BaseException | None = None
    for attempt in range(1, 4):
        try:
            _download_once(url, partial)
            digest, size = validate(partial)
            os.replace(partial, target)
            return {"path": str(target), "sha256": digest, "size": size}
        except urllib.error.HTTPError as exc:
            last_error = exc
            # An invalid full-length or oversized partial may also elicit 416.
            # It has already failed the pinned digest above, so restart cleanly.
            if exc.code == 416:
                partial.unlink(missing_ok=True)
            if attempt == 3:
                break
            time.sleep(2**attempt)
        except (OSError, ValueError, urllib.error.URLError) as exc:
            last_error = exc
            if isinstance(exc, ValueError):
                partial.unlink(missing_ok=True)
            if attempt == 3:
                break
            time.sleep(2**attempt)
    raise RuntimeError(f"download failed after three bounded attempts: {url}: {last_error}")


def safe_relative_name(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"unsafe checkpoint file name: {value!r}")
    return path


def brainlm_checkpoint_spec(model: dict[str, Any]) -> tuple[str, list[dict[str, str]]]:
    revision = model.get("checkpoint_revision")
    files = model.get("checkpoint_files")
    if not isinstance(revision, str) or not REVISION_PATTERN.fullmatch(revision):
        raise ValueError(
            "BrainLM checkpoint download is deferred until configs/models.yaml records "
            "an exact Hugging Face commit as checkpoint_revision"
        )
    if not isinstance(files, list) or not files:
        raise ValueError("BrainLM checkpoint_files with exact SHA-256 values are required")
    validated: list[dict[str, str]] = []
    for item in files:
        if not isinstance(item, dict):
            raise ValueError("BrainLM checkpoint_files entries must be mappings")
        name, url, sha256 = item.get("name"), item.get("url"), item.get("sha256")
        if not all(isinstance(value, str) for value in (name, url, sha256)):
            raise ValueError("BrainLM checkpoint file requires name, url, and sha256")
        safe_relative_name(name)
        if not url.startswith(f"{BRAINLM_URL_PREFIX}{revision}/"):
            raise ValueError("BrainLM checkpoint URL must include the exact checkpoint revision")
        if not SHA256_PATTERN.fullmatch(sha256):
            raise ValueError("BrainLM checkpoint file has invalid SHA-256")
        validated.append({"name": name, "url": url, "sha256": sha256})
    return revision, validated


def write_environment(path: Path, variables: dict[str, str]) -> None:
    lines = [f"export {name}={shlex.quote(value)}" for name, value in sorted(variables.items())]
    atomic_write(path, "\n".join(lines) + "\n")


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--models", required=True)
    value.add_argument("--work-root", required=True)
    value.add_argument("--git", default="git")
    value.add_argument("--stage", choices=("core", "fmri"), required=True)
    value.add_argument("--mode", choices=("check", "dry", "apply"), required=True)
    value.add_argument(
        "--lock-timeout-seconds",
        type=float,
        default=DEFAULT_LOCK_TIMEOUT_SECONDS,
        help="bounded wait for the shared model-cache writer lock (apply mode only)",
    )
    return value


def operate_cache(
    *,
    args: argparse.Namespace,
    models: dict[str, dict[str, Any]],
    cache: Path,
    apply: bool,
) -> int:
    sources: dict[str, dict[str, Any]] = {}
    for name in ("labram_base", "brainlm"):
        model = models[name]
        target = cache / "sources" / name / str(model["revision"])
        sources[name] = ensure_checkout(
            git=args.git,
            name=name,
            repository=str(model["repository"]),
            revision=str(model["revision"]),
            target=target,
            apply=apply,
        )

    labram = models["labram_base"]
    labram_checkpoint = (
        cache
        / "checkpoints"
        / "labram_base"
        / str(labram["revision"])
        / Path(str(labram["checkpoint_url"])).name
    )
    labram_artifact = download_verified(
        url=str(labram["checkpoint_url"]),
        target=labram_checkpoint,
        expected_sha256=(
            str(labram["checkpoint_sha256"])
            if isinstance(labram.get("checkpoint_sha256"), str)
            else None
        ),
        expected_git_blob_sha1=str(labram["checkpoint_git_blob_sha1"]),
        minimum_bytes=1_000_000,
        apply=apply,
    )

    brainlm = models["brainlm"]
    brainlm_checkpoint_dir = cache / "checkpoints" / "brainlm"
    brainlm_artifacts: list[dict[str, Any]] = []
    brainlm_checkpoint_revision: str | None = None
    if args.stage == "fmri":
        brainlm_checkpoint_revision, checkpoint_files = brainlm_checkpoint_spec(brainlm)
        brainlm_checkpoint_dir /= brainlm_checkpoint_revision
        for item in checkpoint_files:
            relative = safe_relative_name(item["name"])
            brainlm_artifacts.append(
                download_verified(
                    url=item["url"],
                    target=brainlm_checkpoint_dir.joinpath(*relative.parts),
                    expected_sha256=item["sha256"],
                    expected_git_blob_sha1=None,
                    minimum_bytes=1,
                    apply=apply,
                )
            )

    manifest = {
        "schema_version": 1,
        "stage": args.stage,
        "models": {
            "labram_base": {
                "repository": labram["repository"],
                "revision": labram["revision"],
                "source": sources["labram_base"],
                "checkpoint": labram_artifact,
                "source_license": labram.get("source_license"),
                "trainable": False,
            },
            "brainlm": {
                "repository": brainlm["repository"],
                "revision": brainlm["revision"],
                "source": sources["brainlm"],
                "checkpoint_status": (
                    "verified_for_fmri" if args.stage == "fmri" else "deferred_until_fmri"
                ),
                "checkpoint_revision": brainlm_checkpoint_revision,
                "checkpoint_files": brainlm_artifacts,
                "usage_license": BRAINLM_LICENSE,
                "commercial_use": False,
                "derivative_redistribution": False,
                "trainable": False,
            },
        },
    }
    if args.mode == "dry":
        print(json.dumps(manifest, indent=2))
        return 0
    if args.mode == "check":
        print(json.dumps(manifest, indent=2))
        return 0

    manifest_path = cache / "MODEL_MANIFEST.json"
    atomic_write(manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    manifest_sha = sha256_file(manifest_path)
    atomic_write(cache / "MODEL_MANIFEST.sha256", f"{manifest_sha}  MODEL_MANIFEST.json\n")
    environment = {
        "NEURAL_MANIFOLDS_MODEL_MANIFEST": str(manifest_path),
        "NEURAL_MANIFOLDS_LABRAM_SOURCE": str(
            cache / "sources" / "labram_base" / str(labram["revision"])
        ),
        "NEURAL_MANIFOLDS_LABRAM_CHECKPOINT": str(labram_checkpoint),
        "NEURAL_MANIFOLDS_BRAINLM_SOURCE": str(
            cache / "sources" / "brainlm" / str(brainlm["revision"])
        ),
        "NEURAL_MANIFOLDS_BRAINLM_LICENSE": BRAINLM_LICENSE,
    }
    if args.stage == "fmri":
        environment["NEURAL_MANIFOLDS_BRAINLM_CHECKPOINT_DIR"] = str(brainlm_checkpoint_dir)
    environment_path = cache / "model_paths.env"
    write_environment(environment_path, environment)
    print(f"model_manifest={manifest_path}")
    print(f"model_manifest_sha256={manifest_sha}")
    print(f"model_environment={environment_path}")
    return 0


def main() -> int:
    args = parser().parse_args()
    config = Path(args.models).resolve(strict=True)
    work_root = Path(args.work_root).resolve(strict=True)
    models = load_models(config)
    cache = work_root / "cache" / "models"
    apply = args.mode == "apply"
    if apply:
        with exclusive_model_cache_lock(cache, timeout=args.lock_timeout_seconds):
            return operate_cache(args=args, models=models, cache=cache, apply=True)
    return operate_cache(args=args, models=models, cache=cache, apply=False)


if __name__ == "__main__":
    raise SystemExit(main())
