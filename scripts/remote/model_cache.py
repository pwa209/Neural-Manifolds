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
import os
import re
import shlex
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

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


def run_git(git: str, *arguments: str, cwd: Path | None = None) -> str:
    result = subprocess.run(
        [git, *arguments],
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode:
        stderr = result.stderr.strip()
        raise RuntimeError(f"git {' '.join(arguments)} failed: {stderr}")
    return result.stdout.strip()


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


def tracked_inventory(git: str, checkout: Path) -> dict[str, Any]:
    output = subprocess.run(
        [git, "ls-files", "-z"],
        cwd=checkout,
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
    names = sorted(item.decode("utf-8") for item in output.split(b"\0") if item)
    files: list[dict[str, Any]] = []
    for name in names:
        path = checkout / name
        if not path.is_file():
            raise FileNotFoundError(f"tracked source file is missing: {path}")
        files.append({"path": name, "sha256": sha256_file(path), "size": path.stat().st_size})
    payload = {"schema_version": 1, "files": files}
    atomic_write(checkout / "SOURCE_MANIFEST.json", json.dumps(payload, indent=2) + "\n")
    return {
        "path": str(checkout / "SOURCE_MANIFEST.json"),
        "sha256": sha256_file(checkout / "SOURCE_MANIFEST.json"),
        "file_count": len(files),
    }


def ensure_checkout(
    *, git: str, name: str, repository: str, revision: str, target: Path, apply: bool
) -> dict[str, Any]:
    if target.is_dir():
        actual = run_git(git, "rev-parse", "HEAD", cwd=target)
        if actual != revision:
            raise RuntimeError(f"cached {name} checkout is {actual}, expected {revision}")
        manifest = target / "SOURCE_MANIFEST.json"
        if not manifest.is_file():
            if not apply:
                raise RuntimeError(f"cached {name} source has no inventory: {manifest}")
            return tracked_inventory(git, target)
        return {
            "path": str(manifest),
            "sha256": sha256_file(manifest),
            "file_count": len(json.loads(manifest.read_text(encoding="utf-8"))["files"]),
        }
    if not apply:
        return {"path": str(target / "SOURCE_MANIFEST.json"), "status": "would_create"}

    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.parent / f".{revision}.partial"
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
    run_git(git, "fetch", "--quiet", "--depth", "1", "origin", revision, cwd=partial)
    run_git(git, "checkout", "--quiet", "--force", "--detach", "FETCH_HEAD", cwd=partial)
    actual = run_git(git, "rev-parse", "HEAD", cwd=partial)
    if actual != revision:
        raise RuntimeError(f"fetched {name} revision {actual}, expected {revision}")
    inventory = tracked_inventory(git, partial)
    partial.rename(target)
    inventory["path"] = str(target / "SOURCE_MANIFEST.json")
    return inventory


def _download_once(url: str, partial: Path, *, timeout: int = 120) -> None:
    existing = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": "neural-manifolds-model-bootstrap/1"}
    if existing:
        headers["Range"] = f"bytes={existing}-"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        status = getattr(response, "status", response.getcode())
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
    def validate(path: Path) -> tuple[str, int]:
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

    if target.is_file():
        digest, size = validate(target)
        return {"path": str(target), "sha256": digest, "size": size}
    if not apply:
        return {"path": str(target), "status": "would_download", "url": url}

    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(f".{target.name}.partial")
    last_error: BaseException | None = None
    for attempt in range(1, 4):
        try:
            _download_once(url, partial)
            digest, size = validate(partial)
            os.replace(partial, target)
            return {"path": str(target), "sha256": digest, "size": size}
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
    return value


def main() -> int:
    args = parser().parse_args()
    config = Path(args.models).resolve(strict=True)
    work_root = Path(args.work_root).resolve(strict=True)
    models = load_models(config)
    cache = work_root / "cache" / "models"
    apply = args.mode == "apply"

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


if __name__ == "__main__":
    raise SystemExit(main())
