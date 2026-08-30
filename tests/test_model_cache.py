from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import urllib.error
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from filelock import FileLock
from filelock import Timeout as FileLockTimeout


def _load_model_cache() -> ModuleType:
    path = Path(__file__).parents[1] / "scripts" / "remote" / "model_cache.py"
    specification = importlib.util.spec_from_file_location("remote_model_cache", path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


model_cache = _load_model_cache()


@pytest.fixture
def pinned_checkout(tmp_path: Path) -> tuple[str, Path, str, Path]:
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    subprocess.run(["git", "init", "--quiet"], cwd=checkout, check=True)
    subprocess.run(
        ["git", "config", "user.email", "model-cache-tests@example.invalid"],
        cwd=checkout,
        check=True,
    )
    subprocess.run(["git", "config", "user.name", "Model Cache Tests"], cwd=checkout, check=True)
    subprocess.run(
        ["git", "remote", "add", "origin", "https://example.invalid/fixture.git"],
        cwd=checkout,
        check=True,
    )
    tracked = checkout / "tracked.py"
    tracked.write_bytes(b"value = 1\n")
    nested = checkout / "package" / "nested.py"
    nested.parent.mkdir()
    nested.write_bytes(b"nested = True\n")
    subprocess.run(["git", "add", "tracked.py", "package/nested.py"], cwd=checkout, check=True)
    subprocess.run(["git", "commit", "--quiet", "-m", "fixture"], cwd=checkout, check=True)
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=checkout,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    model_cache.tracked_inventory("git", checkout)
    return "git", checkout, revision, tracked


def test_reused_checkout_rehashes_every_manifest_entry(
    pinned_checkout: tuple[str, Path, str, Path],
) -> None:
    git, checkout, revision, tracked = pinned_checkout
    manifest = checkout / model_cache.SOURCE_MANIFEST_NAME
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert len(payload["files"]) == 2
    payload["files"][-1]["sha256"] = "0" * 64
    manifest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        model_cache.ensure_checkout(
            git=git,
            name="fixture",
            repository="https://example.invalid/fixture.git",
            revision=revision,
            target=checkout,
            apply=False,
        )

    assert tracked.read_bytes() == b"value = 1\n"


def test_reused_checkout_rejects_source_and_manifest_reblessing(
    pinned_checkout: tuple[str, Path, str, Path],
) -> None:
    git, checkout, revision, tracked = pinned_checkout
    tracked.write_bytes(b"value = 2\n")
    model_cache.tracked_inventory(git, checkout)

    with pytest.raises(RuntimeError, match="tracked changes"):
        model_cache.ensure_checkout(
            git=git,
            name="fixture",
            repository="https://example.invalid/fixture.git",
            revision=revision,
            target=checkout,
            apply=True,
        )


def test_reused_checkout_never_regenerates_a_missing_manifest(
    pinned_checkout: tuple[str, Path, str, Path],
) -> None:
    git, checkout, revision, _tracked = pinned_checkout
    manifest = checkout / model_cache.SOURCE_MANIFEST_NAME
    manifest.unlink()

    with pytest.raises(RuntimeError, match="has no inventory"):
        model_cache.ensure_checkout(
            git=git,
            name="fixture",
            repository="https://example.invalid/fixture.git",
            revision=revision,
            target=checkout,
            apply=True,
        )
    assert not manifest.exists()


def test_reused_checkout_rejects_changed_provenance_remote(
    pinned_checkout: tuple[str, Path, str, Path],
) -> None:
    git, checkout, revision, _tracked = pinned_checkout
    subprocess.run(
        [git, "remote", "set-url", "origin", "https://example.invalid/replaced.git"],
        cwd=checkout,
        check=True,
    )

    with pytest.raises(RuntimeError, match="unexpected remote"):
        model_cache.ensure_checkout(
            git=git,
            name="fixture",
            repository="https://example.invalid/fixture.git",
            revision=revision,
            target=checkout,
            apply=False,
        )


def test_reused_checkout_rejects_untracked_source_file(
    pinned_checkout: tuple[str, Path, str, Path],
) -> None:
    git, checkout, revision, _tracked = pinned_checkout
    (checkout / "shadow_module.py").write_text("override = True\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="untracked or ignored files"):
        model_cache.ensure_checkout(
            git=git,
            name="fixture",
            repository="https://example.invalid/fixture.git",
            revision=revision,
            target=checkout,
            apply=False,
        )


def test_reused_checkout_rejects_git_ignored_source_file(
    pinned_checkout: tuple[str, Path, str, Path],
) -> None:
    git, checkout, revision, _tracked = pinned_checkout
    exclude = checkout / ".git" / "info" / "exclude"
    with exclude.open("a", encoding="utf-8") as stream:
        stream.write("\npackage/__pycache__/\n")
    ignored = checkout / "package" / "__pycache__" / "shadow.pyc"
    ignored.parent.mkdir()
    ignored.write_bytes(b"ignored-but-executable-cache-content")

    with pytest.raises(RuntimeError, match="untracked or ignored files"):
        model_cache.ensure_checkout(
            git=git,
            name="fixture",
            repository="https://example.invalid/fixture.git",
            revision=revision,
            target=checkout,
            apply=False,
        )


def test_reused_checkout_rejects_untracked_symlink(
    pinned_checkout: tuple[str, Path, str, Path],
) -> None:
    git, checkout, revision, tracked = pinned_checkout
    link = checkout / "shadow_module.py"
    try:
        link.symlink_to(tracked.name)
    except OSError as exc:  # Windows may deny symlink creation without developer mode.
        pytest.skip(f"symlink creation unavailable: {exc}")

    with pytest.raises(RuntimeError, match="symbolic link"):
        model_cache.ensure_checkout(
            git=git,
            name="fixture",
            repository="https://example.invalid/fixture.git",
            revision=revision,
            target=checkout,
            apply=False,
        )


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFO creation is unavailable")
def test_reused_checkout_rejects_special_file(
    pinned_checkout: tuple[str, Path, str, Path],
) -> None:
    git, checkout, revision, _tracked = pinned_checkout
    os.mkfifo(checkout / "injection.pipe")

    with pytest.raises(RuntimeError, match="special file"):
        model_cache.ensure_checkout(
            git=git,
            name="fixture",
            repository="https://example.invalid/fixture.git",
            revision=revision,
            target=checkout,
            apply=False,
        )


def test_fresh_inventory_rejects_untracked_file_before_manifest_publication(
    pinned_checkout: tuple[str, Path, str, Path],
) -> None:
    git, checkout, _revision, _tracked = pinned_checkout
    manifest = checkout / model_cache.SOURCE_MANIFEST_NAME
    manifest.unlink()
    (checkout / "shadow_module.py").write_text("override = True\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="untracked or ignored files"):
        model_cache.tracked_inventory(git, checkout)
    assert not manifest.exists()


def test_fresh_inventory_rechecks_coverage_after_manifest_publication(
    pinned_checkout: tuple[str, Path, str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    git, checkout, _revision, _tracked = pinned_checkout
    manifest = checkout / model_cache.SOURCE_MANIFEST_NAME
    manifest.unlink()
    original_atomic_write = model_cache.atomic_write

    def publish_then_inject(path: Path, content: str) -> None:
        original_atomic_write(path, content)
        (checkout / "late_shadow_module.py").write_text("override = True\n", encoding="utf-8")

    monkeypatch.setattr(model_cache, "atomic_write", publish_then_inject)
    with pytest.raises(RuntimeError, match="untracked or ignored files"):
        model_cache.tracked_inventory(git, checkout)
    assert manifest.is_file()


def test_valid_complete_partial_is_published_without_http(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    content = b"complete pinned artifact"
    expected = hashlib.sha256(content).hexdigest()
    target = tmp_path / "artifact.bin"
    partial = tmp_path / ".artifact.bin.partial"
    partial.write_bytes(content)

    def unexpected_download(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("a valid complete partial must not issue an HTTP request")

    monkeypatch.setattr(model_cache, "_download_once", unexpected_download)
    result = model_cache.download_verified(
        url="https://example.invalid/artifact.bin",
        target=target,
        expected_sha256=expected,
        expected_git_blob_sha1=None,
        minimum_bytes=len(content),
        apply=True,
    )

    assert target.read_bytes() == content
    assert not partial.exists()
    assert result == {"path": str(target), "sha256": expected, "size": len(content)}


def test_download_once_appends_only_a_matching_content_range(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    partial = tmp_path / ".artifact.bin.partial"
    partial.write_bytes(b"abc")

    class Response:
        status = 206

        def __init__(self) -> None:
            self.headers = {"Content-Range": "bytes 3-5/6"}
            self.sent = False

        def __enter__(self) -> Response:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        @staticmethod
        def getcode() -> int:
            return 206

        def read(self, _size: int) -> bytes:
            if not self.sent:
                self.sent = True
                return b"def"
            return b""

    def urlopen(request: object, *, timeout: int) -> Response:
        assert timeout == 120
        assert request.get_header("Range") == "bytes=3-"
        return Response()

    monkeypatch.setattr(model_cache.urllib.request, "urlopen", urlopen)
    model_cache._download_once("https://example.invalid/artifact.bin", partial)
    assert partial.read_bytes() == b"abcdef"


def test_http_416_on_invalid_partial_restarts_once_from_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    content = b"verified replacement"
    expected = hashlib.sha256(content).hexdigest()
    target = tmp_path / "artifact.bin"
    partial = tmp_path / ".artifact.bin.partial"
    partial.write_bytes(b"invalid full-length data")
    calls = 0

    def download_once(url: str, destination: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise urllib.error.HTTPError(url, 416, "range not satisfiable", {}, None)
        assert not destination.exists()
        destination.write_bytes(content)

    monkeypatch.setattr(model_cache, "_download_once", download_once)
    monkeypatch.setattr(model_cache.time, "sleep", lambda _seconds: None)
    result = model_cache.download_verified(
        url="https://example.invalid/artifact.bin",
        target=target,
        expected_sha256=expected,
        expected_git_blob_sha1=None,
        minimum_bytes=1,
        apply=True,
    )

    assert calls == 2
    assert result["sha256"] == expected
    assert target.read_bytes() == content


def test_model_cache_writer_lock_is_bounded_and_recoverable(tmp_path: Path) -> None:
    cache = tmp_path / "models"
    cache.mkdir()
    external_lock = FileLock(str(cache / model_cache.MODEL_CACHE_LOCK_NAME))

    with (
        external_lock,
        pytest.raises(RuntimeError, match="locked by another bootstrap"),
        model_cache.exclusive_model_cache_lock(cache, timeout=0.01),
    ):
        pytest.fail("contended writer lock must not be entered")

    with model_cache.exclusive_model_cache_lock(cache, timeout=0.1) as path:
        assert path == cache / model_cache.MODEL_CACHE_LOCK_NAME


def test_model_cache_lock_excludes_another_process(tmp_path: Path) -> None:
    cache = tmp_path / "models"
    child = (
        "import sys\n"
        "from filelock import FileLock, Timeout\n"
        "try:\n"
        "    FileLock(sys.argv[1]).acquire(timeout=0.05)\n"
        "except Timeout:\n"
        "    raise SystemExit(23)\n"
        "raise SystemExit(0)\n"
    )

    with model_cache.exclusive_model_cache_lock(cache, timeout=0.1) as path:
        result = subprocess.run([sys.executable, "-c", child, str(path)], check=False, timeout=5)
    assert result.returncode == 23


def test_apply_holds_writer_lock_through_cache_operation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "models.yaml"
    config.write_text("schema_version: 1\n", encoding="utf-8")
    work_root = tmp_path / "work"
    work_root.mkdir()
    args = SimpleNamespace(
        models=str(config),
        work_root=str(work_root),
        mode="apply",
        lock_timeout_seconds=0.1,
    )

    class FakeParser:
        @staticmethod
        def parse_args() -> SimpleNamespace:
            return args

    def operation(**kwargs: object) -> int:
        cache = kwargs["cache"]
        assert isinstance(cache, Path)
        competitor = FileLock(str(cache / model_cache.MODEL_CACHE_LOCK_NAME))
        with pytest.raises(FileLockTimeout):
            competitor.acquire(timeout=0)
        return 17

    monkeypatch.setattr(model_cache, "parser", FakeParser)
    monkeypatch.setattr(model_cache, "load_models", lambda _path: {})
    monkeypatch.setattr(model_cache, "operate_cache", operation)
    assert model_cache.main() == 17

    released = FileLock(str(work_root / "cache" / "models" / model_cache.MODEL_CACHE_LOCK_NAME))
    with released.acquire(timeout=0):
        pass


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan")])
def test_model_cache_writer_lock_rejects_unbounded_timeout(tmp_path: Path, timeout: float) -> None:
    with (
        pytest.raises(ValueError, match="must be finite and positive"),
        model_cache.exclusive_model_cache_lock(tmp_path / "models", timeout=timeout),
    ):
        pytest.fail("invalid timeout must not acquire a lock")


def test_download_refuses_an_unpinned_artifact(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="pinned artifact digest"):
        model_cache.download_verified(
            url="https://example.invalid/unpinned.bin",
            target=tmp_path / "unpinned.bin",
            expected_sha256=None,
            expected_git_blob_sha1=None,
            minimum_bytes=1,
            apply=False,
        )
