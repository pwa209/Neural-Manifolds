from pathlib import Path

import pytest

from neural_manifolds.data.manifest import (
    MANIFEST_DIRECTORY,
    ManifestError,
    build_manifest,
    validate_release,
    write_completion_marker,
    write_manifest,
)
from neural_manifolds.provenance import sha256_file


def _publish(release: Path) -> None:
    (release / "subject" / "raw.bin").parent.mkdir(parents=True, exist_ok=True)
    (release / "subject" / "raw.bin").write_bytes(b"immutable raw bytes")
    provenance = release / MANIFEST_DIRECTORY / "provenance.json"
    provenance.parent.mkdir(parents=True, exist_ok=True)
    provenance.write_text('{"source":"test"}\n', encoding="utf-8")
    manifest = build_manifest(
        release,
        dataset_id="test_data",
        release_version="1.0.0",
        registry_sha256="a" * 64,
        source={"provenance_sha256": sha256_file(provenance)},
    )
    hashes = write_manifest(release, manifest)
    write_completion_marker(
        release,
        dataset_id="test_data",
        release_version="1.0.0",
        manifest_hashes=hashes,
    )


def test_manifest_detects_content_and_provenance_drift(tmp_path: Path) -> None:
    release = tmp_path / "release"
    _publish(release)
    assert validate_release(release)["valid"] is True

    (release / "subject" / "raw.bin").write_bytes(b"changed")
    with pytest.raises(ManifestError, match=r"size mismatch|SHA-256 mismatch"):
        validate_release(release)

    _publish(release)
    (release / MANIFEST_DIRECTORY / "provenance.json").write_text(
        '{"source":"altered"}\n', encoding="utf-8"
    )
    with pytest.raises(ManifestError, match="provenance"):
        validate_release(release)


def test_manifest_detects_extra_files(tmp_path: Path) -> None:
    release = tmp_path / "release"
    _publish(release)
    (release / "unexpected.txt").write_text("drift", encoding="utf-8")
    with pytest.raises(ManifestError, match="inventory drift"):
        validate_release(release)
