from pathlib import Path
from typing import Any

import pytest

from neural_manifolds.data.acquisition import AcquisitionManager
from neural_manifolds.data.models import DatasetRegistryModel
from neural_manifolds.data.providers import AccessBlocked, Provider
from neural_manifolds.data.registry import DatasetRegistry


class FakeProvider(Provider):
    def check(self) -> dict[str, Any]:
        return {"status": "ready"}

    def materialize(self, staging: Path) -> dict[str, Any]:
        staging.mkdir(parents=True, exist_ok=True)
        (staging / "raw.bin").write_bytes(b"raw")
        return {"metadata": {"license": "CC BY 4.0"}}


def _registry(tmp_path: Path) -> DatasetRegistry:
    model = DatasetRegistryModel.model_validate(
        {
            "schema_version": 1,
            "registry_verified_on": "2026-08-30",
            "defaults": {},
            "datasets": [
                {
                    "id": "example_data",
                    "title": "Example",
                    "role": "test",
                    "modalities": ["eeg"],
                    "source": {
                        "provider": "figshare",
                        "accession": "1",
                        "version": "1",
                        "doi": "10.0000/example.1",
                        "landing_url": "https://example.test/dataset",
                        "api_url": "https://example.test/api/dataset/1",
                    },
                    "license": {
                        "spdx": "CC-BY-4.0",
                        "status": "verified",
                        "source_url": "https://example.test/dataset",
                    },
                    "access": {"mode": "open", "terms_url": "https://example.test/terms"},
                    "validation": {"required_paths": ["raw.bin"], "minimum_bytes": 3},
                    "official_sources": ["https://example.test/dataset"],
                }
            ],
        }
    )
    return DatasetRegistry(model=model, source_path=tmp_path / "registry.yaml", sha256="a" * 64)


def test_acquisition_publishes_then_only_validates_existing_release(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    manager = AcquisitionManager(
        registry, provider_factory=lambda dataset, client: FakeProvider(dataset, client)
    )
    root = tmp_path / "nas"
    first = manager.acquire(registry.datasets[0], root)
    second = manager.acquire(registry.datasets[0], root)
    assert first.status == "published"
    assert second.status == "already_complete"
    assert (root / "example_data" / "1" / ".complete.json").is_file()


def test_dry_run_has_no_filesystem_side_effect(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    manager = AcquisitionManager(registry)
    root = tmp_path / "nas"
    result = manager.acquire(registry.datasets[0], root, dry_run=True)
    assert result.status == "planned"
    assert not root.exists()


def test_non_open_access_is_blocked_before_root_creation(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    dataset = registry.datasets[0].model_copy(
        update={
            "source": registry.datasets[0].source.model_copy(
                update={"provider": "gated", "api_url": "https://example.test/gated"}
            ),
            "access": registry.datasets[0].access.model_copy(
                update={"mode": "account_required", "instructions": "Obtain approval."}
            ),
        }
    )
    root = tmp_path / "nas"
    with pytest.raises(AccessBlocked):
        AcquisitionManager(registry).acquire(dataset, root)
    assert not root.exists()
