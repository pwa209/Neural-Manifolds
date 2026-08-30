from pathlib import Path

import pytest

from neural_manifolds.data.models import assert_no_embedded_secrets
from neural_manifolds.data.registry import load_dataset_registry

REGISTRY = Path(__file__).parents[1] / "configs" / "datasets.yaml"


def test_registry_contains_nine_version_pinned_datasets() -> None:
    registry = load_dataset_registry(REGISTRY)
    assert len(registry.datasets) == 9
    assert len({dataset.id for dataset in registry.datasets}) == 9
    assert all(dataset.source.version != "latest" for dataset in registry.datasets)
    assert registry.get("psiconnect").source.version == "1.2.1"
    assert registry.get("propofol_fmri").source.revision == (
        "9c36d2c59d58fbbced4af6d0413d22a6ea5c4880"
    )


def test_registry_records_real_access_and_licence_restrictions() -> None:
    registry = load_dataset_registry(REGISTRY)
    cogitate = registry.get("cogitate_meeg")
    assert cogitate.access.mode == "account_required"
    assert cogitate.source.provider == "gated"
    osf = registry.get("somatosensory_report_task")
    assert osf.source.mutable_upstream is True
    assert osf.license.spdx == "NOASSERTION"
    assert osf.license.status == "unresolved"


def test_credential_shaped_configuration_is_rejected() -> None:
    with pytest.raises(ValueError, match="credential-shaped"):
        assert_no_embedded_secrets({"source": {"api_token": "do-not-store"}})
    with pytest.raises(ValueError, match="credential-shaped"):
        assert_no_embedded_secrets(
            {"url": "https://example.test/file?X-Amz-Credential=signed-value"}
        )
