"""Keep synthetic unit fixtures independent of the production deployment environment."""

import pytest


@pytest.fixture(autouse=True)
def isolate_deployment_settings(monkeypatch):
    # Tests that exercise these integrations set their own fixture-specific values.
    # Production validation is not disabled: this fixture exists only under pytest.
    for name in (
        "NM_REVISED_EXECUTION",
        "NM_INVENTORY_SEED_STATES",
        "NM_PSICONNECT_SUBSET_MARKER",
        "NEURAL_MANIFOLDS_MODEL_MANIFEST",
        "NEURAL_MANIFOLDS_MODEL_MANIFEST_SHA256",
    ):
        monkeypatch.delenv(name, raising=False)
