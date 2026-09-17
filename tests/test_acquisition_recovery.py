import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_recovery_refuses_download_and_reconciles_only_after_validation(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "recover", Path("scripts/alliance/recover_acquisition.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    dataset = SimpleNamespace(
        id="test", source=SimpleNamespace(version="1"), access=SimpleNamespace(mode="open")
    )
    registry = tmp_path / "registry.yaml"
    registry.write_text("test")
    state_path = tmp_path / "state.json"
    state_path.write_text(
        json.dumps(
            {
                "registry_sha256": module.sha256_file(registry),
                "datasets": {"test": {"status": "downloading"}},
            }
        )
    )
    monkeypatch.setattr(
        module, "load_dataset_registry", lambda _: SimpleNamespace(datasets=[dataset])
    )
    with pytest.raises(ValueError, match="no downloads"):
        module.recover(registry, tmp_path, state_path, "test")
    stage = tmp_path / ".staging/test/1/.acquisition"
    stage.mkdir(parents=True)
    (stage / "COMPLETE.json").write_text("{}")

    def fail(*args):
        raise ValueError("checksum mismatch")

    monkeypatch.setattr(module, "AcquisitionManager", lambda _: SimpleNamespace(acquire=fail))
    with pytest.raises(ValueError, match="checksum mismatch"):
        module.recover(registry, tmp_path, state_path, "test")
    assert json.loads(state_path.read_text())["datasets"]["test"]["status"] == "incomplete"
    from neural_manifolds.data.acquisition import AcquisitionResult

    result = AcquisitionResult("test", "1", "published_recovered_stage", "/test", {})
    monkeypatch.setattr(
        module, "AcquisitionManager", lambda _: SimpleNamespace(acquire=lambda *args: result)
    )
    module.recover(registry, tmp_path, state_path, "test")
    entry = json.loads(state_path.read_text())["datasets"]["test"]
    assert entry["status"] == "complete" and "error" not in entry
