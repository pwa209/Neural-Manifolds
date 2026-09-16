import importlib.util
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts/alliance" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dream_admission_excludes_private_revoked_and_old_amendments():
    module = load_script("resolve_registry")
    text = "Set ID,Accessibility,Latest amendment,Revoked,Data URL\n"
    text += "1,Open,TRUE,FALSE,https://example.org/a\n"
    text += "2,Private,TRUE,FALSE,https://example.org/b\n"
    text += "3,Open,FALSE,FALSE,https://example.org/c\n"
    text += "4,Open,TRUE,TRUE,https://example.org/d\n"
    assert [r["Set ID"] for r in module.eligible_rows(text)] == ["1"]
    with pytest.raises(ValueError, match="schema"):
        module.eligible_rows("wrong,columns\n1,2\n")


def test_v2_has_no_result_gate_or_required_restricted_sources():
    policy = yaml.safe_load((ROOT / "configs/revision_v2.yaml").read_text())
    assert policy["scientific_gates"] is False
    assert policy["five_axes_required"] is False
    assert policy["clinical_required"] is False and policy["fmri_required"] is False
    assert set(policy["initial_datasets"]).isdisjoint(policy["excluded_dataset_ids"])
    assert policy["experience_without_recall"] == "separate_category_not_ordinal"


def test_archive_verifier_rejects_content_drift(tmp_path):
    import hashlib
    import json

    module = load_script("verify_release")
    path = tmp_path / "a.py"
    path.write_text("original")
    hashes = {"a.py": hashlib.sha256(path.read_bytes()).hexdigest()}
    digest = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    (tmp_path / "release_manifest.json").write_text(
        json.dumps({"source_digest": digest, "files": hashes})
    )
    assert module.verify(tmp_path) == digest
    path.write_text("changed")
    with pytest.raises(ValueError, match="Source mismatch"):
        module.verify(tmp_path)


def test_openneuro_budget_counts_content_not_yet_downloaded(tmp_path, monkeypatch):
    from types import SimpleNamespace

    module = load_script("acquire")
    calls = []

    class FakeProvider:
        def check(self):
            return {"status": "ready"}

    class FakeRunner:
        def run(self, argv, **kwargs):
            calls.append(argv)
            return SimpleNamespace(stdout='{"key":"MD5E-s123--abc.edf"}\n')

    monkeypatch.setattr(module, "OpenNeuroProvider", FakeProvider)
    monkeypatch.setattr(module, "CommandRunner", FakeRunner)
    manager = SimpleNamespace(provider=lambda _: FakeProvider())
    dataset = SimpleNamespace(
        id="example",
        source=SimpleNamespace(
            version="1.0.0", repository_url="https://example.org", revision="abc"
        ),
    )
    assert module.source_size(manager, dataset, tmp_path) == 123
    assert ["git", "annex", "find", "--anything", "--json"] in calls
