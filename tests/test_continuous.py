import io
import json
from pathlib import Path

import pytest

from neural_manifolds.continuous.audit import edf_header, inventory, read_records, safe_member
from neural_manifolds.continuous.controller import Controller, valid_receipt
from neural_manifolds.provenance import atomic_write_json, sha256_file


class Scheduler:
    def __init__(self):
        self.calls = []
        self.jobs = {}
        self.uncertain = False

    def submit(self, spec, name, root, account):
        self.calls.append(name)
        self.jobs[name] = str(100 + len(self.calls))
        if self.uncertain:
            raise TimeoutError("response lost after submission")
        return self.jobs[name]

    def find(self, name):
        return [self.jobs[name]] if name in self.jobs else []

    def state(self, job):
        return "PENDING"


@pytest.fixture
def controller(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    atomic_write_json(tmp_path / "release_manifest.json", {"source_digest": "test-source"})
    acquisition = tmp_path / "acquisition.json"
    atomic_write_json(acquisition, {"status": "complete", "datasets": {}})
    registry = tmp_path / "registry.yaml"
    registry.write_text("test")
    return Controller(
        tmp_path / "state", acquisition, registry, "test-account", scheduler=Scheduler()
    )


def test_uncertain_submission_is_adopted_not_duplicated(controller):
    controller.add("inventory", "study", Path("/example"), "identity")
    controller.scheduler.uncertain = True
    controller.submit_ready()
    assert next(iter(controller.state["tasks"].values()))["status"] == "submitting"
    controller.reconcile()
    controller.submit_ready()
    assert len(controller.scheduler.calls) == 1
    assert next(iter(controller.state["tasks"].values()))["status"] == "submitted"


def test_concurrency_and_deterministic_task_identity(controller):
    for i in range(5):
        controller.add("inventory", str(i), Path("/example"), "same")
        controller.add("inventory", str(i), Path("/example"), "same")
    controller.submit_ready()
    assert len(controller.state["tasks"]) == 5
    assert len(controller.scheduler.calls) == 2


def test_completion_requires_hashed_outputs(controller):
    controller.add("inventory", "study", Path("/example"), "identity")
    task = next(iter(controller.state["tasks"].values()))
    out = Path(task["output"])
    out.mkdir()
    artifact = out / "test.json"
    artifact.write_text("{}")
    atomic_write_json(
        out / "receipt.json",
        {
            "status": "complete",
            "source_digest": "test-source",
            "spec_sha256": sha256_file(Path(task["spec"])),
            "artifacts": {str(artifact): sha256_file(artifact)},
        },
    )
    assert valid_receipt(task)
    artifact.write_text("changed")
    assert not valid_receipt(task)


def test_outcome_values_do_not_control_scheduling(controller):
    controller.add("inventory", "null_result", Path("/example"), "identity")
    controller.state["effect"] = 0
    controller.state["pvalue"] = 0.99
    controller.submit_ready()
    assert len(controller.scheduler.calls) == 1


def test_archive_paths_and_missing_report_codes():
    assert safe_member("Data/PSG/a.edf")
    assert not safe_member("../outside")
    assert not safe_member("/outside")
    assert not safe_member("C:\\outside")
    rows = read_records(
        b"Filename,Case ID,Subject ID,Experience,Last sleep stage,Duration\na.edf,a,1,,2,20\n",
        "test",
    )
    assert not rows[0]["primary_report_stage_candidate"]
    assert rows[0]["Experience"] == ""
    assert rows[0]["analysis_ready"] is False


def test_edf_header_no_subject_fields_emitted():
    fixed = bytearray(b" " * 256)
    fixed[184:192] = b"512     "
    fixed[236:244] = b"20      "
    fixed[244:252] = b"1       "
    fixed[252:256] = b"1   "
    channel = bytearray(b" " * 256)
    channel[:16] = b"Cz              "
    channel[216:224] = b"200     "
    header = edf_header(io.BytesIO(fixed + channel))
    assert header["duration_seconds"] == 20
    assert header["channels"] == ["Cz"]
    assert header["sample_rates"] == [200]
    assert "participant" not in header


def test_inventory_never_admits_metadata_as_scientific_qc(tmp_path):
    raw = tmp_path / "raw"
    (raw / ".acquisition").mkdir(parents=True)
    (raw / ".acquisition/COMPLETE.json").write_text("{}")
    (raw / "archive.rar").write_bytes(b"not extracted")
    result = inventory(raw, tmp_path / "inventory.json")
    assert not result["analysis_ready"]
    assert result["issues"][0]["reason"] == "archive_adapter_required"
    assert json.loads((tmp_path / "inventory.json").read_text())["records"] == []
