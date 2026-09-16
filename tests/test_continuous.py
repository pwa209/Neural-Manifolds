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


def test_inventory_completion_schedules_qc_and_recovery(controller):
    controller.add("inventory", "study", Path("/example"), "identity")
    task = next(iter(controller.state["tasks"].values()))
    task["status"] = "complete"
    audit = Path(task["output"]) / "inventory.json"
    atomic_write_json(audit, {"files": []})
    controller.plan()
    controller.plan()
    children = [t for t in controller.state["tasks"].values() if t["kind"] != "inventory"]
    assert {t["kind"] for t in children} == {"signal_qc", "recovery"}
    assert len(children) == 2
    assert all(t["input_identity"] == sha256_file(audit) for t in children)


def test_revised_graph_registers_all_ready_branches(controller, monkeypatch):
    import yaml

    monkeypatch.setenv("NM_REVISED_EXECUTION", "1")
    config = Path("configs/revised_execution.yaml")
    config.parent.mkdir()
    config.write_text(
        yaml.safe_dump({"source_policy": {"study": {}}, "surrogate_repetitions": 100})
    )
    controller.add("revised_cohort", "study", Path("/example"), "cohort-identity")
    cohort = next(iter(controller.state["tasks"].values()))
    cohort["status"] = "complete"
    atomic_write_json(Path(cohort["output"]) / "cohort.json", {"units": []})
    controller.plan()
    measure = next(t for t in controller.state["tasks"].values() if t["kind"] == "revised_measure")
    measure["status"] = "complete"
    atomic_write_json(Path(measure["output"]) / "measurement.json", {"records": []})
    controller.plan()
    controller.plan()
    kinds = [t["kind"] for t in controller.state["tasks"].values()]
    assert kinds.count("revised_controls") == 100
    for kind in (
        "revised_transfer",
        "revised_recovery",
        "revised_sensitivities",
        "revised_perturbation_measure",
    ):
        assert kinds.count(kind) == 1
    perturb = next(
        t for t in controller.state["tasks"].values() if t["kind"] == "revised_perturbation_measure"
    )
    perturb["status"] = "complete"
    atomic_write_json(Path(perturb["output"]) / "perturbation_measurements.json", {})
    controller.plan()
    assert sum(t["kind"] == "revised_robustness" for t in controller.state["tasks"].values()) == 1
    controller.report()
    report = json.loads((controller.root / "progress.json").read_text())
    assert report["revised_core_drivers_implemented"] is True
    assert report["phase_map"]["R6"]["tasks_by_status"]["ready"] == 103


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
    (raw / "archive.7z").write_bytes(b"not extracted")
    result = inventory(raw, tmp_path / "inventory.json")
    assert not result["analysis_ready"]
    assert result["issues"][0]["reason"] == "archive_adapter_required"
    assert json.loads((tmp_path / "inventory.json").read_text())["records"] == []
