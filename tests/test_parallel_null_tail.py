"""Scheduler cutover tests use synthetic paths and a fake Slurm interface."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "alliance" / "parallel_null_tail.py"
SPEC = importlib.util.spec_from_file_location("parallel_null_tail", SCRIPT)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_submit_array_keeps_one_cpu_and_full_array_dependency(tmp_path):
    command = module.submit_array(
        name="parallel1",
        account="example_cpu",
        release=tmp_path / "release",
        output_root=tmp_path / "scratch",
        wrapper=tmp_path / "wrapper.sh",
        plan_path=tmp_path / "plan.json",
        helper=SCRIPT,
        manifest_path=tmp_path / "manifest.json",
        runner=tmp_path / "sealed.py",
        dependency="123",
        count=53,
    )
    assert "--array=0-52%50" in command
    assert "--dependency=afterany:123" in command
    assert "--cpus-per-task=1" in command
    assert "--mem=4G" in command
    assert command[-5:] == [
        "worker",
        "--manifest",
        str(tmp_path / "manifest.json"),
        "--runner",
        str(tmp_path / "sealed.py"),
    ]


def test_cutover_holds_aggregation_before_cancelling_tail(tmp_path, monkeypatch):
    root = tmp_path / "study"
    root.mkdir()
    release = tmp_path / "release"
    release.mkdir()
    runner = root / "sealed.py"
    runner.write_text("# synthetic runner\n")
    wrapper = root / "wrapper.sh"
    wrapper.write_text("# synthetic wrapper\n")
    release_manifest = release / "release_manifest.json"
    release_manifest.write_text("{}")
    plan = {
        "total_replicates": 5000,
        "original_replicates": 100,
        "observed": [{} for _ in range(72)],
        "runner_sha256": module.file_hash(runner),
        "release_manifest_sha256": module.file_hash(release_manifest),
        "analysis_release": str(release.resolve()),
        "output_root": str(tmp_path / "scratch"),
    }
    plan["plan_sha256"] = module.digest(plan)
    plan_path = root / "plan.json"
    plan_path.write_text(json.dumps(plan))
    receipt = {
        "plan_sha256": plan["plan_sha256"],
        "jobs": {
            "retry": {
                "status": "submitted",
                "job_id": "100",
                "environment": {"NM_NULL_RUNNER": str(runner)},
                "command": ["sbatch", "--account=example_cpu", str(wrapper), "worker"],
            },
            "aggregate": {"status": "submitted", "job_id": "200"},
        },
    }
    receipt_path = root / "submission.json"
    receipt_path.write_text(json.dumps(receipt))
    monkeypatch.chdir(release)
    monkeypatch.setenv("USER", "example")
    monkeypatch.setattr(module, "incomplete_seeds", lambda _: [2750, 2751])
    state = {"held": False, "dependency": "afterany:100", "cancelled": set()}
    calls = []

    def fake_live(job_id):
        assert job_id == "100"
        return {
            f"100_{task}": "RUNNING"
            for task in module.TAIL_TASKS
            if f"100_{task}" not in state["cancelled"]
        }

    def fake_fields(job_id):
        assert job_id == "200"
        return {
            "JobState": "PENDING",
            "Priority": "0" if state["held"] else "100",
            "Dependency": state["dependency"],
        }

    def fake_command(argv):
        calls.append(argv)
        if argv[0] == "squeue":
            return ""
        if argv[:2] == ["scontrol", "hold"]:
            state["held"] = True
        elif argv[:2] == ["scontrol", "update"]:
            state["dependency"] = argv[-1].split("=", 1)[1]
        elif argv[:2] == ["scontrol", "release"]:
            state["held"] = False
        elif argv[0] == "scancel":
            state["cancelled"].add(argv[1])
        elif argv[0] == "sbatch":
            return "300" if sum(call[0] == "sbatch" for call in calls) == 1 else "301"
        return ""

    monkeypatch.setattr(module, "live_elements", fake_live)
    monkeypatch.setattr(module, "job_fields", fake_fields)
    monkeypatch.setattr(module, "command", fake_command)
    operations = root / "parallel-tail"
    module.cutover(
        SimpleNamespace(
            plan=plan_path,
            receipt=receipt_path,
            operations=operations,
            max_submitted=1000,
        )
    )
    assert module.read(operations / "submission.json")["stage"] == "active"
    assert module.read(operations / "manifest.json")["seeds"] == [2750, 2751]
    assert next(i for i, call in enumerate(calls) if call[:2] == ["scontrol", "hold"]) < next(
        i for i, call in enumerate(calls) if call[0] == "scancel"
    )
    assert next(i for i, call in enumerate(calls) if call[:2] == ["scontrol", "update"]) < next(
        i for i, call in enumerate(calls) if call[0] == "scancel"
    )
    assert state["dependency"] == "afterany:100:301"
    assert len(state["cancelled"]) == 3
