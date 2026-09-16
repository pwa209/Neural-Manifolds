"""Durable, outcome-independent Slurm progression with bounded concurrency.

Only implemented tasks are submitted. Missing scientific integrations remain
explicit work items, never successful empty stages. The controller runs on the
internet-connected login host; scientific computations run under Slurm.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from filelock import FileLock, Timeout

from neural_manifolds.provenance import atomic_write_json, sha256_file

ACTIVE = {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "SUSPENDED", "REQUEUED"}
TRANSIENT = {"NODE_FAIL", "PREEMPTED", "BOOT_FAIL"}


def now() -> str:
    return datetime.now(UTC).isoformat()


def command(argv: list[str]) -> str:
    return subprocess.run(
        argv, check=True, text=True, capture_output=True, timeout=60
    ).stdout.strip()


class Slurm:
    def find(self, name: str) -> list[str]:
        live = command(["squeue", "--me", "--name", name, "--noheader", "--format=%A"])
        history = command(
            [
                "sacct",
                "--name",
                name,
                "--starttime",
                "2026-09-16",
                "-X",
                "--noheader",
                "--parsable2",
                "--format=JobIDRaw",
            ]
        )
        return sorted(
            {
                x.strip().split("|")[0]
                for x in (live + "\n" + history).splitlines()
                if x.strip().isdigit()
            }
        )

    def state(self, job: str) -> str:
        text = command(["sacct", "-X", "-j", job, "--noheader", "--parsable2", "--format=State"])
        if text:
            return text.splitlines()[0].split("|")[0].split()[0].rstrip("+")
        live = command(["squeue", "-j", job, "--noheader", "--format=%T"])
        return live.splitlines()[0] if live else "UNKNOWN"

    def submit(self, spec: Path, name: str, root: Path, account: str) -> str:
        kind = json.loads(spec.read_text())["kind"]
        resources = {
            "revised_measure": ["--cpus-per-task=4", "--mem=32G", "--time=08:00:00", "--gpus=1"],
            "revised_transfer": ["--cpus-per-task=4", "--mem=32G", "--time=12:00:00"],
            "revised_recovery": ["--cpus-per-task=4", "--mem=16G", "--time=24:00:00"],
            "revised_controls": ["--cpus-per-task=4", "--mem=32G", "--time=12:00:00"],
            "revised_sensitivities": ["--cpus-per-task=4", "--mem=32G", "--time=12:00:00"],
            "revised_tms": ["--cpus-per-task=4", "--mem=64G", "--time=24:00:00"],
            "revised_tactile": ["--cpus-per-task=4", "--mem=64G", "--time=12:00:00"],
            "revised_osf": ["--cpus-per-task=4", "--mem=32G", "--time=08:00:00"],
            "revised_boundary": ["--cpus-per-task=4", "--mem=32G", "--time=12:00:00"],
        }.get(kind, ["--cpus-per-task=2", "--mem=8G", "--time=01:00:00"])
        if kind == "revised_perturbation_measure":
            resources = ["--cpus-per-task=4", "--mem=32G", "--time=12:00:00", "--gpus=1"]
        if kind == "revised_robustness":
            resources = ["--cpus-per-task=4", "--mem=32G", "--time=24:00:00"]
        if kind in {"revised_measure", "revised_perturbation_measure"}:
            account = os.environ["NM_GPU_ACCOUNT"]
        value = command(
            [
                "sbatch",
                "--parsable",
                "--account",
                account,
                "--job-name",
                name,
                *resources,
                "--chdir",
                str(Path.cwd()),
                "--output",
                str(root / "logs" / "%j.log"),
                "--export=ALL",
                "scripts/alliance/task.sbatch",
                str(spec),
            ]
        )
        job = value.split(";")[0]
        if not job.isdigit():
            raise ValueError("Ambiguous Slurm submission response")
        return job


def valid_receipt(task: dict) -> bool:
    path = Path(task["output"]) / "receipt.json"
    if not path.is_file():
        return False
    receipt = json.loads(path.read_text())
    return (
        receipt.get("status") == "complete"
        and receipt.get("spec_sha256") == sha256_file(Path(task["spec"]))
        and receipt.get("source_digest") == task["source_digest"]
        and bool(receipt.get("artifacts"))
        and all(
            Path(p).is_file() and sha256_file(Path(p)) == digest
            for p, digest in receipt["artifacts"].items()
        )
    )


class Controller:
    def __init__(
        self,
        root: Path,
        acquisition: Path,
        registry: Path,
        account: str,
        *,
        scheduler=None,
        maximum_jobs: int = 2,
    ):
        self.root, self.acquisition, self.registry, self.account = (
            root,
            acquisition,
            registry,
            account,
        )
        self.scheduler, self.maximum_jobs = scheduler or Slurm(), maximum_jobs
        self.source = json.loads(Path("release_manifest.json").read_text())["source_digest"]
        for folder in ("specs", "outputs", "logs"):
            (root / folder).mkdir(parents=True, exist_ok=True)
        self.path = root / "state.json"
        self.state = (
            json.loads(self.path.read_text())
            if self.path.exists()
            else {
                "schema_version": 1,
                "source_digest": self.source,
                "tasks": {},
                "acquisition_restarts": 0,
                "scientific_gates": False,
            }
        )
        if self.state["source_digest"] != self.source:
            raise ValueError("Source changed: deploy a distinct controller state directory")

    def save(self):
        self.state.update(updated_at=now(), controller_pid=os.getpid())
        atomic_write_json(self.path, self.state)

    def add(self, kind: str, dataset: str, input_path: Path, identity: str):
        key = hashlib.sha256(f"{self.source}:{kind}:{dataset}:{identity}".encode()).hexdigest()[:20]
        if key in self.state["tasks"]:
            return
        spec_path = self.root / "specs" / f"{key}.json"
        spec = {
            "task_id": key,
            "source_digest": self.source,
            "kind": kind,
            "dataset_id": dataset,
            "input": str(input_path),
            "input_identity": identity,
            "output": str(self.root / "outputs" / key),
        }
        atomic_write_json(spec_path, spec)
        self.state["tasks"][key] = {**spec, "spec": str(spec_path), "status": "ready", "attempt": 0}
        self.save()

    def plan(self):
        acquisition = json.loads(self.acquisition.read_text())
        self.state["acquisition_status"] = acquisition.get("status")
        seeds = json.loads(os.environ.get("NM_INVENTORY_SEED_STATES", "[]"))
        imported = {}
        for seed in seeds:
            for task in json.loads(Path(seed).read_text())["tasks"].values():
                if (
                    task["kind"] == "inventory"
                    and task["status"] == "complete"
                    and valid_receipt(task)
                ):
                    path = Path(task["output"]) / "inventory.json"
                    imported[task["dataset_id"]] = str(path)
        self.state["imported_inventory"] = imported
        for dataset, entry in acquisition["datasets"].items():
            if entry["status"] != "complete":
                continue
            release = Path(entry["result"]["release_path"])
            marker = release / ".acquisition/COMPLETE.json"
            if dataset not in imported:
                self.add("inventory", dataset, release, sha256_file(marker))
        for task in list(self.state["tasks"].values()):
            if task["kind"] == "inventory" and task["status"] == "complete":
                audit = Path(task["output"]) / "inventory.json"
                if os.environ.get("NM_REVISED_EXECUTION") != "1":
                    self.add("recovery", task["dataset_id"], audit, sha256_file(audit))
                    self.add("signal_qc", task["dataset_id"], audit, sha256_file(audit))
        if os.environ.get("NM_REVISED_EXECUTION") == "1":
            self.plan_revised()

    def plan_revised(self):
        import yaml

        policy = yaml.safe_load(Path("configs/revised_execution.yaml").read_text())
        expected = set(policy["source_policy"])
        for dataset, name in self.state.get("imported_inventory", {}).items():
            if dataset in expected:
                path = Path(name)
                self.add("revised_cohort", dataset, path, sha256_file(path))
        acquisition = json.loads(self.acquisition.read_text())
        for dataset, kind in [
            ("propofol_tms_eeg", "revised_tms"),
            ("tactile_detection", "revised_tactile"),
            ("somatosensory_report_task", "revised_osf"),
        ]:
            entry = acquisition["datasets"].get(dataset, {})
            if entry.get("status") == "complete":
                marker = Path(entry["result"]["release_path"]) / ".acquisition/COMPLETE.json"
                self.add(kind, dataset, marker, sha256_file(marker))
        subset = os.environ.get("NM_PSICONNECT_SUBSET_MARKER")
        if subset and Path(subset).is_file():
            self.add("revised_boundary", "psiconnect", Path(subset), sha256_file(subset))
        for task in list(self.state["tasks"].values()):
            if task["kind"] == "revised_perturbation_measure" and task["status"] == "complete":
                path = Path(task["output"]) / "perturbation_measurements.json"
                self.add("revised_robustness", "dream_portfolio", path, sha256_file(path))
            if task["status"] != "complete" or task["dataset_id"] not in expected:
                continue
            if task["kind"] == "inventory":
                path = Path(task["output"]) / "inventory.json"
                self.add("revised_cohort", task["dataset_id"], path, sha256_file(path))
            if task["kind"] == "revised_cohort":
                path = Path(task["output"]) / "cohort.json"
                self.add("revised_measure", task["dataset_id"], path, sha256_file(path))
        measured = {
            t["dataset_id"]: t
            for t in self.state["tasks"].values()
            if t["kind"] == "revised_measure" and t["status"] == "complete"
        }
        if not expected.issubset(measured):
            return
        paths = [Path(measured[d]["output"]) / "measurement.json" for d in sorted(expected)]
        bundle = self.root / "measurement-inputs.json"
        document = {"inputs": [{"path": str(p), "sha256": sha256_file(p)} for p in paths]}
        if bundle.exists() and json.loads(bundle.read_text()) != document:
            raise ValueError("completed_measurement_bundle_changed")
        if not bundle.exists():
            atomic_write_json(bundle, document)
        for kind in [
            "revised_transfer",
            "revised_recovery",
            "revised_sensitivities",
            "revised_perturbation_measure",
        ]:
            self.add(kind, "dream_portfolio", bundle, sha256_file(bundle))
        for index in range(policy["surrogate_repetitions"]):
            self.add("revised_controls", f"dream_null_{index}", bundle, sha256_file(bundle))
        terminal = [
            t
            for t in self.state["tasks"].values()
            if t["kind"]
            in {
                "revised_transfer",
                "revised_recovery",
                "revised_sensitivities",
                "revised_tms",
                "revised_osf",
                "revised_tactile",
                "revised_controls",
                "revised_boundary",
                "revised_robustness",
            }
            and t["status"] == "complete"
        ]
        if terminal:
            names = {
                "revised_transfer": "transfer.json",
                "revised_recovery": "axis_recovery.json",
                "revised_sensitivities": "sensitivities.json",
                "revised_tms": "tms.json",
                "revised_osf": "specificity.json",
                "revised_tactile": "specificity.json",
                "revised_controls": "controls.json",
                "revised_boundary": "boundary.json",
                "revised_robustness": "robustness.json",
            }
            sources = [Path(t["output"]) / names[t["kind"]] for t in terminal]
            document = {
                "inputs": [{"path": str(p), "sha256": sha256_file(p)} for p in sorted(sources)]
            }
            identity = hashlib.sha256(json.dumps(document, sort_keys=True).encode()).hexdigest()
            synthesis = self.root / "bundles" / f"synthesis-{identity}.json"
            atomic_write_json(synthesis, document)
            self.add("revised_synthesis", "dream_portfolio", synthesis, sha256_file(synthesis))

    def reconcile(self):
        for task in self.state["tasks"].values():
            if task["status"] == "submitting":
                jobs = self.scheduler.find(task["job_name"])
                if len(jobs) == 1:
                    task.update(job_id=jobs[0], status="submitted")
                elif len(jobs) > 1:
                    task.update(status="attention", reason="multiple_matching_jobs")
                elif time.time() - task["submitted_at"] > 300:
                    task.update(
                        status="attention", reason="ambiguous_submission_no_automatic_resubmit"
                    )
            if task["status"] not in {"submitted", "running"}:
                continue
            status = self.scheduler.state(task["job_id"])
            task["slurm_state"] = status
            if status in ACTIVE:
                task["status"] = "running" if status == "RUNNING" else "submitted"
            elif status == "COMPLETED":
                task["status"] = "complete" if valid_receipt(task) else "attention"
                if task["status"] == "attention":
                    task["reason"] = "missing_or_invalid_output_receipt"
            elif status in TRANSIENT and task["attempt"] < 3:
                task.update(status="ready", retry_after=time.time() + 300)
            elif status != "UNKNOWN":
                task.update(status="attention", reason=f"scheduler_{status}")
        self.save()

    def submit_ready(self):
        active = sum(
            t["status"] in {"submitting", "submitted", "running"}
            for t in self.state["tasks"].values()
        )
        priority = {
            "revised_cohort": 0,
            "revised_measure": 1,
            "revised_transfer": 2,
            "revised_tms": 3,
            "revised_tactile": 3,
            "revised_osf": 3,
            "revised_sensitivities": 4,
            "revised_recovery": 5,
            "revised_boundary": 6,
            "revised_perturbation_measure": 5,
            "revised_robustness": 6,
            "revised_synthesis": 7,
            "revised_controls": 8,
            "inventory": 9,
        }
        gpu_active = any(
            t["kind"] in {"revised_measure", "revised_perturbation_measure"}
            and t["status"] in {"submitting", "submitted", "running"}
            for t in self.state["tasks"].values()
        )
        for task in sorted(self.state["tasks"].values(), key=lambda t: priority.get(t["kind"], 10)):
            if active >= self.maximum_jobs:
                break
            if task["status"] != "ready" or task.get("retry_after", 0) > time.time():
                continue
            if task["kind"] in {"revised_measure", "revised_perturbation_measure"} and gpu_active:
                continue
            task["attempt"] += 1
            name = f"nm-{task['task_id']}-a{task['attempt']}"
            task.update(status="submitting", job_name=name, submitted_at=time.time())
            self.save()  # Intent precedes sbatch; uncertain outcomes reconcile by unique name.
            try:
                job = self.scheduler.submit(Path(task["spec"]), name, self.root, self.account)
                task.update(status="submitted", job_id=job)
            except Exception as exc:
                task["submission_error"] = str(exc)
            self.save()
            active += 1
            gpu_active = gpu_active or task["kind"] in {
                "revised_measure",
                "revised_perturbation_measure",
            }

    def acquisition_watch(self):
        state = json.loads(self.acquisition.read_text())
        if state.get("status") == "complete":
            return
        try:
            with FileLock(str(self.acquisition) + ".lock", timeout=0):
                pass
        except Timeout:
            self.state["acquisition_worker"] = "lock_held"
            return
        if self.state["acquisition_restarts"] >= 3:
            self.state["acquisition_worker"] = "retry_budget_exhausted"
            return
        if time.time() < self.state.get("next_acquisition_restart", 0):
            return
        self.state["acquisition_restarts"] += 1
        self.state["next_acquisition_restart"] = time.time() + 3600
        self.save()
        # FileLock in acquire.py arbitrates startup races. No credentials saved.
        env = dict(os.environ)
        count = int(env.get("GIT_CONFIG_COUNT", "0"))
        env.update(
            {
                "GIT_CONFIG_COUNT": str(count + 2),
                f"GIT_CONFIG_KEY_{count}": "user.name",
                f"GIT_CONFIG_VALUE_{count}": "Neural Manifolds acquisition",
                f"GIT_CONFIG_KEY_{count + 1}": "user.email",
                f"GIT_CONFIG_VALUE_{count + 1}": "neural-manifolds@localhost",
            }
        )
        with (self.root / "logs" / "acquisition.log").open("ab") as log:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "scripts/alliance/acquire.py",
                    "--registry",
                    str(self.registry),
                    "--root",
                    str(Path(os.environ["NM_PROJECT_ROOT"]) / "raw"),
                    "--state",
                    str(self.acquisition),
                ],
                stdout=log,
                stderr=log,
                stdin=subprocess.DEVNULL,
                env=env,
                start_new_session=True,
            )
        self.state["acquisition_worker"] = {"pid": process.pid, "started_at": now()}

    def report(self):
        tasks = list(self.state["tasks"].values())
        counts = {k: sum(t["status"] == k for t in tasks) for k in {t["status"] for t in tasks}}
        revised = os.environ.get("NM_REVISED_EXECUTION") == "1"
        phase_kinds = {
            "R1_R3": ("inventory", "revised_cohort"),
            "R4": ("revised_recovery",),
            "R5": ("revised_measure",),
            "R6": (
                "revised_transfer",
                "revised_sensitivities",
                "revised_controls",
                "revised_perturbation_measure",
                "revised_robustness",
            ),
            "R7": ("revised_tms",),
            "R8": ("revised_tactile", "revised_osf", "revised_boundary"),
            "R10": ("revised_synthesis",),
        }
        revised_map = {
            phase: {
                "drivers_implemented": True,
                "status": "tasks_registered"
                if any(t["kind"] in kinds for t in tasks)
                else "waiting_for_verified_inputs",
                "tasks_by_status": {
                    status: sum(t["kind"] in kinds and t["status"] == status for t in tasks)
                    for status in sorted(counts)
                },
                "kinds": list(kinds),
            }
            for phase, kinds in phase_kinds.items()
        }
        revised_map.update(
            R0="qualification_required_before_controller_launch",
            R2=self.state.get("acquisition_status"),
            R9="optional_extensions_not_commissioned_by_revised_controller",
        )
        report = {
            "updated_at": now(),
            "source_digest": self.source,
            "task_counts": counts,
            "scientific_gates": False,
            "full_pipeline_implemented": False,
            "revised_core_drivers_implemented": revised,
            "implementation_scope": "core_plus_context_boundary_optional_extensions_separate",
            "phase_map": revised_map
            if revised
            else {
                "R0": "source_qualification_required_before_controller_launch",
                "R1": "open_registry_frozen_independence_audit_pending",
                "R2": self.state.get("acquisition_status"),
                "R3": "archive_inventory_and_sampled_signal_qc_automatic_cohort_audit_pending",
                "R4": "length_matched_ar1_component_automatic_axis_specific_validation_pending",
                "R5": "needs_multicohort_and_fast_track_integration",
                "R6": "tested_nested_prediction_routine_needs_audited_features",
                "R7": "tested_tms_prediction_routine_needs_verified_pairing_and_features",
                "R8": "needs_revised_specificity_and_boundary_drivers",
                "R9": "optional_not_required_for_core_completion",
                "R10": "progress_snapshot_only_final_synthesis_pending",
            },
            "attention": [t for t in tasks if t["status"] == "attention"],
        }
        atomic_write_json(self.root / "progress.json", report)

    def tick(self):
        self.reconcile()
        self.plan()
        self.submit_ready()
        self.acquisition_watch()
        self.report()
        self.save()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--acquisition-state", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--account", required=True)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    with FileLock(str(args.root / "controller.lock"), timeout=0):
        controller = Controller(
            args.root.resolve(),
            args.acquisition_state.resolve(),
            args.registry.resolve(),
            args.account,
        )
        while not (args.root / "STOP").exists():
            try:
                controller.tick()
                atomic_write_json(
                    args.root / "heartbeat.json",
                    {"at": now(), "status": "healthy", "pid": os.getpid()},
                )
            except Exception as exc:
                atomic_write_json(
                    args.root / "heartbeat.json",
                    {"at": now(), "status": "error", "error": str(exc), "pid": os.getpid()},
                )
                if args.once:
                    raise
            if args.once:
                break
            time.sleep(60)


if __name__ == "__main__":
    main()
