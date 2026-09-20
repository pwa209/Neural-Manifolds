"""Queue the sealed continuation with durable, duplicate-resistant Slurm receipts."""

from __future__ import annotations

import argparse
import getpass
import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from filelock import FileLock

from neural_manifolds.provenance import atomic_write_json


def command(args):
    return subprocess.run(args, text=True, capture_output=True, check=True).stdout.strip()


def job_graph(tasks):
    return [
        ("canary-core", [], "24:00:00", ["canary", "--portfolio", "core"], [], 1),
        ("canary-hd", [], "24:00:00", ["canary", "--portfolio", "hd"], [], 1),
        (
            "continuation",
            ["canary-core", "canary-hd"],
            "5-00:00:00",
            ["worker"],
            [f"--array=0-{tasks - 1}%50"],
            tasks,
        ),
        ("retry", ["continuation"], "5-00:00:00", ["worker"], [f"--array=0-{tasks - 1}%50"], tasks),
        ("aggregate", ["retry"], "01:00:00", ["aggregate"], [], 1),
    ]


def submit(args):
    plan = json.loads(args.plan.read_text())
    # The worker rechecks the complete scientific seal and all input hashes.
    if plan["array_tasks"] != 392 or plan["max_parallel_cpus"] != 50:
        raise ValueError("Unexpected fixed continuation resource plan")
    root = args.plan.parent
    receipt_path = root / "submission.json"
    with FileLock(str(root / "submission.lock"), timeout=1):
        receipt = (
            json.loads(receipt_path.read_text())
            if receipt_path.exists()
            else {"plan_sha256": plan["plan_sha256"], "jobs": {}}
        )
        if receipt["plan_sha256"] != plan["plan_sha256"]:
            raise ValueError("Submission receipt belongs to a different plan")
        for value in receipt["jobs"].values():
            if value["status"] == "submitting":
                raise ValueError(
                    "Uncertain previous submission: reconcile with Slurm before retrying"
                )
        graph = job_graph(plan["array_tasks"])
        missing_count = sum(
            n
            for name, _, _, _, _, n in graph
            if receipt["jobs"].get(name, {}).get("status") != "submitted"
        )
        existing = command(
            [
                "squeue",
                "--user",
                getpass.getuser(),
                "--account",
                args.account,
                "--array",
                "--noheader",
                "--format=%i",
            ]
        ).splitlines()
        if len(existing) + missing_count > args.max_submitted:
            raise ValueError("Insufficient association job slots; no new jobs submitted")
        env = {
            "NM_ORIGINAL_RELEASE": plan["analysis_release"],
            "NM_NULL_RUNNER": str(args.runner.resolve()),
            "NM_NULL_PLAN": str(args.plan.resolve()),
        }
        os.environ.update(env)
        for name, parents, walltime, mode, extra, _ in graph:
            if receipt["jobs"].get(name, {}).get("status") == "submitted":
                continue
            argv = [
                "sbatch",
                "--parsable",
                f"--account={args.account}",
                f"--job-name=nm-null5k-{name}",
                "--cpus-per-task=1",
                "--mem=4G",
                f"--time={walltime}",
                "--export=ALL",
                "--open-mode=append",
                f"--chdir={plan['analysis_release']}",
                f"--output={plan['output_root']}/logs/{name}-%A_%a.log",
                *extra,
            ]
            if parents:
                kind = "afterok" if name == "continuation" else "afterany"
                ids = ":".join(receipt["jobs"][p]["job_id"] for p in parents)
                argv.append(f"--dependency={kind}:{ids}")
            argv.extend([str(args.wrapper.resolve()), *mode])
            item = dict(
                status="submitting",
                command=argv,
                environment=env,
                requested_utc=datetime.now(UTC).isoformat(),
            )
            receipt["jobs"][name] = item
            atomic_write_json(receipt_path, receipt)
            try:
                response = command(argv)
            except subprocess.CalledProcessError as exc:
                item.update(status="rejected", error=exc.stderr[-4000:])
                atomic_write_json(receipt_path, receipt)
                raise
            job_id = response.split(";", 1)[0]
            if not job_id.isdigit():
                # Keep 'submitting': do not risk duplicate jobs on ambiguous output.
                raise ValueError(f"Unrecognized sbatch response: {response}")
            item.update(status="submitted", job_id=job_id)
            atomic_write_json(receipt_path, receipt)
            print("SUBMITTED", name, job_id, flush=True)
        print("ALL_PHASES_QUEUED", str(receipt_path), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "runner", "wrapper"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--account", required=True)
    parser.add_argument("--max-submitted", type=int, default=1000)
    submit(parser.parse_args())


if __name__ == "__main__":
    main()
