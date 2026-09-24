"""Recover an unfinished fixed-count null tail with one seed per Slurm task.

This is an operational scheduler amendment. It imports the sealed original
runner for every fit and never edits the scientific plan or observed results.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

try:
    import fcntl
except ModuleNotFoundError:  # Local Windows tests; live cutover runs on Linux.
    fcntl = None

TAIL_TASKS = (213, 215, 217)
MAX_TAIL_SEEDS = 75
MAX_PARALLEL = 50


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + f".tmp-{os.getpid()}")
    with temp.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def command(argv):
    return subprocess.run(argv, check=True, text=True, capture_output=True).stdout.strip()


def job_fields(job_id):
    output = command(["scontrol", "show", "job", "-o", str(job_id)])
    return {part.split("=", 1)[0]: part.split("=", 1)[1] for part in output.split() if "=" in part}


def live_elements(job_id):
    output = command(["squeue", "-j", str(job_id), "--array", "-h", "-o", "%i|%T"])
    return dict(line.split("|", 1) for line in output.splitlines())


def validate_plan(path):
    plan = read(path)
    declared = plan.pop("plan_sha256")
    if digest(plan) != declared:
        raise ValueError("Sealed plan checksum mismatch")
    plan["plan_sha256"] = declared
    if plan["total_replicates"] != 5000 or plan["original_replicates"] != 100:
        raise ValueError("Unexpected fixed null count")
    if len(plan["observed"]) != 72:
        raise ValueError("Incomplete observed family")
    return plan


def record_status(path, plan, portfolio, seed):
    path = Path(path)
    if not path.exists():
        return "missing"
    record = read(path)
    declared = record.pop("record_sha256")
    if digest(record) != declared:
        raise ValueError(f"Checkpoint checksum mismatch: {path}")
    if (
        record["plan_sha256"] != plan["plan_sha256"]
        or record["portfolio"] != portfolio
        or record["replicate"] != seed
        or record["seed"] != plan["policy"]["seed"] + seed
    ):
        raise ValueError(f"Checkpoint identity mismatch: {path}")
    cells = [(row["representation"], row["kind"]) for row in record["rows"]]
    if len(cells) != len(set(cells)) or len(cells) > 9:
        raise ValueError(f"Duplicate or surplus cells: {path}")
    if record["status"] == "complete" and len(cells) == 9:
        return "complete"
    if record["status"] == "partial" and len(cells) < 9:
        return "partial"
    raise ValueError(f"Inconsistent checkpoint state: {path}")


def incomplete_seeds(plan):
    base = Path(plan["output_root"]) / "results"
    for seed in range(100, 5000):
        if record_status(base / f"core-{seed:04d}.json", plan, "core", seed) != "complete":
            raise ValueError(f"Sparse refit incomplete: {seed}")
    pending = [
        seed
        for seed in range(100, 5000)
        if record_status(base / f"hd-{seed:04d}.json", plan, "hd", seed) != "complete"
    ]
    expected = {seed for first in (2750, 2775, 2800) for seed in range(first, first + 25)}
    if not pending or len(pending) > MAX_TAIL_SEEDS or not set(pending) <= expected:
        raise ValueError("Unexpected remaining high-density seed set")
    return pending


def validate_manifest(path, plan):
    manifest = read(path)
    declared = manifest.pop("manifest_sha256")
    if digest(manifest) != declared:
        raise ValueError("Parallel manifest checksum mismatch")
    manifest["manifest_sha256"] = declared
    if (
        manifest["plan_sha256"] != plan["plan_sha256"]
        or manifest["runner_sha256"] != plan["runner_sha256"]
        or manifest["helper_sha256"] != file_hash(__file__)
    ):
        raise ValueError("Parallel manifest identity mismatch")
    seeds = manifest["seeds"]
    if not seeds or len(seeds) > MAX_TAIL_SEEDS or len(seeds) != len(set(seeds)):
        raise ValueError("Invalid parallel seed list")
    if not all(isinstance(seed, int) and 100 <= seed < 5000 for seed in seeds):
        raise ValueError("Out-of-range seed")
    return manifest


def original_module(path, plan):
    path = Path(path).resolve()
    if file_hash(path) != plan["runner_sha256"]:
        raise ValueError("Original runner checksum mismatch")
    spec = importlib.util.spec_from_file_location("sealed_null_runner", path)
    if spec is None or spec.loader is None:
        raise ValueError("Cannot load original runner")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def worker(args):
    from filelock import FileLock

    plan = validate_plan(args.plan)
    manifest = validate_manifest(args.manifest, plan)
    if (
        args.runner is not None
        and Path(args.runner).resolve() != Path(manifest["runner"]).resolve()
    ):
        raise ValueError("Slurm runner argument differs from sealed manifest")
    index = args.index
    if index is None:
        index = int(os.environ["SLURM_ARRAY_TASK_ID"])
    if not 0 <= index < len(manifest["seeds"]):
        raise ValueError("Array index outside sealed parallel manifest")
    seed = manifest["seeds"][index]
    module = original_module(manifest["runner"], plan)
    plan = module.load_plan(args.plan)
    base = Path(plan["output_root"])
    for source in plan["sources"]:
        module.checked(source)
    for portfolio in ("core", "hd"):
        check = module.read(base / "checks" / f"{portfolio}-0000.json")
        module.validate_record(check, plan, portfolio, 0)
        if check["status"] != "complete":
            raise ValueError("Original equivalence canary incomplete")
    with FileLock(str(base / "locks" / f"parallel-hd-{seed:04d}.lock"), timeout=5):
        target = base / "results" / f"hd-{seed:04d}.json"
        if record_status(target, plan, "hd", seed) == "complete":
            print("SEED_ALREADY_COMPLETE", seed, flush=True)
            return
        arrays = module.load_arrays(plan, "hd")
        module.run_seed(plan, "hd", seed, arrays)
        if record_status(target, plan, "hd", seed) != "complete":
            raise ValueError("One-seed worker did not finish")
        print("PARALLEL_SEED_COMPLETE", seed, flush=True)


def submit_array(
    *,
    name,
    account,
    release,
    output_root,
    wrapper,
    plan_path,
    helper,
    manifest_path,
    runner,
    dependency,
    count,
):
    argv = [
        "sbatch",
        "--parsable",
        f"--account={account}",
        f"--job-name=nm-null5k-{name}",
        "--cpus-per-task=1",
        "--mem=4G",
        "--time=1-00:00:00",
        f"--array=0-{count - 1}%{MAX_PARALLEL}",
        f"--dependency=afterany:{dependency}",
        f"--chdir={release}",
        f"--output={output_root}/logs/{name}-%A_%a.log",
        "--export=ALL,NM_ORIGINAL_RELEASE="
        + str(release)
        + ",NM_NULL_RUNNER="
        + str(helper)
        + ",NM_NULL_PLAN="
        + str(plan_path),
        str(wrapper),
        "worker",
        "--manifest",
        str(manifest_path),
        "--runner",
        str(runner),
    ]
    return argv


def cutover(args):
    plan_path = Path(args.plan).resolve()
    plan = validate_plan(plan_path)
    receipt = read(args.receipt)
    if receipt["plan_sha256"] != plan["plan_sha256"]:
        raise ValueError("Submission receipt does not match plan")
    retry = receipt["jobs"]["retry"]
    aggregate = receipt["jobs"]["aggregate"]
    if retry["status"] != "submitted" or aggregate["status"] != "submitted":
        raise ValueError("Original queue receipt is not fully submitted")
    retry_id, aggregate_id = retry["job_id"], aggregate["job_id"]
    runner = Path(retry["environment"]["NM_NULL_RUNNER"]).resolve()
    if file_hash(runner) != plan["runner_sha256"]:
        raise ValueError("Original runner changed")
    wrapper = Path(retry["command"][-2]).resolve()
    if not wrapper.is_file():
        raise ValueError("Original runtime wrapper missing")
    account_args = [arg for arg in retry["command"] if arg.startswith("--account=")]
    if len(account_args) != 1:
        raise ValueError("Ambiguous Slurm account")
    account = account_args[0].split("=", 1)[1]
    release = Path(plan["analysis_release"]).resolve()
    if Path.cwd().resolve() != release or not (release / "release_manifest.json").is_file():
        raise ValueError("Run cutover from the original immutable release")
    if file_hash(release / "release_manifest.json") != plan["release_manifest_sha256"]:
        raise ValueError("Original release manifest changed")
    if (Path(args.receipt).resolve().parent / "null_review_5000.json").exists():
        raise ValueError("Final report already exists")
    active = live_elements(retry_id)
    allowed = {f"{retry_id}_{task}" for task in TAIL_TASKS}
    if (
        not active
        or not set(active) <= allowed
        or any(state != "RUNNING" for state in active.values())
    ):
        raise ValueError(f"Unexpected original retry state: {active}")
    fields = job_fields(aggregate_id)
    if fields.get("JobState") != "PENDING" or retry_id not in fields.get("Dependency", ""):
        raise ValueError("Aggregation job is not waiting for original retry")
    seeds = incomplete_seeds(plan)
    if len(seeds) > 75:
        raise ValueError("Too many outstanding seeds")
    # These arrays are dependency-ordered, so at most one has live workers.
    slots = command(
        [
            "squeue",
            "--user",
            os.environ["USER"],
            "--account",
            account,
            "--array",
            "--noheader",
            "--format=%i",
        ]
    ).splitlines()
    if len(slots) + 2 * len(seeds) > args.max_submitted:
        raise ValueError("Insufficient association job-slot headroom")
    ops = Path(args.operations).resolve()
    if ops.parent != Path(args.receipt).resolve().parent:
        raise ValueError("Operational rescue directory must be beside the receipt")
    if ops.exists():
        raise FileExistsError("Operational rescue directory already exists; reconcile it first")
    ops.mkdir(parents=True)
    with (ops / "cutover.lock").open("w") as lock:
        if fcntl is not None:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = {
            "plan_sha256": plan["plan_sha256"],
            "runner_sha256": plan["runner_sha256"],
            "runner": str(runner),
            "helper_sha256": file_hash(__file__),
            "seeds": seeds,
            "created_utc": datetime.now(UTC).isoformat(),
        }
        manifest["manifest_sha256"] = digest(manifest)
        manifest_path = ops / "manifest.json"
        atomic_json(manifest_path, manifest)
        state_path = ops / "submission.json"
        state = {
            "manifest_sha256": manifest["manifest_sha256"],
            "original_retry": retry_id,
            "aggregate": aggregate_id,
            "seed_count": len(seeds),
            "stage": "planned",
            "jobs": {},
            "created_utc": datetime.now(UTC).isoformat(),
        }
        atomic_json(state_path, state)
        command(["scontrol", "hold", aggregate_id])
        fields = job_fields(aggregate_id)
        if fields.get("JobState") != "PENDING" or fields.get("Priority") != "0":
            raise ValueError("Aggregation hold could not be verified")
        state["stage"] = "aggregate_held"
        atomic_json(state_path, state)
        dependencies = retry_id
        for phase in ("parallel1", "parallel2"):
            argv = submit_array(
                name=phase,
                account=account,
                release=release,
                output_root=plan["output_root"],
                wrapper=wrapper,
                plan_path=plan_path,
                helper=Path(__file__).resolve(),
                manifest_path=manifest_path,
                runner=runner,
                dependency=dependencies,
                count=len(seeds),
            )
            state["jobs"][phase] = {"status": "submitting", "command": argv}
            atomic_json(state_path, state)
            response = command(argv)
            job_id = response.split(";", 1)[0]
            if not job_id.isdigit():
                raise ValueError(f"Ambiguous Slurm submission: {response}")
            state["jobs"][phase].update(status="submitted", job_id=job_id)
            atomic_json(state_path, state)
            dependencies = job_id
        command(
            [
                "scontrol",
                "update",
                f"JobId={aggregate_id}",
                f"Dependency=afterany:{retry_id}:{dependencies}",
            ]
        )
        fields = job_fields(aggregate_id)
        if fields.get("JobState") != "PENDING" or dependencies not in fields.get("Dependency", ""):
            raise ValueError("Aggregation dependency update could not be verified")
        state["stage"] = "aggregate_relinked"
        atomic_json(state_path, state)
        for old_element in sorted(active):
            if old_element in live_elements(retry_id):
                command(["scancel", old_element])
        deadline = time.monotonic() + 60
        while set(live_elements(retry_id)) & allowed and time.monotonic() < deadline:
            time.sleep(2)
        if set(live_elements(retry_id)) & allowed:
            raise ValueError("Old tail still running; aggregate remains held")
        state["stage"] = "old_tail_stopped"
        atomic_json(state_path, state)
        command(["scontrol", "release", aggregate_id])
        fields = job_fields(aggregate_id)
        if fields.get("JobState") != "PENDING" or dependencies not in fields.get("Dependency", ""):
            raise ValueError("Aggregation release or dependency verification failed")
        state["stage"] = "active"
        state["activated_utc"] = datetime.now(UTC).isoformat()
        atomic_json(state_path, state)
        print(
            "PARALLEL_TAIL_ACTIVE",
            json.dumps(
                {
                    "seed_count": len(seeds),
                    "parallel_limit": MAX_PARALLEL,
                    "first_job": state["jobs"]["parallel1"]["job_id"],
                    "retry_job": state["jobs"]["parallel2"]["job_id"],
                    "aggregate_job": aggregate_id,
                }
            ),
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    worker_parser = sub.add_parser("worker")
    worker_parser.add_argument("--plan", required=True, type=Path)
    worker_parser.add_argument("--manifest", required=True, type=Path)
    worker_parser.add_argument("--runner", type=Path)  # Kept in the durable Slurm command.
    worker_parser.add_argument("--index", type=int)
    cutover_parser = sub.add_parser("cutover")
    cutover_parser.add_argument("--plan", required=True, type=Path)
    cutover_parser.add_argument("--receipt", required=True, type=Path)
    cutover_parser.add_argument("--operations", required=True, type=Path)
    cutover_parser.add_argument("--max-submitted", type=int, default=1000)
    args = parser.parse_args()
    if args.mode == "worker":
        worker(args)
    else:
        cutover(args)


if __name__ == "__main__":
    main()
