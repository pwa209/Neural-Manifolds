"""Quota-resilient runner for an explicitly selected, immutable controller release.

Run from the ORIGINAL controller release with its src on PYTHONPATH. This file
can belong to a newer release: its digest is recorded separately from scientific
task provenance. No completed task/source identities are rewritten. Health lives
on a different filesystem so a full project quota cannot kill error reporting.
"""

from __future__ import annotations

import argparse
import errno
import json
import os
import subprocess
import time
from pathlib import Path

from filelock import FileLock

from neural_manifolds.continuous.controller import Controller, Slurm, now
from neural_manifolds.provenance import atomic_write_json


def quota_headroom(text: str) -> tuple[int, int]:
    """Return remaining hard-limit KiB and inodes from one lfs quota row."""
    for line in text.splitlines():
        fields = line.split()
        if fields and fields[0].startswith("/"):
            fields = fields[1:]
        if len(fields) == 8:
            try:
                used, limit, files, file_limit = (int(fields[i].rstrip("*")) for i in (0, 2, 4, 6))
            except ValueError:
                continue
            if limit <= 0 or file_limit <= 0:
                raise ValueError("quota hard limits unavailable")
            return limit - used, file_limit - files
    raise ValueError("unrecognized lfs quota response")


def is_capacity_error(exc: Exception) -> bool:
    return isinstance(exc, OSError) and exc.errno in {errno.EDQUOT, errno.ENOSPC}


def publish_health(project: Path, fallback: Path, health: dict) -> None:
    """Always try both locations; neither failed write ends recovery."""
    for path in (fallback, project):
        try:
            atomic_write_json(path, health)
        except OSError as exc:
            print(f"health_write_failed {path.name}: {exc}", flush=True)


class ReleaseScheduler(Slurm):
    """Use each repaired task's actual source, including any scheduler retry."""

    def submit(self, spec: Path, name: str, root: Path, account: str) -> str:
        original = Path.cwd()
        digest = json.loads(spec.read_text())["source_digest"]
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("Invalid task source digest")
        release = original.parent / digest
        manifest = json.loads((release / "release_manifest.json").read_text())
        if manifest["source_digest"] != digest:
            raise ValueError("Task release identity mismatch")
        try:
            os.chdir(release)
            return super().submit(spec, name, root, account)
        finally:
            os.chdir(original)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--health-root", required=True, type=Path)
    parser.add_argument("--acquisition-state", required=True, type=Path)
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--account", required=True)
    parser.add_argument("--quota-project-id", required=True)
    parser.add_argument("--quota-path", required=True, type=Path)
    parser.add_argument("--minimum-free-files", type=int, default=10000)
    parser.add_argument("--minimum-free-kib", type=int, default=10 * 1024 * 1024)
    parser.add_argument("--maximum-jobs", type=int, default=1)
    parser.add_argument("--interval", type=float, default=60)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.maximum_jobs < 1 or args.interval < 1:
        parser.error("maximum-jobs and interval must be positive")
    args.health_root.mkdir(parents=True, exist_ok=True)
    wrapper = json.loads(
        (Path(__file__).resolve().parents[2] / "release_manifest.json").read_text()
    )
    source = json.loads(Path("release_manifest.json").read_text())["source_digest"]
    base = {
        "pid": os.getpid(),
        "controller_source_digest": source,
        "supervisor_source_digest": wrapper["source_digest"],
    }
    failures = 0
    with FileLock(str(args.root / "controller.lock"), timeout=0):
        while not (args.root / "STOP").exists():
            health = {**base, "at": now()}
            try:
                quota = subprocess.run(
                    ["lfs", "quota", "-p", args.quota_project_id, str(args.quota_path)],
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=30,
                )
                free_kib, free_files = quota_headroom(quota.stdout)
                health.update(free_kib=free_kib, free_files=free_files)
                if free_kib < args.minimum_free_kib or free_files < args.minimum_free_files:
                    health.update(status="waiting_for_quota_headroom")
                else:
                    # Reload durable state after every failure; submission intent
                    # persisted by the controller prevents duplicate jobs.
                    controller = Controller(
                        args.root.resolve(),
                        args.acquisition_state.resolve(),
                        args.registry.resolve(),
                        args.account,
                        scheduler=ReleaseScheduler(),
                        maximum_jobs=args.maximum_jobs,
                    )
                    controller.tick()
                    health.update(status="healthy")
                    failures = 0
            except Exception as exc:
                if is_capacity_error(exc):
                    health.update(status="waiting_for_quota_recovery", error=str(exc))
                else:
                    failures += 1
                    health.update(status="error", error=str(exc), consecutive_errors=failures)
            publish_health(
                args.root / "heartbeat.json", args.health_root / "heartbeat.json", health
            )
            if failures >= 3:
                raise RuntimeError("Repeated non-quota failure requires inspection; see heartbeat")
            if args.once:
                break
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
