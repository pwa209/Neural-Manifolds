"""Revalidate and publish completed staging data on an allocated compute node."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

from filelock import FileLock

from neural_manifolds.data.acquisition import AcquisitionManager
from neural_manifolds.data.registry import load_dataset_registry
from neural_manifolds.provenance import atomic_write_json, sha256_file


def recover(registry_path, root, state_path, dataset_id):
    registry = load_dataset_registry(registry_path)
    dataset = next(d for d in registry.datasets if d.id == dataset_id)
    if dataset.access.mode != "open":
        raise ValueError("Recovery accepts only open datasets")
    candidates = [
        root / dataset.id / dataset.source.version,
        root / ".staging" / dataset.id / dataset.source.version,
    ]
    if not any((p / ".acquisition/COMPLETE.json").is_file() for p in candidates):
        raise ValueError("Recovery requires completed local staging or release; no downloads")
    with FileLock(str(state_path) + ".lock", timeout=1):
        state = json.loads(state_path.read_text())
        if state["registry_sha256"] != sha256_file(registry_path):
            raise ValueError("Registry identity mismatch")
        entry = state["datasets"][dataset_id]
        state.update(status="recovering", pid=os.getpid(), recovery_job=os.getenv("SLURM_JOB_ID"))
        entry["status"] = "validating_publication"
        atomic_write_json(state_path, state)
        try:
            result = AcquisitionManager(registry).acquire(dataset, root)
            entry.update(status="complete", result=asdict(result))
            entry.pop("error", None)
            entry.pop("error_type", None)
        except Exception as exc:
            entry.update(status="incomplete", error_type=type(exc).__name__, error=str(exc))
            state["status"] = "partial"
            atomic_write_json(state_path, state)
            raise
        state["status"] = "partial"
        atomic_write_json(state_path, state)
        print("RECOVERY_PUBLISHED", dataset_id, result.status, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()
    recover(args.registry, args.root, args.state, args.dataset)
