"""Bounded fresh acquisition with durable per-dataset progress and no result gates."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from dataclasses import asdict
from pathlib import Path

from filelock import FileLock

from neural_manifolds.data.acquisition import AcquisitionManager
from neural_manifolds.data.providers import CommandRunner, OpenNeuroProvider
from neural_manifolds.data.registry import load_dataset_registry
from neural_manifolds.provenance import atomic_write_json, sha256_file


def source_size(manager, dataset, raw_root):
    provider = manager.provider(dataset)
    if not isinstance(provider, OpenNeuroProvider):
        discovery = provider.discover()
        if discovery.total_known_bytes is None:
            raise ValueError("Unknown source size; cannot enforce budget")
        return discovery.total_known_bytes
    provider.check()
    stage = raw_root / ".staging" / dataset.id / dataset.source.version
    runner = CommandRunner()
    if not (stage / ".git").exists():
        stage.parent.mkdir(parents=True, exist_ok=True)
        runner.run(["datalad", "clone", dataset.source.repository_url, str(stage)])
    runner.run(["git", "checkout", "--detach", dataset.source.revision], cwd=stage)
    response = runner.run(["git", "annex", "find", "--anything", "--json"], cwd=stage)
    sizes = {}
    for line in response.stdout.splitlines():
        row = json.loads(line)
        key = row.get("key", "")
        match = re.search(r"-s(\d+)(?:-|--)", key)
        if not match:
            raise ValueError("Annex object lacks declared size")
        sizes[key] = int(match.group(1))
    if not sizes:
        raise ValueError("No annex content enumerated")
    return sum(sizes.values())


def run(args):
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    registry = load_dataset_registry(args.registry)
    if any(d.access.mode != "open" for d in registry.datasets):
        raise ValueError("Fresh execution accepts only open datasets")
    state_path = args.state.resolve()
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(state_path) + ".lock", timeout=1):
        identity = sha256_file(args.registry)
        state = (
            json.loads(state_path.read_text())
            if state_path.exists()
            else {"registry_sha256": identity, "datasets": {}, "scientific_gates": False}
        )
        if state["registry_sha256"] != identity:
            raise ValueError("Cannot resume with a different registry")
        state.update(status="running", pid=os.getpid())
        atomic_write_json(state_path, state)
        manager = AcquisitionManager(registry)
        reserved = sum(r.get("reserved_bytes", 0) for r in state["datasets"].values())
        priority = {
            name: index
            for index, name in enumerate(
                [
                    "dream_tononi_serial_awakenings",
                    "propofol_tms_eeg",
                    "tactile_detection",
                    "somatosensory_report_task",
                    "psiconnect",
                ]
            )
        }
        ordered = sorted(registry.datasets, key=lambda d: (priority.get(d.id, 100), d.id))
        selected = getattr(args, "only_datasets", None)
        if selected:
            unknown = set(selected) - {d.id for d in registry.datasets}
            if unknown:
                raise ValueError(f"Unknown requested datasets: {sorted(unknown)}")
            ordered = [d for d in ordered if d.id in selected]
        for dataset in ordered:
            previous = state["datasets"].get(dataset.id, {})
            if previous.get("status") == "complete":
                # A restart must not spend hours rediscovering sealed downloads.
                # Signal workers independently bind their inventory to COMPLETE.
                completed = Path(previous["result"]["release_path"])
                manifest = completed / ".acquisition/manifest.json"
                expected = previous["result"]["details"]["manifest_json_sha256"]
                if (completed / ".acquisition/COMPLETE.json").is_file() and sha256_file(
                    manifest
                ) == expected:
                    continue
                raise ValueError(f"Previously completed acquisition receipt changed: {dataset.id}")
            entry = {**previous, "status": "checking"}
            state["datasets"][dataset.id] = entry
            atomic_write_json(state_path, state)
            try:
                size = source_size(manager, dataset, root)
                addition = max(0, size - previous.get("reserved_bytes", 0))
                if size > args.max_source_bytes or reserved + addition > args.max_total_bytes:
                    raise ValueError("Source exceeds bounded acquisition budget")
                if shutil.disk_usage(root).free < size + args.free_headroom_bytes:
                    raise ValueError("Filesystem free-space headroom insufficient")
                reserved += addition
                entry.update(status="downloading", reserved_bytes=size)
                atomic_write_json(state_path, state)
                entry.update(status="complete", result=asdict(manager.acquire(dataset, root)))
                entry.pop("error", None)
            except Exception as exc:
                entry.update(status="incomplete", error_type=type(exc).__name__, error=str(exc))
            atomic_write_json(state_path, state)
        state["status"] = (
            "complete"
            if all(
                state["datasets"].get(d.id, {}).get("status") == "complete"
                for d in registry.datasets
            )
            else "partial"
        )
        atomic_write_json(state_path, state)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--registry", type=Path, required=True)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--state", type=Path, required=True)
    p.add_argument(
        "--only-datasets",
        nargs="+",
        help="Resume selected sources without changing registry identity",
    )
    p.add_argument("--max-source-bytes", type=int, default=100 * 1024**3)
    p.add_argument("--max-total-bytes", type=int, default=180 * 1024**3)
    p.add_argument("--free-headroom-bytes", type=int, default=100 * 1024**3)
    run(p.parse_args())
