"""Acquire a declared EEG/behaviour subset, never claim the entire MRI release."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path

from filelock import FileLock

from neural_manifolds.provenance import atomic_write_json, sha256_file

REVISION = "8c5d6b077c63d29796409b622ebc300fdec6aadd"
REPOSITORY = "https://github.com/OpenNeuroDatasets/ds006110.git"
EEG = re.compile(
    r"derivatives/EEG/cleaned_RELAX/FieldTrip_format/sub-PC\d+_ses-0[12]_task-(?:meditation|movie|music|rest)_Clean-ft\.mat$"
)


def selected_path(name):
    return bool(
        EEG.fullmatch(name)
        or name
        in {
            "README",
            "dataset_description.json",
            "participants.tsv",
            "participants.json",
            "bids/participants.tsv",
            "bids/participants.json",
        }
        or (name.startswith("phenotype/") and name.endswith((".tsv", ".json")))
    )


def command(args, root=None, timeout=600):
    if args[0] == "git":
        args = [
            "git",
            "-c",
            "user.name=Neural Manifolds acquisition",
            "-c",
            "user.email=neural-manifolds@localhost",
            *args[1:],
        ]
    return subprocess.run(
        args, cwd=root, check=True, text=True, capture_output=True, timeout=timeout
    ).stdout


def acquire(root: Path, maximum_bytes=20 * 1024**3):
    root = root.resolve()
    root.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(root) + ".lock", timeout=0):
        if not (root / ".git").exists():
            command(["git", "clone", "--no-checkout", REPOSITORY, str(root)], timeout=3600)
        origin = command(["git", "config", "--get", "remote.origin.url"], root).strip()
        if origin != REPOSITORY:
            raise ValueError("subset_repository_mismatch")
        command(["git", "checkout", "--detach", REVISION], root)
        command(["git", "annex", "init", "Neural Manifolds public EEG subset"], root)
        names = command(["git", "ls-tree", "-r", "--name-only", REVISION], root).splitlines()
        selected = sorted(n for n in names if selected_path(n))
        if not any(EEG.fullmatch(n) for n in selected):
            raise ValueError("no_official_context_specific_FieldTrip_files")
        keys = {}
        for line in command(["git", "annex", "find", "--anything", "--json"], root).splitlines():
            row = json.loads(line)
            if row["file"] in selected:
                keys[row["file"]] = row["key"]
        sizes = {}
        for key in keys.values():
            match = re.search(r"-s(\d+)(?:-|--)", key)
            if not match:
                raise ValueError("subset_unknown_annex_size")
            sizes[key] = int(match.group(1))
        size = sum(sizes.values())
        if size > maximum_bytes or shutil.disk_usage(root).free < size + 100 * 1024**3:
            raise ValueError("subset_exceeds_storage_bound")
        atomic_write_json(
            root / "SUBSET_PLANNED.json",
            {
                "revision": REVISION,
                "paths": selected,
                "reserved_bytes": size,
                "scope": "EEG_FieldTrip_and_public_behaviour_only_not_full_dataset",
            },
        )
        if keys:
            command(
                ["git", "annex", "get", "--jobs=2", "--", *sorted(keys)], root, timeout=24 * 3600
            )
        files = []
        for name in selected:
            path = root / name
            if not path.is_file():
                raise ValueError("subset_file_not_materialized:" + name)
            digest = sha256_file(path)
            if name in keys and keys[name].startswith("SHA256E-"):
                expected = keys[name].split("--", 1)[1].split(".", 1)[0]
                if digest != expected:
                    raise ValueError("annex_sha256_mismatch:" + name)
            files.append({"path": name, "sha256": digest, "bytes": path.stat().st_size})
        receipt = {
            "revision": REVISION,
            "repository": REPOSITORY,
            "files": files,
            "release": str(root),
            "status": "complete",
            "scope": "declared_subset_only",
            "licence": "CC0-1.0",
            "whole_dataset_complete": False,
        }
        atomic_write_json(root / "SUBSET_COMPLETE.json", receipt)
        return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    acquire(args.root)
