"""Top-level command-line interface for configuration and queued phases."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from pathlib import Path

from neural_manifolds.config import config_sha256, load_study, load_yaml
from neural_manifolds.data.cli import main as data_main
from neural_manifolds.data.registry import load_dataset_registry
from neural_manifolds.phase_runner import PHASES, PhaseContext, run_phase


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="neural-manifolds")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-config")
    validate.add_argument("--study", type=Path, default=Path("configs/study.yaml"))
    validate.add_argument("--datasets", type=Path, default=Path("configs/datasets.yaml"))
    validate.add_argument("--server", type=Path, default=Path("configs/server.yaml"))
    validate.add_argument("--json", action="store_true", dest="json_output")

    run = subparsers.add_parser("run-phase")
    run.add_argument("--phase", required=True, choices=sorted(PHASES))
    run.add_argument("--study", type=Path, required=True)
    run.add_argument("--datasets", type=Path, required=True)
    run.add_argument("--server", type=Path, required=True)
    run.add_argument("--run-id", required=True)

    data = subparsers.add_parser("data")
    data.add_argument("args", nargs=argparse.REMAINDER)
    return parser


def validate_configs(study_path: Path, datasets_path: Path, server_path: Path) -> dict[str, object]:
    study = load_study(study_path)
    registry = load_dataset_registry(datasets_path)
    server = load_yaml(server_path)
    if server.get("scientific_gates") is not False:
        raise ValueError("configs/server.yaml must set scientific_gates: false")
    storage = server.get("storage")
    if not isinstance(storage, dict) or storage.get("raw_data_location") != "canonical_only":
        raise ValueError("server storage must enforce raw_data_location: canonical_only")
    expected_env = {
        "canonical_root": os.environ.get("NEURAL_MANIFOLDS_CANONICAL_ROOT"),
        "work_root": os.environ.get("NEURAL_MANIFOLDS_WORK_ROOT"),
        "checkpoint_root": os.environ.get("NEURAL_MANIFOLDS_CHECKPOINT_ROOT"),
    }
    for key, value in expected_env.items():
        if value is not None and storage.get(key) != value:
            raise ValueError(f"server storage.{key} does not match the queue environment")
    unresolved = [key for key in expected_env if not storage.get(key)]
    return {
        "valid": True,
        "project_status": study.status,
        "scientific_gates": study.scientific_gates,
        "study_sha256": config_sha256(study),
        "dataset_registry_sha256": registry.sha256,
        "dataset_count": len(registry.model.datasets),
        "unresolved_server_roots": unresolved,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "validate-config":
        result = validate_configs(args.study, args.datasets, args.server)
        if args.json_output:
            print(json.dumps(result, indent=2, sort_keys=True))
        else:
            print("configuration valid")
            if result["unresolved_server_roots"]:
                print("server roots unresolved: " + ", ".join(result["unresolved_server_roots"]))
        return 0
    if args.command == "data":
        return data_main(args.args)
    if args.command == "run-phase":
        context = PhaseContext.from_environment(
            phase=args.phase,
            run_id=args.run_id,
            study_path=args.study,
            datasets_path=args.datasets,
            server_path=args.server,
        )
        artifacts = run_phase(context)
        print(json.dumps({"phase": args.phase, "artifacts": [str(path) for path in artifacts]}))
        return 0
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
