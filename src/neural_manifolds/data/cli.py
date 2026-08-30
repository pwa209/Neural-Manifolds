"""Command-line interface for dataset planning, acquisition, and validation."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from .acquisition import AcquisitionError, AcquisitionManager, AcquisitionResult
from .manifest import ManifestError
from .providers import AccessBlocked, ProviderError
from .registry import load_dataset_registry

DEFAULT_CONFIG = Path("configs/datasets.yaml")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="neural-manifolds-data",
        description="Acquire immutable, checksummed raw releases directly to target storage.",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--json", action="store_true", dest="json_output")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="list registry entries without network access")
    list_parser.add_argument("--dataset", action="append", dest="datasets")

    for name, help_text in (
        ("plan", "show target paths without network or filesystem writes"),
        ("acquire", "download, validate, checksum, and atomically publish releases"),
        ("validate", "rehash published releases and reject any drift"),
    ):
        child = subparsers.add_parser(name, help=help_text)
        child.add_argument("--dataset", action="append", dest="datasets")
        child.add_argument(
            "--root",
            type=Path,
            default=None,
            help="absolute raw-data root; or set NEURAL_MANIFOLDS_RAW_ROOT",
        )
        if name == "acquire":
            child.add_argument("--dry-run", action="store_true")
            child.add_argument(
                "--check-only",
                action="store_true",
                help="verify tools/endpoints and enumerate remote files without writing",
            )

    check_parser = subparsers.add_parser(
        "check", help="verify tools/endpoints and enumerate remote files without writing"
    )
    check_parser.add_argument("--dataset", action="append", dest="datasets")
    return parser


def _raw_root(argument: Path | None) -> Path:
    if argument is not None:
        return argument
    value = os.environ.get("NEURAL_MANIFOLDS_RAW_ROOT")
    if not value:
        raise AcquisitionError(
            "provide --root or set the non-secret NEURAL_MANIFOLDS_RAW_ROOT path"
        )
    return Path(value)


def _render(results: list[AcquisitionResult], *, json_output: bool) -> None:
    payload = [asdict(result) for result in results]
    if json_output:
        print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True))
        return
    for result in results:
        path = f" -> {result.release_path}" if result.release_path else ""
        print(
            f"{result.dataset_id}@{result.release_version}: {result.status}{path}",
            flush=True,
        )
        if result.status in {"blocked", "access_blocked"}:
            instructions = result.details.get("instructions") or result.details.get(
                "access", {}
            ).get("instructions")
            if instructions:
                print(f"  {instructions}", flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        registry = load_dataset_registry(args.config)
        selected = registry.select(getattr(args, "datasets", None))
        manager = AcquisitionManager(registry)
        results: list[AcquisitionResult] = []
        failures: list[str] = []
        if args.command == "list":
            for dataset in selected:
                results.append(
                    AcquisitionResult(
                        dataset_id=dataset.id,
                        release_version=dataset.source.version,
                        status="registered",
                        release_path="",
                        details={
                            "provider": dataset.source.provider,
                            "doi": dataset.source.doi,
                            "license": dataset.license.spdx,
                            "license_status": dataset.license.status,
                            "access_mode": dataset.access.mode,
                        },
                    )
                )
        elif args.command == "check" or (
            args.command == "acquire" and getattr(args, "check_only", False)
        ):
            for dataset in selected:
                try:
                    results.append(manager.check(dataset))
                except (ProviderError, OSError) as error:
                    failures.append(f"{dataset.id}: {error}")
        elif args.command == "plan":
            root = _raw_root(args.root)
            results.extend(manager.plan(dataset, root) for dataset in selected)
        elif args.command == "acquire":
            root = _raw_root(args.root)
            for dataset in selected:
                try:
                    results.append(manager.acquire(dataset, root, dry_run=args.dry_run))
                except (AcquisitionError, AccessBlocked, ProviderError, ManifestError) as error:
                    failures.append(f"{dataset.id}: {error}")
        elif args.command == "validate":
            root = _raw_root(args.root)
            for dataset in selected:
                try:
                    results.append(manager.validate(dataset, root))
                except (AcquisitionError, ManifestError) as error:
                    failures.append(f"{dataset.id}: {error}")
        _render(results, json_output=args.json_output)
        if failures:
            for failure in failures:
                print(f"ERROR: {failure}", file=sys.stderr)
            return 2
        if any(result.status in {"blocked", "access_blocked"} for result in results):
            return 3
        return 0
    except (AcquisitionError, ProviderError, ManifestError, KeyError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
