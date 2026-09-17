"""Source-adjudicated supplementary high-density transfer, preserving core outputs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import yaml

from neural_manifolds.continuous.controller import valid_receipt
from neural_manifolds.provenance import atomic_write_json, sha256_file
from neural_manifolds.revised.cohort import build_cohort
from neural_manifolds.revised.driver import load_encoder, verified_bundle


def extension_policy():
    extension = yaml.safe_load(Path("configs/high_density_extension.yaml").read_text())
    policy = yaml.safe_load(Path(extension["base_policy"]).read_text())
    policy["source_policy"][extension["dataset_id"]] = extension["source_rule"]
    return policy, extension


def admitted_cohort(audit, policy, extension):
    result = build_cohort({extension["dataset_id"]: audit}, policy)
    kept = []
    for unit in result["units"]:
        if unit.get("remarks", "").strip():
            result["exclusions"].append(
                {
                    "unit_id": unit["unit_id"],
                    "reason": "source_quality_annotation_excluded_before_signal_analysis",
                }
            )
        elif "high_density" in unit["tracks"]:
            unit["tracks"] = ["high_density"]
            kept.append(unit)
        else:
            result["exclusions"].append(
                {"unit_id": unit["unit_id"], "reason": "no_observed_high_density_track"}
            )
    result["units"] = kept
    result["scope"] = extension["scope"]
    result["limitations"] = extension["limitations"]
    return result


def prepare_measurements(root, inventory, previous_controller, policy, extension):
    from neural_manifolds.revised.measurement import measure_cohort

    state = json.loads((previous_controller / "state.json").read_text())
    old_source = previous_controller.parents[1] / "releases" / state["source_digest"]
    old_manifest = json.loads((old_source / "release_manifest.json").read_text())
    current = json.loads(Path("release_manifest.json").read_text())
    # Reuse only analytically identical implementations. Acquisition changes and
    # this supplementary driver do not change the previously computed features.
    for name, digest in old_manifest["files"].items():
        relevant = (
            name.startswith("src/") and name != "src/neural_manifolds/data/providers.py"
        ) or name == extension["base_policy"]
        if relevant and current["files"].get(name) != digest:
            raise ValueError("analytical_code_changed_cannot_reuse:" + name)
    inventory_task = next(
        (
            t
            for t in state.get("tasks", {}).values()
            if t["kind"] == "inventory"
            and (Path(t["output"]) / "inventory.json").resolve() == inventory.resolve()
        ),
        None,
    )
    # Seed inventory may belong to an earlier controller; validate that producer.
    if inventory_task is None:
        producer_state = json.loads((inventory.parents[2] / "state.json").read_text())
        inventory_task = next(
            t
            for t in producer_state["tasks"].values()
            if (Path(t["output"]) / "inventory.json").resolve() == inventory.resolve()
        )
    if not valid_receipt(inventory_task):
        raise ValueError("inventory_producer_receipt_invalid")
    cohort = admitted_cohort(json.loads(inventory.read_text()), policy, extension)
    cohort["inventory_sha256"] = sha256_file(inventory)
    path = root / "cohort.json"
    atomic_write_json(path, cohort)
    measured = root / "measurement-new"
    measure_cohort(path, measured, policy, load_encoder() if cohort["units"] else None)
    inputs = [measured / "measurement.json"]
    for dataset in extension["reused_datasets"]:
        task = next(
            t
            for t in state["tasks"].values()
            if t["kind"] == "revised_measure"
            and t["dataset_id"] == dataset
            and t["status"] == "complete"
        )
        if not valid_receipt(task):
            raise ValueError("measurement_producer_receipt_invalid")
        original = Path(task["output"]) / "measurement.json"
        data = json.loads(original.read_text())
        view = {
            "records": [r for r in data["records"] if r["track"] == "high_density"],
            "original_artifact": str(original),
            "original_sha256": sha256_file(original),
            "projection": "high_density_track_only_no_feature_refitting_or_label_changes",
        }
        path = root / "reused" / f"{dataset}.json"
        atomic_write_json(path, view)
        inputs.append(path)
    atomic_write_json(
        root / "measurement-inputs.json",
        {"inputs": [{"path": str(p), "sha256": sha256_file(p)} for p in inputs]},
    )


def verified_phase(root, phase, identity, replicate=0):
    folder = root / phase / (str(replicate) if phase == "controls" else "outputs")
    receipt = json.loads((folder / "execution.json").read_text())
    if any(receipt.get(key) != value for key, value in identity.items()):
        raise ValueError("extension_phase_identity_mismatch:" + phase)
    if receipt.get("status") != "executed" or receipt.get("phase") != phase:
        raise ValueError("extension_phase_incomplete:" + phase)
    if receipt.get("replicate") != replicate:
        raise ValueError("extension_replicate_mismatch")
    artifact = Path(receipt["artifact"])
    if sha256_file(artifact) != receipt["sha256"]:
        raise ValueError("extension_artifact_changed:" + phase)
    return artifact


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "phase",
        choices=[
            "measure",
            "transfer",
            "controls",
            "sensitivities",
            "perturb",
            "robustness",
            "synthesis",
        ],
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--inventory", type=Path)
    parser.add_argument("--previous-controller", type=Path)
    parser.add_argument("--replicate", type=int, default=0)
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    policy, extension = extension_policy()
    current = json.loads(Path("release_manifest.json").read_text())
    identity = {
        "source_digest": current["source_digest"],
        "scope": extension["scope"],
        "limitations": extension["limitations"],
        "scientific_gates": False,
    }
    provenance = root / "extension-provenance.json"
    if provenance.exists() and json.loads(provenance.read_text()) != identity:
        raise ValueError("extension_identity_changed")
    atomic_write_json(provenance, identity)
    output = root / args.phase / (str(args.replicate) if args.phase == "controls" else "outputs")
    output.mkdir(parents=True, exist_ok=True)
    if args.phase == "measure":
        prepare_measurements(
            root, args.inventory.resolve(), args.previous_controller.resolve(), policy, extension
        )
        artifact = root / "measurement-inputs.json"
    else:
        verified_phase(root, "measure", identity)
        paths = verified_bundle(root / "measurement-inputs.json")
        if args.phase == "transfer":
            from neural_manifolds.revised.inference import run_transfer

            run_transfer(paths, output, policy)
            artifact = output / "transfer.json"
        elif args.phase == "controls":
            from neural_manifolds.revised.controls import run_controls

            run_controls(paths, output, policy, replicate=args.replicate)
            artifact = output / "controls.json"
        elif args.phase == "sensitivities":
            from neural_manifolds.revised.controls import run_sensitivities

            run_sensitivities(paths, output, policy)
            artifact = output / "sensitivities.json"
        elif args.phase == "perturb":
            from neural_manifolds.revised.robustness import measure_perturbations

            measure_perturbations(paths, output, policy, load_encoder())
            artifact = output / "perturbation_measurements.json"
        elif args.phase == "robustness":
            from neural_manifolds.revised.robustness import run_robustness

            run_robustness(verified_phase(root, "perturb", identity), output, policy)
            artifact = output / "robustness.json"
        else:
            from neural_manifolds.revised.synthesis import synthesize

            evidence = [
                verified_phase(root, p, identity)
                for p in ["transfer", "sensitivities", "robustness"]
            ]
            evidence += [
                verified_phase(root, "controls", identity, i)
                for i in range(policy["surrogate_repetitions"])
            ]
            for p in evidence:
                if not p.is_file():
                    raise ValueError("missing_extension_evidence:" + str(p))
            synthesize(evidence, output, expected_controls=policy["surrogate_repetitions"])
            artifact = output / "evidence.json"
    atomic_write_json(
        output / "execution.json",
        {
            **identity,
            "phase": args.phase,
            "replicate": args.replicate,
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "status": "executed",
            "artifact": str(artifact),
            "sha256": sha256_file(artifact),
        },
    )


if __name__ == "__main__":
    main()
