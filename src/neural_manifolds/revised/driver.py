"""Receipt-bound dispatch for the revised study's scientific jobs."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import yaml

from neural_manifolds.provenance import atomic_write_json, sha256_file


def load_encoder():
    from neural_manifolds.foundation.labram import OfficialLaBraMEncoder

    manifest_path = Path(os.environ["NEURAL_MANIFOLDS_MODEL_MANIFEST"])
    if sha256_file(manifest_path) != os.environ["NEURAL_MANIFOLDS_MODEL_MANIFEST_SHA256"]:
        raise ValueError("model_manifest_checksum_mismatch")
    model = json.loads(manifest_path.read_text())["models"]["labram_base"]
    source_manifest = Path(model["source"]["path"])
    if sha256_file(source_manifest) != model["source"]["sha256"]:
        raise ValueError("encoder_source_manifest_changed")
    # Existing cache verifier checks every source file, not only the manifest.
    from importlib.util import module_from_spec, spec_from_file_location

    spec = spec_from_file_location("nm_model_cache_check", "scripts/remote/model_cache.py")
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.verify_tracked_inventory("git", source_manifest.parent, source_manifest)
    config = yaml.safe_load(Path("configs/models.yaml").read_text())["models"]["labram_base"]
    if model["revision"] != config["revision"]:
        raise ValueError("encoder_source_revision_mismatch")
    return OfficialLaBraMEncoder(
        repository=source_manifest.parent,
        factory=config["factory"],
        checkpoint=model["checkpoint"]["path"],
        checkpoint_sha256=model["checkpoint"]["sha256"],
        device="cuda",
    )


def verified_bundle(path: Path) -> list[Path]:
    bundle = json.loads(path.read_text())
    paths = []
    for item in bundle["inputs"]:
        source = Path(item["path"])
        if sha256_file(source) != item["sha256"]:
            raise ValueError("upstream_scientific_artifact_changed")
        paths.append(source)
    return paths


def run(spec: dict, output: Path) -> list[Path]:
    policy = yaml.safe_load(Path("configs/revised_execution.yaml").read_text())
    source = Path(spec["input"])
    kind = spec["kind"]
    if kind == "revised_cohort":
        from neural_manifolds.revised.cohort import build_cohort

        result = build_cohort({spec["dataset_id"]: json.loads(source.read_text())}, policy)
        result["inventory_sha256"] = sha256_file(source)
        result["policy_sha256"] = sha256_file("configs/revised_execution.yaml")
        path = output / "cohort.json"
        atomic_write_json(path, result)
    elif kind == "revised_measure":
        from neural_manifolds.revised.measurement import measure_cohort

        cohort = json.loads(source.read_text())
        encoder = load_encoder() if cohort["units"] else None
        measure_cohort(source, output, policy, encoder)
        path = output / "measurement.json"
    elif kind == "revised_transfer":
        from neural_manifolds.revised.inference import run_transfer

        run_transfer(verified_bundle(source), output, policy)
        path = output / "transfer.json"
    elif kind == "revised_recovery":
        from neural_manifolds.revised.recovery import run_recovery

        run_recovery(verified_bundle(source), output, policy)
        path = output / "axis_recovery.json"
    elif kind == "revised_controls":
        from neural_manifolds.revised.controls import run_controls

        run_controls(
            verified_bundle(source),
            output,
            policy,
            replicate=int(spec["dataset_id"].rsplit("_", 1)[-1]),
        )
        path = output / "controls.json"
    elif kind == "revised_perturbation_measure":
        from neural_manifolds.revised.robustness import measure_perturbations

        measure_perturbations(verified_bundle(source), output, policy, load_encoder())
        path = output / "perturbation_measurements.json"
    elif kind == "revised_robustness":
        from neural_manifolds.revised.robustness import run_robustness

        run_robustness(source, output, policy)
        path = output / "robustness.json"
    elif kind == "revised_sensitivities":
        from neural_manifolds.revised.controls import run_sensitivities

        run_sensitivities(verified_bundle(source), output, policy)
        path = output / "sensitivities.json"
    elif kind in {"revised_tms", "revised_osf", "revised_tactile"}:
        from neural_manifolds.revised.perturbation import run_tms
        from neural_manifolds.revised.specificity import run_osf, run_tactile

        if source.name != "COMPLETE.json" or source.parent.name != ".acquisition":
            raise ValueError("a_sealed_acquisition_marker_is_required")
        release = source.parent.parent
        {"revised_tms": run_tms, "revised_osf": run_osf, "revised_tactile": run_tactile}[kind](
            release, output, policy
        )
        path = output / ("tms.json" if kind == "revised_tms" else "specificity.json")
    elif kind == "revised_boundary":
        from neural_manifolds.revised.boundary import run_boundary

        run_boundary(source, output, policy)
        path = output / "boundary.json"
    elif kind == "revised_synthesis":
        from neural_manifolds.revised.synthesis import synthesize

        return synthesize(verified_bundle(source), output)
    else:
        raise ValueError(f"Unimplemented revised scientific job: {kind}")
    return [path]
