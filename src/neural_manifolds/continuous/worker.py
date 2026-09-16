"""Bounded compute tasks with explicit technical completion receipts."""

from __future__ import annotations

import argparse
import json
import os
import traceback
from pathlib import Path

import numpy as np

from neural_manifolds.continuous.audit import inventory, observed_lengths
from neural_manifolds.provenance import atomic_write_json, sha256_file


def recovery(audit_path: Path, destination: Path) -> dict:
    """Length-matched AR(1) identifiability smoke, not validation of five axes."""
    lengths = observed_lengths(json.loads(audit_path.read_text()))
    rng = np.random.default_rng(20260916)
    rows = []
    for seconds in lengths:
        for rate in (1, 10):
            count = seconds * rate
            for truth in (0.0, 0.5, 0.9):
                estimates = []
                for _ in range(500):
                    # Burn in to stationarity before observing the short segment.
                    x = np.zeros(count + 1000)
                    noise = rng.normal(size=len(x))
                    for i in range(1, len(x)):
                        x[i] = truth * x[i - 1] + noise[i]
                    x = x[1000:]
                    estimates.append(float(x[:-1] @ x[1:] / (x[:-1] @ x[:-1])))
                error = np.asarray(estimates) - truth
                rows.append(
                    {
                        "seconds": seconds,
                        "samples_per_second": rate,
                        "samples": count,
                        "truth": truth,
                        "bias": float(error.mean()),
                        "rmse": float(np.sqrt(np.mean(error**2))),
                        "repetitions": 500,
                    }
                )
    result = {
        "scope": "ar1_length_recovery_component_only",
        "rows": rows,
        "full_estimability_phase_complete": False,
        "no_result_gate": True,
        "input_sha256": sha256_file(audit_path),
        "missing": [
            "axis_specific_recovery",
            "uncertainty_coverage",
            "observed_artifact_and_missing_channel_simulations",
        ],
    }
    atomic_write_json(destination, result)
    return result


def run(spec_path: Path) -> int:
    spec = json.loads(spec_path.read_text())
    output = Path(spec["output"])
    output.mkdir(parents=True, exist_ok=True)
    receipt = {
        "task_id": spec["task_id"],
        "source_digest": spec["source_digest"],
        "spec_sha256": sha256_file(spec_path),
        "job_id": os.environ.get("SLURM_JOB_ID"),
    }
    try:
        input_path = Path(spec["input"])
        identity_path = (
            input_path / ".acquisition/COMPLETE.json" if spec["kind"] == "inventory" else input_path
        )
        if sha256_file(identity_path) != spec["input_identity"]:
            raise ValueError("Input changed after task planning")
        if spec["kind"] == "inventory":
            inventory(Path(spec["input"]), output / "inventory.json")
            artifacts = [output / "inventory.json"]
        elif spec["kind"] == "recovery":
            recovery(Path(spec["input"]), output / "recovery.json")
            artifacts = [output / "recovery.json"]
        elif spec["kind"] == "signal_qc":
            from neural_manifolds.continuous.signal_qc import run as signal_qc

            signal_qc(Path(spec["input"]), output)
            artifacts = [output / "signal_qc.json"]
        elif spec["kind"].startswith("revised_"):
            from neural_manifolds.revised.driver import run as revised_run

            artifacts = revised_run(spec, output)
        else:
            raise ValueError(f"Unimplemented task kind: {spec['kind']}")
        receipt.update(status="complete", artifacts={str(p): sha256_file(p) for p in artifacts})
        code = 0
    except Exception as exc:
        receipt.update(
            status="failed",
            error=type(exc).__name__,
            detail=str(exc),
            traceback=traceback.format_exc(),
        )
        code = 1
    atomic_write_json(output / "receipt.json", receipt)
    return code


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True, type=Path)
    raise SystemExit(run(parser.parse_args().spec))
