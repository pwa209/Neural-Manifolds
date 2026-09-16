"""Observed-length diagnostics for every implemented candidate dynamics feature."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from neural_manifolds.provenance import atomic_write_json, sha256_file
from neural_manifolds.revised.inference import DYNAMICS, FoldDynamics


def simulate(count, rng, *, coupling=0.0, noise=1.0):
    matrix = np.array([[0.5, coupling, 0.0], [0.0, 0.5, coupling], [coupling, 0.0, 0.5]])
    x = np.zeros((count + 200, 3))
    innovation = rng.normal(scale=noise, size=x.shape)
    for i in range(1, len(x)):
        x[i] = matrix @ x[i - 1] + innovation[i]
    return x[200:]


def as_record(x, segments=None):
    return {
        "trajectory": x,
        "segments": np.zeros(len(x), int) if segments is None else segments,
        "regions": {str(i): x[:, i : i + 1] for i in range(x.shape[1])},
    }


def run_recovery(measurements: list[Path], output: Path, policy: dict, *, bootstrap=100):
    lengths = sorted(
        {
            r["clean_seconds"]
            for p in measurements
            for r in json.loads(p.read_text())["records"]
            if r["status"] == "measured"
        }
    )
    rng = np.random.default_rng(policy["seed"])
    fit_x = simulate(1000, rng, coupling=0.2)
    frame = pd.DataFrame(
        {"unit_id": ["independent_simulation"], "participant_id": ["sim"], "study_group": ["sim"]}
    )
    transform = FoldDynamics(seed=policy["seed"]).fit(
        frame, {"independent_simulation": as_record(fit_x)}
    )
    rows = []
    checkpoint_identity = {
        "inputs": {str(p): sha256_file(p) for p in measurements},
        "seed": policy["seed"],
        "repetitions": policy["recovery_repetitions"],
        "bootstrap": bootstrap,
    }
    for coupling in (0.0, 0.3):
        rng = np.random.default_rng(policy["seed"] + round(coupling * 1000))
        reference, _ = transform.transform_one(as_record(simulate(4000, rng, coupling=coupling)))
        for n in lengths:
            for missing_fraction in (0.0, 0.2):
                checkpoint = (
                    output / "recovery_cells" / f"n{n}-c{coupling}-m{missing_fraction}.json"
                )
                if checkpoint.exists():
                    saved = json.loads(checkpoint.read_text())
                    if saved["identity"] != checkpoint_identity:
                        raise ValueError("recovery_checkpoint_input_changed")
                    rows.extend(saved["rows"])
                    continue
                rng = np.random.default_rng(
                    policy["seed"]
                    + n * 10000
                    + round(coupling * 1000)
                    + round(missing_fraction * 100)
                )
                begin = len(rows)
                estimates = {axis: [] for axis in DYNAMICS}
                coverage = {axis: [] for axis in DYNAMICS}
                for _ in range(policy["recovery_repetitions"]):
                    x = simulate(n, rng, coupling=coupling)
                    keep = rng.random(n) >= missing_fraction
                    indices = np.flatnonzero(keep)
                    if len(indices) < 5:
                        continue
                    observed = x[keep]
                    segments = np.cumsum(np.r_[True, np.diff(indices) != 1])
                    estimate, _ = transform.transform_one(as_record(observed, segments))
                    samples = {axis: [] for axis in DYNAMICS}
                    for _ in range(bootstrap):
                        # Fixed 3-second moving blocks; artificial joins are segmented.
                        starts = rng.integers(
                            0, max(1, len(observed) - 2), size=(len(observed) + 2) // 3
                        )
                        take = np.concatenate(
                            [np.arange(a, min(a + 3, len(observed))) for a in starts]
                        )[: len(observed)]
                        copied_segments = np.cumsum(
                            np.r_[
                                True,
                                (np.diff(take) != 1) | (segments[take[1:]] != segments[take[:-1]]),
                            ]
                        )
                        values, _ = transform.transform_one(
                            as_record(observed[take], copied_segments)
                        )
                        for axis in DYNAMICS:
                            if np.isfinite(values[axis]):
                                samples[axis].append(values[axis])
                    for axis in DYNAMICS:
                        if np.isfinite(estimate[axis]):
                            estimates[axis].append(estimate[axis])
                        if len(samples[axis]) >= 0.8 * bootstrap and np.isfinite(reference[axis]):
                            lo, hi = np.quantile(samples[axis], [0.025, 0.975])
                            coverage[axis].append(float(lo <= reference[axis] <= hi))
                for axis in DYNAMICS:
                    values = np.asarray(estimates[axis])
                    rows.append(
                        {
                            "axis": axis,
                            "observations": n,
                            "coupling": coupling,
                            "missing_fraction": missing_fraction,
                            "target": float(reference[axis]),
                            "estimable_replicates": len(values),
                            "bias": float(np.mean(values - reference[axis]))
                            if len(values)
                            else None,
                            "rmse": float(np.sqrt(np.mean((values - reference[axis]) ** 2)))
                            if len(values)
                            else None,
                            "interval_coverage": float(np.mean(coverage[axis]))
                            if coverage[axis]
                            else None,
                            "intervals_estimable": len(coverage[axis]),
                        }
                    )
                atomic_write_json(
                    checkpoint, {"identity": checkpoint_identity, "rows": rows[begin:]}
                )
    result = {
        "rows": rows,
        "scientific_gates": False,
        "reference": "independent_long_monte_carlo_reference_not_analytic_ground_truth",
        "interval": "three_second_moving_block_percentile_diagnostic",
        "limitations": [
            "coverage_is_measured_not_assumed",
            "simulation_does_not_establish_biological_validity",
            "channel_dropout_and_encoder_input_noise_require_separate_signal_level_validation",
        ],
    }
    atomic_write_json(output / "axis_recovery.json", result)
    return result
