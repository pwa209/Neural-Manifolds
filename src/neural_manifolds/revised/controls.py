"""Refit null controls and explicitly distinct report/stage sensitivities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from neural_manifolds.provenance import atomic_write_json
from neural_manifolds.revised.inference import (
    load_measurements,
    measurement_arrays,
    nested_transfer,
    summarize_predictions,
)


def null_arrays(arrays, kind, rng):
    """Segment-local temporal nulls; never generate transitions across gaps.

    Phase null preserves the *representation* spectrum, not the original EEG
    spectrum. It must never be reported as an EEG spectral control.
    """
    result = {}
    for unit, record in arrays.items():
        trajectory = record["trajectory"].copy()
        regions = {k: v.copy() for k, v in record["regions"].items()}
        for segment in np.unique(record["segments"]):
            positions = np.flatnonzero(record["segments"] == segment)
            if kind == "temporal_permutation":
                order = rng.permutation(len(positions))
                trajectory[positions] = trajectory[positions[order]]
                for values in regions.values():
                    values[positions] = values[positions[order]]
            elif kind == "representation_phase":
                for values in [trajectory, *regions.values()]:
                    fourier = np.fft.rfft(values[positions], axis=0)
                    angle = rng.uniform(-np.pi, np.pi, size=fourier.shape)
                    angle[0] = 0
                    if len(positions) % 2 == 0:
                        angle[-1] = 0
                    values[positions] = np.fft.irfft(
                        fourier * np.exp(1j * angle), n=len(positions), axis=0
                    )
            else:
                raise ValueError("unknown_null")
        result[unit] = {**record, "trajectory": trajectory, "regions": regions}
    return result


def run_controls(paths: list[Path], output: Path, policy: dict, *, replicate=0):
    records = load_measurements(paths)
    rows = []
    for track in sorted({r["track"] for r in records}):
        for representation in ("sensor", "encoder", "time_frequency"):
            frame, arrays = measurement_arrays(
                [r for r in records if r["track"] == track], representation
            )
            if frame.empty:
                continue
            for kind in ("label_permutation", "temporal_permutation", "representation_phase"):
                rng = np.random.default_rng(policy["seed"] + replicate)
                shuffled = frame.copy()
                transformed = arrays
                if kind == "label_permutation":
                    for _, block in shuffled.groupby(
                        ["study_group", "participant_id", "sleep_stage"]
                    ):
                        shuffled.loc[block.index, "experience"] = rng.permutation(block.experience)
                else:
                    transformed = null_arrays(arrays, kind, rng)
                try:
                    fit = nested_transfer(
                        shuffled,
                        transformed,
                        seed=policy["seed"],
                        dimensions=policy["profile_dimensions"],
                        strengths=policy["regularization"],
                    )
                    summary = summarize_predictions(
                        pd.DataFrame(fit["predictions"]), repetitions=20, seed=policy["seed"]
                    )
                    rows.append(
                        dict(
                            track=track,
                            representation=representation,
                            kind=kind,
                            replicate=replicate,
                            summary=summary,
                            unavailable=fit["unavailable"],
                        )
                    )
                except ValueError as exc:
                    rows.append(
                        dict(
                            track=track,
                            representation=representation,
                            kind=kind,
                            replicate=replicate,
                            status="unavailable",
                            reason=str(exc),
                        )
                    )
    result = {
        "rows": rows,
        "full_nested_refit": True,
        "replicate": replicate,
        "exchangeability": "labels exchangeable within participant and stage under null",
        "representation_phase_is_not_signal_level_spectral_control": True,
        "scientific_gates": False,
    }
    atomic_write_json(output / "controls.json", result)
    return result


def exact_electrode_records(records):
    """Electrode provenance is recorded by preprocessing, not inferred from names."""
    return [
        r
        for r in records
        if r["primary_eligible"]
        and r.get("preprocessing", {}).get("position_equivalence") == "native_named_channel"
    ]


def run_sensitivities(paths, output, policy):
    records = load_measurements(paths, primary=False)
    results = {}
    for stage in (2, 5):
        for contrast, codes in [
            ("DE_vs_NE", (2, 0)),
            ("DEWR_vs_NE", (1, 0)),
            ("DE_vs_DEWR", (2, 1)),
        ]:
            if stage == 2 and contrast == "DE_vs_NE":
                continue
            selected = [
                dict(r, experience=int(r["report_code"] == codes[0]))
                for r in records
                if r["sleep_stage"] == stage and r["report_code"] in codes
            ]
            for track in sorted({r["track"] for r in selected}):
                for representation in ("sensor", "encoder", "time_frequency"):
                    frame, arrays = measurement_arrays(
                        [r for r in selected if r["track"] == track], representation
                    )
                    key = f"{stage}:{contrast}:{track}:{representation}"
                    try:
                        fit = nested_transfer(
                            frame,
                            arrays,
                            seed=policy["seed"],
                            dimensions=policy["profile_dimensions"],
                            strengths=policy["regularization"],
                        )
                        fit["summary"] = summarize_predictions(
                            pd.DataFrame(fit["predictions"]),
                            repetitions=policy["bootstrap_repetitions"],
                            seed=policy["seed"],
                        )
                        results[key] = fit
                    except (ValueError, KeyError, AttributeError) as exc:
                        results[key] = {"status": "unavailable", "reason": str(exc)}
    exact = exact_electrode_records(records)
    for track in sorted({r["track"] for r in exact}):
        for representation in ("sensor", "encoder", "time_frequency"):
            frame, arrays = measurement_arrays(
                [r for r in exact if r["track"] == track], representation
            )
            key = f"exclude_approximate_electrodes:{track}:{representation}"
            try:
                fit = nested_transfer(
                    frame,
                    arrays,
                    seed=policy["seed"],
                    dimensions=policy["profile_dimensions"],
                    strengths=policy["regularization"],
                )
                fit["summary"] = summarize_predictions(
                    pd.DataFrame(fit["predictions"]),
                    repetitions=policy["bootstrap_repetitions"],
                    seed=policy["seed"],
                )
                results[key] = fit
            except (ValueError, KeyError, AttributeError) as exc:
                results[key] = {"status": "unavailable", "reason": str(exc)}
    result = {"analyses": results, "scientific_gates": False, "DEWR_is_not_ordinal_midpoint": True}
    atomic_write_json(output / "sensitivities.json", result)
    return result
