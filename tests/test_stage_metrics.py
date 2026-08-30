from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from neural_manifolds.config import load_study
from neural_manifolds.provenance import sha256_file
from neural_manifolds.stages.metrics import run_metrics


def _trajectory(path: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    states = rng.normal(size=(100, 8)).astype(np.float32)
    region_a = (states + rng.normal(scale=0.2, size=states.shape)).astype(np.float32)
    region_b = np.roll(region_a, 1, axis=0)
    with path.open("wb") as stream:
        np.savez_compressed(
            stream,
            global_states=states,
            alignment_regional_frontal=region_a,
            alignment_regional_parietal=region_b,
        )


def test_metric_stage_preserves_participant_split_and_writes_axes(tmp_path: Path) -> None:
    rows = []
    for index in range(5):
        path = tmp_path / f"trajectory-{index}.npz"
        _trajectory(path, index + 10)
        rows.append(
            {
                "unit_id": f"unit-{index}",
                "participant_id": f"sub-{index}",
                "dataset_id": "synthetic",
                "trajectory_path": str(path),
                "trajectory_sha256": sha256_file(path),
                "encoded": True,
                "healthy_wake_reference": True,
                "clinical_holdout": False,
                "condition": "wake",
            }
        )
    manifest = tmp_path / "encoding.parquet"
    pd.DataFrame(rows).to_parquet(manifest, index=False)
    outputs = run_metrics(
        encoding_manifest=manifest,
        output_root=tmp_path / "metrics",
        study=load_study(Path(__file__).parents[1] / "configs" / "study.yaml"),
        state_counts=(3, 4),
        null_repeats=0,
        force_state_fallback=True,
    )
    profiles = pd.read_parquet(outputs[0])
    assert len(profiles) == 5
    assert {"repertoire", "metastability", "directionality", "alignment", "reachability"} <= set(
        profiles
    )
    audit = pd.read_json(outputs[-1], typ="series")
    assert audit["participant_overlap"] == []
    assert audit["scientific_gate_applied"] is False
