from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from neural_manifolds.config import load_study
from neural_manifolds.manifold.profile import FiveAxisProfileEstimator
from neural_manifolds.provenance import sha256_file
from neural_manifolds.stages.metrics import (
    LoadedUnit,
    _load_unit,
    _record,
    _surrogate_record,
    run_metrics,
)


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


def test_explicit_short_interval_uses_encoding_window_contract(tmp_path: Path) -> None:
    path = tmp_path / "final-20-seconds.npz"
    rng = np.random.default_rng(20260830)
    states = rng.normal(size=(19, 8)).astype(np.float32)
    with path.open("wb") as stream:
        np.savez_compressed(
            stream,
            global_states=states,
            alignment_regional_frontal=states,
            alignment_regional_occipital=np.roll(states, 1, axis=0),
        )
    unit = _load_unit(
        {
            "unit_id": "dream-final-20",
            "participant_id": "p01",
            "dataset_id": "dream",
            "trajectory_path": str(path),
            "trajectory_sha256": sha256_file(path),
            "minimum_valid_windows_required": 10,
        }
    )
    assert unit.trajectory.shape[0] == 19


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
    assert bool(audit["alignment_windows_overlap"]) is False
    assert audit["alignment_lags_ms"] == list(range(1000, 10_001, 1000))
    assert audit["unavailable_short_lags_ms"] == list(range(20, 201, 20))
    assert audit["short_lag_status"] == "unavailable_no_independent_sensor_or_csd_track"
    assert audit["profile_input_spaces"] == {
        "repertoire": {
            "record_field": "repertoire_trajectory",
            "space": "untruncated_frozen_encoder_embedding",
            "dimension": 8,
            "discovery_projection_applied": False,
        },
        "dynamics": {
            "record_field": "trajectory",
            "space": "discovery_fitted_pca_projection",
            "dimension": 8,
            "discovery_projection_applied": True,
        },
    }
    assert set(profiles["repertoire_source_dimension"]) == {8}
    assert set(profiles["dynamics_projection_dimension"]) == {8}
    assert set(profiles["pretraining_overlap_status"]) == {"unresolved"}
    assert set(profiles["zero_shot_classification"]) == {"unresolved_not_verified_zero_shot"}
    assert not profiles["zero_shot_verified"].any()
    nulls = pd.read_parquet(outputs[1])
    assert {"pretraining_overlap_status", "zero_shot_verified"} <= set(nulls.columns)
    frozen = joblib.load(outputs[3])
    assert frozen.input_space_audit() == audit["profile_input_spaces"]
    assert frozen.wake_propofol_reference_ is None
    assert audit["clinical_wake_propofol_reference"]["status"] == "unavailable"
    assert audit["pretraining_overlap"]["pretraining_overlap_status"] == "unresolved"
    assert audit["pretraining_overlap"]["zero_shot_verified"] is False
    assert audit["scientific_gate_applied"] is False


class _PrefixProjectionDictionary:
    def project(self, trajectory: np.ndarray) -> np.ndarray:
        return np.asarray(trajectory[:, :32], dtype=np.float64)

    def predict_projected(
        self, projected: np.ndarray, *, segment_ids: np.ndarray | None = None
    ) -> np.ndarray:
        del segment_ids
        return np.arange(len(projected), dtype=np.int64) % 3


def _high_rank_unit(
    *, unit_id: str, base: np.ndarray, extra: np.ndarray, retained_rank: int
) -> LoadedUnit:
    if retained_rank <= 32 or retained_rank > 48:
        raise ValueError("test fixture rank must lie in (32, 48]")
    active_extra = retained_rank - 32
    raw = np.column_stack(
        [base, extra[:, :active_extra], np.zeros((len(base), 64 - retained_rank))]
    )
    regional = {
        "anterior": base[:, :4],
        "posterior": np.roll(base[:, 4:8], 1, axis=0),
    }
    return LoadedUnit(
        unit_id=unit_id,
        participant_id=unit_id,
        dataset_id="synthetic",
        trajectory=raw,
        regional=regional,
        segment_ids=None,
        alignment_segment_ids=None,
        metadata={},
    )


def test_repertoire_remains_untruncated_above_dynamics_rank_32() -> None:
    rng = np.random.default_rng(20260830)
    base = rng.normal(size=(600, 32))
    extra = rng.normal(size=(600, 16))
    dictionary = _PrefixProjectionDictionary()
    low = _record(
        _high_rank_unit(unit_id="rank-40", base=base, extra=extra, retained_rank=40),
        dictionary,  # type: ignore[arg-type]
    )
    high = _record(
        _high_rank_unit(unit_id="rank-48", base=base, extra=extra, retained_rank=48),
        dictionary,  # type: ignore[arg-type]
    )

    assert np.array_equal(low.trajectory, high.trajectory)
    assert np.asarray(low.trajectory).shape[1] == 32
    assert np.asarray(low.repertoire_trajectory).shape[1] == 64

    estimator = FiveAxisProfileEstimator(
        repertoire_shrinkage="none",
        alignment_lags=(1,),
        alignment_rank=2,
        alignment_cv=3,
        standardization="none",
    )
    low_details = estimator._estimate_details(low)
    high_details = estimator._estimate_details(high)

    assert low_details.repertoire.participation_ratio > 32
    assert high_details.repertoire.participation_ratio > (
        low_details.repertoire.participation_ratio + 3
    )
    assert low_details.repertoire.n_features == 64
    assert high_details.repertoire.n_features == 64
    assert low_details.local_dynamics.transition_matrices.shape[-1] == 32
    assert high_details.local_dynamics.transition_matrices.shape[-1] == 32

    rotated = _surrogate_record(
        high,
        dictionary,  # type: ignore[arg-type]
        family="post_encoder_latent_rotation_control",
        seed=17,
    )
    assert np.asarray(rotated.repertoire_trajectory).shape == (600, 64)
    assert np.asarray(rotated.trajectory).shape == (600, 32)
