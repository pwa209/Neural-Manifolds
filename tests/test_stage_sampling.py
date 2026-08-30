from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml

from neural_manifolds.config import SamplingConfig, load_study
from neural_manifolds.dynamics.state_dictionary import fit_state_dictionary
from neural_manifolds.manifold.profile import (
    FiveAxisProfileEstimator,
    ManifoldProfile,
    ManifoldRecord,
)
from neural_manifolds.stages.sampling import (
    _duration_template,
    _track_runs,
    run_sampling_sensitivity,
)


class BoundaryCheckingEstimator(FiveAxisProfileEstimator):
    """Small frozen test double that asserts the stage's boundary contract."""

    def profile(self, record: ManifoldRecord | dict[str, Any]) -> ManifoldProfile:
        assert isinstance(record, ManifoldRecord)
        trajectory = np.asarray(record.trajectory, dtype=float)
        repertoire = np.asarray(record.repertoire_trajectory, dtype=float)
        segments = np.asarray(record.segment_ids)
        fine_segments = np.asarray(record.alignment_segment_ids)
        assert len(trajectory) == len(segments)
        assert len(repertoire) == len(segments)
        assert trajectory.shape[1] == 2
        assert repertoire.shape[1] == 3
        assert np.array_equal(np.unique(segments), np.arange(len(np.unique(segments))))
        assert np.all(np.diff(segments) >= 0)
        assert np.array_equal(np.unique(fine_segments), np.arange(len(np.unique(fine_segments))))
        assert np.all(np.diff(fine_segments) >= 0)
        base = float(np.mean(trajectory[:, 0]))
        spread = float(np.std(trajectory[:, 0]))
        axes = np.asarray(
            [
                base,
                base + spread,
                base + 0.1 * len(np.unique(segments)),
                base + 0.01 * len(fine_segments),
                base + float(np.mean(trajectory[:, -1])),
            ]
        )
        return ManifoldProfile(
            values=axes,
            raw_values=axes,
            details=None,  # type: ignore[arg-type]
            standardized=False,
            name=record.name,
        )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _event_archive(path: Path, *, participant: int, condition: str) -> np.ndarray:
    trials: list[np.ndarray] = []
    regions_a: list[np.ndarray] = []
    regions_b: list[np.ndarray] = []
    segment_ids: list[np.ndarray] = []
    condition_offset = 0.35 if condition == "condition_b" else 0.0
    for trial in range(3):
        time = np.arange(40, dtype=float)
        base = participant * 3.0 + condition_offset + trial * 0.4
        trajectory = np.column_stack([base + time * 0.01, base * 0.5 + np.sin(time / 5), time / 40])
        trials.append(trajectory)
        regions_a.append(trajectory[:, :2] + 0.05)
        regions_b.append(trajectory[:, 1:] - 0.05)
        segment_ids.append(np.full(len(time), trial, dtype=np.int64))
    states = np.concatenate(trials)
    segments = np.concatenate(segment_ids)
    np.savez_compressed(
        path,
        global_states=states,
        alignment_regional_frontal=np.concatenate(regions_a),
        alignment_regional_posterior=np.concatenate(regions_b),
        segment_ids=segments,
        alignment_segment_ids=segments,
    )
    return states


def _build_inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path, Any]:
    study = load_study(Path("configs/study.yaml"))
    study = study.model_copy(
        update={
            "sampling": SamplingConfig(
                equalise_windows=True,
                repeats=3,
                reliability_seconds=[3, 4, 199],
            )
        }
    )
    rows: list[dict[str, Any]] = []
    discovery: list[np.ndarray] = []
    for participant in (1, 2):
        for condition in ("condition_a", "condition_b"):
            unit_id = f"p{participant}-{condition}"
            archive = tmp_path / f"{unit_id}.npz"
            discovery.append(_event_archive(archive, participant=participant, condition=condition))
            rows.append(
                {
                    "unit_id": unit_id,
                    "participant_id": f"p{participant}",
                    "dataset_id": "synthetic",
                    "condition": condition,
                    "encoded": True,
                    "event_aggregated": True,
                    "selector_json": json.dumps({"kind": "event_epoch"}),
                    "trajectory_path": str(archive),
                    "trajectory_sha256": _sha256(archive),
                }
            )
    # An unmatched participant is retained as an audit row, not a stage failure.
    unmatched = tmp_path / "p3-condition_a.npz"
    discovery.append(_event_archive(unmatched, participant=3, condition="condition_a"))
    rows.append(
        {
            "unit_id": "p3-condition_a",
            "participant_id": "p3",
            "dataset_id": "synthetic",
            "condition": "condition_a",
            "encoded": True,
            "event_aggregated": True,
            "selector_json": json.dumps({"kind": "event_epoch"}),
            "trajectory_path": str(unmatched),
            "trajectory_sha256": _sha256(unmatched),
        }
    )
    manifest = tmp_path / "encoding.parquet"
    pd.DataFrame(rows).to_parquet(manifest, index=False)
    dictionary = fit_state_dictionary(
        discovery,
        discovery[:2],
        rank=2,
        state_counts=[2],
        seeds=[17],
        minimum_stability_ami=0.7,
        force_fallback=True,
    )
    dictionary_path = tmp_path / "dictionary.joblib"
    estimator_path = tmp_path / "estimator.joblib"
    joblib.dump(dictionary, dictionary_path)
    joblib.dump(BoundaryCheckingEstimator(), estimator_path)
    contrasts = tmp_path / "contrasts.yaml"
    contrasts.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "datasets": {
                    "synthetic": {
                        "contrasts": [
                            {
                                "id": "a_vs_b",
                                "positive": ["condition_a"],
                                "reference": ["condition_b"],
                                "match_within": "participant_id",
                            }
                        ]
                    }
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return manifest, dictionary_path, estimator_path, contrasts, study


def test_sampling_is_deterministic_matched_and_boundary_safe(tmp_path: Path) -> None:
    manifest, dictionary, estimator, contrasts, study = _build_inputs(tmp_path)
    first = run_sampling_sensitivity(
        encoding_manifest=manifest,
        state_dictionary_path=dictionary,
        profile_estimator_path=estimator,
        contrasts_path=contrasts,
        output_root=tmp_path / "first",
        study=study,
    )
    second = run_sampling_sensitivity(
        encoding_manifest=manifest,
        state_dictionary_path=dictionary,
        profile_estimator_path=estimator,
        contrasts_path=contrasts,
        output_root=tmp_path / "second",
        study=study,
    )

    repeats = pd.read_parquet(first[0]).sort_values(
        ["analysis", "duration_seconds", "profile_id", "repeat"], na_position="first"
    )
    repeated = pd.read_parquet(second[0]).sort_values(
        ["analysis", "duration_seconds", "profile_id", "repeat"], na_position="first"
    )
    pd.testing.assert_frame_equal(repeats.reset_index(drop=True), repeated.reset_index(drop=True))
    equal = repeats[repeats["analysis"] == "equal_window"]
    assert len(equal) == 4 * 3
    assert set(equal["condition_levels"]) == {"condition_a", "condition_b"}
    assert equal["matched_coarse_windows"].nunique() == 1
    assert equal["matched_fine_windows"].nunique() == 1
    assert set(equal["matched_segments"]) == {3}
    assert set(equal["pretraining_overlap_status"]) == {"unresolved"}
    assert not equal["zero_shot_verified"].any()

    averaged = pd.read_parquet(first[1])
    assert set(averaged["pretraining_overlap_status"]) == {"unresolved"}
    assert not averaged["zero_shot_verified"].any()

    reliability = repeats[repeats["analysis"] == "reliability"]
    assert set(reliability["duration_seconds"]) == {3.0, 4.0}
    assert np.all(reliability["matched_effective_seconds"] >= reliability["duration_seconds"])
    curves = pd.read_parquet(first[2])
    assert len(curves) == 2 * 2 * 5
    assert set(curves["status"]).issubset({"available", "unavailable"})
    assert set(curves["pretraining_overlap_status"]) == {"unresolved"}
    assert not curves["zero_shot_verified"].any()

    audit = json.loads(first[3].read_text(encoding="utf-8"))
    assert audit["state_dictionary_refit"] is False
    assert audit["profile_estimator_refit"] is False
    assert audit["profile_input_spaces"] == {
        "repertoire": {
            "record_field": "repertoire_trajectory",
            "space": "untruncated_frozen_encoder_embedding",
            "dimension": 3,
            "discovery_projection_applied": False,
        },
        "dynamics": {
            "record_field": "trajectory",
            "space": "discovery_fitted_pca_projection",
            "dimension": 2,
            "discovery_projection_applied": True,
        },
    }
    assert audit["scientific_gate_applied"] is False
    assert audit["pretraining_overlap"]["pretraining_overlap_status"] == "unresolved"
    assert audit["pretraining_overlap"]["zero_shot_verified"] is False
    assert any("every contrast arm" in row.get("reason", "") for row in audit["rows"])
    assert any(
        row.get("duration_seconds") == 199.0 and row["status"] == "unavailable"
        for row in audit["rows"]
    )


def test_duration_and_track_helpers_respect_gaps_and_insufficient_data() -> None:
    starts = np.asarray([0.0, 0.02, 0.04, 0.0, 0.02, 1.0, 1.02])
    segments = np.asarray([0, 0, 0, 1, 1, 1, 1])
    assert _track_runs(segments, starts, expected_step=0.02) == [
        ("0", 0, 3),
        ("1", 3, 5),
        ("1", 5, 7),
    ]
    assert _duration_template(
        (40, 40, 40), duration_seconds=3, window_seconds=1, step_seconds=0.02
    ) == (40, 12)
    assert not _duration_template(
        (40, 40, 40), duration_seconds=99, window_seconds=1, step_seconds=0.02
    )
