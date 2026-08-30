from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from neural_manifolds import stage_units
from neural_manifolds.config import load_study
from neural_manifolds.manifold.profile import ManifoldRecord
from neural_manifolds.provenance import sha256_file
from neural_manifolds.stage_units import (
    _aggregate_event_rows,
    _segment_ids_from_retained_starts,
    encode_analysis_units,
)
from neural_manifolds.stages import metrics
from neural_manifolds.stages.metrics import _combine_records

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("track", "starts", "step", "expected"),
    [
        ("coarse", [0, 200, 400, 800, 1000], 200, [0, 0, 0, 1, 1]),
        ("alignment", [0, 4, 8, 16, 20], 4, [0, 0, 0, 1, 1]),
    ],
)
def test_middle_artifact_gap_starts_a_new_temporal_segment(
    track: str,
    starts: list[int],
    step: int,
    expected: list[int],
) -> None:
    del track
    observed = _segment_ids_from_retained_starts(np.asarray(starts), expected_step_samples=step)
    assert observed.tolist() == expected


def test_encoder_writes_gap_aware_ids_for_coarse_and_alignment_tracks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "unit-raw.fif"
    source.write_bytes(b"synthetic")
    preprocessing = tmp_path / "preprocessing.parquet"
    labels = tmp_path / "labels.parquet"
    study = load_study(ROOT / "configs" / "study.yaml")
    study = study.model_copy(
        update={
            "preprocessing": study.preprocessing.model_copy(update={"minimum_valid_windows": 2})
        }
    )
    selector = {"kind": "full_recording"}
    preprocessing_receipt = tmp_path / "unit-raw.provenance.json"
    stage_units._write_derivative_receipt(
        receipt=preprocessing_receipt,
        marker=stage_units.PREPROCESSED_UNIT_MARKER,
        inputs={
            "unit_id": "unit-1",
            **stage_units._preprocessing_input_fingerprints(study, selector),
        },
        output=source,
        metadata={"duration_seconds": 100.0},
    )
    pd.DataFrame(
        [
            {
                "unit_id": "unit-1",
                "preprocessed_path": str(source),
                "preprocessed_sha256": sha256_file(source),
                "preprocessing_provenance_path": str(preprocessing_receipt),
                "preprocessing_provenance_sha256": sha256_file(preprocessing_receipt),
                "selector_json": json.dumps(selector),
                "eligible": True,
            }
        ]
    ).to_parquet(preprocessing, index=False)
    pd.DataFrame(
        [
            {
                "unit_id": "unit-1",
                "participant_id": "sub-1",
                "dataset_id": "synthetic",
                "condition": "wake",
                "selector_json": json.dumps({"kind": "full_recording"}),
            }
        ]
    ).to_parquet(labels, index=False)

    class FakeRaw:
        def __init__(self) -> None:
            self.info = {"sfreq": 200.0}

    fake_mne = SimpleNamespace(io=SimpleNamespace(read_raw_fif=lambda *args, **kwargs: FakeRaw()))
    monkeypatch.setitem(sys.modules, "mne", fake_mne)
    monkeypatch.setattr(
        stage_units,
        "_model_environment",
        lambda: (tmp_path, tmp_path / "checkpoint.pth", "f" * 64),
    )
    monkeypatch.setattr(
        stage_units,
        "_model_fingerprint",
        lambda *args: {"model": "synthetic", "sha256": "f" * 64},
    )
    monkeypatch.setattr(stage_units, "OfficialLaBraMEncoder", lambda **kwargs: object())

    def fake_encode(
        raw: object,
        encoder: object,
        *,
        window_seconds: float,
        step_seconds: float,
    ) -> tuple[SimpleNamespace, np.ndarray, SimpleNamespace]:
        del raw, encoder, step_seconds
        if window_seconds == 2.0:
            starts = np.asarray([0, 200, 600, 800])
            keep = np.asarray([True, True, False, True, True])
        else:
            starts = np.asarray([0, 200, 400, 800, 1000])
            keep = np.asarray([True, True, True, False, True, True])
        states = np.arange(len(starts) * 3, dtype=np.float32).reshape(len(starts), 3)
        encoded = SimpleNamespace(
            global_states=states,
            regional_states={"anterior": states, "posterior": states + 1},
        )
        return encoded, starts, SimpleNamespace(keep=keep)

    monkeypatch.setattr(stage_units, "_encode_windows", fake_encode)
    manifest, flow_path = encode_analysis_units(
        preprocessing_manifest=preprocessing,
        labels_manifest=labels,
        output_root=tmp_path / "encoded",
        study=study,
    )
    row = pd.read_parquet(manifest).iloc[0]
    with np.load(row["trajectory_path"], allow_pickle=False) as archive:
        assert archive["segment_ids"].tolist() == [0, 0, 1, 1]
        assert archive["alignment_segment_ids"].tolist() == [0, 0, 0, 1, 1]
    assert row["coarse_segments"] == 2
    assert row["alignment_segments"] == 2
    assert row["pretraining_overlap_status"] == "unresolved"
    assert row["zero_shot_classification"] == "unresolved_not_verified_zero_shot"
    assert not bool(row["zero_shot_verified"])
    flow = json.loads(flow_path.read_text(encoding="utf-8"))
    assert flow["pretraining_overlap"]["pretraining_overlap_status"] == "unresolved"
    assert flow["pretraining_overlap"]["zero_shot_verified"] is False


def test_event_aggregation_preserves_middle_gap_and_trial_boundaries(tmp_path: Path) -> None:
    rows = []
    for trial, segment_ids in enumerate(([0, 0, 1, 1], [0, 0, 0, 1])):
        trajectory = np.arange(12, dtype=np.float32).reshape(4, 3) + trial
        path = tmp_path / f"trial-{trial}.npz"
        with path.open("wb") as stream:
            np.savez_compressed(
                stream,
                alignment_global_states=trajectory,
                alignment_regional_anterior=trajectory,
                alignment_regional_posterior=trajectory + 1,
                alignment_segment_ids=np.asarray(segment_ids, dtype=np.int32),
            )
        receipt = tmp_path / f"trial-{trial}.encoding.json"
        stage_units._write_derivative_receipt(
            receipt=receipt,
            marker=stage_units.ENCODED_UNIT_MARKER,
            inputs={"unit_id": f"trial-{trial}"},
            output=path,
            metadata={},
        )
        rows.append(
            {
                "unit_id": f"trial-{trial}",
                "participant_id": "sub-1",
                "dataset_id": "synthetic",
                "condition": "event",
                "selector_json": json.dumps({"kind": "event_epoch"}),
                "encoded": True,
                "trajectory_path": str(path),
                "trajectory_sha256": sha256_file(path),
                "encoding_provenance_path": str(receipt),
                "encoding_provenance_sha256": sha256_file(receipt),
            }
        )
    aggregated, issues = _aggregate_event_rows(
        pd.DataFrame(rows), destination=tmp_path, minimum_trials=2
    )
    assert issues == []
    assert len(aggregated) == 1
    with np.load(aggregated.iloc[0]["trajectory_path"], allow_pickle=False) as archive:
        expected = [0, 0, 1, 1, 2, 2, 2, 3]
        assert archive["segment_ids"].tolist() == expected
        assert archive["alignment_segment_ids"].tolist() == expected
    assert aggregated.iloc[0]["temporal_segments"] == 4


def test_alignment_lag_plan_refuses_overlapping_or_subwindow_embeddings() -> None:
    study = load_study(ROOT / "configs" / "study.yaml")
    assert metrics._alignment_lag_indices(study) == tuple(range(1, 11))

    overlapping = study.model_copy(
        update={
            "representation": study.representation.model_copy(
                update={"alignment_step_seconds": 0.02}
            )
        }
    )
    with pytest.raises(ValueError, match="non-overlapping LaBraM windows"):
        metrics._alignment_lag_indices(overlapping)

    subwindow = study.model_copy(
        update={
            "metrics": {
                **study.metrics,
                "alignment": {**study.metrics["alignment"], "lags_ms": [200]},
            }
        }
    )
    with pytest.raises(ValueError, match="sub-window alignment lags are unavailable"):
        metrics._alignment_lag_indices(subwindow)


def test_combining_reference_records_preserves_every_artifact_gap() -> None:
    first = ManifoldRecord(
        trajectory=np.zeros((4, 2)),
        states=np.arange(4),
        regional_trajectories={
            "anterior": np.zeros((5, 2)),
            "posterior": np.ones((5, 2)),
        },
        repertoire_trajectory=np.zeros((4, 4)),
        segment_ids=np.asarray([0, 0, 1, 1]),
        alignment_segment_ids=np.asarray([0, 0, 0, 1, 1]),
    )
    second = ManifoldRecord(
        trajectory=np.zeros((3, 2)),
        states=np.arange(3),
        regional_trajectories={
            "anterior": np.zeros((4, 2)),
            "posterior": np.ones((4, 2)),
        },
        repertoire_trajectory=np.ones((3, 4)),
        segment_ids=None,
        alignment_segment_ids=np.asarray([7, 7, 8, 8]),
    )

    combined = _combine_records([first, second], name="participant")

    assert np.asarray(combined.trajectory).shape == (7, 2)
    assert np.asarray(combined.repertoire_trajectory).shape == (7, 4)
    assert np.asarray(combined.segment_ids).tolist() == [0, 0, 1, 1, 2, 2, 2]
    assert np.asarray(combined.alignment_segment_ids).tolist() == [
        0,
        0,
        0,
        1,
        1,
        2,
        2,
        3,
        3,
    ]
