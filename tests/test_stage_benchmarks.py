from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from neural_manifolds.config import load_study
from neural_manifolds.manifold.profile import AXIS_NAMES
from neural_manifolds.stages import benchmarks as benchmark_stage
from neural_manifolds.stages.benchmarks import CONVENTIONAL_FEATURES, run_benchmarks
from neural_manifolds.stages.models import run_models


class _FakeRaw:
    def __init__(self, data: np.ndarray, sfreq: float) -> None:
        self._data = data
        self.info = {"sfreq": sfreq}
        self.closed = False

    def get_data(self) -> np.ndarray:
        return self._data

    def close(self) -> None:
        self.closed = True


def _signal(seed: int, *, sfreq: float = 200.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    times = np.arange(1200) / sfreq
    return np.stack(
        [
            np.sin(2 * np.pi * (8 + channel) * times + channel / 5)
            + rng.normal(scale=0.15, size=len(times))
            for channel in range(4)
        ]
    )


def test_benchmark_stage_publishes_only_scalars_and_explicit_unavailable_status(
    tmp_path: Path, monkeypatch: object
) -> None:
    paths = []
    for name in ("rest", "event", "clinical"):
        path = tmp_path / f"{name}-raw.fif"
        path.write_bytes(name.encode())
        paths.append(path)
    rows = [
        {
            "unit_id": "rest",
            "participant_id": "synthetic:sub-01",
            "dataset_id": "synthetic",
            "condition": "wake",
            "encoded": True,
            "preprocessed_path": str(paths[0]),
            "selector_json": json.dumps({"kind": "full_recording"}),
            "clinical_holdout": False,
            "secondary_fmri": False,
            "modality": "eeg",
        },
        {
            "unit_id": "event",
            "participant_id": "synthetic:sub-01",
            "dataset_id": "synthetic",
            "condition": "hit",
            "encoded": True,
            "preprocessed_path": str(paths[1]),
            "selector_json": json.dumps(
                {
                    "kind": "event_epoch",
                    "event_onset_seconds": 1.0,
                    "epoch_start_offset_seconds": -0.2,
                    "epoch_stop_offset_seconds": 0.8,
                }
            ),
            "clinical_holdout": False,
            "secondary_fmri": False,
            "modality": "eeg",
        },
        {
            "unit_id": "clinical",
            "participant_id": "synthetic:sub-02",
            "dataset_id": "synthetic",
            "condition": "mcs",
            "encoded": True,
            "preprocessed_path": str(paths[2]),
            "selector_json": json.dumps({"kind": "full_recording"}),
            "clinical_holdout": True,
            "secondary_fmri": False,
            "modality": "eeg",
        },
    ]
    manifest = tmp_path / "encoding.parquet"
    pd.DataFrame(rows).to_parquet(manifest, index=False)
    seen: list[Path] = []

    def fake_read(path: Path) -> _FakeRaw:
        seen.append(path)
        return _FakeRaw(_signal(10), 200.0)

    monkeypatch.setattr(benchmark_stage, "_read_raw_fif", fake_read)  # type: ignore[attr-defined]
    benchmark_path, audit_path = run_benchmarks(
        encoding_manifest=manifest,
        output_root=tmp_path / "metrics",
    )

    output = pd.read_parquet(benchmark_path)
    assert output["unit_id"].tolist() == ["rest"]
    assert output[list(CONVENTIONAL_FEATURES)].apply(np.isfinite).all().all()
    assert output.loc[0, "wsmi_status"] == "unavailable_no_validated_backend"
    assert output.loc[0, "microstates_status"] == "unavailable_no_validated_backend"
    assert output.loc[0, "pcist_status"] == "unavailable_no_validated_backend"
    assert output[["wsmi", "microstates", "pcist"]].isna().all().all()
    assert not {"preprocessed_path", "source_path", "trajectory_path"}.intersection(output)
    assert seen == [paths[0].resolve()]

    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert audit["completed_rows"] == 1
    assert audit["skipped_rows"]["event_unit"] == 1
    assert audit["skipped_rows"]["clinical_holdout"] == 1
    assert audit["raw_or_array_artifacts_published"] is False
    assert audit["scientific_gate_applied"] is False


def test_models_add_participant_held_out_conventional_baseline(tmp_path: Path) -> None:
    rng = np.random.default_rng(21)
    profile_rows = []
    benchmark_rows = []
    for participant in range(12):
        for label, condition in enumerate(("low", "high")):
            profile_rows.append(
                {
                    "unit_id": f"unit-{participant}-{condition}",
                    "participant_id": f"synthetic:sub-{participant:02d}",
                    "dataset_id": "synthetic",
                    "condition": condition,
                    "prediction_evaluation_eligible": participant >= 3,
                    **dict(zip(AXIS_NAMES, rng.normal(size=5) + label * 0.4, strict=True)),
                }
            )
            conventional = rng.normal(size=len(CONVENTIONAL_FEATURES)) + label * 0.6
            benchmark_rows.append(
                {
                    "unit_id": f"unit-{participant}-{condition}",
                    "participant_id": f"synthetic:sub-{participant:02d}",
                    "dataset_id": "synthetic",
                    "condition": condition,
                    "benchmark_status": "computed",
                    **dict(zip(CONVENTIONAL_FEATURES, conventional, strict=True)),
                }
            )
    profiles = tmp_path / "profiles.parquet"
    benchmarks = tmp_path / "benchmarks.parquet"
    pd.DataFrame(profile_rows).to_parquet(profiles, index=False)
    pd.DataFrame(benchmark_rows).to_parquet(benchmarks, index=False)
    contrasts = tmp_path / "contrasts.yaml"
    contrasts.write_text(
        """schema_version: 1
contrasts:
  - id: high-versus-low
    datasets: [synthetic]
    label_column: condition
    positive: high
    negative: low
    design: within_participant
""",
        encoding="utf-8",
    )

    outputs = run_models(
        profiles_path=profiles,
        benchmarks_path=benchmarks,
        contrasts_path=contrasts,
        output_root=tmp_path / "models",
        study=load_study(Path(__file__).parents[1] / "configs" / "study.yaml"),
        repetitions=99,
    )
    predictions = pd.read_parquet(outputs[2])
    conventional = predictions[predictions["model"] == "conventional_scalar"]
    assert len(conventional) == 1
    assert conventional.iloc[0]["n_participants"] == 9
    assert conventional.iloc[0]["n_fixed_transform_participants"] == 3
    assert set(conventional.iloc[0]["features"]) == set(CONVENTIONAL_FEATURES)
    assert 0 <= conventional.iloc[0]["auroc"] <= 1

    audit = json.loads(outputs[-1].read_text(encoding="utf-8"))
    assert audit["benchmarks_sha256"]
    assert audit["conventional_prediction_completed"] == 1
