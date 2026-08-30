from __future__ import annotations

import hashlib
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
    def __init__(
        self,
        data: np.ndarray,
        sfreq: float,
        ch_names: tuple[str, ...] = ("F3", "C3", "P3", "O1"),
    ) -> None:
        self._data = data
        self.info = {"sfreq": sfreq}
        self.ch_names = list(ch_names)
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


def _expected_cell_key(dataset: str, participant: str, condition: str) -> str:
    payload = json.dumps(
        [dataset, participant, condition],
        ensure_ascii=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()


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
    assert output["unit_id"].tolist() == ["rest", "event", "clinical"]
    output = output.set_index("unit_id")
    assert output.loc["rest", list(CONVENTIONAL_FEATURES)].apply(np.isfinite).all()
    assert output.loc["rest", "legacy_conventional_status"] == "available"
    assert output.loc["rest", "benchmark_status"] == "computed"
    assert output.loc["rest", "wsmi_status"] == "available_validated_deterministic"
    assert np.isfinite(output.loc["rest", "wsmi"])
    assert output.loc["rest", "wsmi_order"] == 3
    assert output.loc["rest", "wsmi_lag_samples"] == 6
    assert output.loc["rest", "wsmi_lowpass_hz"] == 10.0
    assert output.loc["rest", "wsmi_minimum_symbol_samples"] == 180
    assert output.loc["rest", "microstates_status"] == (
        "unavailable_no_representation_partition_in_encoding_manifest"
    )

    assert output.loc["event", "benchmark_status"] == "unavailable"
    assert output.loc["event", "legacy_conventional_status"] == "unavailable"
    assert output.loc["event", list(CONVENTIONAL_FEATURES)].isna().all()
    assert output.loc["event", "wsmi_status"] == "unavailable_signal_not_computed"
    assert output.loc["event", "wsmi_reason"] == (
        "event_unit_has_no_valid_four_second_continuous_benchmark_signal"
    )
    assert output.loc["clinical", "benchmark_status"] == "not_applicable"
    assert output.loc["clinical", "legacy_conventional_status"] == "not_applicable"
    assert output.loc["clinical", list(CONVENTIONAL_FEATURES)].isna().all()
    assert output.loc["clinical", "benchmark_reason"] == (
        "clinical_holdout_excluded_from_healthy_benchmark_fit"
    )
    assert output["pcist_status"].eq("unavailable_no_validated_backend").all()
    assert output[["microstates", "pcist"]].isna().all().all()
    assert output.loc["rest", "participant_condition_cell_status"] == (
        "complete_all_expected_units"
    )
    assert output.loc["event", "participant_condition_cell_status"] == ("incomplete_expected_units")
    assert output.loc["clinical", "participant_condition_cell_status"] == "not_applicable"
    rest_key = _expected_cell_key("synthetic", "synthetic:sub-01", "wake")
    event_key = _expected_cell_key("synthetic", "synthetic:sub-01", "hit")
    clinical_key = _expected_cell_key("synthetic", "synthetic:sub-02", "mcs")
    assert output.loc["rest", "participant_condition_cell_key_sha256"] == rest_key
    assert output.loc["event", "participant_condition_cell_key_sha256"] == event_key
    assert output.loc["clinical", "participant_condition_cell_key_sha256"] == clinical_key
    assert not {"preprocessed_path", "source_path", "trajectory_path"}.intersection(output)
    assert seen == [paths[0].resolve()]

    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert audit["published_rows"] == 3
    assert audit["completed_rows"] == 1
    assert audit["unavailable_rows"] == 1
    assert audit["not_applicable_rows"] == 1
    assert audit["skipped_rows"]["event_unit"] == 1
    assert audit["skipped_rows"]["clinical_holdout"] == 1
    assert audit["wsmi"]["available_rows"] == 1
    assert audit["wsmi"]["minimum_symbol_samples"] == 180
    assert audit["microstates"]["status"] == "unavailable"
    assert audit["participant_condition_cell_contract"]["expected_cells"] == 2
    assert audit["participant_condition_cell_contract"]["complete_cells"] == 1
    assert audit["participant_condition_cell_contract"]["incomplete_cells"] == 1
    assert audit["participant_condition_cell_contract"]["complete_cell_keys_sha256"] == [rest_key]
    assert audit["participant_condition_cell_contract"]["incomplete_cell_keys_sha256"] == [
        event_key
    ]
    assert audit["participant_condition_cell_contract"]["conventional_prediction_status"] == (
        "unavailable_requires_consumer_fail_closed_on_incomplete_cells"
    )
    assert audit["raw_or_array_artifacts_published"] is False
    assert audit["scientific_gate_applied"] is False


def test_microstates_fit_discovery_participants_only_then_apply_frozen(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    specifications = [
        ("discovery-a", "representation_discovery", "awake", 1),
        ("discovery-b", "representation_discovery", "propofol", 2),
        ("discovery-c", "representation_discovery", "awake", 3),
        ("validation-a", "representation_validation", "propofol", 4),
        ("evaluation-a", "representation_evaluation", "awake", 5),
    ]
    signals: dict[Path, np.ndarray] = {}
    rows: list[dict[str, object]] = []
    for participant, partition, condition, seed in specifications:
        path = (tmp_path / f"{participant}-raw.fif").resolve()
        path.write_bytes(participant.encode())
        signals[path] = _signal(seed)
        rows.append(
            {
                "unit_id": f"unit-{participant}",
                "participant_id": f"synthetic:{participant}",
                "dataset_id": "synthetic",
                "condition": condition,
                "encoded": True,
                "preprocessed_path": str(path),
                "selector_json": json.dumps({"kind": "full_recording"}),
                "clinical_holdout": False,
                "secondary_fmri": False,
                "modality": "eeg",
                "representation_partition": partition,
            }
        )
    manifest = tmp_path / "encoding-with-partitions.parquet"
    pd.DataFrame(rows).to_parquet(manifest, index=False)
    opened: list[Path] = []

    def fake_read(path: Path) -> _FakeRaw:
        resolved = path.resolve()
        opened.append(resolved)
        return _FakeRaw(signals[resolved].copy(), 200.0)

    fitted_participants: set[str] = set()
    real_model = benchmark_stage.FrozenMicrostateModel

    class _FitSpy(real_model):
        def fit(self, participant_maps: object) -> object:
            fitted_participants.update(participant_maps)  # type: ignore[arg-type]
            return super().fit(participant_maps)  # type: ignore[arg-type]

    monkeypatch.setattr(benchmark_stage, "_read_raw_fif", fake_read)  # type: ignore[attr-defined]
    monkeypatch.setattr(benchmark_stage, "FrozenMicrostateModel", _FitSpy)  # type: ignore[attr-defined]
    benchmark_path, audit_path = run_benchmarks(
        encoding_manifest=manifest,
        output_root=tmp_path / "metrics-with-microstates",
    )

    output = pd.read_parquet(benchmark_path).set_index("participant_id")
    assert fitted_participants == {
        "synthetic:discovery-a",
        "synthetic:discovery-b",
        "synthetic:discovery-c",
    }
    discovery = output.loc[
        ["synthetic:discovery-a", "synthetic:discovery-b", "synthetic:discovery-c"]
    ]
    held_out = output.loc[["synthetic:validation-a", "synthetic:evaluation-a"]]
    assert discovery["microstates_status"].eq("available_frozen_discovery_in_sample").all()
    assert held_out["microstates_status"].eq("available_frozen_out_of_sample").all()
    assert (
        output[
            [
                "microstate_transition_entropy",
                "microstate_global_explained_variance",
                "microstate_median_duration_seconds",
            ]
        ]
        .apply(np.isfinite)
        .all()
        .all()
    )
    assert len(opened) == 2 * len(specifications)

    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert audit["microstates"]["status"] == "frozen"
    assert audit["microstates"]["discovery_participants"] == 3
    assert audit["microstates"]["fit_partition"] == "representation_discovery"
    assert audit["microstates"]["fit_label_fields"] == []
    assert audit["microstates"]["participant_partition_overlap"] is False
    assert audit["microstates"]["per_condition_clustering_performed"] is False
    assert audit["microstates"]["refit_after_discovery"] is False
    assert audit["participant_condition_cell_contract"]["complete_cells"] == 5
    assert audit["participant_condition_cell_contract"]["incomplete_cells"] == 0
    expected_keys = sorted(
        _expected_cell_key("synthetic", f"synthetic:{participant}", condition)
        for participant, _, condition, _ in specifications
    )
    assert audit["participant_condition_cell_contract"]["complete_cell_keys_sha256"] == (
        expected_keys
    )


def test_microstates_fail_closed_when_participant_crosses_partitions() -> None:
    rows = [
        {
            "benchmark_status": "computed",
            "microstates_status": "pending_frozen_discovery_branch",
        },
        {
            "benchmark_status": "computed",
            "microstates_status": "pending_frozen_discovery_branch",
        },
    ]
    candidates = [
        {
            "row_index": 0,
            "participant_id": "synthetic:sub-01",
            "partition": "representation_discovery",
        },
        {
            "row_index": 1,
            "participant_id": "synthetic:sub-01",
            "partition": "representation_evaluation",
        },
    ]
    audit = benchmark_stage._run_microstate_branch(  # type: ignore[attr-defined]
        rows,
        candidates,
        partition_column_present=True,
    )
    assert audit["status"] == "unavailable"
    assert audit["reason"] == "unavailable_participant_crosses_representation_partitions"
    assert all(row["microstates_status"] == audit["reason"] for row in rows)


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
    assert conventional.iloc[0]["estimand_role"] == "secondary_sensitivity"
    assert conventional.iloc[0]["estimand_id"] == (
        "all_available_participant_condition_sensitivity"
    )
    assert conventional.iloc[0]["sampling_basis"] == (
        "all_available_participant_condition_profiles"
    )
    assert conventional.iloc[0]["pretraining_overlap_status"] == "unresolved"
    assert not bool(conventional.iloc[0]["zero_shot_verified"])
    assert conventional.iloc[0]["baseline_match_status"] == (
        "available_exact_participant_condition_cells"
    )
    assert bool(conventional.iloc[0]["baseline_cell_keys_verified_identical"])
    assert conventional.iloc[0]["baseline_matched_cell_count"] == 24
    five_axis = predictions[
        (predictions["model"] == "five_axis")
        & (predictions["estimand_role"] == "secondary_sensitivity")
    ].iloc[0]
    assert conventional.iloc[0]["baseline_match_id"] == five_axis["baseline_match_id"]
    assert conventional.iloc[0]["baseline_cell_key_sha256"] == five_axis["baseline_cell_key_sha256"]
    assert len(conventional.iloc[0]["baseline_cell_key_sha256"]) == 64
    assert conventional.iloc[0]["n_observations"] == five_axis["n_observations"]
    assert (
        conventional.iloc[0]["n_evaluation_participants"] == five_axis["n_evaluation_participants"]
    )
    assert conventional.iloc[0]["prediction_equivalence_status"] in {
        "equivalent",
        "not_equivalent",
    }
    assert conventional.iloc[0]["prediction_equivalence_smallest_auc_difference"] == 0.05
    assert np.isfinite(conventional.iloc[0]["prediction_equivalence_ci_low"])
    assert np.isfinite(conventional.iloc[0]["prediction_equivalence_ci_high"])

    audit = json.loads(outputs[-1].read_text(encoding="utf-8"))
    assert audit["benchmarks_sha256"]
    assert audit["conventional_prediction_completed"] == 1
    assert audit["conventional_baseline_exact_cell_match_required"] is True


def test_conventional_baseline_fails_closed_when_one_profile_cell_is_missing(
    tmp_path: Path,
) -> None:
    rng = np.random.default_rng(22)
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
            if not (participant == 11 and condition == "high"):
                benchmark_rows.append(
                    {
                        "unit_id": f"unit-{participant}-{condition}",
                        "participant_id": f"synthetic:sub-{participant:02d}",
                        "dataset_id": "synthetic",
                        "condition": condition,
                        "benchmark_status": "computed",
                        **dict(
                            zip(
                                CONVENTIONAL_FEATURES,
                                rng.normal(size=len(CONVENTIONAL_FEATURES)) + label * 0.5,
                                strict=True,
                            )
                        ),
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
    assert "five_axis" in set(predictions["model"])
    assert "conventional_scalar" not in set(predictions["model"])
    audit = json.loads(outputs[-1].read_text(encoding="utf-8"))
    unavailable = [
        issue for issue in audit["issues"] if issue["component"] == "conventional_scalar_prediction"
    ]
    assert len(unavailable) == 1
    assert unavailable[0]["status"] == "unavailable"
    assert "missing=1" in unavailable[0]["error"]
    assert audit["conventional_prediction_completed"] == 0
