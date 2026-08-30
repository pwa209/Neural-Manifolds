from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from neural_manifolds import phase_runner
from neural_manifolds.config import StudyConfig, config_sha256, load_study
from neural_manifolds.manifold.profile import AXIS_NAMES
from neural_manifolds.provenance import sha256_file
from neural_manifolds.stages.benchmarks import CONVENTIONAL_FEATURES
from neural_manifolds.stages.representation_controls import run_representation_controls


def _cell_key(dataset_id: str, participant_id: str, condition: str) -> str:
    payload = json.dumps(
        [dataset_id, participant_id, condition],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _study() -> StudyConfig:
    study = load_study(Path("configs/study.yaml"))
    return study.model_copy(
        update={
            "statistics": study.statistics.model_copy(
                update={
                    "participant_stratified_folds": 3,
                    "participant_bootstrap_repetitions": 12,
                    "permutation_repetitions": 12,
                }
            )
        }
    )


def _inputs(
    tmp_path: Path,
    *,
    datasets: int = 3,
    participants_per_dataset: int = 5,
    complete_benchmarks: bool = True,
) -> dict[str, Path]:
    rng = np.random.default_rng(947)
    profile_rows: list[dict[str, object]] = []
    benchmark_rows: list[dict[str, object]] = []
    for dataset_index in range(datasets):
        dataset_id = f"dataset-{dataset_index}"
        for participant_index in range(participants_per_dataset):
            participant_id = f"participant-{participant_index}"
            participant_shift = rng.normal(scale=0.08, size=len(AXIS_NAMES))
            for condition_index, condition in enumerate(("wake", "altered")):
                axis_values = (
                    dataset_index * 1.5
                    + participant_shift
                    + condition_index * 0.03
                    + rng.normal(scale=0.02, size=len(AXIS_NAMES))
                )
                profile_rows.append(
                    {
                        "dataset_id": dataset_id,
                        "participant_id": participant_id,
                        "condition": condition,
                        **dict(zip(AXIS_NAMES, axis_values, strict=True)),
                    }
                )
                conventional = dataset_index * 0.9 + rng.normal(
                    scale=0.04, size=len(CONVENTIONAL_FEATURES)
                )
                expected = complete_benchmarks or not (
                    dataset_index == datasets - 1
                    and participant_index == participants_per_dataset - 1
                    and condition == "altered"
                )
                benchmark_rows.append(
                    {
                        "dataset_id": dataset_id,
                        "participant_id": participant_id,
                        "condition": condition,
                        "benchmark_status": "computed" if expected else "unavailable",
                        "benchmark_cell_expected": True,
                        "participant_condition_cell_key_sha256": _cell_key(
                            dataset_id, participant_id, condition
                        ),
                        **{
                            feature: float(value) if expected else np.nan
                            for feature, value in zip(
                                CONVENTIONAL_FEATURES, conventional, strict=True
                            )
                        },
                    }
                )
    profiles = tmp_path / "profiles.parquet"
    benchmarks = tmp_path / "benchmarks.parquet"
    pd.DataFrame(profile_rows).to_parquet(profiles, index=False)
    pd.DataFrame(benchmark_rows).to_parquet(benchmarks, index=False)

    trajectory = tmp_path / "trajectory.npz"
    trajectory.write_bytes(b"hash-bound-coordinate-trajectory")
    trajectory_sha256 = sha256_file(trajectory)
    study = _study()
    representation_sha256 = hashlib.sha256(
        json.dumps(
            study.representation.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode()
    ).hexdigest()
    model_fingerprint = {"checkpoint_sha256": "a" * 64}
    model_fingerprint_sha256 = hashlib.sha256(
        json.dumps(
            model_fingerprint,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode()
    ).hexdigest()
    encoding_receipt = tmp_path / "trajectory.encoding.json"
    encoding_receipt.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "provenance_marker": "neural_manifolds.encoded_unit.v1",
                "inputs": {
                    "representation_config_sha256": representation_sha256,
                    "study_config_sha256": config_sha256(study),
                    "model": model_fingerprint,
                },
                "output": {
                    "path": str(trajectory),
                    "sha256": trajectory_sha256,
                    "size": trajectory.stat().st_size,
                },
                "metadata": {},
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    encoding_manifest = tmp_path / "encoding-manifest.parquet"
    pd.DataFrame(
        [
            {
                "encoded": True,
                "trajectory_path": str(trajectory),
                "trajectory_sha256": trajectory_sha256,
                "encoding_provenance_path": str(encoding_receipt),
                "encoding_provenance_sha256": sha256_file(encoding_receipt),
            }
        ]
    ).to_parquet(encoding_manifest, index=False)
    encoding_flow = tmp_path / "encoding-flow.json"
    encoding_flow.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "checkpoint_sha256": "a" * 64,
                "model_fingerprint_sha256": model_fingerprint_sha256,
            }
        ),
        encoding="utf-8",
    )
    return {
        "profiles": profiles,
        "benchmarks": benchmarks,
        "encoding_manifest": encoding_manifest,
        "encoding_flow": encoding_flow,
        "models": Path("configs/models.yaml").resolve(),
    }


def _run(tmp_path: Path, inputs: dict[str, Path], *, name: str = "controls") -> tuple[Path, ...]:
    return run_representation_controls(
        profiles_path=inputs["profiles"],
        benchmarks_path=inputs["benchmarks"],
        encoding_manifest_path=inputs["encoding_manifest"],
        encoding_flow_path=inputs["encoding_flow"],
        models_path=inputs["models"],
        output_root=tmp_path / name,
        study=_study(),
        participant_folds=3,
        bootstrap_repetitions=12,
        permutation_repetitions=12,
    )


def test_availability_claims_only_rehashed_primary_trajectory(tmp_path: Path) -> None:
    paths = _run(tmp_path, _inputs(tmp_path))
    availability = pd.read_parquet(paths[0]).set_index("control_id")

    assert set(availability) == {
        "primary_model_id",
        "checkpoint_sha256",
        "model_fingerprint_sha256",
        "source_revision",
        "factory",
        "trajectory_inventory_sha256",
        "trajectory_count",
        "encoding_receipt_inventory_sha256",
        "representation_config_sha256",
        "input_artifact_hashes_sha256",
        "scientific_gate_applied",
        "result_selection_applied",
        "control_family",
        "requested_layer",
        "requested_pooling",
        "requested_checkpoint_size",
        "status",
        "reason",
        "exact_hash_pinned_backend_configured",
        "trajectory_generated",
        "implemented_comparator_reference_json",
    }
    primary = availability.loc["primary_layer_pooling"]
    assert primary["status"] == "available"
    assert bool(primary["exact_hash_pinned_backend_configured"])
    assert bool(primary["trajectory_generated"])
    assert primary["checkpoint_sha256"] == "a" * 64
    assert len(primary["trajectory_inventory_sha256"]) == 64
    assert int(primary["trajectory_count"]) == 1

    unavailable = availability.drop(index="primary_layer_pooling")
    assert unavailable["status"].eq("unavailable").all()
    assert (~unavailable["trajectory_generated"]).all()
    assert (~unavailable["exact_hash_pinned_backend_configured"]).all()
    assert (
        "full_coordinate_trajectory_backend" in availability.loc["pca_coordinate_control", "reason"]
    )
    spectral_references = json.loads(
        availability.loc[
            "time_frequency_coordinate_control", "implemented_comparator_reference_json"
        ]
    )
    assert "relative_band_power" in spectral_references
    assert "weighted_symbolic_mutual_information" in spectral_references

    audit = json.loads(paths[3].read_text(encoding="utf-8"))
    assert audit["available_coordinate_trajectory_controls"] == ["primary_layer_pooling"]
    assert audit["full_pca_coordinate_trajectory_claimed"] is False
    assert audit["full_time_frequency_coordinate_trajectory_claimed"] is False
    assert audit["scientific_gate_applied"] is False


def test_dataset_identity_is_deterministic_participant_separated_and_plus_one(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path)
    first_paths = _run(tmp_path, inputs, name="first")
    second_paths = _run(tmp_path, inputs, name="second")
    first = pd.read_parquet(first_paths[1]).sort_values("feature_set").reset_index(drop=True)
    second = pd.read_parquet(second_paths[1]).sort_values("feature_set").reset_index(drop=True)
    pd.testing.assert_frame_equal(first, second)

    assert set(first["feature_set"]) == {
        "five_axis_participant_condition",
        "conventional_exact_eligible_cells",
    }
    assert first["status"].eq("available").all()
    assert first["balanced_accuracy"].between(0, 1).all()
    assert first["auroc"].between(0, 1).all()
    assert first["balanced_accuracy_bootstrap_status"].eq("available").all()
    assert first["auroc_bootstrap_status"].eq("available").all()
    for row in first.to_dict(orient="records"):
        successful = row["balanced_accuracy_permutation_successful_repetitions"]
        extreme = row["balanced_accuracy_permutation_extreme_count"]
        assert row["balanced_accuracy_permutation_pvalue_plus_one"] == (extreme + 1) / (
            successful + 1
        )
        successful_auc = row["auroc_permutation_successful_repetitions"]
        extreme_auc = row["auroc_permutation_extreme_count"]
        assert row["auroc_permutation_pvalue_plus_one"] == (extreme_auc + 1) / (successful_auc + 1)
        assert row["participant_key_scope"] == "dataset_id_plus_participant_id"
        assert bool(row["participant_equal_weighting"])
        assert not bool(row["participant_level_predictions_published"])

    folds = pd.read_parquet(first_paths[2])
    assert len(folds) == 6
    assert folds["participant_sets_disjoint"].all()
    assert folds["imputer_fit_scope"].eq("fold_training_participants_only").all()
    assert folds["scaler_fit_scope"].eq("fold_training_participants_only").all()
    assert folds["classifier_fit_scope"].eq("fold_training_participants_only").all()
    assert folds["train_participant_set_sha256"].str.len().eq(64).all()
    assert folds["test_participant_set_sha256"].str.len().eq(64).all()
    forbidden = {
        "participant_id",
        "participant_key",
        "unit_id",
        "trajectory_path",
        "source_path",
    }
    assert forbidden.isdisjoint(first.columns)
    assert forbidden.isdisjoint(folds.columns)


def test_incomplete_benchmark_cell_is_explicit_and_not_silently_imputed(
    tmp_path: Path,
) -> None:
    paths = _run(tmp_path, _inputs(tmp_path, complete_benchmarks=False))
    diagnostics = pd.read_parquet(paths[1]).set_index("feature_set")
    five_axis = diagnostics.loc["five_axis_participant_condition"]
    conventional = diagnostics.loc["conventional_exact_eligible_cells"]

    assert five_axis["status"] == "available"
    assert conventional["status"] == "available"
    contract = json.loads(conventional["cell_contract_json"])
    assert contract["incomplete_cells"] == 1
    assert contract["excluded_profile_cells"] == 1
    assert contract["matched_cells"] == int(conventional["n_cells"])
    assert contract["matched_cell_inventory_sha256"] == conventional["cell_inventory_sha256"]


def test_insufficient_participants_return_structured_unavailable_rows(tmp_path: Path) -> None:
    paths = _run(
        tmp_path,
        _inputs(tmp_path, datasets=2, participants_per_dataset=1),
    )
    diagnostics = pd.read_parquet(paths[1])
    assert diagnostics["status"].eq("unavailable").all()
    assert diagnostics["reason"].str.contains("fewer_than_two_participants").all()
    assert diagnostics["balanced_accuracy"].isna().all()
    assert diagnostics["auroc"].isna().all()
    assert pd.read_parquet(paths[2]).empty


def test_benchmark_identity_hash_mismatch_fails_closed(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    benchmarks = pd.read_parquet(inputs["benchmarks"])
    benchmarks.loc[0, "participant_condition_cell_key_sha256"] = "0" * 64
    benchmarks.to_parquet(inputs["benchmarks"], index=False)

    with pytest.raises(ValueError, match="cell hash does not match"):
        _run(tmp_path, inputs)


def test_primary_availability_fails_closed_on_representation_receipt_mismatch(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path)
    manifest = pd.read_parquet(inputs["encoding_manifest"])
    receipt = Path(str(manifest.loc[0, "encoding_provenance_path"]))
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    payload["inputs"]["representation_config_sha256"] = "f" * 64
    receipt.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    manifest.loc[0, "encoding_provenance_sha256"] = sha256_file(receipt)
    manifest.to_parquet(inputs["encoding_manifest"], index=False)

    with pytest.raises(ValueError, match="representation configuration changed"):
        _run(tmp_path, inputs)


def test_unpinned_runtime_is_reported_unavailable_without_a_scientific_gate(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path)
    flow = json.loads(inputs["encoding_flow"].read_text(encoding="utf-8"))
    flow["model_fingerprint_sha256"] = None
    inputs["encoding_flow"].write_text(json.dumps(flow), encoding="utf-8")

    paths = _run(tmp_path, inputs)
    availability = pd.read_parquet(paths[0]).set_index("control_id")
    primary = availability.loc["primary_layer_pooling"]
    assert primary["status"] == "unavailable"
    assert primary["reason"] == ("encoding_flow_lacks_exact_checkpoint_or_model_fingerprint_sha256")
    assert not bool(primary["trajectory_generated"])
    assert pd.read_parquet(paths[1])["status"].eq("available").all()
    audit = json.loads(paths[3].read_text(encoding="utf-8"))
    assert audit["scientific_gate_applied"] is False


def test_metrics_phase_publishes_representation_controls_after_benchmarks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_root = tmp_path / "run"
    output_root = run_root / "metrics"
    for relative in (
        "encode/encoding-manifest.parquet",
        "encode/encoding-flow.json",
        "preprocess/preprocessing-manifest.parquet",
        "preprocess/cohort-labels.parquet",
    ):
        path = run_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"input")

    def artifacts(*names: str) -> tuple[Path, ...]:
        output: list[Path] = []
        for name in names:
            path = output_root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(name.encode())
            output.append(path)
        return tuple(output)

    metric_artifacts = artifacts(
        "profiles.parquet",
        "nulls.parquet",
        "dictionary.joblib",
        "estimator.joblib",
        "metric-audit.json",
    )
    sampling_artifacts = artifacts("sampling/profiles.parquet", "sampling/audit.json")
    channel_artifacts = artifacts("channel/profiles.parquet", "channel/audit.json")
    combined_nulls = artifacts("all-null-profiles.parquet")[0]
    benchmark_artifacts = artifacts("benchmarks.parquet", "benchmark-audit.json")
    representation_artifacts = artifacts(
        "representation-controls/availability.parquet",
        "representation-controls/diagnostics.parquet",
        "representation-controls/folds.parquet",
        "representation-controls/audit.json",
    )
    monkeypatch.setattr(phase_runner, "load_study", lambda _path: _study())
    monkeypatch.setattr(phase_runner, "run_metrics", lambda **_kwargs: metric_artifacts)
    monkeypatch.setattr(
        phase_runner,
        "run_sampling_sensitivity",
        lambda **_kwargs: sampling_artifacts,
    )
    monkeypatch.setattr(
        phase_runner,
        "run_preencoder_channel_permutation_control",
        lambda **_kwargs: SimpleNamespace(
            profiles_path=channel_artifacts[0], audit_path=channel_artifacts[1]
        ),
    )
    monkeypatch.setattr(phase_runner, "combine_null_profile_tables", lambda *_args: combined_nulls)
    benchmark_completed = False

    def fake_benchmarks(**_kwargs: object) -> tuple[Path, ...]:
        nonlocal benchmark_completed
        benchmark_completed = True
        return benchmark_artifacts

    monkeypatch.setattr(phase_runner, "run_benchmarks", fake_benchmarks)
    captured: dict[str, object] = {}

    def fake_controls(**kwargs: object) -> tuple[Path, ...]:
        assert benchmark_completed
        captured.update(kwargs)
        return representation_artifacts

    monkeypatch.setattr(phase_runner, "run_representation_controls", fake_controls)
    context = SimpleNamespace(
        study_path=Path("configs/study.yaml").resolve(),
        run_root=run_root,
        output_directory=lambda: output_root,
    )

    published = phase_runner.run_metric_stage(context)

    assert captured["profiles_path"] == metric_artifacts[0]
    assert captured["benchmarks_path"] == benchmark_artifacts[0]
    assert captured["encoding_manifest_path"] == (run_root / "encode" / "encoding-manifest.parquet")
    assert captured["encoding_flow_path"] == run_root / "encode" / "encoding-flow.json"
    assert list(representation_artifacts) == published[-4:]
