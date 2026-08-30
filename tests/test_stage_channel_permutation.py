from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import numpy as np
import pandas as pd
import pytest

from neural_manifolds.config import StudyConfig, load_study
from neural_manifolds.foundation.base import EncodedBatch
from neural_manifolds.provenance import sha256_file
from neural_manifolds.stages import channel_permutation
from neural_manifolds.stages.channel_permutation import (
    FAMILY,
    NULL_PROFILE_COLUMNS,
    ChannelPermutationError,
    combine_null_profile_tables,
    run_preencoder_channel_permutation_control,
)
from neural_manifolds.stages.metrics import run_metrics

CHANNEL_NAMES = ("Fp1", "Fp2", "C3", "C4")


class FakeRaw:
    def __init__(self, data: np.ndarray) -> None:
        self._data = np.asarray(data, dtype=np.float64)
        self.ch_names = list(CHANNEL_NAMES)
        self.info = {"sfreq": 200.0}

    def get_data(self) -> np.ndarray:
        return self._data.copy()


class FakeFrozenLaBraM:
    metadata: ClassVar[dict[str, Any]] = {
        "encoder": "fake_labram_for_test",
        "weights_frozen": True,
        "checkpoint_sha256": "f" * 64,
    }

    def __init__(self, labels_opened: Any | None = None) -> None:
        self.calls: list[tuple[np.ndarray, tuple[str, ...], tuple[str, ...]]] = []
        self._labels_opened = labels_opened or (lambda: False)

    def encode(
        self,
        windows_volts: np.ndarray,
        channel_names: tuple[str, ...],
        *,
        metadata_fields: tuple[str, ...] = (),
    ) -> EncodedBatch:
        assert not self._labels_opened(), "labels were opened before frozen inference"
        assert metadata_fields == ()
        windows = np.asarray(windows_volts, dtype=np.float64)
        names = tuple(channel_names)
        self.calls.append((windows.copy(), names, tuple(metadata_fields)))
        channel_means = windows.mean(axis=2)
        global_states = np.column_stack(
            (
                channel_means,
                windows.mean(axis=(1, 2)),
                windows.std(axis=(1, 2)),
                np.mean(np.diff(windows, axis=2), axis=(1, 2)),
                np.sqrt(np.mean(windows**2, axis=(1, 2))),
            )
        ).astype(np.float32)
        anterior = global_states + np.linspace(0.0, 0.2, len(global_states))[:, None]
        posterior = np.roll(global_states, 1, axis=0) + 0.03 * global_states
        patch_count = windows.shape[2] // 200
        tokens = np.broadcast_to(
            global_states[:, None, None, :],
            (len(windows), windows.shape[1], patch_count, global_states.shape[1]),
        ).copy()
        return EncodedBatch(
            global_states=global_states,
            regional_states={"anterior": anterior, "posterior": posterior},
            patch_tokens=tokens,
            channel_names=names,
            metadata=dict(self.metadata),
        )


def _study() -> StudyConfig:
    source = Path(__file__).parents[1] / "configs" / "study.yaml"
    study = load_study(source)
    representation = study.representation.model_copy(
        update={
            "alignment_step_seconds": 1.0,
            "dynamics_rank": 6,
        }
    )
    preprocessing = study.preprocessing.model_copy(
        update={
            "minimum_event_trials_per_condition": 3,
            "minimum_valid_windows": 10,
        }
    )
    metrics = {
        **study.metrics,
        "alignment": {**study.metrics["alignment"], "lags_ms": [1000, 2000]},
    }
    return study.model_copy(
        update={
            "representation": representation,
            "preprocessing": preprocessing,
            "metrics": metrics,
        }
    )


def _frozen_profiler(tmp_path: Path, study: StudyConfig) -> tuple[Path, Path]:
    rows = []
    for index in range(5):
        rng = np.random.default_rng(100 + index)
        trajectory = rng.normal(size=(140, 8)).astype(np.float32)
        anterior = (trajectory + rng.normal(scale=0.1, size=trajectory.shape)).astype(np.float32)
        posterior = (np.roll(anterior, 1, axis=0) + 0.05 * trajectory).astype(np.float32)
        path = tmp_path / f"reference-{index}.npz"
        with path.open("wb") as stream:
            np.savez_compressed(
                stream,
                global_states=trajectory,
                alignment_regional_anterior=anterior,
                alignment_regional_posterior=posterior,
            )
        rows.append(
            {
                "unit_id": f"reference-{index}",
                "participant_id": f"reference-sub-{index}",
                "dataset_id": "synthetic",
                "trajectory_path": str(path),
                "trajectory_sha256": sha256_file(path),
                "encoded": True,
                "healthy_wake_reference": True,
                "clinical_holdout": False,
                "condition": "wake",
            }
        )
    manifest = tmp_path / "reference-encoding.parquet"
    pd.DataFrame(rows).to_parquet(manifest, index=False)
    outputs = run_metrics(
        encoding_manifest=manifest,
        output_root=tmp_path / "reference-metrics",
        study=study,
        state_counts=(3,),
        null_repeats=0,
        force_state_fallback=True,
    )
    return outputs[2], outputs[3]


def _signal_and_labels(
    tmp_path: Path,
) -> tuple[Path, Path, dict[str, np.ndarray], dict[str, Path]]:
    rng = np.random.default_rng(90210)
    data: dict[str, np.ndarray] = {
        "rest-unit": rng.normal(scale=2e-6, size=(4, 24_000)),
        "event-0": rng.normal(scale=2e-6, size=(4, 2_400)),
        "event-1": rng.normal(scale=2e-6, size=(4, 2_400)),
        "event-2": rng.normal(scale=2e-6, size=(4, 2_400)),
    }
    # Channel-specific offsets make the row permutation directly observable.
    for values in data.values():
        values += np.arange(4, dtype=float)[:, None] * 1e-5
    paths: dict[str, Path] = {}
    preprocessing_rows = []
    label_rows = []
    for unit_id in data:
        path = tmp_path / f"{unit_id}.fif"
        path.write_bytes(f"preprocessed:{unit_id}".encode())
        paths[unit_id] = path
        is_event = unit_id.startswith("event")
        selector = {
            "kind": "event_epoch" if is_event else "full_recording",
            "event_onset_seconds": 1.0 if is_event else None,
            "epoch_start_offset_seconds": 0.0 if is_event else None,
            "epoch_stop_offset_seconds": 12.0 if is_event else None,
        }
        preprocessing_rows.append(
            {
                "unit_id": unit_id,
                "modality": "eeg",
                "selector_json": json.dumps(selector),
                "preprocessed_path": str(path),
                "preprocessed_sha256": sha256_file(path),
                "eligible": True,
            }
        )
        label_rows.append(
            {
                "unit_id": unit_id,
                "participant_id": "synthetic:event-sub" if is_event else "synthetic:rest-sub",
                "dataset_id": "synthetic",
                "condition": "event-target" if is_event else "wake",
                "modality": "eeg",
                "healthy_wake_reference": not is_event,
                "clinical_holdout": False,
                "metadata_status": "verified",
            }
        )
    preprocessing = tmp_path / "preprocessing.parquet"
    labels = tmp_path / "labels.parquet"
    pd.DataFrame(preprocessing_rows).to_parquet(preprocessing, index=False)
    pd.DataFrame(label_rows).to_parquet(labels, index=False)
    return preprocessing, labels, data, paths


def test_true_preencoder_permutation_label_firewall_segments_and_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    study = _study()
    dictionary, estimator = _frozen_profiler(tmp_path, study)
    preprocessing, labels, data, _paths = _signal_and_labels(tmp_path)
    label_state = {"opened": False}
    real_read_parquet = pd.read_parquet

    def guarded_read(path: Any, *args: Any, **kwargs: Any) -> pd.DataFrame:
        if Path(path).resolve() == labels.resolve():
            label_state["opened"] = True
        return real_read_parquet(path, *args, **kwargs)

    monkeypatch.setattr(channel_permutation.pd, "read_parquet", guarded_read)
    monkeypatch.setattr(
        channel_permutation,
        "detect_artifact_windows",
        lambda windows, sfreq: SimpleNamespace(keep=np.ones(len(windows), dtype=bool)),
    )
    backend = FakeFrozenLaBraM(lambda: label_state["opened"])
    artifacts = run_preencoder_channel_permutation_control(
        preprocessing_manifest=preprocessing,
        labels_manifest=labels,
        state_dictionary_path=dictionary,
        profile_estimator_path=estimator,
        output_root=tmp_path / "control",
        study=study,
        repeats=2,
        encoder=backend,
        raw_loader=lambda path: FakeRaw(data[path.stem]),
    )

    assert artifacts.repeats == 2
    assert len(backend.calls) == 10
    first_windows, names, metadata_fields = backend.calls[0]
    assert names == CHANNEL_NAMES
    assert metadata_fields == ()
    original = data["event-0"][:, :200]
    assert not np.array_equal(first_windows[0], original)
    assert sorted(np.mean(first_windows[0], axis=1)) == pytest.approx(
        sorted(np.mean(original, axis=1))
    )

    profiles = pd.read_parquet(artifacts.profiles_path)
    assert len(profiles) == 4
    assert set(profiles["family"]) == {FAMILY}
    assert set(profiles["repeat"]) == {0, 1}
    assert profiles["unit_id"].equals(profiles["profile_id"])
    assert profiles["seed"].equals(profiles["repeat_seed"])
    assert set(NULL_PROFILE_COLUMNS).issubset(profiles.columns)
    assert set(
        ("repertoire", "metastability", "directionality", "alignment", "reachability")
    ) <= set(profiles)
    assert not {
        "preprocessed_path",
        "source_path",
        "trajectory_path",
        "permutation",
    }.intersection(profiles.columns)

    repeat_audit = json.loads(
        (artifacts.repeats_root / "repeat-0000" / "repeat-audit.json").read_text(encoding="utf-8")
    )
    event_audit = next(
        row for row in repeat_audit["profile_segment_audit"] if row["event_aggregated"]
    )
    assert event_audit["trial_count"] == 3
    assert len(event_audit["coarse_segment_lengths"]) == 3
    assert len(event_audit["alignment_segment_lengths"]) == 3
    assert event_audit["cross_trial_transitions_allowed"] is False
    assert event_audit["cross_trial_alignment_pairs_allowed"] is False
    assert all(
        row["channel_names_passed_unchanged"]
        and row["permutation_nonidentity"]
        and row["label_fields_consumed"] == []
        for row in repeat_audit["unit_permutation_audit"]
    )

    first_profiles = profiles.copy()
    calls_before_restart = len(backend.calls)
    label_state["opened"] = False
    second = run_preencoder_channel_permutation_control(
        preprocessing_manifest=preprocessing,
        labels_manifest=labels,
        state_dictionary_path=dictionary,
        profile_estimator_path=estimator,
        output_root=tmp_path / "control",
        study=study,
        repeats=2,
        encoder=backend,
        raw_loader=lambda path: FakeRaw(data[path.stem]),
    )
    assert len(backend.calls) == calls_before_restart
    pd.testing.assert_frame_equal(first_profiles, pd.read_parquet(second.profiles_path))
    root_audit = json.loads(second.audit_path.read_text(encoding="utf-8"))
    assert all(row["reused"] for row in root_audit["repeat_artifacts"])
    assert root_audit["profile_input_spaces"] == {
        "repertoire": {
            "record_field": "repertoire_trajectory",
            "space": "untruncated_frozen_encoder_embedding",
            "dimension": 8,
            "discovery_projection_applied": False,
        },
        "dynamics": {
            "record_field": "trajectory",
            "space": "discovery_fitted_pca_projection",
            "dimension": 6,
            "discovery_projection_applied": True,
        },
    }
    assert root_audit["intervention"]["signal_rows_permuted_before_encoder"] is True
    assert root_audit["intervention"]["post_encoder_latent_rotation"] is False


def test_rejects_label_leakage_before_any_encoder_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    study = _study()
    dictionary, estimator = _frozen_profiler(tmp_path, study)
    preprocessing, labels, data, _paths = _signal_and_labels(tmp_path)
    frame = pd.read_parquet(preprocessing)
    frame["condition"] = "leaked"
    frame.to_parquet(preprocessing, index=False)
    backend = FakeFrozenLaBraM()
    with pytest.raises(ChannelPermutationError, match="label fields are forbidden"):
        run_preencoder_channel_permutation_control(
            preprocessing_manifest=preprocessing,
            labels_manifest=labels,
            state_dictionary_path=dictionary,
            profile_estimator_path=estimator,
            output_root=tmp_path / "control",
            study=study,
            repeats=1,
            encoder=backend,
            raw_loader=lambda path: FakeRaw(data[path.stem]),
        )
    assert backend.calls == []


def test_requires_explicit_positive_repeat_count(tmp_path: Path) -> None:
    with pytest.raises(ChannelPermutationError, match="explicit positive integer"):
        run_preencoder_channel_permutation_control(
            preprocessing_manifest=tmp_path / "missing-preprocessing.parquet",
            labels_manifest=tmp_path / "missing-labels.parquet",
            state_dictionary_path=tmp_path / "missing-dictionary.joblib",
            profile_estimator_path=tmp_path / "missing-estimator.joblib",
            output_root=tmp_path / "control",
            study=_study(),
            repeats=0,
            encoder=FakeFrozenLaBraM(),
        )


def test_canonical_null_profile_combiner_is_atomic_and_schema_exact(tmp_path: Path) -> None:
    axes = {
        "repertoire": 1.0,
        "metastability": 2.0,
        "directionality": 3.0,
        "alignment": 4.0,
        "reachability": 5.0,
    }
    base = tmp_path / "base.parquet"
    channel = tmp_path / "channel.parquet"
    pd.DataFrame(
        [
            {
                "unit_id": "base-unit",
                "participant_id": "dataset:sub-01",
                "dataset_id": "dataset",
                "family": "phase_randomization",
                "repeat": 0,
                "seed": 11,
                **axes,
            }
        ]
    ).to_parquet(base, index=False)
    pd.DataFrame(
        [
            {
                "unit_id": "channel-unit",
                "participant_id": "dataset:sub-02",
                "dataset_id": "dataset",
                "family": FAMILY,
                "repeat": 0,
                "seed": 23,
                "profile_id": "detail-not-published-in-combined-table",
                **axes,
            }
        ]
    ).to_parquet(channel, index=False)

    destination = combine_null_profile_tables(
        base, channel, tmp_path / "combined" / "null-profiles.parquet"
    )
    combined = pd.read_parquet(destination)
    assert list(combined.columns) == list(NULL_PROFILE_COLUMNS)
    assert combined[["unit_id", "family"]].to_dict(orient="records") == [
        {"unit_id": "base-unit", "family": "phase_randomization"},
        {"unit_id": "channel-unit", "family": FAMILY},
    ]
