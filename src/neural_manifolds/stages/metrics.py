"""Participant-safe five-axis metric, null, and reliability stage."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from neural_manifolds.config import StudyConfig, config_sha256
from neural_manifolds.dynamics.state_dictionary import StateDictionary, fit_state_dictionary
from neural_manifolds.foundation.overlap import (
    OVERLAP_ARTIFACT_COLUMNS,
    ensure_pretraining_overlap_columns,
    summarize_pretraining_overlap,
)
from neural_manifolds.manifold.clinical_reference import FrozenWakePropofolLikelihoodRatio
from neural_manifolds.manifold.profile import AXIS_NAMES, FiveAxisProfileEstimator, ManifoldRecord
from neural_manifolds.manifold.surrogates import (
    block_permutation,
    covariance_matched_surrogate,
    dwell_matched_state_surrogate,
    phase_randomized_surrogate,
    rotate_channels,
)
from neural_manifolds.provenance import atomic_write_json, sha256_file
from neural_manifolds.tms_separation import assert_no_direct_tms


@dataclass(frozen=True)
class LoadedUnit:
    unit_id: str
    participant_id: str
    dataset_id: str
    trajectory: np.ndarray
    regional: Mapping[str, np.ndarray]
    segment_ids: np.ndarray | None
    alignment_segment_ids: np.ndarray | None
    metadata: Mapping[str, Any]


def _atomic_parquet(frame: pd.DataFrame, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, destination)
    return destination


def _atomic_joblib(value: Any, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    joblib.dump(value, temporary, compress=3)
    os.replace(temporary, destination)
    return destination


def _unit_key(row: Mapping[str, Any]) -> str:
    for key in ("unit_id", "recording_id"):
        value = row.get(key)
        if isinstance(value, str) and value:
            return value
    raise ValueError("encoding manifest row has no unit_id or recording_id")


def _load_unit(row: Mapping[str, Any]) -> LoadedUnit:
    path = Path(str(row["trajectory_path"])).resolve(strict=True)
    expected = row.get("trajectory_sha256")
    if isinstance(expected, str) and sha256_file(path) != expected:
        raise ValueError(f"trajectory checksum mismatch: {path}")
    with np.load(path, allow_pickle=False) as archive:
        if "global_states" not in archive:
            raise ValueError(f"trajectory has no global_states: {path}")
        trajectory = np.asarray(archive["global_states"], dtype=np.float64)
        fine_names = [name for name in archive.files if name.startswith("alignment_regional_")]
        prefix = "alignment_regional_" if len(fine_names) >= 2 else "regional_"
        regional = {
            name.removeprefix(prefix): np.asarray(archive[name], dtype=np.float64)
            for name in archive.files
            if name.startswith(prefix)
        }
        segment_ids = (
            np.asarray(archive["segment_ids"], dtype=np.int64) if "segment_ids" in archive else None
        )
        alignment_segment_ids = (
            np.asarray(archive["alignment_segment_ids"], dtype=np.int64)
            if "alignment_segment_ids" in archive
            else None
        )
    declared_minimum = row.get("minimum_valid_windows_required")
    if isinstance(declared_minimum, (int, float, np.generic)) and np.isfinite(declared_minimum):
        minimum_windows = max(1, int(declared_minimum))
    else:
        minimum_windows = 20
    if trajectory.ndim != 2 or trajectory.shape[0] < minimum_windows:
        raise ValueError(f"trajectory is too short or malformed: {path}")
    if len(regional) < 2 or len({values.shape[0] for values in regional.values()}) != 1:
        raise ValueError(f"at least two time-aligned regional trajectories are required: {path}")
    safe_metadata = {
        key: value
        for key, value in row.items()
        if key
        not in {
            "trajectory_path",
            "preprocessed_path",
            "source_path",
            "events_path",
            "channels_path",
        }
        and (value is None or isinstance(value, (str, int, float, bool, np.generic)))
    }
    return LoadedUnit(
        unit_id=_unit_key(row),
        participant_id=str(row["participant_id"]),
        dataset_id=str(row["dataset_id"]),
        trajectory=trajectory,
        regional=regional,
        segment_ids=segment_ids,
        alignment_segment_ids=alignment_segment_ids,
        metadata=safe_metadata,
    )


def _stable_order(value: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def _unit_sequences(unit: LoadedUnit) -> list[np.ndarray]:
    if unit.segment_ids is None:
        return [unit.trajectory]
    segments = np.asarray(unit.segment_ids)
    return [unit.trajectory[segments == value] for value in pd.unique(segments)]


def _reference_split(
    units: Sequence[LoadedUnit],
    *,
    seed: int,
    validation_fraction: float = 0.20,
    evaluation_fraction: float = 0.20,
) -> tuple[set[str], set[str], set[str]]:
    participants = sorted(
        {unit.participant_id for unit in units}, key=lambda x: _stable_order(x, seed)
    )
    if len(participants) < 4:
        raise ValueError("at least four healthy-reference participants are required")
    n_validation = max(1, round(len(participants) * validation_fraction))
    n_evaluation = max(1, round(len(participants) * evaluation_fraction))
    while n_validation + n_evaluation > len(participants) - 2:
        if n_evaluation >= n_validation and n_evaluation > 1:
            n_evaluation -= 1
        elif n_validation > 1:
            n_validation -= 1
        else:
            raise ValueError("healthy-reference split cannot retain two discovery participants")
    validation = set(participants[:n_validation])
    evaluation = set(participants[n_validation : n_validation + n_evaluation])
    discovery = set(participants[n_validation + n_evaluation :])
    if discovery & validation or discovery & evaluation or validation & evaluation:
        raise AssertionError("participant leakage in reference split")
    return discovery, validation, evaluation


def _combined_segment_runs(
    records: Sequence[ManifoldRecord],
    *,
    attribute: str,
    lengths: Sequence[int],
) -> np.ndarray:
    """Concatenate records without reconnecting any existing temporal break."""

    combined: list[np.ndarray] = []
    next_segment = 0
    for record, length in zip(records, lengths, strict=True):
        source = getattr(record, attribute)
        segments = np.zeros(length, dtype=np.int64) if source is None else np.asarray(source)
        if segments.ndim != 1 or len(segments) != length:
            raise ValueError(f"{attribute} must align with its record")
        run_starts = np.r_[True, segments[1:] != segments[:-1]]
        local = np.cumsum(run_starts, dtype=np.int64) - 1
        local += next_segment
        combined.append(local)
        next_segment = int(local[-1]) + 1
    return np.concatenate(combined)


def _combine_records(records: Sequence[ManifoldRecord], *, name: str) -> ManifoldRecord:
    if not records:
        raise ValueError("cannot combine an empty record collection")
    region_names = set(records[0].regional_trajectories)
    for record in records[1:]:
        region_names.intersection_update(record.regional_trajectories)
    if len(region_names) < 2:
        raise ValueError("combined record has fewer than two shared regions")
    global_lengths = [len(np.asarray(record.states)) for record in records]
    alignment_lengths = [
        len(np.asarray(record.regional_trajectories[next(iter(sorted(region_names)))]))
        for record in records
    ]
    repertoire_trajectories = [
        np.asarray(
            record.trajectory
            if record.repertoire_trajectory is None
            else record.repertoire_trajectory
        )
        for record in records
    ]
    if len({value.shape[1] for value in repertoire_trajectories}) != 1:
        raise ValueError("combined record has inconsistent repertoire dimensions")
    return ManifoldRecord(
        trajectory=np.concatenate([np.asarray(record.trajectory) for record in records]),
        states=np.concatenate([np.asarray(record.states) for record in records]),
        regional_trajectories={
            region: np.concatenate(
                [np.asarray(record.regional_trajectories[region]) for record in records]
            )
            for region in sorted(region_names)
        },
        repertoire_trajectory=np.concatenate(repertoire_trajectories),
        segment_ids=_combined_segment_runs(
            records,
            attribute="segment_ids",
            lengths=global_lengths,
        ),
        alignment_segment_ids=_combined_segment_runs(
            records,
            attribute="alignment_segment_ids",
            lengths=alignment_lengths,
        ),
        name=name,
    )


def _record(unit: LoadedUnit, dictionary: StateDictionary) -> ManifoldRecord:
    projected = dictionary.project(unit.trajectory)
    return ManifoldRecord(
        trajectory=projected,
        states=dictionary.predict_projected(projected, segment_ids=unit.segment_ids),
        regional_trajectories=unit.regional,
        repertoire_trajectory=unit.trajectory,
        segment_ids=unit.segment_ids,
        alignment_segment_ids=unit.alignment_segment_ids,
        name=unit.unit_id,
    )


def _within_segments(
    values: np.ndarray,
    segment_ids: np.ndarray | None,
    transform: Any,
    *,
    seed: int,
) -> np.ndarray:
    """Apply a temporal null independently within every contiguous segment."""

    array = np.asarray(values)
    if segment_ids is None:
        return np.asarray(transform(array, random_state=seed))
    segments = np.asarray(segment_ids)
    if segments.ndim != 1 or len(segments) != len(array):
        raise ValueError("segment_ids must align with the temporal null input")
    boundaries = np.r_[0, np.flatnonzero(segments[1:] != segments[:-1]) + 1, len(segments)]
    output = np.empty_like(array, dtype=np.float64)
    for index, (start, stop) in enumerate(pairwise(boundaries)):
        output[start:stop] = transform(array[start:stop], random_state=seed + index)
    return output


def _profile_row(
    unit: LoadedUnit,
    profile: Any,
    *,
    state_method: str,
) -> dict[str, Any]:
    details = profile.details
    output: dict[str, Any] = {
        **unit.metadata,
        "unit_id": unit.unit_id,
        "participant_id": unit.participant_id,
        "dataset_id": unit.dataset_id,
        "n_windows": int(unit.trajectory.shape[0]),
        "state_method": state_method,
        "repertoire_source_dimension": int(details.repertoire.n_features),
        "dynamics_projection_dimension": int(details.local_dynamics.transition_matrices.shape[-1]),
        "repertoire_effective_rank": details.repertoire.effective_rank,
        "metastability_median_dwell_seconds": details.metastability.median_dwell,
        "metastability_switching_rate": details.metastability.switching_rate,
        "metastability_recurrence_probability": details.metastability.recurrence_probability,
        "directionality_flux_asymmetry": details.directionality.flux_asymmetry,
        "alignment_best_pair_mean": details.alignment.mean_shared_predictive_variance,
        "reachability_effective_rank": details.reachability.effective_rank,
    }
    for index, axis in enumerate(AXIS_NAMES):
        output[axis] = float(profile.values[index])
        output[f"{axis}_raw"] = float(profile.raw_values[index])
    return output


def _surrogate_record(
    record: ManifoldRecord,
    dictionary: StateDictionary,
    *,
    family: str,
    seed: int,
) -> ManifoldRecord:
    trajectory = np.asarray(record.trajectory)
    repertoire_trajectory = np.asarray(
        record.trajectory if record.repertoire_trajectory is None else record.repertoire_trajectory
    )
    regional = {name: np.asarray(value) for name, value in record.regional_trajectories.items()}
    if family == "phase_randomization":
        trajectory_null = _within_segments(
            trajectory,
            np.asarray(record.segment_ids) if record.segment_ids is not None else None,
            phase_randomized_surrogate,
            seed=seed,
        )
        repertoire_null = _within_segments(
            repertoire_trajectory,
            np.asarray(record.segment_ids) if record.segment_ids is not None else None,
            phase_randomized_surrogate,
            seed=seed,
        )
        regional_null = {
            name: _within_segments(
                value,
                (
                    np.asarray(record.alignment_segment_ids)
                    if record.alignment_segment_ids is not None
                    else None
                ),
                phase_randomized_surrogate,
                seed=seed + index + 1,
            )
            for index, (name, value) in enumerate(sorted(regional.items()))
        }
        states = dictionary.predict_projected(
            trajectory_null,
            segment_ids=np.asarray(record.segment_ids) if record.segment_ids is not None else None,
        )
    elif family == "blockwise_temporal_permutation":
        block = max(2, min(20, trajectory.shape[0] // 5))
        trajectory_null = block_permutation(
            trajectory,
            block_size=block,
            segment_ids=record.segment_ids,
            random_state=seed,
        )
        repertoire_null = block_permutation(
            repertoire_trajectory,
            block_size=block,
            segment_ids=record.segment_ids,
            random_state=seed,
        )
        regional_null = {
            name: block_permutation(
                value,
                block_size=max(2, min(50, value.shape[0] // 5)),
                segment_ids=record.alignment_segment_ids,
                random_state=seed + index + 1,
            )
            for index, (name, value) in enumerate(sorted(regional.items()))
        }
        states = dictionary.predict_projected(
            trajectory_null,
            segment_ids=np.asarray(record.segment_ids) if record.segment_ids is not None else None,
        )
    elif family == "post_encoder_latent_rotation_control":
        trajectory_null = rotate_channels(trajectory, random_state=seed)
        repertoire_null = rotate_channels(repertoire_trajectory, random_state=seed)
        regional_null = {
            name: rotate_channels(value, random_state=seed + index + 1)
            for index, (name, value) in enumerate(sorted(regional.items()))
        }
        states = dictionary.predict_projected(
            trajectory_null,
            segment_ids=np.asarray(record.segment_ids) if record.segment_ids is not None else None,
        )
    elif family == "covariance_dwell_matched_state_space":
        trajectory_null = covariance_matched_surrogate(trajectory, random_state=seed)
        repertoire_null = covariance_matched_surrogate(repertoire_trajectory, random_state=seed)
        states = dwell_matched_state_surrogate(
            record.states, segment_ids=record.segment_ids, random_state=seed + 1
        )
        regional_null = {
            name: covariance_matched_surrogate(value, random_state=seed + index + 2)
            for index, (name, value) in enumerate(sorted(regional.items()))
        }
    else:
        raise ValueError(f"unknown null family: {family}")
    return ManifoldRecord(
        trajectory=trajectory_null,
        states=states,
        regional_trajectories=regional_null,
        repertoire_trajectory=repertoire_null,
        segment_ids=record.segment_ids,
        alignment_segment_ids=record.alignment_segment_ids,
        name=f"{record.name}:{family}:{seed}",
    )


def _alignment_lag_indices(study: StudyConfig) -> tuple[int, ...]:
    """Map honest, non-overlapping encoder-window lags to trajectory indices."""

    window_ms = study.representation.alignment_window_seconds * 1000.0
    step_ms = study.representation.alignment_step_seconds * 1000.0
    if abs(step_ms - window_ms) > 1e-9:
        raise ValueError(
            "alignment trajectories must use non-overlapping LaBraM windows; "
            "no independent high-temporal-resolution sensor/CSD track is implemented"
        )
    configured = study.metrics.get("alignment", {}).get("lags_ms")
    if not isinstance(configured, list) or not configured:
        raise ValueError("metrics.alignment.lags_ms must be a non-empty list")
    indices: list[int] = []
    for value in configured:
        lag_ms = float(value)
        if lag_ms + 1e-9 < window_ms:
            raise ValueError(
                "sub-window alignment lags are unavailable without an independent "
                "high-temporal-resolution sensor/CSD track"
            )
        lag_index = round(lag_ms / step_ms)
        if abs(lag_ms - lag_index * step_ms) > 1e-9:
            raise ValueError("alignment lags must lie exactly on the non-overlapping grid")
        indices.append(lag_index)
    if len(set(indices)) != len(indices):
        raise ValueError("alignment lags map to duplicate trajectory indices")
    return tuple(sorted(indices))


def run_metrics(
    *,
    encoding_manifest: str | Path,
    output_root: str | Path,
    study: StudyConfig,
    state_counts: Sequence[int] | None = None,
    minimum_stability_ami: float | None = None,
    null_repeats: int | None = None,
    force_state_fallback: bool = False,
) -> tuple[Path, Path, Path, Path, Path]:
    """Estimate participant-condition profiles without treating windows as subjects."""

    frame = pd.read_parquet(encoding_manifest)
    assert_no_direct_tms(frame, stage="general metric input")
    required = {"participant_id", "dataset_id", "trajectory_path", "encoded"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"encoding manifest is missing {sorted(missing)}")
    frame = ensure_pretraining_overlap_columns(frame, default_model_id="labram_base")
    eligible = frame[frame["encoded"].astype(bool)].copy()
    if "clinical_holdout" in eligible:
        eligible = eligible[~eligible["clinical_holdout"].fillna(False).astype(bool)]
    if "secondary_fmri" in eligible:
        eligible = eligible[~eligible["secondary_fmri"].fillna(False).astype(bool)]
    units = [_load_unit(row) for row in eligible.to_dict(orient="records")]
    if not units:
        raise RuntimeError("no non-clinical encoded analysis units are available")
    references = [
        unit
        for unit in units
        if bool(unit.metadata.get("healthy_wake_reference", False))
        and str(unit.metadata.get("modality", "")) != "tms-eeg"
        and str(unit.metadata.get("acquisition", "")) != "tms"
    ]
    if not references:
        raise RuntimeError("adapter manifest identified no healthy wake reference units")
    discovery_ids, validation_ids, evaluation_ids = _reference_split(
        references, seed=study.random_seeds[0]
    )
    discovery_sequences = [
        sequence
        for unit in references
        if unit.participant_id in discovery_ids
        for sequence in _unit_sequences(unit)
    ]
    validation_sequences = [
        sequence
        for unit in references
        if unit.participant_id in validation_ids
        for sequence in _unit_sequences(unit)
    ]
    configured_range = study.metrics["metastability"]["state_range"]
    counts = tuple(state_counts or range(int(configured_range[0]), int(configured_range[1]) + 1))
    stability = float(
        minimum_stability_ami
        if minimum_stability_ami is not None
        else study.metrics["metastability"]["minimum_solution_ami"]
    )
    configured_initialisations = int(study.metrics["metastability"]["required_initialisations"])
    seeds = tuple(study.random_seeds)
    while len(seeds) < configured_initialisations:
        seeds += (seeds[-1] + 1009,)
    dictionary = fit_state_dictionary(
        discovery_sequences,
        validation_sequences,
        rank=study.representation.dynamics_rank,
        state_counts=counts,
        seeds=seeds[:configured_initialisations],
        minimum_stability_ami=stability,
        force_fallback=force_state_fallback,
    )
    records = {unit.unit_id: _record(unit, dictionary) for unit in units}
    reference_by_participant: list[ManifoldRecord] = []
    for participant in sorted(discovery_ids):
        participant_records = [
            records[unit.unit_id] for unit in references if unit.participant_id == participant
        ]
        reference_by_participant.append(
            _combine_records(participant_records, name=f"reference:{participant}")
        )
    lag_indices = _alignment_lag_indices(study)
    estimator = FiveAxisProfileEstimator(
        sample_interval=study.representation.harmonised_step_seconds,
        alignment_lags=lag_indices,
        alignment_rank=min(3, study.representation.dynamics_rank),
        alignment_cv=study.statistics.participant_stratified_folds,
        reachability_horizon=int(study.metrics["reachability"]["horizon_steps"]),
        gramian_regularization=float(study.metrics["reachability"]["regularisation"]),
        standardization="zscore",
    ).fit(reference_by_participant)

    rows = []
    for unit in units:
        row = _profile_row(
            unit,
            estimator.profile(records[unit.unit_id]),
            state_method=dictionary.method,
        )
        if unit.participant_id in discovery_ids:
            partition = "representation_discovery"
        elif unit.participant_id in validation_ids:
            partition = "representation_validation"
        elif unit.participant_id in evaluation_ids:
            partition = "representation_evaluation"
        else:
            partition = "not_used_for_representation"
        row["representation_partition"] = partition
        row["prediction_evaluation_eligible"] = partition in {
            "representation_evaluation",
            "not_used_for_representation",
        }
        rows.append(row)
    profile_frame = pd.DataFrame(rows)
    clinical_reference_status: dict[str, Any]
    try:
        clinical_reference = FrozenWakePropofolLikelihoodRatio().fit(profile_frame)
        estimator.wake_propofol_reference_ = clinical_reference
        clinical_reference_status = clinical_reference.audit()
    except ValueError as error:
        # Unit tests and partial exploratory runs may not contain both reference
        # conditions.  The metric stage remains useful, but clinical transfer
        # will fail closed rather than manufacture a reference distribution.
        estimator.wake_propofol_reference_ = None
        clinical_reference_status = {
            "status": "unavailable",
            "reason": str(error),
        }
    destination = Path(output_root)
    destination.mkdir(parents=True, exist_ok=True)
    profiles_path = _atomic_parquet(profile_frame, destination / "profiles.parquet")
    dictionary_path = _atomic_joblib(dictionary, destination / "state-dictionary.joblib")
    estimator_path = _atomic_joblib(estimator, destination / "profile-estimator.joblib")

    repeats = study.sampling.repeats if null_repeats is None else int(null_repeats)
    if repeats < 0:
        raise ValueError("null_repeats cannot be negative")
    null_rows: list[dict[str, Any]] = []
    null_errors: list[dict[str, Any]] = []
    families = (
        "phase_randomization",
        "blockwise_temporal_permutation",
        "post_encoder_latent_rotation_control",
        "covariance_dwell_matched_state_space",
    )
    for unit in units:
        record = records[unit.unit_id]
        for family_index, family in enumerate(families):
            for repeat in range(repeats):
                seed = int(seeds[repeat % len(seeds)] + 100_003 * family_index + repeat)
                try:
                    profile = estimator.profile(
                        _surrogate_record(record, dictionary, family=family, seed=seed)
                    )
                    null_rows.append(
                        {
                            "unit_id": unit.unit_id,
                            "participant_id": unit.participant_id,
                            "dataset_id": unit.dataset_id,
                            **{
                                column: unit.metadata.get(column)
                                for column in OVERLAP_ARTIFACT_COLUMNS
                            },
                            "family": family,
                            "repeat": repeat,
                            "seed": seed,
                            **profile.as_dict(),
                        }
                    )
                except (RuntimeError, ValueError, np.linalg.LinAlgError) as error:
                    null_errors.append(
                        {
                            "unit_id": unit.unit_id,
                            "family": family,
                            "repeat": repeat,
                            "error": f"{type(error).__name__}: {error}",
                        }
                    )
    nulls_path = _atomic_parquet(
        pd.DataFrame(
            null_rows,
            columns=[
                "unit_id",
                "participant_id",
                "dataset_id",
                *OVERLAP_ARTIFACT_COLUMNS,
                "family",
                "repeat",
                "seed",
                *AXIS_NAMES,
            ],
        ),
        destination / "null-profiles.parquet",
    )
    audit_path = destination / "metric-audit.json"
    atomic_write_json(
        audit_path,
        {
            "schema_version": 1,
            "study_sha256": config_sha256(study),
            "encoding_manifest": str(Path(encoding_manifest).resolve(strict=True)),
            "encoding_manifest_sha256": sha256_file(encoding_manifest),
            "units": len(units),
            "participants": len({unit.participant_id for unit in units}),
            "healthy_reference_participants": len({unit.participant_id for unit in references}),
            "healthy_reference_excludes_direct_tms": True,
            "reference_discovery_participants": sorted(discovery_ids),
            "reference_validation_participants": sorted(validation_ids),
            "reference_evaluation_participants": sorted(evaluation_ids),
            "participant_overlap": sorted(
                (discovery_ids & validation_ids)
                | (discovery_ids & evaluation_ids)
                | (validation_ids & evaluation_ids)
            ),
            "state_dictionary": dictionary.audit(),
            "profile_input_spaces": estimator.input_space_audit(),
            "pretraining_overlap": summarize_pretraining_overlap(profile_frame),
            "clinical_wake_propofol_reference": clinical_reference_status,
            "alignment_lags_ms": list(study.metrics["alignment"]["lags_ms"]),
            "alignment_lag_indices": list(lag_indices),
            "alignment_window_seconds": study.representation.alignment_window_seconds,
            "alignment_step_seconds": study.representation.alignment_step_seconds,
            "alignment_windows_overlap": False,
            "unavailable_short_lags_ms": list(
                study.metrics["alignment"].get("unavailable_short_lags_ms", [])
            ),
            "short_lag_status": study.metrics["alignment"].get("short_lag_status"),
            "null_families": list(families),
            "null_repeats": repeats,
            "null_errors": null_errors,
            "scientific_gate_applied": False,
        },
    )
    return profiles_path, nulls_path, dictionary_path, estimator_path, audit_path
