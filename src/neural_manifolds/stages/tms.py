"""Direct TMS perturbational validation with participant-level linkage."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from neural_manifolds.config import StudyConfig
from neural_manifolds.preprocessing.eeg import (
    canonicalize_channel_name,
    detect_artifact_windows,
    detect_bad_channels,
)
from neural_manifolds.preprocessing.tms import (
    fit_shared_perturbational_trajectories,
    interpolate_continuous_pulses,
    trajectory_outcomes,
)
from neural_manifolds.provenance import atomic_write_json, sha256_file
from neural_manifolds.stage_processing import infer_mains_frequency, read_raw_recording


def _atomic_parquet(frame: pd.DataFrame, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, destination)
    return destination


def _load_epochs(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path.resolve(strict=True), allow_pickle=False) as archive:
        epochs = np.asarray(archive["epochs"], dtype=np.float64)
        times = np.asarray(archive["times_seconds"], dtype=np.float64)
    if epochs.ndim != 3 or epochs.shape[2] != times.size:
        raise ValueError(f"invalid TMS epoch archive: {path}")
    if not np.all(np.isfinite(epochs)) or not np.all(np.isfinite(times)):
        raise ValueError(f"non-finite TMS epoch archive: {path}")
    return epochs, times


def build_tms_epoch_manifest(
    *,
    cohort_labels: str | Path,
    output_root: str | Path,
    study: StudyConfig,
) -> tuple[Path, Path]:
    """Epoch ds005620 pulses, interpolate before filtering, and reject blind to condition."""

    try:
        import mne
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("install neural-manifolds[eeg]") from exc
    labels = pd.read_parquet(cohort_labels)
    selected = labels[
        (labels["dataset_id"] == "propofol_tms_eeg") & (labels["modality"] == "tms-eeg")
    ]
    if selected.empty:
        raise RuntimeError("cohort manifest contains no ds005620 TMS-EEG units")
    destination = Path(output_root)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for row in selected.to_dict(orient="records"):
        try:
            raw = read_raw_recording(row["source_path"])
            raw.load_data()
            events, event_id = mne.events_from_annotations(raw, verbose="ERROR")
            pulse_codes = [
                code
                for description, code in event_id.items()
                if "r128" in description.lower().replace(" ", "")
            ]
            if not pulse_codes:
                raise ValueError("no audited Response/R128 pulse marker found")
            pulse_events = events[np.isin(events[:, 2], pulse_codes)]
            raw.pick("eeg")
            rename = {name: canonicalize_channel_name(name) for name in raw.ch_names}
            raw.rename_channels(rename)
            keep = [name for name in study.preprocessing.canonical_channels if name in raw.ch_names]
            if len(keep) < study.preprocessing.minimum_canonical_channels:
                raise ValueError(f"only {len(keep)} canonical TMS channels")
            raw.pick(keep)
            pulse_samples = pulse_events[:, 0].astype(np.int64) - int(raw.first_samp)
            interpolated = interpolate_continuous_pulses(
                raw.get_data(),
                pulse_samples,
                float(raw.info["sfreq"]),
                start_seconds=-0.005,
                stop_seconds=0.015,
            )
            clean = mne.io.RawArray(
                interpolated,
                raw.info.copy(),
                first_samp=int(raw.first_samp),
                verbose="ERROR",
            )
            clean.set_annotations(raw.annotations.copy())
            bad = detect_bad_channels(
                interpolated,
                float(clean.info["sfreq"]),
            )
            if (
                len(bad.bad_indices) / len(keep)
                > study.preprocessing.maximum_interpolation_fraction
            ):
                raise ValueError("too many bad TMS channels for interpolation")
            clean.info["bads"] = [clean.ch_names[index] for index in bad.bad_indices]
            clean.set_montage("standard_1020", on_missing="warn", verbose="ERROR")
            if clean.info["bads"]:
                clean.interpolate_bads(reset_bads=True, verbose="ERROR")
            clean.set_eeg_reference("average", projection=False, verbose="ERROR")
            clean.filter(
                l_freq=study.preprocessing.highpass_hz,
                h_freq=study.preprocessing.lowpass_hz,
                method="fir",
                phase="zero-double",
                verbose="ERROR",
            )
            notch = infer_mains_frequency(raw)
            if notch < float(clean.info["sfreq"]) / 2:
                clean.notch_filter([notch], verbose="ERROR")
            clean, pulse_events = clean.resample(
                study.preprocessing.target_sampling_hz,
                events=pulse_events,
                verbose="ERROR",
            )
            epochs = mne.Epochs(
                clean,
                pulse_events,
                event_id=None,
                tmin=-0.5,
                tmax=1.0,
                baseline=None,
                preload=True,
                reject_by_annotation=True,
                verbose="ERROR",
            )
            trial_artifact = detect_artifact_windows(
                epochs.get_data(copy=True), float(epochs.info["sfreq"])
            )
            accepted = epochs.get_data(copy=True)[trial_artifact.keep]
            archive = destination / "epochs" / f"{row['unit_id']}.npz"
            archive.parent.mkdir(parents=True, exist_ok=True)
            temporary = archive.with_name(f".{archive.name}.tmp")
            with temporary.open("wb") as stream:
                np.savez_compressed(
                    stream,
                    epochs=accepted.astype(np.float32),
                    times_seconds=epochs.times.astype(np.float64),
                )
            os.replace(temporary, archive)
            rows.append(
                {
                    "unit_id": row["unit_id"],
                    "participant_id": row["participant_id"],
                    "condition": row["condition"],
                    "epochs_path": str(archive),
                    "epochs_sha256": sha256_file(archive),
                    "pulse_trials_total": len(epochs),
                    "pulse_trials_retained": len(accepted),
                    "pulse_trials_rejected": int(np.count_nonzero(~trial_artifact.keep)),
                }
            )
        except (ValueError, RuntimeError, OSError) as error:
            failures.append(
                {
                    "unit_id": row["unit_id"],
                    "error": f"{type(error).__name__}: {error}",
                }
            )
    if not rows:
        raise RuntimeError("all TMS epoch preparations failed")
    manifest_path = _atomic_parquet(pd.DataFrame(rows), destination / "tms-epoch-manifest.parquet")
    audit_path = destination / "tms-epoch-audit.json"
    atomic_write_json(
        audit_path,
        {
            "schema_version": 1,
            "pulse_marker": "Response/R128",
            "pulse_interpolation_before_filtering_seconds": [-0.005, 0.015],
            "pulse_interpolation_domain": "continuous_eeg_before_filtering_and_epoching",
            "epoch_seconds": [-0.5, 1.0],
            "failures": failures,
        },
    )
    return manifest_path, audit_path


def run_tms_validation(
    *,
    tms_manifest: str | Path,
    profiles_path: str | Path,
    output_root: str | Path,
    study: StudyConfig,
) -> tuple[Path, Path, Path, Path]:
    """Estimate direct outcomes and relate them to passive reachability."""

    manifest = pd.read_parquet(tms_manifest)
    required = {"participant_id", "condition", "epochs_path"}
    missing = required.difference(manifest.columns)
    if missing:
        raise ValueError(f"TMS manifest is missing {sorted(missing)}")
    outcomes: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for participant, participant_frame in manifest.groupby("participant_id"):
        loaded: dict[str, np.ndarray] = {}
        times: np.ndarray | None = None
        try:
            for row in participant_frame.to_dict(orient="records"):
                condition = str(row["condition"])
                epochs, current_times = _load_epochs(Path(str(row["epochs_path"])))
                if times is None:
                    times = current_times
                elif times.shape != current_times.shape or not np.allclose(times, current_times):
                    raise ValueError("TMS conditions use different time grids")
                loaded[condition] = epochs
            if len(loaded) < 2 or times is None:
                raise ValueError("participant has fewer than two TMS conditions")
            trial_count = min(value.shape[0] for value in loaded.values())
            if trial_count < study.preprocessing.minimum_event_trials_per_condition:
                raise ValueError(
                    f"only {trial_count} matched trials; need "
                    f"{study.preprocessing.minimum_event_trials_per_condition}"
                )
            rng = np.random.default_rng(
                study.random_seeds[
                    int.from_bytes(str(participant).encode(), "little") % len(study.random_seeds)
                ]
            )
            matched = {
                name: value[np.sort(rng.choice(value.shape[0], trial_count, replace=False))]
                for name, value in loaded.items()
            }
            trajectories = fit_shared_perturbational_trajectories(
                matched,
                times,
                baseline=(-0.5, -0.05),
                sample_step_seconds=0.005,
                rank=study.representation.dynamics_rank,
                random_state=study.random_seeds[0],
            )
            for condition, trajectory in trajectories.items():
                trajectory_times = np.linspace(
                    float(times[0]), float(times[-1]), len(trajectory.mean_trajectory)
                )
                displacement = np.linalg.norm(
                    trajectory.mean_trajectory - trajectory.baseline_centroid, axis=1
                )
                trajectory_rows.extend(
                    {
                        "participant_id": participant,
                        "condition": condition,
                        "time_ms": float(time * 1000.0),
                        "trajectory_value": float(value),
                    }
                    for time, value in zip(trajectory_times, displacement, strict=True)
                )
                outcomes.append(
                    {
                        "participant_id": participant,
                        "condition": condition,
                        "matched_trials": trial_count,
                        **trajectory_outcomes(trajectory, times, post_interval=(0.015, 1.0)),
                    }
                )
        except (ValueError, RuntimeError, OSError, np.linalg.LinAlgError) as error:
            failures.append(
                {
                    "participant_id": participant,
                    "error": f"{type(error).__name__}: {error}",
                }
            )
    if not outcomes:
        raise RuntimeError("no participant produced valid TMS outcomes")
    outcome_frame = pd.DataFrame(outcomes)
    profiles = pd.read_parquet(profiles_path)
    if "condition" not in profiles:
        raise ValueError("profile table has no condition for TMS linkage")
    required_profile_fields = {"dataset_id", "modality", "acquisition"}
    missing_profile_fields = required_profile_fields.difference(profiles.columns)
    if missing_profile_fields:
        raise ValueError(
            "profile table cannot prove a passive TMS linkage; missing "
            f"{sorted(missing_profile_fields)}"
        )
    passive_rows = profiles[
        (profiles["dataset_id"] == "propofol_tms_eeg")
        & (profiles["modality"] == "eeg")
        & (profiles["acquisition"] != "tms")
    ].copy()
    if passive_rows.empty:
        raise RuntimeError("no non-TMS ds005620 profiles are available for passive reachability")
    passive = passive_rows.groupby(["participant_id", "condition"], as_index=False)[
        "reachability"
    ].mean()
    linked = outcome_frame.merge(passive, on=["participant_id", "condition"], how="left")
    associations: list[dict[str, Any]] = []
    for outcome in (
        "maximum_displacement",
        "occupied_log_volume",
        "spatial_differentiation",
        "recovery_half_time_seconds",
    ):
        paired = (
            linked[linked["condition"].isin(["awake", "propofol_sedation"])]
            .groupby(["participant_id", "condition"], as_index=False)[["reachability", outcome]]
            .mean()
        )
        reachability_wide = paired.pivot(
            index="participant_id", columns="condition", values="reachability"
        )
        outcome_wide = paired.pivot(index="participant_id", columns="condition", values=outcome)
        needed = {"awake", "propofol_sedation"}
        if needed <= set(reachability_wide) and needed <= set(outcome_wide):
            deltas = pd.DataFrame(
                {
                    "passive_delta": (
                        reachability_wide["awake"] - reachability_wide["propofol_sedation"]
                    ),
                    "direct_delta": outcome_wide["awake"] - outcome_wide["propofol_sedation"],
                }
            ).dropna()
        else:
            deltas = pd.DataFrame(columns=["passive_delta", "direct_delta"])
        if len(deltas) >= 5:
            coefficient, p_value = stats.spearmanr(deltas["passive_delta"], deltas["direct_delta"])
            associations.append(
                {
                    "predictor": "passive_reachability",
                    "outcome": outcome,
                    "estimate": float(coefficient),
                    "p_value": float(p_value),
                    "n_participants": len(deltas),
                    "contrast": "awake_minus_propofol_sedation",
                    "test": "spearman_participant_level_within_condition_delta",
                }
            )
    destination = Path(output_root)
    outcomes_path = _atomic_parquet(linked, destination / "tms-outcomes.parquet")
    associations_path = _atomic_parquet(
        pd.DataFrame(associations), destination / "tms-associations.parquet"
    )
    trajectory_path = _atomic_parquet(
        pd.DataFrame(trajectory_rows), destination / "tms-trajectory.parquet"
    )
    audit_path = destination / "tms-audit.json"
    atomic_write_json(
        audit_path,
        {
            "schema_version": 1,
            "tms_manifest_sha256": sha256_file(tms_manifest),
            "profiles_sha256": sha256_file(profiles_path),
            "pulse_interpolation_seconds": [-0.005, 0.015],
            "pulse_interpolation_repeated_after_filtering": False,
            "trajectory_step_seconds": 0.005,
            "participants_attempted": int(manifest["participant_id"].nunique()),
            "participants_completed": int(outcome_frame["participant_id"].nunique()),
            "passive_profile_rule": {
                "dataset_id": "propofol_tms_eeg",
                "modality": "eeg",
                "acquisition_excluded": "tms",
            },
            "association_unit": "participant_awake_minus_propofol_sedation_delta",
            "failures": failures,
            "pcist_status": "not_computed_without_a_validated_pcist_backend",
            "scientific_gate_applied": False,
        },
    )
    return outcomes_path, associations_path, trajectory_path, audit_path
