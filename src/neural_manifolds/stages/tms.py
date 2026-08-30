"""Direct TMS perturbational validation with participant-level linkage."""

from __future__ import annotations

import json
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
    conventional_tms_eeg_outcomes,
    early_post_pulse_burden,
    fit_shared_perturbational_trajectories,
    interpolate_continuous_pulses,
    trajectory_outcomes,
)
from neural_manifolds.provenance import atomic_write_json, sha256_file
from neural_manifolds.stage_processing import infer_mains_frequency, read_raw_recording
from neural_manifolds.tms_separation import direct_tms_mask

DIRECT_TRAJECTORY_OUTCOMES = (
    "maximum_displacement",
    "occupied_log_volume",
    "spatial_differentiation",
    "recovery_half_time_seconds",
)
CONVENTIONAL_TMS_OUTCOMES = (
    "tep_peak_global_field_power_uv",
    "tep_peak_global_field_power_latency_seconds",
    "tep_mean_global_field_power_uv",
    "tep_global_field_power_auc_uv_seconds",
    "tep_mean_absolute_amplitude_uv",
    "sensor_spread_fraction",
    "sensor_propagation_latency_range_seconds",
    "sensor_propagation_latency_iqr_seconds",
)


def _atomic_parquet(frame: pd.DataFrame, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, destination)
    return destination


def _load_epochs(
    path: Path,
    *,
    expected_sha256: str,
    expected_channel_order: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    observed_sha256 = sha256_file(path.resolve(strict=True))
    if observed_sha256 != expected_sha256:
        raise ValueError(
            f"TMS epoch archive hash mismatch for {path}: "
            f"expected {expected_sha256}, observed {observed_sha256}"
        )
    with np.load(path.resolve(strict=True), allow_pickle=False) as archive:
        epochs = np.asarray(archive["epochs"], dtype=np.float64)
        times = np.asarray(archive["times_seconds"], dtype=np.float64)
        channel_names = np.asarray(archive["channel_names"])
    if epochs.ndim != 3 or epochs.shape[2] != times.size:
        raise ValueError(f"invalid TMS epoch archive: {path}")
    if channel_names.ndim != 1 or channel_names.size != epochs.shape[1]:
        raise ValueError(f"invalid TMS channel order in epoch archive: {path}")
    channels = tuple(str(value) for value in channel_names.tolist())
    if not channels or any(not value for value in channels) or len(set(channels)) != len(channels):
        raise ValueError(f"empty or duplicate TMS channel names in epoch archive: {path}")
    if channels != expected_channel_order:
        raise ValueError(
            f"TMS channel order mismatch for {path}: "
            f"expected {expected_channel_order}, observed {channels}"
        )
    if not np.all(np.isfinite(epochs)) or not np.all(np.isfinite(times)):
        raise ValueError(f"non-finite TMS epoch archive: {path}")
    if times.ndim != 1 or times.size < 2 or np.any(np.diff(times) <= 0):
        raise ValueError(f"invalid TMS epoch time grid: {path}")
    return epochs, times, channels


def _auxiliary_channel_inventory(raw: Any) -> dict[str, Any]:
    names = [str(value) for value in getattr(raw, "ch_names", [])]
    getter = getattr(raw, "get_channel_types", None)
    if not callable(getter):
        return {
            "auxiliary_channel_status": "unavailable_channel_type_metadata",
            "auxiliary_channel_inventory_json": json.dumps({}, separators=(",", ":")),
            "ica_auxiliary_support_status": "unavailable_channel_type_metadata",
        }
    try:
        channel_types = [str(value).lower() for value in getter()]
    except (RuntimeError, ValueError, TypeError) as error:
        return {
            "auxiliary_channel_status": f"unavailable_channel_type_error:{type(error).__name__}",
            "auxiliary_channel_inventory_json": json.dumps({}, separators=(",", ":")),
            "ica_auxiliary_support_status": "unavailable_channel_type_metadata",
        }
    if len(channel_types) != len(names):
        raise ValueError("raw channel type inventory does not align with channel names")
    auxiliary = {
        kind: [
            name
            for name, channel_type in zip(names, channel_types, strict=True)
            if channel_type == kind
        ]
        for kind in ("eog", "emg", "ecg")
    }
    present = any(auxiliary.values())
    ica_support = bool(auxiliary["eog"] or auxiliary["ecg"])
    return {
        "auxiliary_channel_status": (
            "available_auxiliary_channels_present"
            if present
            else "available_metadata_no_auxiliary_channels"
        ),
        "auxiliary_channel_inventory_json": json.dumps(
            auxiliary, sort_keys=True, separators=(",", ":")
        ),
        "ica_auxiliary_support_status": (
            "available_eog_or_ecg_reference"
            if ica_support
            else "unavailable_no_eog_or_ecg_reference"
        ),
    }


def _epoch_selection(epochs: Any, *, event_count: int) -> np.ndarray:
    declared = getattr(epochs, "selection", None)
    selection = (
        np.arange(len(epochs), dtype=np.int64)
        if declared is None
        else np.asarray(declared, dtype=np.int64)
    )
    if selection.shape != (len(epochs),):
        raise ValueError("TMS epoch selection does not match the retained epoch count")
    if (
        np.any(selection < 0)
        or np.any(selection >= event_count)
        or len(np.unique(selection)) != len(selection)
    ):
        raise ValueError("TMS epoch selection contains invalid or duplicate event indices")
    return selection


def _epoch_drop_reasons(epochs: Any, *, event_count: int) -> list[str]:
    declared = getattr(epochs, "drop_log", None)
    if declared is None:
        return ["unavailable_drop_log_not_exposed"] * event_count
    if len(declared) != event_count:
        raise ValueError("TMS epoch drop log does not match the pulse-event count")
    reasons: list[str] = []
    for entry in declared:
        if not isinstance(entry, tuple) or not all(isinstance(reason, str) for reason in entry):
            raise ValueError("TMS epoch drop log contains an invalid reason entry")
        reasons.append("|".join(entry) if entry else "retained_by_epoch_constructor")
    return reasons


def _channel_quality_value(result: Any, field: str, index: int) -> float:
    values = getattr(result, field, None)
    if values is None:
        return float("nan")
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or index >= len(array):
        raise ValueError(f"TMS channel-quality field {field} is misaligned")
    return float(array[index])


def _participant_delta_associations(linked: pd.DataFrame) -> list[dict[str, Any]]:
    associations: list[dict[str, Any]] = []
    families = {
        **{outcome: "direct_latent_trajectory" for outcome in DIRECT_TRAJECTORY_OUTCOMES},
        **{outcome: "conventional_tep_gfp_or_sensor" for outcome in CONVENTIONAL_TMS_OUTCOMES},
    }
    for outcome, family in families.items():
        if outcome not in linked:
            raise ValueError(f"TMS outcome table lacks declared outcome {outcome}")
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
        estimate = float("nan")
        p_value = float("nan")
        if len(deltas) < 5:
            status = "unavailable_fewer_than_five_complete_participant_deltas"
        elif deltas["passive_delta"].nunique() < 2 or deltas["direct_delta"].nunique() < 2:
            status = "unavailable_zero_variance_participant_delta"
        else:
            coefficient, probability = stats.spearmanr(
                deltas["passive_delta"], deltas["direct_delta"]
            )
            if np.isfinite(coefficient) and np.isfinite(probability):
                estimate = float(coefficient)
                p_value = float(probability)
                status = "available"
            else:
                status = "unavailable_nonfinite_spearman_result"
        associations.append(
            {
                "predictor": "passive_reachability",
                "outcome": outcome,
                "outcome_family": family,
                "estimate": estimate,
                "p_value": p_value,
                "n_participants": len(deltas),
                "contrast": "awake_minus_propofol_sedation",
                "test": "spearman_participant_level_within_condition_delta",
                "status": status,
            }
        )
    return associations


def select_direct_tms_units(cohort_labels: pd.DataFrame) -> pd.DataFrame:
    """Select pulse-aware TMS units while preserving their raw-file lineage."""

    required = {
        "unit_id",
        "participant_id",
        "dataset_id",
        "modality",
        "condition",
        "source_path",
    }
    missing = required.difference(cohort_labels.columns)
    if missing:
        raise ValueError(f"cohort label manifest is missing {sorted(missing)}")
    selected = cohort_labels[
        cohort_labels["dataset_id"].eq("propofol_tms_eeg") & direct_tms_mask(cohort_labels)
    ].copy()
    return selected


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
    selected = select_direct_tms_units(labels)
    if selected.empty:
        raise RuntimeError("cohort manifest contains no ds005620 TMS-EEG units")
    if selected["unit_id"].duplicated().any():
        raise ValueError("cohort manifest contains duplicate direct-TMS unit IDs")
    destination = Path(output_root)
    rows: list[dict[str, Any]] = []
    channel_rows: list[dict[str, Any]] = []
    trial_rows: list[dict[str, Any]] = []
    trial_channel_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for row in selected.to_dict(orient="records"):
        try:
            source_path = Path(str(row["source_path"])).resolve(strict=True)
            source_hash = sha256_file(source_path)
            raw = read_raw_recording(source_path)
            raw.load_data()
            auxiliary = _auxiliary_channel_inventory(raw)
            events, event_id = mne.events_from_annotations(raw, verbose="ERROR")
            pulse_codes = [
                code
                for description, code in event_id.items()
                if "r128" in description.lower().replace(" ", "")
            ]
            if not pulse_codes:
                raise ValueError("no audited Response/R128 pulse marker found")
            pulse_events = events[np.isin(events[:, 2], pulse_codes)]
            pulse_events_source = pulse_events.copy()
            raw.pick("eeg")
            original_eeg_names = tuple(str(value) for value in raw.ch_names)
            rename = {name: canonicalize_channel_name(name) for name in raw.ch_names}
            raw.rename_channels(rename)
            original_by_canonical = {
                canonicalize_channel_name(original): original for original in original_eeg_names
            }
            keep = [name for name in study.preprocessing.canonical_channels if name in raw.ch_names]
            if len(keep) < study.preprocessing.minimum_canonical_channels:
                raise ValueError(f"only {len(keep)} canonical TMS channels")
            raw.pick(keep)
            channel_order = tuple(str(value) for value in raw.ch_names)
            if len(set(channel_order)) != len(channel_order):
                raise ValueError("canonical TMS channel order contains duplicates")
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
            if len(pulse_events) != len(pulse_events_source):
                raise RuntimeError("TMS resampling changed the number of pulse events")
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
            pulse_trials_total = len(pulse_events)
            pulse_trials_after_epoching = len(epochs)
            if pulse_trials_after_epoching > pulse_trials_total:
                raise RuntimeError("TMS epoch count exceeds the pulse-event count")
            epoch_data = epochs.get_data(copy=True)
            burden = early_post_pulse_burden(epoch_data, epochs.times)
            trial_artifact = detect_artifact_windows(epoch_data, float(epochs.info["sfreq"]))
            artifact_keep = np.asarray(trial_artifact.keep, dtype=bool)
            if artifact_keep.shape != (pulse_trials_after_epoching,):
                raise RuntimeError("TMS artifact mask does not match the epoched pulse count")
            selection = _epoch_selection(epochs, event_count=pulse_trials_total)
            drop_reasons = _epoch_drop_reasons(epochs, event_count=pulse_trials_total)
            accepted = epoch_data[artifact_keep]
            pulse_trials_annotation_or_boundary_rejected = (
                pulse_trials_total - pulse_trials_after_epoching
            )
            pulse_trials_artifact_rejected = int(np.count_nonzero(~artifact_keep))
            pulse_trials_rejected = (
                pulse_trials_annotation_or_boundary_rejected + pulse_trials_artifact_rejected
            )
            if pulse_trials_total != len(accepted) + pulse_trials_rejected:
                raise RuntimeError("TMS pulse accounting is not exhaustive")
            local_channel_rows = [
                {
                    "unit_id": row["unit_id"],
                    "participant_id": row["participant_id"],
                    "condition": row["condition"],
                    "channel_index": channel_index,
                    "source_channel_name": original_by_canonical[channel],
                    "canonical_channel_name": channel,
                    "bad_channel_detected": channel_index in set(bad.bad_indices.tolist()),
                    "bad_channel_interpolated": channel_index in set(bad.bad_indices.tolist()),
                    "flat_fraction": _channel_quality_value(bad, "flat_fraction", channel_index),
                    "log_variance_z": _channel_quality_value(bad, "log_variance_z", channel_index),
                    "high_frequency_z": _channel_quality_value(
                        bad, "high_frequency_z", channel_index
                    ),
                    "kurtosis_z": _channel_quality_value(bad, "kurtosis_z", channel_index),
                    "correlation_z": _channel_quality_value(bad, "correlation_z", channel_index),
                }
                for channel_index, channel in enumerate(channel_order)
            ]
            epoch_by_event = {
                int(event_index): epoch_index
                for epoch_index, event_index in enumerate(selection.tolist())
            }
            retained_archive_indices = np.full(pulse_trials_after_epoching, -1, dtype=np.int64)
            retained_archive_indices[artifact_keep] = np.arange(len(accepted), dtype=np.int64)
            local_trial_rows: list[dict[str, Any]] = []
            local_trial_channel_rows: list[dict[str, Any]] = []
            for pulse_index in range(pulse_trials_total):
                epoch_index = epoch_by_event.get(pulse_index)
                if epoch_index is None:
                    status = "annotation_or_boundary_rejected"
                    archive_index: int | None = None
                    summary = {
                        "early_rms_uv_median": float("nan"),
                        "early_to_baseline_rms_ratio_median": float("nan"),
                        "early_derivative_rms_uv_per_second_median": float("nan"),
                        "early_burden_status": "unavailable_trial_not_epoched",
                    }
                else:
                    retained = bool(artifact_keep[epoch_index])
                    status = "retained" if retained else "artifact_rejected"
                    archive_index = int(retained_archive_indices[epoch_index]) if retained else None
                    summary = {
                        "early_rms_uv_median": float(np.median(burden.early_rms_uv[epoch_index])),
                        "early_to_baseline_rms_ratio_median": float(
                            np.median(burden.early_to_baseline_rms_ratio[epoch_index])
                        ),
                        "early_derivative_rms_uv_per_second_median": float(
                            np.median(burden.early_derivative_rms_uv_per_second[epoch_index])
                        ),
                        "early_burden_status": "available_not_used_as_scientific_gate",
                    }
                    for channel_index, channel in enumerate(channel_order):
                        local_trial_channel_rows.append(
                            {
                                "unit_id": row["unit_id"],
                                "participant_id": row["participant_id"],
                                "condition": row["condition"],
                                "pulse_index": pulse_index,
                                "epoch_index": epoch_index,
                                "canonical_channel_name": channel,
                                "trial_status": status,
                                "baseline_rms_uv": float(
                                    burden.baseline_rms_uv[epoch_index, channel_index]
                                ),
                                "early_rms_uv": float(
                                    burden.early_rms_uv[epoch_index, channel_index]
                                ),
                                "early_to_baseline_rms_ratio": float(
                                    burden.early_to_baseline_rms_ratio[epoch_index, channel_index]
                                ),
                                "early_derivative_rms_uv_per_second": float(
                                    burden.early_derivative_rms_uv_per_second[
                                        epoch_index, channel_index
                                    ]
                                ),
                            }
                        )
                local_trial_rows.append(
                    {
                        "unit_id": row["unit_id"],
                        "participant_id": row["participant_id"],
                        "condition": row["condition"],
                        "pulse_index": pulse_index,
                        "source_event_sample": int(pulse_events_source[pulse_index, 0]),
                        "resampled_event_sample": int(pulse_events[pulse_index, 0]),
                        "epoch_index": epoch_index,
                        "retained_archive_index": archive_index,
                        "trial_status": status,
                        "epoch_constructor_status": drop_reasons[pulse_index],
                        **summary,
                    }
                )
            archive = destination / "epochs" / f"{row['unit_id']}.npz"
            archive.parent.mkdir(parents=True, exist_ok=True)
            temporary = archive.with_name(f".{archive.name}.tmp")
            with temporary.open("wb") as stream:
                np.savez_compressed(
                    stream,
                    epochs=accepted.astype(np.float32),
                    times_seconds=epochs.times.astype(np.float64),
                    channel_names=np.asarray(channel_order, dtype="U"),
                )
            os.replace(temporary, archive)
            if sha256_file(source_path) != source_hash:
                raise RuntimeError("TMS source changed during epoch preparation")
            channel_rows.extend(local_channel_rows)
            trial_rows.extend(local_trial_rows)
            trial_channel_rows.extend(local_trial_channel_rows)
            rows.append(
                {
                    "unit_id": row["unit_id"],
                    "participant_id": row["participant_id"],
                    "condition": row["condition"],
                    "source_path": str(source_path),
                    "source_sha256": source_hash,
                    "epochs_path": str(archive),
                    "epochs_sha256": sha256_file(archive),
                    "channel_order_json": json.dumps(channel_order, separators=(",", ":")),
                    "channel_count": len(channel_order),
                    "pulse_trials_total": pulse_trials_total,
                    "pulse_trials_retained": len(accepted),
                    "pulse_trials_rejected": pulse_trials_rejected,
                    "pulse_trials_annotation_or_boundary_rejected": (
                        pulse_trials_annotation_or_boundary_rejected
                    ),
                    "pulse_trials_artifact_rejected": pulse_trials_artifact_rejected,
                    "input_eeg_channel_count": len(original_eeg_names),
                    "early_post_pulse_interval_seconds": "[0.02,0.05]",
                    "early_post_pulse_rms_uv_median": float(np.median(burden.early_rms_uv)),
                    "early_post_pulse_derivative_rms_uv_per_second_median": float(
                        np.median(burden.early_derivative_rms_uv_per_second)
                    ),
                    **auxiliary,
                    "auxiliary_channels_used_for_cleaning": False,
                    "executed_preprocessing_steps_json": json.dumps(
                        [
                            "continuous_pulse_gap_interpolation",
                            "deterministic_bad_channel_detection",
                            "bad_channel_interpolation_when_flagged",
                            "average_reference",
                            "zero_phase_fir_bandpass",
                            "mains_notch_when_below_nyquist",
                            "resample",
                            "epoch",
                            "deterministic_trial_artifact_detection",
                        ],
                        separators=(",", ":"),
                    ),
                    "ica_dependency_status": "available_mne",
                    "ica_executed": False,
                    "ica_execution_status": (
                        "not_executed_no_validated_two_pass_component_selection"
                    ),
                    "ssp_executed": False,
                    "ssp_dependency_status": "available_mne",
                    "ssp_execution_status": ("not_executed_no_validated_tms_projector_definition"),
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
    channel_provenance_path = _atomic_parquet(
        pd.DataFrame(channel_rows), destination / "tms-channel-provenance.parquet"
    )
    trial_provenance_path = _atomic_parquet(
        pd.DataFrame(trial_rows), destination / "tms-trial-provenance.parquet"
    )
    trial_channel_qc_path = _atomic_parquet(
        pd.DataFrame(trial_channel_rows), destination / "tms-trial-channel-qc.parquet"
    )
    audit_path = destination / "tms-epoch-audit.json"
    atomic_write_json(
        audit_path,
        {
            "schema_version": 1,
            "cohort_labels_sha256": sha256_file(cohort_labels),
            "pulse_marker": "Response/R128",
            "pulse_interpolation_before_filtering_seconds": [-0.005, 0.015],
            "pulse_interpolation_domain": "continuous_eeg_before_filtering_and_epoching",
            "epoch_seconds": [-0.5, 1.0],
            "general_encoder_status": "omitted_requires_dedicated_tms_preprocessing",
            "executed_cleaning_scope": (
                "pulse_gap_interpolation_bad_channel_handling_average_reference_"
                "bandpass_notch_resample_and_label_blind_trial_rejection"
            ),
            "auxiliary_channels_used_for_cleaning": False,
            "ica_status": "not_implemented",
            "ica_dependency_status": "available_mne",
            "ica_execution_status": ("not_executed_no_validated_two_pass_component_selection"),
            "ssp_dependency_status": "available_mne",
            "ssp_execution_status": "not_executed_no_validated_tms_projector_definition",
            "auxiliary_channel_status_by_run": sorted(
                {str(row["auxiliary_channel_status"]) for row in rows}
            ),
            "epoch_archives_include_channel_order": True,
            "pulse_accounting": (
                "total=retained+annotation_or_boundary_rejected+artifact_rejected"
            ),
            "raw_lineage_retained": True,
            "per_run_provenance_rows": len(rows),
            "channel_provenance": {
                "path": str(channel_provenance_path),
                "sha256": sha256_file(channel_provenance_path),
                "rows": len(channel_rows),
            },
            "trial_provenance": {
                "path": str(trial_provenance_path),
                "sha256": sha256_file(trial_provenance_path),
                "rows": len(trial_rows),
            },
            "trial_channel_early_burden": {
                "path": str(trial_channel_qc_path),
                "sha256": sha256_file(trial_channel_qc_path),
                "rows": len(trial_channel_rows),
                "interval_seconds": [0.02, 0.05],
                "interpretation": "residual_eeg_muscle_or_artifact_burden_not_source_specific",
                "used_as_scientific_gate": False,
            },
            "failures": failures,
            "scientific_gate_applied": False,
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
    required = {
        "unit_id",
        "participant_id",
        "condition",
        "epochs_path",
        "epochs_sha256",
        "channel_order_json",
        "pulse_trials_total",
        "pulse_trials_retained",
        "pulse_trials_rejected",
    }
    missing = required.difference(manifest.columns)
    if missing:
        raise ValueError(f"TMS manifest is missing {sorted(missing)}")
    if manifest["unit_id"].duplicated().any():
        raise ValueError("TMS manifest contains duplicate unit IDs")
    outcomes: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for participant, participant_frame in manifest.groupby("participant_id"):
        loaded_runs: dict[str, list[np.ndarray]] = {}
        run_counts: dict[str, int] = {}
        run_unit_ids: dict[str, list[str]] = {}
        run_epoch_hashes: dict[str, list[str]] = {}
        run_early_burden: dict[str, list[float]] = {}
        run_auxiliary_statuses: dict[str, list[str]] = {}
        run_ica_statuses: dict[str, list[str]] = {}
        run_ssp_statuses: dict[str, list[str]] = {}
        pulse_accounting: dict[str, dict[str, int]] = {}
        times: np.ndarray | None = None
        channel_order: tuple[str, ...] | None = None
        try:
            ordered_frame = participant_frame.sort_values(["condition", "unit_id"])
            for row in ordered_frame.to_dict(orient="records"):
                condition = str(row["condition"])
                try:
                    declared_channels = json.loads(str(row["channel_order_json"]))
                except json.JSONDecodeError as error:
                    raise ValueError("TMS manifest has invalid channel_order_json") from error
                if not isinstance(declared_channels, list) or not all(
                    isinstance(value, str) and value for value in declared_channels
                ):
                    raise ValueError("TMS manifest channel order must be a non-empty string list")
                expected_channels = tuple(declared_channels)
                epochs, current_times, current_channels = _load_epochs(
                    Path(str(row["epochs_path"])),
                    expected_sha256=str(row["epochs_sha256"]),
                    expected_channel_order=expected_channels,
                )
                total = int(row["pulse_trials_total"])
                retained = int(row["pulse_trials_retained"])
                rejected = int(row["pulse_trials_rejected"])
                if min(total, retained, rejected) < 0:
                    raise ValueError("TMS manifest contains a negative pulse count")
                if total != retained + rejected or retained != epochs.shape[0]:
                    raise ValueError("TMS manifest pulse counts disagree with the epoch archive")
                if times is None:
                    times = current_times
                elif times.shape != current_times.shape or not np.allclose(times, current_times):
                    raise ValueError("TMS runs or conditions use different time grids")
                if channel_order is None:
                    channel_order = current_channels
                elif channel_order != current_channels:
                    raise ValueError("TMS runs or conditions use different channel orders")
                loaded_runs.setdefault(condition, []).append(epochs)
                run_counts[condition] = run_counts.get(condition, 0) + 1
                run_unit_ids.setdefault(condition, []).append(str(row["unit_id"]))
                run_epoch_hashes.setdefault(condition, []).append(str(row["epochs_sha256"]))
                burden_value = row.get("early_post_pulse_rms_uv_median")
                if isinstance(burden_value, (int, float, np.generic)) and np.isfinite(burden_value):
                    run_early_burden.setdefault(condition, []).append(float(burden_value))
                auxiliary_status = row.get("auxiliary_channel_status")
                if isinstance(auxiliary_status, str) and auxiliary_status:
                    run_auxiliary_statuses.setdefault(condition, []).append(auxiliary_status)
                ica_status = row.get("ica_execution_status")
                if isinstance(ica_status, str) and ica_status:
                    run_ica_statuses.setdefault(condition, []).append(ica_status)
                ssp_status = row.get("ssp_execution_status")
                if isinstance(ssp_status, str) and ssp_status:
                    run_ssp_statuses.setdefault(condition, []).append(ssp_status)
                accounting = pulse_accounting.setdefault(
                    condition,
                    {"total": 0, "retained": 0, "rejected": 0},
                )
                accounting["total"] += total
                accounting["retained"] += retained
                accounting["rejected"] += rejected
            loaded = {
                condition: np.concatenate(condition_runs, axis=0)
                for condition, condition_runs in loaded_runs.items()
            }
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
                conventional = conventional_tms_eeg_outcomes(
                    matched[condition],
                    times,
                    baseline=(-0.5, -0.05),
                    post_interval=(0.020, 0.300),
                )
                outcomes.append(
                    {
                        "participant_id": participant,
                        "condition": condition,
                        "matched_trials": trial_count,
                        "source_run_count": run_counts[condition],
                        "source_unit_ids_json": json.dumps(
                            run_unit_ids[condition], separators=(",", ":")
                        ),
                        "source_epoch_sha256s_json": json.dumps(
                            run_epoch_hashes[condition], separators=(",", ":")
                        ),
                        "pulse_trials_total": pulse_accounting[condition]["total"],
                        "pulse_trials_retained": pulse_accounting[condition]["retained"],
                        "pulse_trials_rejected": pulse_accounting[condition]["rejected"],
                        "channel_order_json": json.dumps(channel_order, separators=(",", ":")),
                        "early_post_pulse_rms_uv_run_median": (
                            float(np.median(run_early_burden[condition]))
                            if run_early_burden.get(condition)
                            else float("nan")
                        ),
                        "auxiliary_channel_statuses_json": json.dumps(
                            sorted(set(run_auxiliary_statuses.get(condition, []))),
                            separators=(",", ":"),
                        ),
                        "ica_execution_statuses_json": json.dumps(
                            sorted(set(run_ica_statuses.get(condition, []))),
                            separators=(",", ":"),
                        ),
                        "ssp_execution_statuses_json": json.dumps(
                            sorted(set(run_ssp_statuses.get(condition, []))),
                            separators=(",", ":"),
                        ),
                        **trajectory_outcomes(trajectory, times, post_interval=(0.015, 1.0)),
                        **conventional,
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
    associations = _participant_delta_associations(linked)
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
            "source_runs_attempted": len(manifest),
            "epoch_archive_sha256_verified_before_load": True,
            "repeated_condition_runs_concatenated_before_trial_matching": True,
            "channel_order_required_identical_across_runs_and_conditions": True,
            "pulse_rejection_accounting_verified": True,
            "direct_trajectory_outcomes": list(DIRECT_TRAJECTORY_OUTCOMES),
            "conventional_tep_gfp_sensor_outcomes": list(CONVENTIONAL_TMS_OUTCOMES),
            "conventional_baseline": {
                "baseline_seconds": [-0.5, -0.05],
                "post_seconds": [0.02, 0.3],
                "gfp_definition": "across_sensor_population_standard_deviation_of_mean_tep",
                "sensor_activation_threshold": "baseline_median_plus_3_mad",
                "propagation_scope": "sensor_level_temporal_spread_not_source_or_causal",
            },
            "passive_profile_rule": {
                "dataset_id": "propofol_tms_eeg",
                "modality": "eeg",
                "acquisition_excluded": "tms",
            },
            "association_unit": "participant_awake_minus_propofol_sedation_delta",
            "association_rows_expected": len(DIRECT_TRAJECTORY_OUTCOMES)
            + len(CONVENTIONAL_TMS_OUTCOMES),
            "associations_include_explicit_unavailable_status": True,
            "failures": failures,
            "pcist_status": "not_computed_without_a_validated_pcist_backend",
            "scientific_gate_applied": False,
        },
    )
    return outcomes_path, associations_path, trajectory_path, audit_path
