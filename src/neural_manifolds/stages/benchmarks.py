"""Conventional EEG scalar benchmarks on label-defined analysis units.

The stage emits an explicit status row for every encoding-manifest unit.  It
computes eligible continuous EEG units, marks unsupported units unavailable or
not applicable, and deliberately does not write samples, spectra, connectivity
matrices, or other signal-derived arrays.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from math import factorial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from neural_manifolds.benchmarks import (
    DEFAULT_BANDS,
    FrozenMicrostateModel,
    microstate_peak_maps,
    normalized_lempel_ziv,
    permutation_entropy,
    relative_band_power,
    spectral_exponent,
    weighted_phase_lag_index,
    weighted_symbolic_mutual_information,
)
from neural_manifolds.provenance import atomic_write_json, sha256_file

BAND_POWER_FEATURES = tuple(f"relative_band_power_{name}" for name in DEFAULT_BANDS)
CONVENTIONAL_FEATURES = (
    *BAND_POWER_FEATURES,
    "spectral_exponent",
    "permutation_entropy_median",
    "normalized_lempel_ziv",
    "wpli_mean",
)
MICROSTATE_FEATURES = (
    "microstate_transition_entropy",
    "microstate_global_explained_variance",
    "microstate_median_duration_seconds",
)
WSMI_ORDER = 3
WSMI_LAG_SECONDS = 0.032
WSMI_MINIMUM_SYMBOL_SAMPLES = 5 * (factorial(WSMI_ORDER) ** 2)
UNAVAILABLE_METHODS = {
    "pcist": "unavailable_no_validated_backend",
}
_EVENT_SELECTOR_KINDS = frozenset({"event_epoch", "pre_epoched"})
_PATH_OR_SIGNAL_FIELDS = frozenset(
    {
        "trajectory_path",
        "preprocessed_path",
        "source_path",
        "source_file",
        "events_path",
        "channels_path",
        "selector_json",
    }
)


def _atomic_parquet(frame: pd.DataFrame, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, destination)
    return destination


def _read_raw_fif(path: Path) -> Any:
    try:
        import mne
    except ImportError as exc:  # pragma: no cover - production dependency guard
        raise RuntimeError("install neural-manifolds[eeg] for MNE FIF loading") from exc
    return mne.io.read_raw_fif(path, preload=True, verbose="ERROR")


def _selector_kind(row: Mapping[str, Any]) -> str:
    value = row.get("selector_json")
    if isinstance(value, str) and value:
        try:
            selector = json.loads(value)
        except json.JSONDecodeError as error:
            raise ValueError("selector_json is not valid JSON") from error
        if not isinstance(selector, dict) or not isinstance(selector.get("kind"), str):
            raise ValueError("selector_json has no string kind")
        return str(selector["kind"])
    if isinstance(value, Mapping) and isinstance(value.get("kind"), str):
        return str(value["kind"])
    raise ValueError("encoding manifest row has no audited selector kind")


def _is_true(value: Any) -> bool:
    if value is None:
        return False
    try:
        if bool(pd.isna(value)):
            return False
    except (TypeError, ValueError):
        pass
    return bool(value)


def _safe_metadata(row: Mapping[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in row.items():
        if (
            key in _PATH_OR_SIGNAL_FIELDS
            or key in CONVENTIONAL_FEATURES
            or key.endswith("_path")
            or key.endswith("_file")
        ):
            continue
        if value is None or isinstance(value, (str, int, float, bool, np.generic)):
            output[key] = value.item() if isinstance(value, np.generic) else value
    return output


def _cell_key(row: Mapping[str, Any]) -> str:
    payload = json.dumps(
        [
            str(row.get("dataset_id", "")),
            str(row.get("participant_id", "")),
            str(row.get("condition", "")),
        ],
        ensure_ascii=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _blank_result(
    row: Mapping[str, Any],
    *,
    status: str,
    reason: str,
    cell_expected: bool,
) -> dict[str, Any]:
    return {
        **_safe_metadata(row),
        **{feature: np.nan for feature in CONVENTIONAL_FEATURES},
        "legacy_conventional_status": status,
        "legacy_conventional_reason": reason,
        "wsmi": np.nan,
        "wsmi_status": "unavailable_signal_not_computed",
        "wsmi_reason": reason,
        "wsmi_order": WSMI_ORDER,
        "wsmi_lag_seconds": WSMI_LAG_SECONDS,
        "wsmi_lag_samples": np.nan,
        "wsmi_symbol_samples": np.nan,
        "wsmi_channel_pairs": np.nan,
        "wsmi_lowpass_hz": 10.0,
        "wsmi_minimum_symbol_samples": WSMI_MINIMUM_SYMBOL_SAMPLES,
        "microstates": np.nan,
        **{feature: np.nan for feature in MICROSTATE_FEATURES},
        "microstates_status": "unavailable_signal_not_computed",
        "microstates_reason": reason,
        "pcist": np.nan,
        "pcist_status": UNAVAILABLE_METHODS["pcist"],
        "benchmark_status": status,
        "benchmark_reason": reason,
        "benchmark_cell_expected": cell_expected,
        "participant_condition_cell_key_sha256": _cell_key(row),
    }


def _features(data: np.ndarray, sfreq: float) -> dict[str, float]:
    values = np.asarray(data, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("benchmark data must contain at least two channels")
    if values.shape[1] < max(256, round(4 * sfreq)):
        raise ValueError("benchmark data must contain at least four seconds")
    if not np.all(np.isfinite(values)):
        raise ValueError("benchmark data contain non-finite samples")
    powers = relative_band_power(values, sfreq)
    phase_entropy = np.asarray([permutation_entropy(channel) for channel in values])
    wpli = weighted_phase_lag_index(values, sfreq)
    upper = wpli[np.triu_indices(wpli.shape[0], k=1)]
    if upper.size == 0:
        raise ValueError("wPLI requires at least two channels")
    result = {
        **{f"relative_band_power_{name}": float(value) for name, value in powers.items()},
        "spectral_exponent": spectral_exponent(values, sfreq),
        "permutation_entropy_median": float(np.median(phase_entropy)),
        "normalized_lempel_ziv": normalized_lempel_ziv(values),
        "wpli_mean": float(np.mean(upper)),
    }
    if set(result) != set(CONVENTIONAL_FEATURES) or not np.all(np.isfinite(list(result.values()))):
        raise RuntimeError("conventional benchmark calculation is incomplete or non-finite")
    return result


def _empty_frame() -> pd.DataFrame:
    columns = [
        "unit_id",
        "participant_id",
        "dataset_id",
        "condition",
        *CONVENTIONAL_FEATURES,
        "legacy_conventional_status",
        "legacy_conventional_reason",
        "wsmi",
        "wsmi_status",
        "wsmi_reason",
        "wsmi_order",
        "wsmi_lag_seconds",
        "wsmi_lag_samples",
        "wsmi_symbol_samples",
        "wsmi_channel_pairs",
        "wsmi_lowpass_hz",
        "wsmi_minimum_symbol_samples",
        "microstates",
        *MICROSTATE_FEATURES,
        "microstates_status",
        "microstates_reason",
        "pcist",
        "pcist_status",
        "benchmark_status",
        "benchmark_reason",
        "benchmark_cell_expected",
        "participant_condition_cell_key_sha256",
        "participant_condition_cell_status",
    ]
    return pd.DataFrame(columns=columns)


def _microstates_unavailable(
    rows: list[dict[str, Any]],
    *,
    reason: str,
) -> dict[str, Any]:
    for row in rows:
        if row.get("benchmark_status") == "computed" and row.get("microstates_status") in {
            "pending_frozen_discovery_branch",
            "unavailable_signal_not_computed",
        }:
            row["microstates_status"] = reason
            row["microstates_reason"] = reason
    return {
        "status": "unavailable",
        "reason": reason,
        "per_condition_clustering_performed": False,
        "fit_label_fields": [],
    }


def _run_microstate_branch(
    rows: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    *,
    partition_column_present: bool,
) -> dict[str, Any]:
    if not partition_column_present:
        return _microstates_unavailable(
            rows,
            reason="unavailable_no_representation_partition_in_encoding_manifest",
        )
    if not candidates:
        return _microstates_unavailable(rows, reason="unavailable_no_computed_eeg_units")
    allowed = {
        "representation_discovery",
        "representation_validation",
        "representation_evaluation",
        "not_used_for_representation",
    }
    if any(candidate["partition"] not in allowed for candidate in candidates):
        return _microstates_unavailable(
            rows,
            reason="unavailable_invalid_or_missing_representation_partition",
        )
    participant_partitions: dict[str, set[str]] = {}
    for candidate in candidates:
        participant_partitions.setdefault(candidate["participant_id"], set()).add(
            candidate["partition"]
        )
    if any(len(partitions) != 1 for partitions in participant_partitions.values()):
        return _microstates_unavailable(
            rows,
            reason="unavailable_participant_crosses_representation_partitions",
        )
    active_candidates = [
        candidate
        for candidate in candidates
        if candidate["partition"] != "not_used_for_representation"
    ]
    for candidate in candidates:
        if candidate["partition"] == "not_used_for_representation":
            output = rows[candidate["row_index"]]
            output["microstates_status"] = "not_applicable_not_used_for_representation"
            output["microstates_reason"] = "unit_not_assigned_to_representation_partition"
    if not active_candidates:
        return _microstates_unavailable(
            rows,
            reason="unavailable_no_discovery_validation_or_evaluation_units",
        )
    channel_orders = {candidate["channel_order"] for candidate in active_candidates}
    if len(channel_orders) != 1 or not next(iter(channel_orders)):
        return _microstates_unavailable(
            rows,
            reason="unavailable_inconsistent_or_missing_frozen_channel_order",
        )
    discovery_maps: dict[str, list[np.ndarray]] = {}
    for candidate in active_candidates:
        if candidate["partition"] == "representation_discovery":
            maps = candidate.get("discovery_maps")
            if not isinstance(maps, np.ndarray):
                return _microstates_unavailable(
                    rows,
                    reason="unavailable_discovery_gfp_maps_not_computed",
                )
            discovery_maps.setdefault(candidate["participant_id"], []).append(maps)
    try:
        model = FrozenMicrostateModel(n_states=4).fit(
            {
                participant: np.concatenate(parts, axis=0)
                for participant, parts in discovery_maps.items()
            }
        )
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
        return _microstates_unavailable(
            rows,
            reason=f"unavailable_frozen_discovery_fit_failed:{type(error).__name__}:{error}",
        )
    failures: list[dict[str, str]] = []
    for candidate in active_candidates:
        output = rows[candidate["row_index"]]
        raw = None
        try:
            path = candidate["raw_path"]
            if sha256_file(path) != candidate["raw_sha256"]:
                raise ValueError("preprocessed FIF changed before frozen microstate application")
            raw = _read_raw_fif(path)
            channel_order = tuple(str(value) for value in getattr(raw, "ch_names", ()))
            if channel_order != candidate["channel_order"]:
                raise ValueError("microstate application channel order changed")
            score = model.score(
                np.asarray(raw.get_data(), dtype=np.float64),
                float(raw.info["sfreq"]),
            )
            output.update(score)
            output["microstates"] = score["microstate_transition_entropy"]
            output["microstates_status"] = (
                "available_frozen_discovery_in_sample"
                if candidate["partition"] == "representation_discovery"
                else "available_frozen_out_of_sample"
            )
            output["microstates_reason"] = None
        except (OSError, RuntimeError, ValueError, KeyError, np.linalg.LinAlgError) as error:
            output["microstates_status"] = (
                f"unavailable_frozen_application_failed:{type(error).__name__}"
            )
            output["microstates_reason"] = str(error)
            failures.append(
                {
                    "unit_id": str(candidate["unit_id"]),
                    "error": f"{type(error).__name__}: {error}",
                }
            )
        finally:
            close = getattr(raw, "close", None)
            if callable(close):
                close()
    return {
        **model.audit(),
        "participant_partition_overlap": False,
        "application_failures": failures,
        "per_condition_clustering_performed": False,
        "fit_label_fields": [],
    }


def run_benchmarks(
    *,
    encoding_manifest: str | Path,
    output_root: str | Path,
) -> tuple[Path, Path]:
    """Compute fixed conventional scalar comparators without scientific gating."""

    manifest_path = Path(encoding_manifest).resolve(strict=True)
    frame = pd.read_parquet(manifest_path)
    required = {
        "unit_id",
        "participant_id",
        "dataset_id",
        "condition",
        "encoded",
        "preprocessed_path",
        "selector_json",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"encoding manifest is missing {sorted(missing)}")
    unit_ids = frame["unit_id"].astype(str)
    if unit_ids.eq("").any() or unit_ids.duplicated().any():
        raise ValueError("encoding manifest unit_id values must be non-empty and unique")

    rows: list[dict[str, Any]] = []
    microstate_candidates: list[dict[str, Any]] = []
    wsmi_available = 0
    wsmi_unavailable = 0
    issues: list[dict[str, str]] = []
    skipped = {
        "not_encoded": 0,
        "event_unit": 0,
        "clinical_holdout": 0,
        "secondary_fmri": 0,
        "missing_preprocessed_path": 0,
    }
    for row in frame.to_dict(orient="records"):
        identity = {
            "unit_id": str(row.get("unit_id", "")),
            "participant_id": str(row.get("participant_id", "")),
            "dataset_id": str(row.get("dataset_id", "")),
        }
        encoded = _is_true(row.get("encoded"))
        excluded_clinical = _is_true(row.get("clinical_holdout"))
        excluded_fmri = _is_true(row.get("secondary_fmri")) or row.get("modality") == "fmri"
        cell_expected = encoded and not excluded_clinical and not excluded_fmri
        try:
            if not encoded:
                skipped["not_encoded"] += 1
                rows.append(
                    _blank_result(
                        row,
                        status="not_applicable",
                        reason="not_encoded_for_five_axis_estimand",
                        cell_expected=False,
                    )
                )
                continue
            if _selector_kind(row) in _EVENT_SELECTOR_KINDS or _is_true(
                row.get("event_aggregated")
            ):
                skipped["event_unit"] += 1
                rows.append(
                    _blank_result(
                        row,
                        status="unavailable",
                        reason="event_unit_has_no_valid_four_second_continuous_benchmark_signal",
                        cell_expected=cell_expected,
                    )
                )
                continue
            if excluded_clinical:
                skipped["clinical_holdout"] += 1
                rows.append(
                    _blank_result(
                        row,
                        status="not_applicable",
                        reason="clinical_holdout_excluded_from_healthy_benchmark_fit",
                        cell_expected=False,
                    )
                )
                continue
            if excluded_fmri:
                skipped["secondary_fmri"] += 1
                rows.append(
                    _blank_result(
                        row,
                        status="not_applicable",
                        reason="secondary_fmri_has_no_eeg_benchmark_signal",
                        cell_expected=False,
                    )
                )
                continue
            raw_path_value = row.get("preprocessed_path")
            if not isinstance(raw_path_value, str) or not raw_path_value:
                skipped["missing_preprocessed_path"] += 1
                rows.append(
                    _blank_result(
                        row,
                        status="unavailable",
                        reason="missing_preprocessed_fif_path",
                        cell_expected=cell_expected,
                    )
                )
                continue
            raw_path = Path(raw_path_value).resolve(strict=True)
            expected_hash = row.get("preprocessed_sha256")
            observed_hash = sha256_file(raw_path)
            if isinstance(expected_hash, str) and expected_hash and observed_hash != expected_hash:
                raise ValueError("preprocessed FIF checksum mismatch")
            raw = _read_raw_fif(raw_path)
            try:
                data = np.asarray(raw.get_data(), dtype=np.float64)
                sfreq = float(raw.info["sfreq"])
                features = _features(data, sfreq)
                n_channels, n_samples = data.shape
                try:
                    wsmi = weighted_symbolic_mutual_information(
                        data,
                        sfreq,
                        order=WSMI_ORDER,
                        lag_seconds=WSMI_LAG_SECONDS,
                    )
                    wsmi_fields = {
                        "wsmi": wsmi.value,
                        "wsmi_status": "available_validated_deterministic",
                        "wsmi_reason": None,
                        "wsmi_order": wsmi.order,
                        "wsmi_lag_seconds": wsmi.lag_samples / sfreq,
                        "wsmi_lag_samples": wsmi.lag_samples,
                        "wsmi_symbol_samples": wsmi.symbol_samples,
                        "wsmi_channel_pairs": wsmi.channel_pairs,
                        "wsmi_lowpass_hz": wsmi.lowpass_hz,
                        "wsmi_minimum_symbol_samples": wsmi.minimum_symbol_samples,
                    }
                    wsmi_available += 1
                except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
                    wsmi_fields = {
                        "wsmi": np.nan,
                        "wsmi_status": f"unavailable:{type(error).__name__}",
                        "wsmi_reason": str(error),
                        "wsmi_order": WSMI_ORDER,
                        "wsmi_lag_seconds": WSMI_LAG_SECONDS,
                        "wsmi_lag_samples": max(1, round(WSMI_LAG_SECONDS * sfreq)),
                        "wsmi_symbol_samples": np.nan,
                        "wsmi_channel_pairs": int(n_channels * (n_channels - 1) / 2),
                        "wsmi_lowpass_hz": 10.0,
                        "wsmi_minimum_symbol_samples": WSMI_MINIMUM_SYMBOL_SAMPLES,
                    }
                    wsmi_unavailable += 1
                channel_order = tuple(str(value) for value in getattr(raw, "ch_names", ()))
                partition_value = row.get("representation_partition")
                partition = partition_value if isinstance(partition_value, str) else ""
                discovery_maps: np.ndarray | None = None
                if partition == "representation_discovery" and channel_order:
                    try:
                        discovery_maps = microstate_peak_maps(data, sfreq)
                    except (ValueError, RuntimeError, np.linalg.LinAlgError):
                        discovery_maps = None
            finally:
                close = getattr(raw, "close", None)
                if callable(close):
                    close()
            result = {
                **_blank_result(
                    row,
                    status="computed",
                    reason="all_legacy_conventional_features_available",
                    cell_expected=cell_expected,
                ),
                **features,
                "n_benchmark_channels": int(n_channels),
                "n_benchmark_samples": int(n_samples),
                "benchmark_duration_seconds": float(n_samples / sfreq),
                "legacy_conventional_status": "available",
                "legacy_conventional_reason": None,
                **wsmi_fields,
                "microstates_status": "pending_frozen_discovery_branch",
                "microstates_reason": None,
                "benchmark_status": "computed",
                "benchmark_reason": None,
            }
            row_index = len(rows)
            rows.append(result)
            microstate_candidates.append(
                {
                    "row_index": row_index,
                    "unit_id": identity["unit_id"],
                    "participant_id": identity["participant_id"],
                    "partition": partition,
                    "raw_path": raw_path,
                    "raw_sha256": observed_hash,
                    "channel_order": channel_order,
                    "discovery_maps": discovery_maps,
                }
            )
        except (OSError, RuntimeError, ValueError, KeyError) as error:
            issues.append(
                {
                    **identity,
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            rows.append(
                _blank_result(
                    row,
                    status="unavailable",
                    reason=f"{type(error).__name__}: {error}",
                    cell_expected=cell_expected,
                )
            )

    microstate_audit = _run_microstate_branch(
        rows,
        microstate_candidates,
        partition_column_present="representation_partition" in frame,
    )
    cell_groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if bool(row.get("benchmark_cell_expected")):
            cell_groups.setdefault(str(row["participant_condition_cell_key_sha256"]), []).append(
                row
            )
        else:
            row["participant_condition_cell_status"] = "not_applicable"
    complete_cells: list[str] = []
    incomplete_cells: list[str] = []
    for key, cell_rows in cell_groups.items():
        complete = all(row.get("benchmark_status") == "computed" for row in cell_rows)
        status = "complete_all_expected_units" if complete else "incomplete_expected_units"
        for row in cell_rows:
            row["participant_condition_cell_status"] = status
        (complete_cells if complete else incomplete_cells).append(key)
    destination = Path(output_root)
    output_frame = pd.DataFrame(rows) if rows else _empty_frame()
    benchmark_path = _atomic_parquet(output_frame, destination / "benchmarks.parquet")
    audit_path = destination / "benchmark-audit.json"
    atomic_write_json(
        audit_path,
        {
            "schema_version": 1,
            "encoding_manifest_sha256": sha256_file(manifest_path),
            "manifest_rows": len(frame),
            "published_rows": len(rows),
            "completed_rows": sum(row.get("benchmark_status") == "computed" for row in rows),
            "unavailable_rows": sum(row.get("benchmark_status") == "unavailable" for row in rows),
            "not_applicable_rows": sum(
                row.get("benchmark_status") == "not_applicable" for row in rows
            ),
            "failed_rows": len(issues),
            "skipped_rows": skipped,
            "issues": issues,
            "published_features": [
                *CONVENTIONAL_FEATURES,
                "wsmi",
                "microstates",
                *MICROSTATE_FEATURES,
            ],
            "legacy_conventional_prediction_features": list(CONVENTIONAL_FEATURES),
            "wsmi": {
                "status": "implemented_validated_deterministic",
                "order": WSMI_ORDER,
                "lag_seconds": WSMI_LAG_SECONDS,
                "excluded_pair_weights": [
                    "identical_patterns",
                    "sign_reversed_patterns",
                ],
                "channel_pair_aggregation": "median_upper_triangle",
                "lowpass_hz": 10.0,
                "lowpass": "fourth_order_zero_phase_butterworth",
                "minimum_symbol_samples": WSMI_MINIMUM_SYMBOL_SAMPLES,
                "available_rows": wsmi_available,
                "unavailable_rows": wsmi_unavailable,
            },
            "microstates": microstate_audit,
            "unavailable_methods": UNAVAILABLE_METHODS,
            "analysis_unit": "participant_condition_analysis_unit",
            "participant_condition_cell_contract": {
                "expected_cells": len(cell_groups),
                "complete_cells": len(complete_cells),
                "incomplete_cells": len(incomplete_cells),
                "complete_cell_keys_sha256": sorted(complete_cells),
                "incomplete_cell_keys_sha256": sorted(incomplete_cells),
                "conventional_prediction_status": (
                    "ready_exact_encoding_manifest_cells"
                    if not incomplete_cells
                    else "unavailable_requires_consumer_fail_closed_on_incomplete_cells"
                ),
                "consumer_requirement": (
                    "prediction_cell_keys_must_equal_corresponding_five_axis_estimand_cell_keys"
                ),
            },
            "raw_or_array_artifacts_published": False,
            "path_fields_published": False,
            "scientific_gate_applied": False,
        },
    )
    return benchmark_path, audit_path
