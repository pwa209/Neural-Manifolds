"""Participant-level inference for the secondary BrainLM fMRI triangulation.

All signal encoding and discovery-only axis calibration happen before this
module consumes condition or timing labels.  Runs are first collapsed within
participant-condition cells; resampling then treats participants, never runs or
windows, as the independent observations.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

AXES = ("R", "M", "D", "A")
PARTITIONS = ("discovery", "validation", "test")
MINIMUM_PAIRED_PARTICIPANTS = 2

CONTRASTS: tuple[dict[str, str], ...] = (
    {
        "contrast_id": "propofol_vs_wake",
        "positive_arm": "propofol",
        "negative_arm": "wake",
        "definition": (
            "verified positive effect-site propofol exposure minus verified "
            "zero-exposure healthy-wake reference"
        ),
    },
    {
        "contrast_id": "post_lor_vs_post_ror",
        "positive_arm": "post_lor_unresponsive",
        "negative_arm": "post_ror_responsive",
        "definition": (
            "explicit timing-table post-LOR unresponsive segment minus explicit "
            "timing-table post-ROR responsive segment"
        ),
    },
)

LABEL_FIELDS_CONSUMED = (
    "condition",
    "task",
    "run_id",
    "metadata_status",
    "healthy_wake_reference",
    "effect_site_concentration_min",
    "effect_site_concentration_mean",
    "effect_site_concentration_max",
    "lor_tr_csv",
    "ror_tr_csv",
    "lor_volume",
    "ror_volume",
    "timing_index_origin",
    "volume_start",
    "volume_stop",
)


class FMRIInferenceError(ValueError):
    """Raised when participant-level inference would violate its frozen contract."""


@dataclass(frozen=True)
class FMRIPairedInference:
    paired_differences: pd.DataFrame
    estimates: pd.DataFrame
    ledger: dict[str, Any]


def _present(value: object) -> bool:
    return value is not None and not pd.isna(value) and str(value) != ""


def _integer(value: object, *, field: str) -> int:
    if not _present(value):
        raise FMRIInferenceError(f"explicit timing condition is missing {field}")
    number = float(value)
    if not np.isfinite(number) or number != np.floor(number):
        raise FMRIInferenceError(f"explicit timing field {field} must be an integer")
    return int(number)


def _run_number(value: object) -> int | None:
    if not _present(value):
        return None
    number = float(value)
    if not np.isfinite(number) or number != np.floor(number):
        raise FMRIInferenceError("run_id must be an integer for timing-labelled fMRI units")
    return int(number)


def _inference_condition(row: dict[str, Any]) -> str:
    """Resolve timing-sensitive conditions only from explicit audited labels."""

    condition = str(row.get("condition", ""))
    timing_conditions = {
        "responsive_induction",
        "behaviorally_unresponsive",
        "responsive_recovery",
    }
    if condition not in timing_conditions:
        return condition
    if str(row.get("metadata_status")) != "verified":
        raise FMRIInferenceError(
            f"timing-sensitive condition {condition!r} must have metadata_status='verified'"
        )
    if str(row.get("task")) != "imagery":
        raise FMRIInferenceError(f"timing-sensitive condition {condition!r} must be imagery")
    run = _run_number(row.get("run_id"))
    origin = _integer(row.get("timing_index_origin"), field="timing_index_origin")
    if origin not in {0, 1}:
        raise FMRIInferenceError("timing_index_origin must be exactly 0 or 1")
    start = _integer(row.get("volume_start"), field="volume_start")
    stop = _integer(row.get("volume_stop"), field="volume_stop")
    if condition == "responsive_induction":
        if run != 2:
            raise FMRIInferenceError("responsive_induction must originate from imagery run 2")
        boundary = _integer(row.get("lor_volume"), field="lor_volume")
        _integer(row.get("lor_tr_csv"), field="lor_tr_csv")
        if stop != boundary:
            raise FMRIInferenceError("responsive_induction does not end at the explicit LOR")
        return "pre_lor_responsive"
    if condition == "behaviorally_unresponsive" and run == 2:
        boundary = _integer(row.get("lor_volume"), field="lor_volume")
        _integer(row.get("lor_tr_csv"), field="lor_tr_csv")
        if start != boundary:
            raise FMRIInferenceError("post-LOR segment does not start at the explicit LOR")
        return "post_lor_unresponsive"
    if condition == "behaviorally_unresponsive" and run == 3:
        boundary = _integer(row.get("ror_volume"), field="ror_volume")
        _integer(row.get("ror_tr_csv"), field="ror_tr_csv")
        if stop != boundary:
            raise FMRIInferenceError("pre-ROR segment does not end at the explicit ROR")
        return "pre_ror_unresponsive"
    if condition == "responsive_recovery":
        if run != 3:
            raise FMRIInferenceError("responsive_recovery must originate from imagery run 3")
        boundary = _integer(row.get("ror_volume"), field="ror_volume")
        _integer(row.get("ror_tr_csv"), field="ror_tr_csv")
        if start != boundary:
            raise FMRIInferenceError("post-ROR segment does not start at the explicit ROR")
        return "post_ror_responsive"
    raise FMRIInferenceError(
        "behaviorally_unresponsive timing label must originate from imagery run 2 or 3"
    )


def _single_or_missing(group: pd.DataFrame, column: str) -> Any:
    if column not in group:
        return None
    values = [value for value in group[column].tolist() if _present(value)]
    unique = pd.unique(pd.Series(values, dtype="object"))
    if len(unique) > 1:
        raise FMRIInferenceError(
            f"participant-condition cell has inconsistent {column}: {unique.tolist()}"
        )
    return unique[0] if len(unique) else None


def collapse_runs_within_participant_condition(
    unit_summaries: pd.DataFrame,
    *,
    metric_columns: list[str],
) -> pd.DataFrame:
    """Give each participant-condition one row after equal-weight run collapse."""

    required = {"unit_id", "participant_id", "dataset_id", "partition", "condition"}
    missing = required.difference(unit_summaries.columns)
    if missing:
        raise FMRIInferenceError(f"fMRI unit summaries are missing {sorted(missing)}")
    if not metric_columns or not set(metric_columns) <= set(unit_summaries.columns):
        raise FMRIInferenceError("participant-condition collapse has invalid metric columns")
    if unit_summaries.empty:
        raise FMRIInferenceError("participant-condition collapse received no units")
    frame = unit_summaries.copy()
    if frame["unit_id"].astype(str).duplicated().any():
        raise FMRIInferenceError("fMRI unit summaries contain duplicate unit_id values")
    frame["inference_condition"] = [
        _inference_condition(row) for row in frame.to_dict(orient="records")
    ]
    participant_partitions = frame.groupby("participant_id")["partition"].nunique()
    if (participant_partitions != 1).any():
        leaked = sorted(participant_partitions[participant_partitions != 1].index.astype(str))
        raise FMRIInferenceError(f"participants cross fMRI inference partitions: {leaked}")
    group_columns = [
        "participant_id",
        "dataset_id",
        "partition",
        "condition",
        "inference_condition",
    ]
    frame = frame.sort_values([*group_columns, "unit_id"], kind="stable")
    rows: list[dict[str, Any]] = []
    for identity, group in frame.groupby(group_columns, sort=True, dropna=False):
        metrics = group[metric_columns].apply(pd.to_numeric, errors="coerce")
        if metrics.isna().any().any() or not np.all(np.isfinite(metrics.to_numpy(dtype=float))):
            raise FMRIInferenceError("participant-condition metrics must be finite")
        unit_ids = sorted(group["unit_id"].astype(str))
        row = dict(zip(group_columns, identity, strict=True))
        row.update(metrics.mean(axis=0).to_dict())
        timing_label_verified = str(row["inference_condition"]) in {
            "pre_lor_responsive",
            "post_lor_unresponsive",
            "pre_ror_unresponsive",
            "post_ror_responsive",
        }
        row.update(
            {
                "unit_count": len(group),
                "run_count": (
                    int(group["run_id"].astype(str).nunique()) if "run_id" in group else len(group)
                ),
                "source_unit_ids_sha256": hashlib.sha256(
                    "\0".join(unit_ids).encode("utf-8")
                ).hexdigest(),
                "timing_label_verified": timing_label_verified,
                "timing_label_source": (
                    "official_lor_ror_table_with_explicit_index_origin"
                    if timing_label_verified
                    else None
                ),
                "task": _single_or_missing(group, "task"),
                "metadata_status": _single_or_missing(group, "metadata_status"),
                "healthy_wake_reference": _single_or_missing(group, "healthy_wake_reference"),
                "timing_index_origin": _single_or_missing(group, "timing_index_origin"),
                "lor_tr_csv": _single_or_missing(group, "lor_tr_csv"),
                "ror_tr_csv": _single_or_missing(group, "ror_tr_csv"),
                "lor_volume": _single_or_missing(group, "lor_volume"),
                "ror_volume": _single_or_missing(group, "ror_volume"),
            }
        )
        for column, reducer in (
            ("effect_site_concentration_min", "min"),
            ("effect_site_concentration_mean", "mean"),
            ("effect_site_concentration_max", "max"),
        ):
            if column not in group:
                row[column] = np.nan
                continue
            values = pd.to_numeric(group[column], errors="coerce")
            if values.isna().any() or not np.all(np.isfinite(values.to_numpy(dtype=float))):
                raise FMRIInferenceError(f"participant-condition cell has invalid {column}")
            row[column] = float(getattr(values, reducer)())
        rows.append(row)
    result = pd.DataFrame(rows)
    return result.sort_values(group_columns, kind="stable").reset_index(drop=True)


def _truth(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and value in {0, 1}:
        return bool(value)
    if isinstance(value, str) and value.lower() in {"true", "false"}:
        return value.lower() == "true"
    return False


def _arm(row: dict[str, Any], contrast_id: str) -> str | None:
    if contrast_id == "post_lor_vs_post_ror":
        if (
            not _truth(row.get("timing_label_verified"))
            or row.get("timing_label_source") != "official_lor_ror_table_with_explicit_index_origin"
        ):
            return None
        condition = str(row.get("inference_condition"))
        if condition == "post_lor_unresponsive":
            return "post_lor_unresponsive"
        if condition == "post_ror_responsive":
            return "post_ror_responsive"
        return None
    if contrast_id != "propofol_vs_wake":  # pragma: no cover - frozen definitions
        raise FMRIInferenceError(f"unknown fMRI contrast {contrast_id!r}")
    if str(row.get("metadata_status")) != "verified":
        return None
    maximum = row.get("effect_site_concentration_max")
    if not _present(maximum) or not np.isfinite(float(maximum)) or float(maximum) < 0:
        return None
    if _truth(row.get("healthy_wake_reference")) and np.isclose(float(maximum), 0.0):
        return "wake"
    if not _truth(row.get("healthy_wake_reference")) and float(maximum) > 0.0:
        return "propofol"
    return None


def _stable_seed(base_seed: int, *parts: str) -> int:
    payload = "\0".join((str(base_seed), *parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _empty_paired_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "participant_id",
            "dataset_id",
            "partition",
            "contrast_id",
            "contrast_definition",
            "positive_arm",
            "negative_arm",
            "axis",
            "positive_value",
            "negative_value",
            "difference",
            "positive_condition_cells",
            "negative_condition_cells",
        ]
    )


def infer_paired_condition_contrasts(
    condition_cells: pd.DataFrame,
    *,
    bootstrap_repetitions: int,
    permutation_repetitions: int,
    random_seed: int,
    minimum_pairs: int = MINIMUM_PAIRED_PARTICIPANTS,
) -> FMRIPairedInference:
    """Estimate partition-stratified paired contrasts with participant resampling."""

    required = {"participant_id", "dataset_id", "partition", "inference_condition", *AXES}
    missing = required.difference(condition_cells.columns)
    if missing:
        raise FMRIInferenceError(f"fMRI condition cells are missing {sorted(missing)}")
    if bootstrap_repetitions <= 0 or permutation_repetitions <= 0:
        raise FMRIInferenceError("fMRI resampling repetitions must be positive")
    if minimum_pairs < 2:
        raise FMRIInferenceError("fMRI paired inference requires at least two participants")
    if not set(condition_cells["partition"].astype(str)) <= set(PARTITIONS):
        raise FMRIInferenceError("fMRI inference contains an unknown partition")
    participant_partitions = condition_cells.groupby("participant_id")["partition"].nunique()
    if (participant_partitions != 1).any():
        raise FMRIInferenceError("fMRI paired inference cannot pool participants across splits")
    for axis in AXES:
        values = pd.to_numeric(condition_cells[axis], errors="coerce")
        if values.isna().any() or not np.all(np.isfinite(values.to_numpy(dtype=float))):
            raise FMRIInferenceError(f"fMRI axis {axis} contains non-finite values")

    paired_rows: list[dict[str, Any]] = []
    estimate_rows: list[dict[str, Any]] = []
    pairing_records: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    ordered_cells = condition_cells.sort_values(
        ["participant_id", "partition", "inference_condition"], kind="stable"
    )
    records = ordered_cells.to_dict(orient="records")
    for contrast in CONTRASTS:
        contrast_id = contrast["contrast_id"]
        positive_arm = contrast["positive_arm"]
        negative_arm = contrast["negative_arm"]
        classified: list[dict[str, Any]] = []
        for row in records:
            arm = _arm(row, contrast_id)
            if arm is not None:
                classified.append({**row, "contrast_arm": arm})
        classified_frame = pd.DataFrame(classified)
        for partition in PARTITIONS:
            if classified_frame.empty:
                partition_frame = classified_frame
            else:
                partition_frame = classified_frame[
                    classified_frame["partition"].astype(str).eq(partition)
                ]
            arm_values: dict[tuple[str, str], dict[str, Any]] = {}
            if not partition_frame.empty:
                for identity, group in partition_frame.groupby(
                    ["participant_id", "contrast_arm"], sort=True
                ):
                    participant_id, arm = (str(identity[0]), str(identity[1]))
                    datasets = sorted(set(group["dataset_id"].astype(str)))
                    if len(datasets) != 1:
                        raise FMRIInferenceError(
                            "a paired fMRI participant cannot span multiple datasets"
                        )
                    arm_values[(participant_id, arm)] = {
                        "dataset_id": datasets[0],
                        "condition_cells": len(group),
                        **{axis: float(group[axis].to_numpy(dtype=float).mean()) for axis in AXES},
                    }
            positive_participants = {
                participant for participant, arm in arm_values if arm == positive_arm
            }
            negative_participants = {
                participant for participant, arm in arm_values if arm == negative_arm
            }
            paired_participants = sorted(positive_participants & negative_participants)
            for participant_id in paired_participants:
                positive = arm_values[(participant_id, positive_arm)]
                negative = arm_values[(participant_id, negative_arm)]
                if positive["dataset_id"] != negative["dataset_id"]:
                    raise FMRIInferenceError("paired fMRI arms have inconsistent dataset IDs")
                for axis in AXES:
                    positive_value = float(positive[axis])
                    negative_value = float(negative[axis])
                    paired_rows.append(
                        {
                            "participant_id": participant_id,
                            "dataset_id": positive["dataset_id"],
                            "partition": partition,
                            "contrast_id": contrast_id,
                            "contrast_definition": contrast["definition"],
                            "positive_arm": positive_arm,
                            "negative_arm": negative_arm,
                            "axis": axis,
                            "positive_value": positive_value,
                            "negative_value": negative_value,
                            "difference": positive_value - negative_value,
                            "positive_condition_cells": positive["condition_cells"],
                            "negative_condition_cells": negative["condition_cells"],
                        }
                    )
            status = (
                "available" if len(paired_participants) >= minimum_pairs else "insufficient_pairs"
            )
            pairing_records.append(
                {
                    "contrast_id": contrast_id,
                    "partition": partition,
                    "status": status,
                    "minimum_pairs": minimum_pairs,
                    "positive_participants": sorted(positive_participants),
                    "negative_participants": sorted(negative_participants),
                    "paired_participants": paired_participants,
                    "unpaired_positive_participants": sorted(
                        positive_participants - negative_participants
                    ),
                    "unpaired_negative_participants": sorted(
                        negative_participants - positive_participants
                    ),
                }
            )
            if status != "available":
                issues.append(
                    {
                        "issue_id": f"fmri:{contrast_id}:{partition}:insufficient-pairs",
                        "scope": "fmri_participant_condition_inference",
                        "contrast_id": contrast_id,
                        "partition": partition,
                        "status": "unavailable_insufficient_pairs",
                        "severity": "report_unavailable",
                        "technical_gate": False,
                        "scientific_gate": False,
                        "observed_pairs": len(paired_participants),
                        "required_pairs": minimum_pairs,
                        "message": (
                            "Participant-level paired inference is unavailable in this partition; "
                            "no run- or window-level substitution is permitted."
                        ),
                    }
                )
            for axis in AXES:
                differences = np.asarray(
                    [
                        arm_values[(participant, positive_arm)][axis]
                        - arm_values[(participant, negative_arm)][axis]
                        for participant in paired_participants
                    ],
                    dtype=np.float64,
                )
                observed = float(np.mean(differences)) if differences.size else np.nan
                interval_low = np.nan
                interval_high = np.nan
                p_value = np.nan
                extreme_count: int | None = None
                if status == "available":
                    bootstrap_rng = np.random.default_rng(
                        _stable_seed(random_seed, contrast_id, partition, axis, "bootstrap")
                    )
                    bootstrap_indices = bootstrap_rng.integers(
                        0,
                        differences.size,
                        size=(bootstrap_repetitions, differences.size),
                    )
                    bootstrap = differences[bootstrap_indices].mean(axis=1)
                    interval_low, interval_high = np.quantile(bootstrap, [0.025, 0.975])
                    permutation_rng = np.random.default_rng(
                        _stable_seed(random_seed, contrast_id, partition, axis, "permutation")
                    )
                    signs = permutation_rng.integers(
                        0,
                        2,
                        size=(permutation_repetitions, differences.size),
                    )
                    signs = signs * 2.0 - 1.0
                    null = (signs * differences).mean(axis=1)
                    extreme_count = int(
                        np.count_nonzero(np.abs(null) >= abs(observed) - np.finfo(float).eps)
                    )
                    p_value = (extreme_count + 1.0) / (permutation_repetitions + 1.0)
                estimate_rows.append(
                    {
                        "contrast_id": contrast_id,
                        "contrast_definition": contrast["definition"],
                        "partition": partition,
                        "analysis_role": (
                            "calibration_partition_descriptive"
                            if partition == "discovery"
                            else "held_out_partition_inference"
                        ),
                        "axis": axis,
                        "positive_arm": positive_arm,
                        "negative_arm": negative_arm,
                        "effect_definition": "positive_arm_minus_negative_arm",
                        "status": status,
                        "paired_participants": len(paired_participants),
                        "participant_mean_difference": observed,
                        "bootstrap_interval_low": float(interval_low),
                        "bootstrap_interval_high": float(interval_high),
                        "bootstrap_repetitions": (
                            bootstrap_repetitions if status == "available" else 0
                        ),
                        "permutation_pvalue_two_sided_plus_one": float(p_value),
                        "permutation_extreme_count": extreme_count,
                        "permutation_repetitions": (
                            permutation_repetitions if status == "available" else 0
                        ),
                        "inference_unit": "participant",
                    }
                )
    paired = pd.DataFrame(paired_rows) if paired_rows else _empty_paired_frame()
    if not paired.empty:
        paired = paired.sort_values(
            ["contrast_id", "partition", "axis", "participant_id"], kind="stable"
        ).reset_index(drop=True)
    estimates = pd.DataFrame(estimate_rows).sort_values(
        ["contrast_id", "partition", "axis"], kind="stable"
    )
    status_counts = estimates["status"].value_counts().sort_index().to_dict()
    ledger = {
        "schema_version": 1,
        "inference_unit": "participant",
        "run_collapse": "equal_weight_within_participant_condition_before_pairing",
        "partitions_analyzed_separately": list(PARTITIONS),
        "pooled_cross_partition_inference": False,
        "contrasts": list(CONTRASTS),
        "axes": list(AXES),
        "minimum_pairs_for_resampling": minimum_pairs,
        "minimum_pairs_is_result_gate": False,
        "bootstrap": {
            "scheme": "participant_pairs_with_replacement",
            "repetitions": bootstrap_repetitions,
            "interval": "percentile_95",
        },
        "permutation": {
            "scheme": "within_participant_paired_difference_sign_flip",
            "repetitions": permutation_repetitions,
            "two_sided": True,
            "plus_one": True,
        },
        "random_seed": random_seed,
        "label_fields_consumed_after_encoding": list(LABEL_FIELDS_CONSUMED),
        "pairing": pairing_records,
        "status_counts": status_counts,
        "issues": issues,
        "cross_modal_evidence": {
            "relationship": "independent_cohort_triangulation",
            "verified_participant_mapping": None,
            "participant_level_eeg_fmri_correlation_performed": False,
            "reason": "No explicit verified EEG-fMRI participant mapping was supplied.",
        },
        "scientific_gate_applied": False,
        "result_threshold_applied": False,
    }
    return FMRIPairedInference(
        paired_differences=paired,
        estimates=estimates,
        ledger=ledger,
    )


__all__ = [
    "AXES",
    "CONTRASTS",
    "FMRIInferenceError",
    "FMRIPairedInference",
    "collapse_runs_within_participant_condition",
    "infer_paired_condition_contrasts",
]
