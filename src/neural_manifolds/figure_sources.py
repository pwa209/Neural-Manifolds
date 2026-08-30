"""Transform stage artifacts into strict, hash-grounded figure source bundles."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from neural_manifolds.manifold.profile import AXIS_NAMES
from neural_manifolds.provenance import sha256_file

AXIS_LETTERS = dict(zip(AXIS_NAMES, ("R", "M", "D", "A", "P"), strict=True))


def _write(frame: pd.DataFrame, path: Path) -> Path:
    if frame.empty:
        raise ValueError(f"figure source table would be empty: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)
    return path


def _levels(value: object) -> tuple[str, ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(str(item) for item in value)
    return (str(value),)


def _load_contrast_specs(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    if not isinstance(document, Mapping) or document.get("schema_version") != 1:
        raise ValueError("contrast configuration must use schema_version 1")
    datasets = document.get("datasets")
    if not isinstance(datasets, Mapping):
        raise ValueError("contrast configuration has no datasets mapping")
    specs: list[dict[str, Any]] = []
    for dataset_id, dataset_value in datasets.items():
        if not isinstance(dataset_value, Mapping):
            continue
        raw_contrasts = dataset_value.get("contrasts", ())
        if not isinstance(raw_contrasts, Sequence):
            raise ValueError(f"contrasts for {dataset_id} must be a sequence")
        for raw_value in raw_contrasts:
            if not isinstance(raw_value, Mapping):
                raise ValueError(f"contrast for {dataset_id} must be a mapping")
            if "continuous_covariate" in raw_value:
                continue
            if "conditions" in raw_value:
                conditions = _levels(raw_value["conditions"])
                if len(conditions) != 2:
                    raise ValueError(
                        f"contrast {raw_value.get('id')} must declare exactly two conditions"
                    )
                positive, negative = (conditions[0],), (conditions[1],)
            else:
                positive = _levels(raw_value.get("positive"))
                negative = _levels(raw_value.get("reference"))
            if positive == ("None",) or negative == ("None",):
                raise ValueError(f"contrast {raw_value.get('id')} lacks positive/reference levels")
            match_value = raw_value.get("match_on", raw_value.get("match_within", ()))
            matching = _levels(match_value) if match_value else ()
            specs.append(
                {
                    "contrast": str(raw_value["id"]),
                    "dataset_id": str(dataset_id),
                    "positive": positive,
                    "negative": negative,
                    "subset": dict(raw_value.get("subset", {})),
                    "matching": tuple(field for field in matching if field != "participant_id"),
                }
            )
    if not specs:
        raise ValueError("contrast configuration has no binary figure contrasts")
    return specs


def _contrast_side(
    frame: pd.DataFrame,
    *,
    levels: tuple[str, ...],
    label: str,
    group_columns: Sequence[str],
    value_columns: Sequence[str],
) -> pd.DataFrame:
    selected = frame[frame["condition"].astype(str).isin(levels)]
    means = selected.groupby(list(group_columns), as_index=False, dropna=False)[
        list(value_columns)
    ].mean()
    counts = (
        selected.groupby(list(group_columns), as_index=False, dropna=False)
        .size()
        .rename(columns={"size": f"n_{label}_units"})
    )
    return means.merge(counts, on=list(group_columns), validate="one_to_one")


def _participant_contrast_effects(
    frame: pd.DataFrame,
    *,
    specs: Sequence[Mapping[str, Any]],
    value_columns: Sequence[str],
    extra_group_columns: Sequence[str] = (),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return true within-participant positive-minus-reference effects.

    Matching factors are respected before participant-level strata are averaged.
    A contrast that cannot be formed remains visible in the returned status table.
    """

    required = {"participant_id", "dataset_id", "condition", *value_columns}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"profile table lacks contrast columns {sorted(missing)}")
    rows: list[pd.DataFrame] = []
    status_rows: list[dict[str, Any]] = []
    for spec in specs:
        contrast = str(spec["contrast"])
        dataset_id = str(spec["dataset_id"])
        subset = frame[frame["dataset_id"].astype(str).eq(dataset_id)].copy()
        reason: str | None = None
        for field, expected in dict(spec.get("subset", {})).items():
            if field not in subset:
                reason = f"missing subset field {field}"
                subset = subset.iloc[0:0]
                break
            allowed = expected if isinstance(expected, list) else [expected]
            subset = subset[subset[field].isin(allowed)]
        matching = tuple(str(value) for value in spec.get("matching", ()))
        absent_matching = sorted(set(matching).difference(subset.columns))
        if absent_matching:
            reason = f"missing matching fields {absent_matching}"
            subset = subset.iloc[0:0]
        positive_levels = tuple(str(value) for value in spec["positive"])
        negative_levels = tuple(str(value) for value in spec["negative"])
        subset = subset[subset["condition"].astype(str).isin((*positive_levels, *negative_levels))]
        group_columns = [
            "participant_id",
            "dataset_id",
            *matching,
            *extra_group_columns,
        ]

        positive = _contrast_side(
            subset,
            levels=positive_levels,
            label="positive",
            group_columns=group_columns,
            value_columns=value_columns,
        )
        negative = _contrast_side(
            subset,
            levels=negative_levels,
            label="negative",
            group_columns=group_columns,
            value_columns=value_columns,
        )
        paired = positive.merge(
            negative,
            on=group_columns,
            how="inner",
            suffixes=("_positive", "_negative"),
            validate="one_to_one",
        )
        if paired.empty:
            status_rows.append(
                {
                    "contrast": contrast,
                    "dataset_id": dataset_id,
                    "status": "unavailable",
                    "reason": reason or "no participant has both contrast sides",
                    "n_participants": 0,
                }
            )
            continue
        for value in value_columns:
            paired[value] = paired[f"{value}_positive"] - paired[f"{value}_negative"]
        participant_group = ["participant_id", "dataset_id", *extra_group_columns]
        aggregations: dict[str, str] = {
            **{str(value): "mean" for value in value_columns},
            "n_positive_units": "sum",
            "n_negative_units": "sum",
        }
        participant = paired.groupby(participant_group, as_index=False, dropna=False).agg(
            aggregations
        )
        participant["contrast"] = contrast
        participant["positive_conditions"] = "|".join(positive_levels)
        participant["negative_conditions"] = "|".join(negative_levels)
        strata = paired.groupby(participant_group, as_index=False, dropna=False).size()
        participant = participant.merge(
            strata.rename(columns={"size": "matched_strata"}),
            on=participant_group,
            validate="one_to_one",
        )
        rows.append(participant)
        status_rows.append(
            {
                "contrast": contrast,
                "dataset_id": dataset_id,
                "status": "computed",
                "reason": None,
                "n_participants": int(participant["participant_id"].nunique()),
            }
        )
    effects = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return effects, pd.DataFrame(status_rows)


def _long_effects(frame: pd.DataFrame) -> pd.DataFrame:
    id_columns = [column for column in frame.columns if column not in AXIS_NAMES]
    result = frame.melt(
        id_vars=id_columns,
        value_vars=list(AXIS_NAMES),
        var_name="axis",
        value_name="value",
    )
    result["axis"] = result["axis"].map(AXIS_LETTERS)
    return result


def _augment_null_metadata(nulls: pd.DataFrame, profiles: pd.DataFrame) -> pd.DataFrame:
    if "unit_id" not in nulls or "unit_id" not in profiles:
        raise ValueError("profiles and null profiles require unit_id for condition provenance")
    if profiles["unit_id"].duplicated().any():
        raise ValueError("profile table contains duplicate unit_id values")
    null_identity = nulls[["unit_id", "participant_id", "dataset_id"]].copy()
    metadata = profiles.drop(columns=list(AXIS_NAMES)).copy()
    augmented = nulls.drop(columns=["participant_id", "dataset_id"]).merge(
        metadata,
        on="unit_id",
        how="left",
        validate="many_to_one",
    )
    if augmented["participant_id"].isna().any() or augmented["dataset_id"].isna().any():
        raise ValueError("a null profile has no observed unit metadata")
    observed_identity = augmented[["unit_id", "participant_id", "dataset_id"]]
    comparison = null_identity.merge(
        observed_identity.drop_duplicates(),
        on="unit_id",
        suffixes=("_null", "_observed"),
        validate="many_to_one",
    )
    for field in ("participant_id", "dataset_id"):
        if (
            not comparison[f"{field}_null"]
            .astype(str)
            .equals(comparison[f"{field}_observed"].astype(str))
        ):
            raise ValueError(f"null profile {field} differs from its observed unit")
    return augmented


def _robustness_effects(
    *,
    observed: pd.DataFrame,
    nulls: pd.DataFrame,
    specs: Sequence[Mapping[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    observed_effects, observed_status = _participant_contrast_effects(
        observed,
        specs=specs,
        value_columns=AXIS_NAMES,
    )
    if observed_effects.empty:
        raise ValueError("no observed participant contrast effect is available for robustness")
    augmented = _augment_null_metadata(nulls, observed)
    null_effects, null_status = _participant_contrast_effects(
        augmented,
        specs=specs,
        value_columns=AXIS_NAMES,
        extra_group_columns=("family", "repeat", "seed"),
    )
    if null_effects.empty:
        raise ValueError("no null participant contrast effect is available for robustness")
    observed_long = _long_effects(observed_effects).rename(columns={"value": "observed_effect"})
    null_long = _long_effects(null_effects).rename(columns={"value": "null_effect"})
    keys = ["participant_id", "dataset_id", "contrast", "axis"]
    provenance_columns = [
        "family",
        "repeat",
        "seed",
        "positive_conditions",
        "negative_conditions",
        "n_positive_units",
        "n_negative_units",
        "matched_strata",
    ]
    robustness = observed_long[[*keys, "observed_effect"]].merge(
        null_long[[*keys, *provenance_columns, "null_effect"]],
        on=keys,
        how="inner",
        validate="one_to_many",
    )
    robustness["observed_minus_null"] = robustness["observed_effect"] - robustness["null_effect"]
    direction = np.sign(robustness["observed_effect"].to_numpy(dtype=float))
    robustness["signed_effect_survival"] = (
        robustness["observed_minus_null"].to_numpy(dtype=float) * direction
    )
    robustness["analysis"] = robustness["family"].astype(str)
    robustness["metric"] = robustness["axis"].astype(str)
    robustness["value"] = robustness["signed_effect_survival"]
    statuses = pd.concat(
        [
            observed_status.assign(source="observed"),
            null_status.assign(source="null"),
        ],
        ignore_index=True,
    )
    return robustness, statuses


def prepare_clinical_figure_source(
    *, clinical_profiles_path: str | Path, output_root: str | Path
) -> Path:
    """Build a held-out clinical source bundle without inventing missing endpoints."""

    source = Path(clinical_profiles_path).resolve(strict=True)
    source_hash = sha256_file(source)
    frame = pd.read_parquet(source)
    required = {"participant_id", "dataset_id", "diagnosis", *AXIS_NAMES}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"clinical profiles lack figure columns {sorted(missing)}")
    columns = ["participant_id", "dataset_id", "diagnosis", *AXIS_NAMES]
    for optional in ("crs_r_total", "regime_preservation_score"):
        if optional in frame:
            columns.append(optional)
    clinical = frame[columns].rename(columns=AXIS_LETTERS)
    if "crs_r_total" not in clinical:
        clinical["crs_r_total"] = np.nan
    if "regime_preservation_score" not in clinical:
        raise ValueError(
            "clinical profiles lack the locked regime_preservation_score; "
            "the figure source builder will not derive a replacement endpoint"
        )
    clinical["source_artifact_sha256"] = source_hash
    destination = Path(output_root)
    _write(clinical, destination / "clinical_profiles.parquet")
    return destination


def prepare_fmri_figure_source(*, fmri_profiles_path: str | Path, output_root: str | Path) -> Path:
    """Build the optional four-axis fMRI bundle or fail on an incompatible schema."""

    source = Path(fmri_profiles_path).resolve(strict=True)
    source_hash = sha256_file(source)
    frame = pd.read_parquet(source)
    required = {"participant_id", "dataset_id", "condition", "R", "M", "D", "A"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(
            "fMRI supplement requires an explicitly calibrated R/M/D/A profile table; "
            f"missing {sorted(missing)}"
        )
    fmri = frame[["participant_id", "dataset_id", "condition", "R", "M", "D", "A"]].copy()
    fmri["source_artifact_sha256"] = source_hash
    destination = Path(output_root)
    _write(fmri, destination / "fmri_profiles.parquet")
    return destination


def prepare_figure_sources(
    *,
    profiles_path: str | Path,
    nulls_path: str | Path,
    contrasts_path: str | Path,
    tms_outcomes_path: str | Path,
    tms_trajectory_path: str | Path,
    output_root: str | Path,
) -> tuple[Path, Path, Path]:
    """Create exact participant/source schemas consumed by healthy manuscript figures."""

    destination = Path(output_root)
    profiles_source = Path(profiles_path).resolve(strict=True)
    profiles_hash = sha256_file(profiles_source)
    profiles = pd.read_parquet(profiles_source)
    required = {"unit_id", "participant_id", "dataset_id", "condition", *AXIS_NAMES}
    missing = required.difference(profiles.columns)
    if missing:
        raise ValueError(f"profiles lack figure columns {sorted(missing)}")
    profile_table = profiles[["participant_id", "dataset_id", "condition", *AXIS_NAMES]].rename(
        columns=AXIS_LETTERS
    )
    profile_table["source_artifact_sha256"] = profiles_hash
    profiles_bundle = destination / "profiles"
    _write(profile_table, profiles_bundle / "profiles.parquet")

    contrast_source = Path(contrasts_path).resolve(strict=True)
    contrast_hash = sha256_file(contrast_source)
    specs = _load_contrast_specs(contrast_source)
    observed_effects, contrast_status = _participant_contrast_effects(
        profiles,
        specs=specs,
        value_columns=AXIS_NAMES,
    )
    if observed_effects.empty:
        raise ValueError("no participant-level binary contrast can be computed for figures")
    content = _long_effects(observed_effects)
    content["source_artifact_sha256"] = profiles_hash
    content["contrast_config_sha256"] = contrast_hash

    nulls_source = Path(nulls_path).resolve(strict=True)
    nulls_hash = sha256_file(nulls_source)
    nulls = pd.read_parquet(nulls_source)
    null_required = {
        "unit_id",
        "participant_id",
        "dataset_id",
        "family",
        "repeat",
        "seed",
        *AXIS_NAMES,
    }
    if not null_required <= set(nulls):
        raise ValueError(f"null profiles lack figure columns {sorted(null_required - set(nulls))}")
    robustness, robustness_status = _robustness_effects(
        observed=profiles,
        nulls=nulls,
        specs=specs,
    )
    robustness["source_artifact_sha256"] = nulls_hash
    robustness["observed_profiles_sha256"] = profiles_hash
    robustness["contrast_config_sha256"] = contrast_hash
    status = pd.concat(
        [
            contrast_status.assign(analysis_table="content_report"),
            robustness_status.assign(analysis_table="robustness"),
        ],
        ignore_index=True,
    )
    status["source_artifact_sha256"] = profiles_hash
    status["contrast_config_sha256"] = contrast_hash
    models_bundle = destination / "models"
    _write(content, models_bundle / "content_report.parquet")
    _write(robustness, models_bundle / "robustness.parquet")
    _write(status, models_bundle / "contrast_status.parquet")

    tms_source = Path(tms_outcomes_path).resolve(strict=True)
    tms_hash = sha256_file(tms_source)
    tms = pd.read_parquet(tms_source)
    tms_required = {"participant_id", "condition", "reachability", "maximum_displacement"}
    if not tms_required <= set(tms):
        raise ValueError(f"TMS outcomes lack figure columns {sorted(tms_required - set(tms))}")
    participants = tms[
        ["participant_id", "condition", "reachability", "maximum_displacement"]
    ].rename(
        columns={
            "reachability": "passive_reachability",
            "maximum_displacement": "direct_response",
        }
    )
    participants["dataset_id"] = "propofol_tms_eeg"
    participants["source_artifact_sha256"] = tms_hash
    trajectory_source = Path(tms_trajectory_path).resolve(strict=True)
    trajectory_hash = sha256_file(trajectory_source)
    trajectory = pd.read_parquet(trajectory_source)
    trajectory["dataset_id"] = "propofol_tms_eeg"
    trajectory["source_artifact_sha256"] = trajectory_hash
    tms_bundle = destination / "tms"
    _write(participants, tms_bundle / "tms_participants.parquet")
    _write(trajectory, tms_bundle / "tms_trajectory.parquet")
    return profiles_bundle, models_bundle, tms_bundle
