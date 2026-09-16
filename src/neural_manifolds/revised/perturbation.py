"""Audited spontaneous/TMS pairing and nested participant-held-out prediction."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from neural_manifolds.adapters.datasets import PropofolTMSEEGAdapter
from neural_manifolds.config import load_study
from neural_manifolds.preprocessing.tms import conventional_tms_eeg_outcomes
from neural_manifolds.provenance import atomic_write_json, sha256_file
from neural_manifolds.revised.inference import CONVENTIONAL, DYNAMICS, FoldDynamics
from neural_manifolds.revised.measurement import (
    conventional_features,
    prepare_window,
    sensor_trajectory,
    window_qc,
)
from neural_manifolds.stage_processing import read_raw_recording
from neural_manifolds.stages.tms import build_tms_epoch_manifest
from neural_manifolds.statistics.study_transfer import independent_weights, validate_units


def pair_sessions(units) -> tuple[list, list]:
    pairs, unavailable = [], []
    for pulse in [u for u in units if u.modality == "tms-eeg"]:
        # Exact BIDS entities. Never pair by row order, neighbouring sessions or labels.
        candidates = [
            u
            for u in units
            if u.modality == "eeg"
            and u.participant_id == pulse.participant_id
            and u.session_id == pulse.session_id
            and u.run_id == pulse.run_id
            and u.variables["task"] == pulse.variables["task"]
            and u.variables["acquisition"]
            == ("EC" if pulse.variables["task"] == "awake" else "rest")
        ]
        if len(candidates) == 1:
            pairs.append((candidates[0], pulse))
        else:
            unavailable.append(
                {
                    "unit_id": pulse.unit_id,
                    "reason": "passive_session_match_not_unique",
                    "matches": len(candidates),
                }
            )
    return pairs, unavailable


def passive_measure(source: Path, policy: dict) -> tuple[dict, dict]:
    raw = read_raw_recording(source)
    windows, segment_ids, features = [], [], []
    try:
        duration = raw.n_times / raw.info["sfreq"]
        # Fixed first two minutes, or all complete intervals if shorter.
        for index, stop in enumerate(np.arange(20, min(120, duration) + 0.001, 20)):
            x, _ = prepare_window(
                raw, {"stop_seconds": stop}, policy["high_density_channels"], policy
            )
            blocks, reasons = window_qc(x, policy["sampling_hz"], policy)
            keep = np.array([not r for r in reasons])
            if keep.sum() < policy["minimum_clean_seconds"]:
                continue
            selected = blocks[keep]
            positions = np.flatnonzero(keep)
            segments = index * 100 + np.cumsum(np.r_[True, np.diff(positions) != 1])
            windows.append(sensor_trajectory(selected))
            segment_ids.append(segments)
            features.append(conventional_features(selected, policy["sampling_hz"]))
    finally:
        raw.close()
    if not windows:
        raise ValueError("no_eligible_passive_intervals")
    x = np.concatenate(windows)
    return {k: float(np.mean([r[k] for r in features])) for k in CONVENTIONAL}, {
        "trajectory": x,
        "segments": np.concatenate(segment_ids),
        "regions": {str(i): x[:, i : i + 1] for i in range(x.shape[1])},
    }


def nested_tms(
    frame: pd.DataFrame, arrays: dict, outcome: str, *, alphas=(0.1, 1.0, 10.0), seed=20260916
):
    validate_units(frame, outcome)
    if frame.participant_id.nunique() < 4:
        raise ValueError("TMS_requires_four_independent_participants")
    base_columns = [*CONVENTIONAL, "sedation"]
    base = pd.DataFrame(frame.conventional.tolist(), index=frame.index)
    base["sedation"] = frame.sedation
    predictions, audit = [], []
    for held in sorted(frame.participant_id.unique()):
        train, test = frame[frame.participant_id != held], frame[frame.participant_id == held]
        partitions = []
        for a, b in GroupKFold(min(5, train.participant_id.nunique())).split(
            train, groups=train.participant_id
        ):
            fit, validation = train.iloc[a], train.iloc[b]
            transform = FoldDynamics(seed=seed).fit(fit, arrays)
            dynamics, _ = transform.transform(pd.concat([fit, validation]), arrays)
            partitions.append((fit, validation, base.join(dynamics)))
        transform = FoldDynamics(seed=seed).fit(train, arrays)
        dynamics, _ = transform.transform(pd.concat([train, test]), arrays)
        all_features = base.join(dynamics)
        for model_name, candidate_columns in [
            ("condition_conventional", base_columns),
            ("plus_dynamics", base_columns + DYNAMICS),
        ]:
            choices = []
            for alpha in alphas:
                losses = []
                for fit, validation, features in partitions:
                    columns = [
                        c for c in candidate_columns if features.loc[fit.index, c].notna().any()
                    ]
                    model = make_pipeline(SimpleImputer(), StandardScaler(), Ridge(alpha=alpha))
                    model.fit(
                        features.loc[fit.index, columns],
                        fit[outcome],
                        ridge__sample_weight=independent_weights(fit),
                    )
                    errors = validation[outcome].to_numpy() - model.predict(
                        features.loc[validation.index, columns]
                    )
                    losses.append(
                        float(np.average(errors**2, weights=independent_weights(validation)))
                    )
                choices.append((float(np.mean(losses)), alpha))
            _, alpha = min(choices)
            columns = [
                c for c in candidate_columns if all_features.loc[train.index, c].notna().any()
            ]
            model = make_pipeline(SimpleImputer(), StandardScaler(), Ridge(alpha=alpha))
            model.fit(
                all_features.loc[train.index, columns],
                train[outcome],
                ridge__sample_weight=independent_weights(train),
            )
            predicted = model.predict(all_features.loc[test.index, columns])
            for (_, row), estimate in zip(test.iterrows(), predicted, strict=True):
                predictions.append(
                    {
                        "unit_id": row.unit_id,
                        "participant_id": held,
                        "condition": row.condition,
                        "model": model_name,
                        "outcome": outcome,
                        "observed": float(row[outcome]),
                        "predicted": float(estimate),
                    }
                )
            audit.append(
                {
                    "held_participant": held,
                    "model": model_name,
                    "alpha": alpha,
                    "training_units": transform.training_ids_,
                    "inner_losses": choices,
                }
            )
    return {"predictions": predictions, "audit": audit}


def paired_error_summary(predictions: list, *, repetitions=1000, seed=20260916):
    frame = pd.DataFrame(predictions)
    frame["error"] = (frame.observed - frame.predicted) ** 2
    pivot = frame.pivot(
        index=["unit_id", "participant_id", "condition"], columns="model", values="error"
    ).dropna()
    delta = (
        (pivot.plus_dynamics - pivot.condition_conventional)
        .rename("incremental_squared_error")
        .reset_index()
    )
    people = delta.groupby("participant_id").incremental_squared_error.mean().to_numpy()
    rng = np.random.default_rng(seed)
    draws = [
        float(rng.choice(people, len(people), replace=True).mean()) for _ in range(repetitions)
    ]
    conditions = delta.groupby("condition").incremental_squared_error.mean().to_dict()
    diagnostics = []
    changes = []
    cells = frame.groupby(["participant_id", "condition", "model"], as_index=False)[
        ["observed", "predicted"]
    ].mean()
    for (condition, model), block in cells.groupby(["condition", "model"]):
        finite = block[["observed", "predicted"]].dropna()
        variable = len(finite) >= 4 and (finite.nunique() > 1).all()
        diagnostics.append(
            {
                "condition": condition,
                "model": model,
                "participants": len(finite),
                "mean_calibration_error": float((finite.predicted - finite.observed).mean()),
                "held_out_spearman": float(spearmanr(finite.observed, finite.predicted).statistic)
                if variable
                else None,
            }
        )
    for model, block in cells.groupby("model"):
        wide = block.pivot(
            index="participant_id", columns="condition", values=["observed", "predicted"]
        )
        if {"awake", "propofol_sedation"}.issubset(set(block.condition)):
            d = pd.DataFrame(
                {
                    k: wide[k]["propofol_sedation"] - wide[k]["awake"]
                    for k in ("observed", "predicted")
                }
            ).dropna()
            changes.append(
                {
                    "model": model,
                    "participants": len(d),
                    "sedation_minus_awake_observed": float(d.observed.mean()),
                    "sedation_minus_awake_predicted": float(d.predicted.mean()),
                    "change_mse": float(np.mean((d.predicted - d.observed) ** 2)),
                    "paired_values": d.reset_index().to_dict("records"),
                }
            )
    return {
        "incremental_squared_error": float(people.mean()),
        "interval_95": np.quantile(draws, [0.025, 0.975]).tolist(),
        "within_condition_errors": conditions,
        "within_condition_calibration_and_association": diagnostics,
        "within_participant_changes": changes,
        "participants": len(people),
        "uncertainty": "participant_bootstrap_conditional_on_fitted_predictions",
    }


def run_tms(release: Path, output: Path, policy: dict):
    output.mkdir(parents=True, exist_ok=True)
    participants = pd.read_csv(release / "participants.tsv", sep="\t")
    files = [p.relative_to(release).as_posix() for p in release.glob("sub-*/eeg/*_eeg.vhdr")]
    units = PropofolTMSEEGAdapter().adapt(participants, files)
    pairs, unavailable = pair_sessions(units)
    labels = pd.DataFrame(
        [
            dict(
                u.model_dump(),
                source_path=str(release / u.source_file),
                acquisition=u.variables["acquisition"],
            )
            for _, u in pairs
        ]
    )
    if labels.empty:
        raise ValueError("no_verified_spontaneous_tms_pairs")
    label_path = output / "tms-labels.parquet"
    labels.to_parquet(label_path, index=False)
    manifest_path, audit_path = build_tms_epoch_manifest(
        cohort_labels=label_path,
        output_root=output / "pulse_qc",
        study=load_study("configs/study.yaml"),
    )
    manifests = pd.read_parquet(manifest_path).set_index("unit_id")
    rows, arrays = [], {}
    for spontaneous, pulse in pairs:
        if pulse.unit_id not in manifests.index:
            unavailable.append({"unit_id": pulse.unit_id, "reason": "pulse_qc_unavailable"})
            continue
        try:
            conventional, trajectory = passive_measure(release / spontaneous.source_file, policy)
            record = manifests.loc[pulse.unit_id]
            if sha256_file(record.epochs_path) != record.epochs_sha256:
                raise ValueError("TMS_epochs_changed")
            with np.load(record.epochs_path, allow_pickle=False) as archive:
                epochs, times = archive["epochs"], archive["times_seconds"]
            if len(epochs) < 10:
                raise ValueError("fewer_than_ten_accepted_pulses")
            outcomes = conventional_tms_eeg_outcomes(epochs, times)
            mean = epochs.mean(axis=0)
            baseline = (times >= -0.5) & (times <= -0.05)
            mean -= mean[:, baseline].mean(axis=1, keepdims=True)
            post = (times >= 0.02) & (times <= 0.3)
            eigenvalues = np.linalg.eigvalsh(np.cov(mean[:, post]))
            outcomes["response_differentiation_pr"] = float(
                eigenvalues.sum() ** 2 / max(np.square(eigenvalues).sum(), 1e-30)
            )
            gfp = mean.std(axis=0)
            threshold = np.quantile(gfp[baseline], 0.99)
            active = np.flatnonzero((times >= 0.02) & (gfp > threshold))
            outcomes["response_recovery_last_crossing_seconds"] = (
                float(times[active[-1]]) if len(active) else 0.02
            )
            outcomes["recovery_right_censored"] = bool(len(active) and active[-1] == len(times) - 1)
            rows.append(
                {
                    "unit_id": pulse.unit_id,
                    "participant_id": pulse.participant_id,
                    "study_group": "propofol_tms",
                    "condition": pulse.condition,
                    "sedation": int(pulse.variables["task"] != "awake"),
                    "conventional": conventional,
                    **outcomes,
                    "passive_unit_id": spontaneous.unit_id,
                    "passive_source_sha256": sha256_file(release / spontaneous.source_file),
                    "tms_epochs_sha256": record.epochs_sha256,
                }
            )
            arrays[pulse.unit_id] = trajectory
        except (ValueError, OSError, RuntimeError) as exc:
            unavailable.append({"unit_id": pulse.unit_id, "reason": str(exc)})
    results = {}
    frame = pd.DataFrame(rows)
    for outcome in [
        "sensor_spread_fraction",
        "response_differentiation_pr",
        "response_recovery_last_crossing_seconds",
    ]:
        try:
            selected = frame[~frame.recovery_right_censored] if "recovery" in outcome else frame
            result = nested_tms(selected, arrays, outcome, seed=policy["seed"])
            result["summary"] = paired_error_summary(
                result["predictions"],
                repetitions=policy["bootstrap_repetitions"],
                seed=policy["seed"],
            )
            results[outcome] = result
        except (ValueError, KeyError, AttributeError) as exc:
            results[outcome] = {"status": "unavailable", "reason": str(exc)}
    result = {
        "analyses": results,
        "paired_records": rows,
        "unavailable": unavailable,
        "pulse_qc_audit": {"path": str(audit_path), "sha256": sha256_file(audit_path)},
        "limitations": [
            "no_sham_control",
            "residual_sensory_and_muscle_contamination",
            "passive_predictive_proxy_not_causal_control",
            "recovery_censored_records_reported_separately",
            "stimulation_site_intensity_and_sensory_confound_adjustment_not_identified_from_file_names",
        ],
        "scientific_gates": False,
    }
    atomic_write_json(output / "tms.json", result)
    return result
