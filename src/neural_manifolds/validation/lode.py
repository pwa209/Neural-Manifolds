"""LODE method-level external validation; no zero-shot model-transfer claim."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import tempfile
from collections import Counter
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from neural_manifolds.continuous.audit import safe_member
from neural_manifolds.provenance import atomic_write_json, sha256_file
from neural_manifolds.revised.controls import null_arrays
from neural_manifolds.revised.inference import measurement_arrays, nested_transfer
from neural_manifolds.revised.measurement import (
    conventional_features,
    prepare_window,
    sensor_trajectory,
    window_qc,
)
from neural_manifolds.stage_processing import read_raw_recording
from neural_manifolds.statistics.study_transfer import independent_weights


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()[:24]


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _config(path: Path) -> tuple[dict, str]:
    policy = yaml.safe_load(path.read_text(encoding="utf-8"))
    if policy["schema_version"] != 1 or policy["null_kinds"] != [
        "label_permutation",
        "temporal_permutation",
    ]:
        raise ValueError("unexpected_validation_policy")
    if policy["reference_mode"] != "native_bipolar_no_rereference":
        raise ValueError("LODE_must_keep_native_bipolar_reference")
    return policy, sha256_file(path)


def _candidate_rows(records_csv: Path, archive: Path, policy: dict) -> tuple[list[dict], dict]:
    rows = list(csv.DictReader(io.StringIO(records_csv.read_text(encoding="utf-8-sig"))))
    required = {"Filename", "Case ID", "Subject ID", "Experience", "Last sleep stage", "Duration"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError("LODE_records_schema_changed")
    case_counts = Counter(r["Case ID"].strip() for r in rows)
    with ZipFile(archive) as z:
        entries = {m.filename: m for m in z.infolist() if not m.is_dir()}
    selected, exclusions = [], Counter()
    wanted = {policy["source_channel_order"].index(channel) + 1 for channel in policy["primary_channels"]}
    for row in rows:
        if row["Last sleep stage"].strip() != policy["stage_code"]:
            exclusions["not_N2"] += 1
            continue
        report = row["Experience"].strip()
        if report not in {policy["experience_code"], policy["no_experience_code"]}:
            exclusions["not_clear_E_or_NE"] += 1
            continue
        case, person, filename = (row[k].strip() for k in ("Case ID", "Subject ID", "Filename"))
        member = str(PurePosixPath("Data/PSG") / filename.replace("\\", "/"))
        try:
            if not case or case_counts[case] != 1 or not person:
                raise ValueError("missing_or_duplicate_identifier")
            if not safe_member(member) or member not in entries or not member.lower().endswith(".edf"):
                raise ValueError("missing_or_unsafe_EDF_link")
            duration = float(row["Duration"])
            if not np.isfinite(duration) or duration < policy["dream_seconds"]:
                raise ValueError("short_or_invalid_duration")
            if entries[member].file_size <= 0 or entries[member].file_size > 2 * 1024**3:
                raise ValueError("EDF_member_size_outside_bound")
            remarks = row.get("Remarks", "").strip()
            if not remarks.startswith("Bad channels:"):
                raise ValueError("bad_channel_metadata_unrecognized")
            bad = {int(x) for x in re.findall(r"\b[1-7]\b", remarks)}
            if bad & wanted:
                raise ValueError("primary_channel_source_flagged_bad")
        except ValueError as exc:
            exclusions[str(exc)] += 1
            continue
        selected.append(
            {
                "unit_id": _hash(f"lode:{case}"),
                "participant_id": _hash(f"lode:{person}"),
                "member": member,
                "bytes": entries[member].file_size,
                "duration": duration,
                "sleep_stage": 2,
                "experience": int(report == policy["experience_code"]),
                "report_code": int(report),
                "remarks_present": bool(row.get("Remarks", "").strip()),
            }
        )
    if len({r["unit_id"] for r in selected}) != len(selected):
        raise ValueError("pseudonymous_case_collision")
    return sorted(selected, key=lambda r: r["unit_id"]), dict(exclusions)


def preflight(config: Path, records_csv: Path, archive: Path, output: Path) -> dict:
    policy, config_sha = _config(config)
    for name, path in [("records_csv", records_csv), ("data_zip", archive)]:
        if sha256_file(path) != policy["source_sha256"][name]:
            raise ValueError(f"LODE_{name}_checksum_changed")
    candidates, exclusions = _candidate_rows(records_csv, archive, policy)
    if not candidates:
        raise ValueError("no_linked_N2_clear_report_records")
    result = {
        "scope": "metadata_preflight_before_EEG_signal_or_outcome_analysis",
        "config_sha256": config_sha,
        "records_sha256": sha256_file(records_csv),
        "archive_sha256": sha256_file(archive),
        "archive_path": str(archive.resolve()),
        "candidates": candidates,
        "metadata_exclusions": exclusions,
        "candidate_counts": dict(Counter(r["report_code"] for r in candidates)),
        "participant_count": len({r["participant_id"] for r in candidates}),
        "scientific_gates": False,
    }
    atomic_write_json(output / "preflight.json", result)
    return result


def _verified_preflight(config: Path, output: Path) -> tuple[dict, dict]:
    policy, config_sha = _config(config)
    pre = _read(output / "preflight.json")
    if pre["config_sha256"] != config_sha:
        raise ValueError("validation_config_changed_after_preflight")
    return policy, pre


def _measure_one(candidate: dict, archive: Path, policy: dict, output: Path, pre_sha: str) -> dict:
    identity = candidate["unit_id"]
    checkpoint = output / "measure" / f"{identity}.json"
    if checkpoint.exists():
        existing = _read(checkpoint)
        if existing["preflight_sha256"] != pre_sha:
            raise ValueError("measurement_checkpoint_input_changed")
        if existing.get("array_path") and sha256_file(existing["array_path"]) != existing["array_sha256"]:
            raise ValueError("measurement_array_changed")
        return existing
    result = {k: candidate[k] for k in ("unit_id", "participant_id", "sleep_stage", "experience", "report_code")}
    result.update(
        preflight_sha256=pre_sha,
        study_group="lode_lucca",
        primary_eligible=True,
        track="bipolar_two_derivation_external",
        participant_alias_status="source_native_identifier",
    )
    try:
        with ZipFile(archive) as z:
            member = z.getinfo(candidate["member"])
            if member.file_size != candidate["bytes"]:
                raise ValueError("EDF_member_size_changed")
            with tempfile.TemporaryDirectory(prefix="nm-lode-", dir=output / "tmp") as directory:
                source = Path(directory) / "recording.edf"
                with z.open(member) as reader, source.open("xb") as writer:
                    total = 0
                    while block := reader.read(1024 * 1024):
                        total += len(block)
                        if total > candidate["bytes"]:
                            raise ValueError("EDF_member_exceeded_bound")
                        writer.write(block)
                if total != candidate["bytes"]:
                    raise ValueError("EDF_member_truncated")
                raw = read_raw_recording(source)
                try:
                    values, audit = prepare_window(
                        raw,
                        {"stop_seconds": candidate["duration"], "timing_policy": "final_140_seconds_to_awakening"},
                        policy["primary_channels"],
                        policy,
                    )
                finally:
                    raw.close()
        windows, reasons = window_qc(values, policy["sampling_hz"], policy)
        keep = np.asarray([not r for r in reasons], dtype=bool)
        result.update(clean_seconds=int(keep.sum()), window_reasons=reasons, preprocessing=audit)
        if int(keep.sum()) < policy["minimum_clean_seconds"]:
            raise ValueError("insufficient_clean_seconds")
        indices = np.flatnonzero(keep)
        segments = np.cumsum(np.r_[True, np.diff(indices) != 1])
        selected = windows[keep]
        result["conventional"] = conventional_features(selected, policy["sampling_hz"])
        array_path = output / "arrays" / f"{identity}.npz"
        array_path.parent.mkdir(parents=True, exist_ok=True)
        with array_path.with_suffix(".tmp").open("wb") as stream:
            np.savez_compressed(
                stream,
                sensor=sensor_trajectory(selected),
                segments=segments,
                seconds=indices,
                channels=np.asarray(audit["channels"]),
            )
        array_path.with_suffix(".tmp").replace(array_path)
        result.update(status="measured", array_path=str(array_path), array_sha256=sha256_file(array_path))
    except (ValueError, OSError, RuntimeError, KeyError) as exc:
        result.update(status="unavailable", reason=f"{type(exc).__name__}:{exc}")
    atomic_write_json(checkpoint, result)
    return result


def measure(config: Path, output: Path, chunk: int, chunks: int) -> dict:
    policy, pre = _verified_preflight(config, output)
    if not 0 <= chunk < chunks or chunks != 10:
        raise ValueError("fixed_measurement_chunk_count_is_ten")
    archive = Path(pre["archive_path"])
    if not archive.is_file() or archive.stat().st_size <= 0:
        raise ValueError("source_archive_missing")
    if sha256_file(archive) != pre["archive_sha256"]:
        raise ValueError("source_archive_changed_after_preflight")
    (output / "tmp").mkdir(parents=True, exist_ok=True)
    pre_sha = sha256_file(output / "preflight.json")
    selected = pre["candidates"][chunk::chunks]
    rows = [_measure_one(c, archive, policy, output, pre_sha) for c in selected]
    receipt = {"chunk": chunk, "preflight_sha256": pre_sha, "units": len(rows), "measured": sum(r["status"] == "measured" for r in rows)}
    atomic_write_json(output / "measure" / f"chunk-{chunk:02d}.json", receipt)
    return receipt


def assemble(config: Path, output: Path) -> dict:
    _, pre = _verified_preflight(config, output)
    pre_sha = sha256_file(output / "preflight.json")
    for chunk in range(10):
        receipt = _read(output / "measure" / f"chunk-{chunk:02d}.json")
        if receipt["chunk"] != chunk or receipt["preflight_sha256"] != pre_sha:
            raise ValueError("measurement_chunk_receipt_changed")
    rows = []
    for candidate in pre["candidates"]:
        row = _read(output / "measure" / f"{candidate['unit_id']}.json")
        if row["preflight_sha256"] != pre_sha:
            raise ValueError("measurement_record_input_changed")
        rows.append(row)
    selected = [r for r in rows if r["status"] == "measured"]
    result = {
        "preflight_sha256": pre_sha,
        "records": rows,
        "measured_counts": dict(Counter(r["report_code"] for r in selected)),
        "measured_participants": len({r["participant_id"] for r in selected}),
        "participants_with_both_labels": sum(
            len({r["experience"] for r in selected if r["participant_id"] == person}) == 2
            for person in {r["participant_id"] for r in selected}
        ),
        "unavailable_reasons": dict(Counter(r.get("reason", "") for r in rows if r["status"] != "measured")),
        "scope": "label_blind_signal_QC_then_metadata_counts_not_model_results",
    }
    atomic_write_json(output / "measurement.json", result)
    return result


def _analysis_inputs(config: Path, output: Path) -> tuple[dict, pd.DataFrame, dict]:
    policy, _ = _verified_preflight(config, output)
    measurement = _read(output / "measurement.json")
    if measurement["preflight_sha256"] != sha256_file(output / "preflight.json"):
        raise ValueError("measurement_summary_input_changed")
    selected = [r for r in measurement["records"] if r["status"] == "measured"]
    frame, arrays = measurement_arrays(selected, "sensor")
    if frame.empty or frame.experience.nunique() != 2:
        raise ValueError("both_report_categories_required_after_QC")
    persons = sorted(frame.participant_id.unique().tolist())
    if len(persons) < policy["outer_participant_folds"]:
        raise ValueError("too_few_participants_for_fixed_folds")
    rng = np.random.default_rng(policy["seed"])
    rng.shuffle(persons)
    mapping = {person: i % policy["outer_participant_folds"] for i, person in enumerate(persons)}
    frame["validation_fold"] = frame.participant_id.map(mapping).astype(int)
    if frame.groupby("participant_id").validation_fold.nunique().max() != 1:
        raise ValueError("participant_leakage_across_folds")
    return policy, frame, arrays


def _scored(frame: pd.DataFrame, arrays: dict, policy: dict, *, all_models: bool) -> dict:
    families = policy["secondary_descriptive_models"] if all_models else ["shared_dynamics"]
    fitted = nested_transfer(
        frame,
        arrays,
        outer="validation_fold",
        seed=policy["seed"],
        families=families,
        strengths=policy["regularization"],
        dimensions=policy["profile_dimensions"],
    )
    predictions = pd.DataFrame(fitted["predictions"])
    if fitted["unavailable"] or predictions.empty:
        raise ValueError(f"incomplete_nested_fit:{fitted['unavailable']}")
    if set(predictions.model) != set(families):
        raise ValueError("missing_model_predictions")
    expected = set(frame.unit_id)
    for _, block in predictions.groupby("model"):
        if len(block) != len(frame) or set(block.unit_id) != expected:
            raise ValueError("incomplete_held_out_predictions")
    constants = []
    for fold in sorted(frame.validation_fold.unique()):
        train, test = frame[frame.validation_fold != fold], frame[frame.validation_fold == fold]
        probability = float(np.average(train.experience, weights=independent_weights(train)))
        probability = float(np.clip(probability, 1e-6, 1 - 1e-6))
        for row in test.itertuples():
            constants.append(
                {"unit_id": row.unit_id, "participant_id": row.participant_id,
                 "study_group": row.study_group, "experience": int(row.experience),
                 "model": "training_constant", "probability": probability,
                 "held_group": int(fold)}
            )
    predictions = pd.concat([predictions, pd.DataFrame(constants)], ignore_index=True)
    losses = {}
    for family, block in predictions.groupby("model"):
        y = block.experience.to_numpy(int)
        p = block.probability.to_numpy(float)
        w = independent_weights(block)
        losses[family] = {
            "log_loss": float(log_loss(y, p, labels=[0, 1], sample_weight=w)),
            "brier": float(brier_score_loss(y, p, sample_weight=w)),
            "auroc": float(roc_auc_score(y, p, sample_weight=w)) if len(set(y)) == 2 else None,
        }
    return {
        "predictions": _jsonable(predictions.to_dict(orient="records")),
        "losses": losses,
        "dynamic_minus_training_constant_log_loss": losses["shared_dynamics"]["log_loss"] - losses["training_constant"]["log_loss"],
        "dynamic_absolute_log_loss": losses["shared_dynamics"]["log_loss"],
        "tuning": _jsonable(fitted["tuning"]),
        "outer_unit": "participant_disjoint_label_blind_five_fold",
    }


def _paired_participant_intervals(predictions: list[dict], repetitions: int, seed: int) -> list[dict]:
    table = pd.DataFrame(predictions)
    table["loss"] = -table.experience * np.log(np.clip(table.probability, 1e-12, 1 - 1e-12)) - (
        1 - table.experience
    ) * np.log1p(-np.clip(table.probability, 1e-12, 1 - 1e-12))
    by_person = table.groupby(["participant_id", "model"]).loss.mean().unstack()
    rng = np.random.default_rng(seed)
    rows = []
    for comparator in sorted(set(by_person) - {"shared_dynamics"}):
        difference = (by_person["shared_dynamics"] - by_person[comparator]).dropna().to_numpy(float)
        if len(difference) != len(by_person):
            raise ValueError("paired_participant_prediction_missing")
        draws = rng.choice(difference, size=(repetitions, len(difference)), replace=True).mean(axis=1)
        rows.append({"comparator": comparator, "dynamics_minus_comparator_log_loss": float(difference.mean()),
                     "interval_95_conditional_on_fitted_predictions": np.quantile(draws, [0.025, 0.975]).tolist(),
                     "independent_participants": len(difference)})
    return rows


def observed(config: Path, output: Path) -> dict:
    try:
        policy, frame, arrays = _analysis_inputs(config, output)
        result = _scored(frame, arrays, policy, all_models=True)
        result["paired_intervals"] = _paired_participant_intervals(
            result["predictions"], policy["participant_cluster_bootstrap_repetitions"], policy["seed"]
        )
        result["status"] = "analysed"
    except ValueError as exc:
        result = {"status": "unavailable", "reason": str(exc)}
        frame = pd.DataFrame(columns=["unit_id", "participant_id", "experience", "validation_fold"])
    result.update(
        config_sha256=sha256_file(config),
        measurement_sha256=sha256_file(output / "measurement.json"),
        n_observations=len(frame),
        n_participants=frame.participant_id.nunique(),
        label_counts=_jsonable(frame.experience.value_counts().to_dict()),
        participants_with_both_labels=int((frame.groupby("participant_id").experience.nunique() == 2).sum()) if not frame.empty else 0,
        fold_map=_jsonable(frame.set_index("unit_id").validation_fold.to_dict()) if not frame.empty else {},
        external_method_replication_not_zero_shot_transfer=True,
    )
    atomic_write_json(output / "observed.json", result)
    return result


def null_chunk(config: Path, output: Path, task: int) -> dict:
    policy, _ = _verified_preflight(config, output)
    obs = _read(output / "observed.json")
    if obs["config_sha256"] != sha256_file(config) or obs["measurement_sha256"] != sha256_file(output / "measurement.json"):
        raise ValueError("observed_input_changed")
    if obs["status"] != "analysed":
        return {"status": "skipped_observed_unavailable"}
    _, frame, arrays = _analysis_inputs(config, output)
    frame["validation_fold"] = frame.unit_id.map(obs["fold_map"]).astype(int)
    n = policy["null_replicates_per_family"]
    size = policy["null_chunk_size"]
    chunks = (n + size - 1) // size
    if not 0 <= task < chunks * 2:
        raise ValueError("invalid_null_array_task")
    kind = policy["null_kinds"][task // chunks]
    start = (task % chunks) * size
    rows = []
    for replicate in range(start, min(start + size, n)):
        path = output / "null" / f"{kind}-{replicate:04d}.json"
        if path.exists():
            row = _read(path)
            if row["observed_sha256"] != sha256_file(output / "observed.json"):
                raise ValueError("null_checkpoint_input_changed")
        else:
            rng = np.random.default_rng(policy["seed"] + 1_000_003 * (task // chunks + 1) + replicate)
            shuffled = frame.copy()
            transformed = arrays
            if kind == "label_permutation":
                for _, block in shuffled.groupby(["participant_id", "sleep_stage"]):
                    shuffled.loc[block.index, "experience"] = rng.permutation(block.experience.to_numpy())
            else:
                transformed = null_arrays(arrays, kind, rng)
            try:
                scored = _scored(shuffled, transformed, policy, all_models=False)
                row = {
                    "status": "analysed", "kind": kind, "replicate": replicate,
                    "statistic": scored[
                        "dynamic_minus_training_constant_log_loss"
                        if kind == "label_permutation" else "dynamic_absolute_log_loss"
                    ],
                }
            except ValueError as exc:
                row = {"status": "unavailable", "kind": kind, "replicate": replicate, "reason": str(exc)}
            row["observed_sha256"] = sha256_file(output / "observed.json")
            atomic_write_json(path, row)
        rows.append(row)
    result = {"task": task, "kind": kind, "start": start, "count": len(rows), "analysed": sum(r["status"] == "analysed" for r in rows)}
    atomic_write_json(output / "null" / f"task-{task:03d}.json", result)
    return result


def _holm_two(pvalues: list[float]) -> list[float]:
    order = np.argsort(pvalues)
    adjusted = [0.0, 0.0]
    adjusted[int(order[0])] = min(1.0, 2 * pvalues[int(order[0])])
    adjusted[int(order[1])] = min(1.0, max(adjusted[int(order[0])], pvalues[int(order[1])]))
    return adjusted


def finalize(config: Path, output: Path) -> dict:
    policy, _ = _verified_preflight(config, output)
    obs = _read(output / "observed.json")
    n = policy["null_replicates_per_family"]
    tests = []
    for kind, metric in [
        ("label_permutation", "dynamic_minus_training_constant_log_loss"),
        ("temporal_permutation", "dynamic_absolute_log_loss"),
    ]:
        rows = []
        for index in range(n):
            path = output / "null" / f"{kind}-{index:04d}.json"
            if path.exists():
                row = _read(path)
                if row["observed_sha256"] != sha256_file(output / "observed.json") or row["kind"] != kind or row["replicate"] != index:
                    raise ValueError("null_checkpoint_identity_mismatch")
                rows.append(row)
        valid = [r["statistic"] for r in rows if r["status"] == "analysed"]
        complete = obs["status"] == "analysed" and len(rows) == n and len(valid) == n
        observed_value = obs.get(metric)
        p = (1 + sum(value <= observed_value for value in valid)) / (n + 1) if complete else None
        tests.append({"kind": kind, "metric": metric, "observed": observed_value,
                      "requested_refits": n, "received_refits": len(rows), "valid_refits": len(valid),
                      "p_lower_plus_one": p, "null_median": float(np.median(valid)) if valid else None,
                      "status": "complete" if complete else "incomplete"})
    if all(t["status"] == "complete" for t in tests):
        adjusted = _holm_two([t["p_lower_plus_one"] for t in tests])
        for test, value in zip(tests, adjusted, strict=True):
            test["holm_two_test_p"] = value
    result = {
        "status": "complete" if all(t["status"] == "complete" for t in tests) else "incomplete",
        "tests": tests,
        "observed_status": obs["status"],
        "config_sha256": sha256_file(config),
        "preflight_sha256": sha256_file(output / "preflight.json"),
        "measurement_sha256": sha256_file(output / "measurement.json"),
        "observed_sha256": sha256_file(output / "observed.json"),
        "scientific_gates": False,
        "separate_from_original_72_test_family": True,
        "non_preregistered_post_original_results_dataset_selection": True,
    }
    atomic_write_json(output / "final.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("preflight", "measure", "assemble", "observed", "null", "finalize"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--records-csv", type=Path)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--index", type=int)
    args = parser.parse_args()
    if args.stage == "preflight":
        if args.records_csv is None or args.archive is None:
            parser.error("preflight requires --records-csv and --archive")
        result = preflight(args.config, args.records_csv, args.archive, args.output)
    elif args.stage == "measure":
        if args.index is None:
            parser.error("measure requires --index")
        result = measure(args.config, args.output, args.index, 10)
    elif args.stage == "assemble":
        result = assemble(args.config, args.output)
    elif args.stage == "observed":
        result = observed(args.config, args.output)
    elif args.stage == "null":
        if args.index is None:
            parser.error("null requires --index")
        result = null_chunk(args.config, args.output, args.index)
    else:
        result = finalize(args.config, args.output)
    print(json.dumps(_jsonable({k: v for k, v in result.items() if k not in {"candidates", "records", "predictions", "tuning", "fold_map"}}), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
