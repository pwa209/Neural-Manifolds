"""Independent DREAM report-sensitivity extension, not strict consciousness validation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import tempfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from zipfile import BadZipFile, ZipFile

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import log_loss

from neural_manifolds.continuous.audit import safe_member
from neural_manifolds.provenance import atomic_write_json, sha256_file
from neural_manifolds.revised.cohort import normalized_channel
from neural_manifolds.revised.controls import null_arrays
from neural_manifolds.revised.inference import (
    measurement_arrays,
    nested_transfer,
    summarize_predictions,
)
from neural_manifolds.revised.measurement import (
    conventional_features,
    prepare_window,
    sensor_trajectory,
    window_qc,
)
from neural_manifolds.stage_processing import read_raw_recording
from neural_manifolds.statistics.study_transfer import independent_weights
from neural_manifolds.validation.lode import _holm_two, _jsonable


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _policy(config: Path) -> tuple[dict, str]:
    p = yaml.safe_load(config.read_text(encoding="utf-8"))
    if p["schema_version"] != 1 or p["null_kinds"] != ["label_permutation", "temporal_permutation"]:
        raise ValueError("unexpected_dream_recall_policy")
    if p["reference_mode"] != "observed_common_average" or p["channels"] != ["F3", "C3", "O1"]:
        raise ValueError("unreviewed_dream_recall_montage")
    return p, sha256_file(config)


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()[:24]


def _archive_path(root: Path, source: dict) -> Path:
    matches = list(
        (root / "constituents" / f"set-{source['set_id']}").glob(f"*/{source['archive_name']}")
    )
    if len(matches) != 1 or not matches[0].is_file():
        raise ValueError(f"source_archive_not_unique:{source['set_id']}")
    path = matches[0].resolve()
    if path.stat().st_size != source["archive_bytes"]:
        raise ValueError(f"source_archive_size_changed:{source['set_id']}")
    if source.get("published_md5"):
        digest = hashlib.md5(usedforsecurity=False)
        with path.open("rb") as stream:
            while block := stream.read(4 * 1024 * 1024):
                digest.update(block)
        if digest.hexdigest() != source["published_md5"]:
            raise ValueError(f"source_archive_differs_from_published_checksum:{source['set_id']}")
    return path


def _edf_header_channels(archive: ZipFile, member: str) -> list[str]:
    with archive.open(member) as stream:
        header = stream.read(8192)
    if len(header) < 256:
        raise ValueError("truncated_edf_header")
    try:
        count = int(header[252:256].strip())
    except ValueError as exc:
        raise ValueError("invalid_edf_channel_count") from exc
    if not 1 <= count <= 256 or len(header) < 256 + count * 16:
        raise ValueError("invalid_edf_header_length")
    return [
        header[256 + i * 16 : 256 + (i + 1) * 16].decode("ascii", "replace").strip()
        for i in range(count)
    ]


def _candidate_rows(
    name: str, source: dict, archive_path: Path, policy: dict
) -> tuple[list[dict], dict, str]:
    exclusions = Counter()
    with ZipFile(archive_path) as archive:
        members = {i.filename: i for i in archive.infolist() if not i.is_dir()}
        records_names = [n for n in members if n.endswith("Records.csv")]
        if len(records_names) != 1:
            raise ValueError(f"records_csv_not_unique:{name}")
        records_name = records_names[0]
        records_bytes = archive.read(records_name)
        rows = list(csv.DictReader(io.StringIO(records_bytes.decode("utf-8-sig"))))
        required = {
            "Filename",
            "Case ID",
            "Subject ID",
            "Experience",
            "Last sleep stage",
            "Duration",
        }
        if not rows or not required.issubset(rows[0]):
            raise ValueError(f"records_schema_changed:{name}")
        linked = []
        seen_filenames = set()
        for row in rows:
            if row["Last sleep stage"].strip() != policy["stage_code"]:
                exclusions["not_N2"] += 1
                continue
            if row["Experience"].strip() not in {policy["positive_code"], policy["negative_code"]}:
                exclusions["not_binary_report"] += 1
                continue
            filename = row["Filename"].strip().replace("\\", "/")
            if source["selection"] == "nrem_targeted_n2_only" and not re.search(
                r"_NREM\.edf$", filename, re.I
            ):
                exclusions["not_targeted_NREM_awakening"] += 1
                continue
            person = row["Subject ID"].strip()
            member = str(PurePosixPath(records_name).parent / "Data/PSG" / filename)
            if not filename or not person or not safe_member(member) or member not in members:
                exclusions["missing_or_unsafe_record_link"] += 1
                continue
            if filename in seen_filenames:
                exclusions["duplicate_filename"] += 1
                continue
            seen_filenames.add(filename)
            info = members[member]
            try:
                duration = float(row["Duration"])
                if (
                    not np.isfinite(duration)
                    or duration < policy["dream_seconds"] + source["timing_offset_seconds"]
                ):
                    raise ValueError("short_or_invalid_duration")
                if not 0 < info.file_size <= policy["max_edf_member_bytes"]:
                    raise ValueError("EDF_member_size_outside_bound")
                channels = _edf_header_channels(archive, member)
                mapped = [
                    normalized_channel(source["channel_aliases"].get(ch, ch)) for ch in channels
                ]
                if any(mapped.count(ch) != 1 for ch in policy["channels"]):
                    raise ValueError("required_channel_absent_or_ambiguous_in_EDF_header")
                remarks = row.get("Remarks", "")
                if re.search(r"(?:^|[^A-Z0-9])(?:F3|C3|O1|01)(?:[^A-Z0-9]|$)", remarks.upper()):
                    raise ValueError("target_channel_flagged_in_source_remarks")
            except (BadZipFile, EOFError) as exc:
                exclusions[f"unreadable_published_EDF_member:{member}:{type(exc).__name__}"] += 1
                continue
            except ValueError as exc:
                exclusions[str(exc)] += 1
                continue
            linked.append(
                {
                    "unit_id": _hash(f"dream-recall:{name}:{filename}"),
                    "participant_id": _hash(f"dream-recall:{name}:{person}"),
                    "dataset_id": name,
                    "study_group": name,
                    "laboratory": source["laboratory"],
                    "experience": int(row["Experience"].strip() == policy["positive_code"]),
                    "report_code": int(row["Experience"].strip()),
                    "sleep_stage": 2,
                    "member": member,
                    "bytes": info.file_size,
                    "crc32": info.CRC,
                    "duration": duration,
                    "archive_path": str(archive_path),
                    "archive_bytes": archive_path.stat().st_size,
                    "timing_offset_seconds": source["timing_offset_seconds"],
                    "timing_caveat": source["timing_caveat"],
                    "report_construct": source["report_construct"],
                    "channel_aliases": source["channel_aliases"],
                }
            )
    # One awakening per person avoids treating repeated reports as independent people.
    by_person = defaultdict(list)
    for row in linked:
        by_person[row["participant_id"]].append(row)
    selected = []
    for person in sorted(by_person):
        choices = sorted(by_person[person], key=lambda r: r["member"])
        selected.append(choices[0])
        exclusions["additional_awakening_same_person"] += len(choices) - 1
    digest = hashlib.sha256(records_bytes).hexdigest()
    return sorted(selected, key=lambda r: r["unit_id"]), dict(exclusions), digest


def preflight(config: Path, source_root: Path, output: Path) -> dict:
    policy, config_sha = _policy(config)
    candidates, source_audit = [], {}
    for name, source in policy["sources"].items():
        path = _archive_path(source_root, source)
        rows, exclusions, records_sha = _candidate_rows(name, source, path, policy)
        candidates.extend(rows)
        source_audit[name] = {
            "candidate_units": len(rows),
            "candidate_people": len({r["participant_id"] for r in rows}),
            "labels": dict(Counter(r["report_code"] for r in rows)),
            "exclusions": exclusions,
            "records_sha256": records_sha,
            "archive_bytes": path.stat().st_size,
            "report_construct": source["report_construct"],
            "timing_caveat": source["timing_caveat"],
        }
    if len({r["unit_id"] for r in candidates}) != len(candidates):
        raise ValueError("pseudonymous_unit_collision")
    result = {
        "scope": "source_metadata_and_EDF_headers_before_signal_analysis",
        "config_sha256": config_sha,
        "registry": policy["registry"],
        "candidates": candidates,
        "sources": source_audit,
        "candidate_people": len({r["participant_id"] for r in candidates}),
        "scientific_gates": False,
    }
    atomic_write_json(output / "preflight.json", result)
    return result


def _preflight(config: Path, output: Path) -> tuple[dict, dict]:
    policy, config_sha = _policy(config)
    pre = _read(output / "preflight.json")
    if pre["config_sha256"] != config_sha:
        raise ValueError("config_changed_after_preflight")
    return policy, pre


def _measure_one(row: dict, policy: dict, output: Path, pre_sha: str) -> dict:
    checkpoint = output / "measure" / f"{row['unit_id']}.json"
    if checkpoint.exists():
        result = _read(checkpoint)
        if result["preflight_sha256"] != pre_sha:
            raise ValueError("measurement_checkpoint_input_changed")
        if result.get("array_path") and sha256_file(result["array_path"]) != result["array_sha256"]:
            raise ValueError("measurement_array_changed")
        return result
    result = {
        k: row[k]
        for k in (
            "unit_id",
            "participant_id",
            "dataset_id",
            "study_group",
            "laboratory",
            "experience",
            "report_code",
            "sleep_stage",
            "timing_caveat",
            "report_construct",
        )
    }
    result.update(
        preflight_sha256=pre_sha,
        primary_eligible=True,
        track="three_sensor_dream_recall",
        participant_alias_status="source_native_identifier",
    )
    try:
        path = Path(row["archive_path"])
        if path.stat().st_size != row["archive_bytes"]:
            raise ValueError("archive_size_changed_after_preflight")
        with ZipFile(path) as archive:
            member = archive.getinfo(row["member"])
            if member.file_size != row["bytes"] or row["crc32"] != member.CRC:
                raise ValueError("EDF_member_identity_changed")
            with tempfile.TemporaryDirectory(prefix="nm-recall-", dir=output / "tmp") as directory:
                source = Path(directory) / "recording.edf"
                with archive.open(member) as reader, source.open("xb") as writer:
                    total = 0
                    while block := reader.read(4 * 1024 * 1024):
                        total += len(block)
                        if total > row["bytes"]:
                            raise ValueError("EDF_member_exceeded_bound")
                        writer.write(block)
                if total != row["bytes"]:
                    raise ValueError("EDF_member_truncated")
                raw = read_raw_recording(source)
                try:
                    values, audit = prepare_window(
                        raw,
                        {
                            "stop_seconds": row["duration"] - row["timing_offset_seconds"],
                            "channel_aliases": row["channel_aliases"],
                            "timing_policy": "source_report_linked_record_end",
                        },
                        policy["channels"],
                        policy,
                    )
                finally:
                    raw.close()
        windows, reasons = window_qc(values, policy["sampling_hz"], policy)
        keep = np.asarray([not reason for reason in reasons])
        result.update(
            clean_seconds=int(keep.sum()),
            window_reasons=reasons,
            preprocessing=_jsonable(audit),
        )
        if keep.sum() < policy["minimum_clean_seconds"]:
            raise ValueError("insufficient_clean_seconds")
        indices = np.flatnonzero(keep)
        segments = np.cumsum(np.r_[True, np.diff(indices) != 1])
        selected = windows[keep]
        result["conventional"] = conventional_features(selected, policy["sampling_hz"])
        array_path = output / "arrays" / f"{row['unit_id']}.npz"
        array_path.parent.mkdir(parents=True, exist_ok=True)
        with array_path.with_suffix(".tmp").open("wb") as stream:
            np.savez_compressed(
                stream,
                sensor=sensor_trajectory(selected),
                segments=segments,
                seconds=indices,
                channels=np.asarray(policy["channels"]),
            )
        array_path.with_suffix(".tmp").replace(array_path)
        result.update(
            status="measured", array_path=str(array_path), array_sha256=sha256_file(array_path)
        )
    except (ValueError, OSError, RuntimeError, KeyError, BadZipFile, EOFError) as exc:
        result.update(status="unavailable", reason=f"{type(exc).__name__}:{exc}")
    atomic_write_json(checkpoint, _jsonable(result))
    return result


def measure(config: Path, output: Path, index: int) -> dict:
    policy, pre = _preflight(config, output)
    if not 0 <= index < 12:
        raise ValueError("fixed_measurement_array_has_twelve_tasks")
    (output / "tmp").mkdir(parents=True, exist_ok=True)
    pre_sha = sha256_file(output / "preflight.json")
    rows = [_measure_one(row, policy, output, pre_sha) for row in pre["candidates"][index::12]]
    result = {
        "index": index,
        "preflight_sha256": pre_sha,
        "units": len(rows),
        "measured": sum(r["status"] == "measured" for r in rows),
    }
    atomic_write_json(output / "measure" / f"task-{index:02d}.json", result)
    return result


def assemble(config: Path, output: Path) -> dict:
    _, pre = _preflight(config, output)
    pre_sha = sha256_file(output / "preflight.json")
    for index in range(12):
        task = _read(output / "measure" / f"task-{index:02d}.json")
        if task["index"] != index or task["preflight_sha256"] != pre_sha:
            raise ValueError("measurement_task_receipt_changed")
    rows = [_read(output / "measure" / f"{r['unit_id']}.json") for r in pre["candidates"]]
    if any(r["preflight_sha256"] != pre_sha for r in rows):
        raise ValueError("measurement_record_input_changed")
    good = [r for r in rows if r["status"] == "measured"]
    result = {
        "preflight_sha256": pre_sha,
        "records": rows,
        "measured_people": len(good),
        "by_dataset": {
            name: {
                "people": sum(r["dataset_id"] == name for r in good),
                "labels": dict(Counter(r["report_code"] for r in good if r["dataset_id"] == name)),
            }
            for name in pre["sources"]
        },
        "unavailable_reasons": dict(
            Counter(r.get("reason", "") for r in rows if r["status"] != "measured")
        ),
        "scope": "label_blind_signal_QC_and_counts_not_model_results",
    }
    atomic_write_json(output / "measurement.json", result)
    return result


def _inputs(config: Path, output: Path) -> tuple[dict, pd.DataFrame, dict]:
    policy, _ = _preflight(config, output)
    measured = _read(output / "measurement.json")
    if measured["preflight_sha256"] != sha256_file(output / "preflight.json"):
        raise ValueError("measurement_summary_input_changed")
    selected = [r for r in measured["records"] if r["status"] == "measured"]
    frame, arrays = measurement_arrays(selected, "sensor")
    if frame.empty or frame.study_group.nunique() < 3 or frame.experience.nunique() != 2:
        raise ValueError("insufficient_datasets_or_report_categories_after_QC")
    if frame.groupby("participant_id").size().max() != 1:
        raise ValueError("repeated_participant_in_analysis")
    return policy, frame, arrays


def _score(frame: pd.DataFrame, arrays: dict, policy: dict, *, all_models: bool) -> dict:
    families = (
        ["spectral_arousal", "nonlinear_scalar", "conventional_multivariate", "shared_dynamics"]
        if all_models
        else ["shared_dynamics"]
    )
    fit = nested_transfer(
        frame,
        arrays,
        outer="study_group",
        seed=policy["seed"],
        families=families,
        strengths=policy["regularization"],
        dimensions=policy["profile_dimensions"],
    )
    pred = pd.DataFrame(fit["predictions"])
    if fit["unavailable"] or pred.empty:
        raise ValueError(f"incomplete_nested_fit:{fit['unavailable']}")
    expected = set(frame.unit_id)
    for _, block in pred.groupby("model"):
        if len(block) != len(frame) or set(block.unit_id) != expected:
            raise ValueError("incomplete_held_out_predictions")
    constants = []
    for held in sorted(frame.study_group.unique()):
        train, test = frame[frame.study_group != held], frame[frame.study_group == held]
        probability = float(
            np.clip(
                np.average(train.experience, weights=independent_weights(train)), 1e-6, 1 - 1e-6
            )
        )
        for row in test.itertuples():
            constants.append(
                {
                    "unit_id": row.unit_id,
                    "participant_id": row.participant_id,
                    "study_group": row.study_group,
                    "experience": int(row.experience),
                    "model": "training_constant",
                    "probability": probability,
                    "held_group": held,
                }
            )
    pred = pd.concat([pred, pd.DataFrame(constants)], ignore_index=True)
    losses = {}
    for model, block in pred.groupby("model"):
        losses[model] = float(
            log_loss(
                block.experience,
                block.probability,
                labels=[0, 1],
                sample_weight=independent_weights(block),
            )
        )
    return {
        "predictions": _jsonable(pred.to_dict("records")),
        "pooled_equal_dataset_log_loss": losses,
        "dynamic_minus_training_constant_log_loss": losses["shared_dynamics"]
        - losses["training_constant"],
        "dynamic_absolute_log_loss": losses["shared_dynamics"],
        "tuning": _jsonable(fit["tuning"]),
        "by_dataset": _jsonable(summarize_predictions(pred, repetitions=1000, seed=policy["seed"]))
        if all_models
        else None,
        "outer_unit": "leave_one_dataset_out_with_all_transforms_fit_inside_folds",
    }


def observed(config: Path, output: Path) -> dict:
    try:
        policy, frame, arrays = _inputs(config, output)
        result = _score(frame, arrays, policy, all_models=True)
        result["status"] = "analysed"
    except ValueError as exc:
        result = {"status": "unavailable", "reason": str(exc)}
        frame = pd.DataFrame(columns=["unit_id", "study_group", "participant_id", "experience"])
    result.update(
        config_sha256=sha256_file(config),
        measurement_sha256=sha256_file(output / "measurement.json"),
        n_people=len(frame),
        n_datasets=frame.study_group.nunique(),
        label_counts=_jsonable(frame.experience.value_counts().to_dict()),
        dataset_counts=_jsonable(frame.study_group.value_counts().to_dict()),
        explicit_experience_vs_no_experience_validation=False,
        exploratory_post_original_results_selection=True,
    )
    atomic_write_json(output / "observed.json", result)
    return result


def null_chunk(config: Path, output: Path, index: int) -> dict:
    policy, _ = _preflight(config, output)
    obs = _read(output / "observed.json")
    if obs["config_sha256"] != sha256_file(config) or obs["measurement_sha256"] != sha256_file(
        output / "measurement.json"
    ):
        raise ValueError("observed_input_changed")
    if obs["status"] != "analysed":
        return {"status": "skipped_observed_unavailable"}
    _, frame, arrays = _inputs(config, output)
    n, size = policy["null_replicates_per_family"], policy["null_chunk_size"]
    chunks = (n + size - 1) // size
    if not 0 <= index < 2 * chunks:
        raise ValueError("invalid_null_array_index")
    kind = policy["null_kinds"][index // chunks]
    start = (index % chunks) * size
    rows = []
    for replicate in range(start, min(start + size, n)):
        path = output / "null" / f"{kind}-{replicate:04d}.json"
        if path.exists():
            row = _read(path)
            if row["observed_sha256"] != sha256_file(output / "observed.json"):
                raise ValueError("null_checkpoint_input_changed")
        else:
            rng = np.random.default_rng(
                policy["seed"] + 1_000_003 * (index // chunks + 1) + replicate
            )
            shuffled = frame.copy()
            transformed = arrays
            if kind == "label_permutation":
                for _, block in shuffled.groupby("study_group"):
                    shuffled.loc[block.index, "experience"] = rng.permutation(
                        block.experience.to_numpy()
                    )
            else:
                transformed = null_arrays(arrays, kind, rng)
            try:
                fit = _score(shuffled, transformed, policy, all_models=False)
                statistic = fit[
                    "dynamic_minus_training_constant_log_loss"
                    if kind == "label_permutation"
                    else "dynamic_absolute_log_loss"
                ]
                row = {
                    "status": "analysed",
                    "kind": kind,
                    "replicate": replicate,
                    "statistic": statistic,
                }
            except ValueError as exc:
                row = {
                    "status": "unavailable",
                    "kind": kind,
                    "replicate": replicate,
                    "reason": str(exc),
                }
            row["observed_sha256"] = sha256_file(output / "observed.json")
            atomic_write_json(path, row)
        rows.append(row)
    result = {
        "index": index,
        "kind": kind,
        "start": start,
        "count": len(rows),
        "analysed": sum(r["status"] == "analysed" for r in rows),
    }
    atomic_write_json(output / "null" / f"task-{index:03d}.json", result)
    return result


def finalize(config: Path, output: Path) -> dict:
    policy, _ = _preflight(config, output)
    obs = _read(output / "observed.json")
    n = policy["null_replicates_per_family"]
    tests = []
    for kind, metric in (
        ("label_permutation", "dynamic_minus_training_constant_log_loss"),
        ("temporal_permutation", "dynamic_absolute_log_loss"),
    ):
        rows = []
        for index in range(n):
            path = output / "null" / f"{kind}-{index:04d}.json"
            if path.exists():
                row = _read(path)
                if (
                    row["observed_sha256"] != sha256_file(output / "observed.json")
                    or row["kind"] != kind
                    or row["replicate"] != index
                ):
                    raise ValueError("null_checkpoint_identity_changed")
                rows.append(row)
        valid = [r["statistic"] for r in rows if r["status"] == "analysed"]
        complete = obs["status"] == "analysed" and len(rows) == len(valid) == n
        p = (1 + sum(x <= obs[metric] for x in valid)) / (n + 1) if complete else None
        tests.append(
            {
                "kind": kind,
                "metric": metric,
                "observed": obs.get(metric),
                "requested_refits": n,
                "received_refits": len(rows),
                "valid_refits": len(valid),
                "p_lower_plus_one": p,
                "status": "complete" if complete else "incomplete",
            }
        )
    if all(t["status"] == "complete" for t in tests):
        for test, value in zip(
            tests, _holm_two([t["p_lower_plus_one"] for t in tests]), strict=True
        ):
            test["holm_two_test_p"] = value
    result = {
        "status": "complete" if all(t["status"] == "complete" for t in tests) else "incomplete",
        "tests": tests,
        "observed_status": obs["status"],
        "config_sha256": sha256_file(config),
        "preflight_sha256": sha256_file(output / "preflight.json"),
        "measurement_sha256": sha256_file(output / "measurement.json"),
        "observed_sha256": sha256_file(output / "observed.json"),
        "distinct_from_strict_E_NE_and_original_72_test_family": True,
        "scientific_gates": False,
    }
    atomic_write_json(output / "final.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage", choices=("preflight", "measure", "assemble", "observed", "null", "finalize")
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--index", type=int)
    args = parser.parse_args()
    if args.stage == "preflight":
        if args.source_root is None:
            parser.error("preflight requires --source-root")
        result = preflight(args.config, args.source_root, args.output)
    elif args.stage == "measure":
        if args.index is None:
            parser.error("measure requires --index")
        result = measure(args.config, args.output, args.index)
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
    print(
        json.dumps(
            _jsonable(
                {
                    k: v
                    for k, v in result.items()
                    if k not in {"candidates", "records", "predictions", "tuning"}
                }
            ),
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
