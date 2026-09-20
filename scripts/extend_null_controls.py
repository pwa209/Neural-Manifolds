"""Fixed-count continuation of reviewed nulls, using an unchanged analysis release.

Run on the data host. The plan is sealed before continuation; original seeds,
observed statistics and the complete test family are retained. No significance
stopping and no replacement seeds for failed fits. Outputs are aggregate only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

TRACKS = {"core": "sparse", "hd": "high_density"}
REPS = ("sensor", "encoder", "time_frequency")
KINDS = ("label_permutation", "temporal_permutation", "representation_phase")
METRICS = (
    "absolute_dynamic_log_loss",
    "conventional_multivariate",
    "nonlinear_scalar",
    "spectral_arousal",
)
TOTAL = 5000
ORIGINAL = 100
CHUNK = 25


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def atomic(path, value):
    from neural_manifolds.provenance import atomic_write_json

    # Strict serialization first: unavailable/nonfinite results must not become data.
    json.dumps(value, allow_nan=False)
    atomic_write_json(path, value)


def checked(record):
    path = Path(record["path"])
    if sha(path) != record["sha256"]:
        raise ValueError(f"Input changed: {path.name}")
    return path


def pointer(path):
    return {"path": str(Path(path).resolve()), "sha256": sha(path)}


def family_keys():
    return {(p, r, k, m) for p in TRACKS for r in REPS for k in KINDS for m in METRICS}


def compact(summary):
    models = [r for r in summary["models"] if r["model"] == "shared_dynamics"]
    if len(models) != 3 or len({r["study_group"] for r in models}) != 3:
        raise ValueError("All three held-out studies are required")
    result = {"absolute_dynamic_log_loss": float(np.mean([r["log_loss"] for r in models]))}
    for metric in METRICS[1:]:
        rows = [r for r in summary["paired_comparisons"] if r["baseline"] == metric]
        if len(rows) != 1 or rows[0]["independent_studies"] != 3:
            raise ValueError("Missing or duplicated paired comparator")
        result[metric] = float(rows[0]["dynamic_minus_baseline_log_loss"])
    if not np.isfinite(list(result.values())).all():
        raise ValueError("Nonfinite statistic")
    return result


def chunk_spec(task):
    chunks = (TOTAL - ORIGINAL + CHUNK - 1) // CHUNK
    if not 0 <= task < chunks * 2:
        raise ValueError("Invalid array index")
    port = ("core", "hd")[task % 2]
    start = ORIGINAL + (task // 2) * CHUNK
    return port, range(start, min(start + CHUNK, TOTAL))


def validate_family(rows):
    keys = [(r["portfolio"], r["representation"], r["kind"], r["metric"]) for r in rows]
    if len(keys) != 72 or set(keys) != family_keys():
        raise ValueError("The complete, unique 72-test family is required")


def load_plan(path):
    plan = read(path)
    declared = plan.pop("plan_sha256")
    if identity(plan) != declared:
        raise ValueError("Sealed plan changed")
    if plan["total_replicates"] != TOTAL or plan["original_replicates"] != ORIGINAL:
        raise ValueError("Fixed refit count changed")
    validate_family(plan["observed"])
    plan["plan_sha256"] = declared
    if sha(__file__) != plan["runner_sha256"]:
        raise ValueError("Continuation runner changed")
    if str(Path.cwd().resolve()) != plan["analysis_release"]:
        raise ValueError("Wrong original analysis release")
    if sha(Path.cwd() / "release_manifest.json") != plan["release_manifest_sha256"]:
        raise ValueError("Original source manifest changed")
    return plan


def prepare(args):
    import yaml

    root = args.root.resolve()
    if root.exists():
        raise FileExistsError("Use a new continuation directory")
    review = read(args.review)
    if sha(args.review) != args.review_sha256:
        raise ValueError("Reviewed evidence checksum mismatch")
    observed = []
    for row in review["null_controls"]:
        track, rep = row["analysis"].split(":")
        if track != TRACKS[row["portfolio"]] or row["null_replicates"] != ORIGINAL:
            raise ValueError("Unexpected reviewed cohort/count")
        observed.append(
            {
                "portfolio": row["portfolio"],
                "representation": rep,
                "kind": row["kind"],
                "metric": row["metric"],
                "observed": row["observed"],
                "original_p": row["p_lower"],
            }
        )
    validate_family(observed)
    source_manifest = read(Path.cwd() / "release_manifest.json")
    records = []
    sources = []
    for port, evidence in [("core", args.core_evidence), ("hd", args.hd_evidence)]:
        expected = [r for r in review["sources"] if r["label"] == port]
        if len(expected) != 1 or sha(evidence) != expected[0]["sha256"]:
            raise ValueError("Observed synthesis changed")
        sources.append(pointer(evidence))
        for entry in read(evidence)["evidence"]:
            if entry["artifact"] not in {"controls.json", "transfer.json"}:
                continue
            source = checked({"path": entry["source_path"], "sha256": entry["sha256"]})
            sources.append(pointer(source))
            if entry["artifact"] != "controls.json":
                continue
            result = read(source)
            if not result.get("full_nested_refit"):
                raise ValueError("Original controls are not nested refits")
            for row in result["rows"]:
                if row["track"] != TRACKS[port]:
                    continue
                if "summary" not in row:
                    raise ValueError("Original null cell missing")
                records.append(
                    {
                        "portfolio": port,
                        "representation": row["representation"],
                        "kind": row["kind"],
                        "replicate": row["replicate"],
                        "statistics": compact(row["summary"]),
                    }
                )
    keys = {(r["portfolio"], r["representation"], r["kind"], r["replicate"]) for r in records}
    expected = {(p, r, k, i) for p in TRACKS for r in REPS for k in KINDS for i in range(ORIGINAL)}
    if keys != expected or len(records) != len(expected):
        raise ValueError("Original 100 refits incomplete or duplicated")
    for row in observed:
        values = [
            r["statistics"][row["metric"]]
            for r in records
            if all(r[k] == row[k] for k in ("portfolio", "representation", "kind"))
        ]
        if not np.isclose(
            (1 + np.count_nonzero(np.array(values) <= row["observed"])) / 101,
            row["original_p"],
            atol=1e-12,
            rtol=0,
        ):
            raise ValueError("Original refits do not reproduce reviewed tails")
    policy_path = Path.cwd() / "configs/revised_execution.yaml"
    policy = yaml.safe_load(policy_path.read_text())
    # The extension reused analytically identical inference, controls and dependencies.
    hd_manifest = read(args.hd_release / "release_manifest.json")
    for name, digest in source_manifest["files"].items():
        relevant = name.startswith(
            ("src/neural_manifolds/manifold/", "src/neural_manifolds/statistics/")
        )
        relevant |= name in {
            "src/neural_manifolds/revised/inference.py",
            "src/neural_manifolds/revised/controls.py",
            "configs/revised_execution.yaml",
        }
        if relevant and (
            hd_manifest["files"].get(name) != digest or sha(args.hd_release / name) != digest
        ):
            raise ValueError(f"Analytical implementations differ: {name}")
    bundles = {p: pointer(getattr(args, f"{p}_bundle")) for p in TRACKS}
    for bundle in bundles.values():
        for item in read(checked(bundle))["inputs"]:
            checked(item)
    plan = dict(
        schema_version=1,
        created_utc=datetime.now(UTC).isoformat(),
        status="exploratory_fixed_count_continuation_not_preregistration",
        total_replicates=TOTAL,
        original_replicates=ORIGINAL,
        seed_rule="original policy seed + replicate index, indices 0..4999",
        policy=policy,
        analysis_release=str(Path.cwd().resolve()),
        release_manifest_sha256=sha(Path.cwd() / "release_manifest.json"),
        runner_sha256=sha(__file__),
        sources=sources,
        review=pointer(args.review),
        measurement_bundles=bundles,
        observed=observed,
        original_rows=records,
        chunk_size=CHUNK,
        array_tasks=len(range(0, TOTAL - ORIGINAL, CHUNK)) * 2,
        max_parallel_cpus=50,
        no_significance_stopping=True,
        retain_failed_seed=True,
        partial_results_are_not_final_p_values=True,
        output_root=str(args.output_root.resolve()),
    )
    plan["plan_sha256"] = identity(plan)
    root.mkdir(parents=True)
    for sub in ("logs", "results", "locks", "checks"):
        (args.output_root / sub).mkdir(parents=True, exist_ok=True)
    atomic(root / "plan.json", plan)
    print("PLAN_SEALED", plan["plan_sha256"], "tasks", plan["array_tasks"], flush=True)


def load_arrays(plan, port):
    from neural_manifolds.revised.driver import verified_bundle
    from neural_manifolds.revised.inference import load_measurements, measurement_arrays

    paths = verified_bundle(checked(plan["measurement_bundles"][port]))
    records = [r for r in load_measurements(paths) if r["track"] == TRACKS[port]]
    return {rep: measurement_arrays(records, rep) for rep in REPS}


def fit_cell(frame, arrays, policy, replicate, kind):
    import pandas as pd

    from neural_manifolds.revised.controls import null_arrays
    from neural_manifolds.revised.inference import nested_transfer, summarize_predictions

    rng = np.random.default_rng(policy["seed"] + replicate)
    shuffled = frame.copy()
    transformed = arrays
    if kind == "label_permutation":
        for _, block in shuffled.groupby(["study_group", "participant_id", "sleep_stage"]):
            shuffled.loc[block.index, "experience"] = rng.permutation(block.experience)
    else:
        transformed = null_arrays(arrays, kind, rng)
    fit = nested_transfer(
        shuffled,
        transformed,
        seed=policy["seed"],
        dimensions=policy["profile_dimensions"],
        strengths=policy["regularization"],
    )
    predictions = pd.DataFrame(fit["predictions"])
    expected_models = {"shared_dynamics", *METRICS[1:]}
    counts = predictions.groupby("model").unit_id.agg(["count", "nunique"])
    if (
        fit["unavailable"]
        or set(counts.index) != expected_models
        or not (counts == len(frame)).all().all()
        or set(predictions.unit_id) != set(frame.unit_id)
    ):
        raise ValueError("Incomplete nested held-out predictions; retain failed seed")
    return compact(summarize_predictions(predictions, repetitions=20, seed=policy["seed"]))


def validate_record(record, plan, port, replicate):
    declared = record.get("record_sha256")
    if identity({k: v for k, v in record.items() if k != "record_sha256"}) != declared:
        raise ValueError("Checkpoint checksum mismatch")
    if (
        record["plan_sha256"] != plan["plan_sha256"]
        or record["portfolio"] != port
        or record["replicate"] != replicate
        or record["seed"] != plan["policy"]["seed"] + replicate
    ):
        raise ValueError("Checkpoint identity mismatch")
    if record["status"] not in {"partial", "complete"}:
        raise ValueError("Unknown checkpoint status")
    keys = [(r["representation"], r["kind"]) for r in record["rows"]]
    if len(set(keys)) != len(keys) or not set(keys) <= {(r, k) for r in REPS for k in KINDS}:
        raise ValueError("Checkpoint duplicate or unknown null cells")
    for row in record["rows"]:
        if (
            set(row["statistics"]) != set(METRICS)
            or not np.isfinite(list(row["statistics"].values())).all()
        ):
            raise ValueError("Checkpoint statistics invalid")
    if record["status"] == "complete" and len(keys) != 9:
        raise ValueError("Incomplete checkpoint marked complete")


def save_record(path, record):
    record = {k: v for k, v in record.items() if k != "record_sha256"}
    record["record_sha256"] = identity(record)
    atomic(path, record)


def run_seed(plan, port, replicate, arrays, *, canary=False):
    base = Path(plan["output_root"])
    path = base / ("checks" if canary else "results") / f"{port}-{replicate:04d}.json"
    if path.exists():
        record = read(path)
        validate_record(record, plan, port, replicate)
    else:
        record = dict(
            plan_sha256=plan["plan_sha256"],
            portfolio=port,
            replicate=replicate,
            seed=plan["policy"]["seed"] + replicate,
            status="partial",
            rows=[],
        )
    done = {(r["representation"], r["kind"]) for r in record["rows"]}
    for rep in REPS:
        for kind in KINDS:
            if (rep, kind) in done:
                continue
            start = time.monotonic()
            values = fit_cell(*arrays[rep], plan["policy"], replicate, kind)
            if canary:
                original = [
                    r
                    for r in plan["original_rows"]
                    if r["portfolio"] == port
                    and r["replicate"] == replicate
                    and r["representation"] == rep
                    and r["kind"] == kind
                ]
                if len(original) != 1 or not all(
                    np.isclose(values[m], original[0]["statistics"][m], atol=1e-8, rtol=1e-7)
                    for m in METRICS
                ):
                    raise ValueError(f"Canary differs from immutable original: {port} {rep} {kind}")
            record["rows"].append(
                dict(
                    representation=rep,
                    kind=kind,
                    statistics=values,
                    elapsed_seconds=time.monotonic() - start,
                )
            )
            record["status"] = "complete" if len(record["rows"]) == 9 else "partial"
            save_record(path, record)
            print(
                "CELL_COMPLETE",
                port,
                replicate,
                rep,
                kind,
                round(time.monotonic() - start, 2),
                flush=True,
            )
    return path


def work(args, canary=False):
    from filelock import FileLock

    plan = load_plan(args.plan)
    port, seeds = (args.portfolio, range(1)) if canary else chunk_spec(args.task)
    base = Path(plan["output_root"])
    for item in plan["sources"]:
        # Evidence and original control artifacts are immutable, checked before computation.
        checked(item)
    with FileLock(
        str(base / "locks" / (f"canary-{port}.lock" if canary else f"task-{args.task}.lock")),
        timeout=1,
    ):
        if not canary:
            for p in TRACKS:
                check = read(base / "checks" / f"{p}-0000.json")
                validate_record(check, plan, p, 0)
                if check["status"] != "complete":
                    raise ValueError("Engineering equivalence check has not passed")
        pending = []
        for seed in seeds:
            path = base / ("checks" if canary else "results") / f"{port}-{seed:04d}.json"
            if path.exists():
                record = read(path)
                validate_record(record, plan, port, seed)
                if record["status"] == "complete":
                    continue
            pending.append(seed)
        if pending:
            arrays = load_arrays(plan, port)
            for seed in pending:
                run_seed(plan, port, seed, arrays, canary=canary)
        print("CHUNK_COMPLETE", port, args.task, len(seeds), flush=True)


def holm(values):
    values = np.asarray(values, float)
    order = np.argsort(values)
    result = np.empty(len(values))
    result[order] = np.minimum(
        1, np.maximum.accumulate(values[order] * np.arange(len(values), 0, -1))
    )
    return result.tolist()


def aggregate(args):
    from scipy.stats import binomtest

    plan = load_plan(args.plan)
    root = args.plan.parent
    base = Path(plan["output_root"])
    for item in plan["sources"]:
        checked(item)
    values = {key: {} for key in family_keys()}
    for row in plan["original_rows"]:
        for metric, value in row["statistics"].items():
            values[row["portfolio"], row["representation"], row["kind"], metric][
                row["replicate"]
            ] = value
    missing = []
    for port in TRACKS:
        for seed in range(ORIGINAL, TOTAL):
            path = base / "results" / f"{port}-{seed:04d}.json"
            if not path.exists():
                missing.append([port, seed, "missing"])
                continue
            row = read(path)
            validate_record(row, plan, port, seed)
            if row["status"] != "complete":
                missing.append([port, seed, "partial"])
                continue
            for cell in row["rows"]:
                for metric, value in cell["statistics"].items():
                    values[port, cell["representation"], cell["kind"], metric][seed] = value
    if missing:
        atomic(
            root / "incomplete.json",
            {
                "plan_sha256": plan["plan_sha256"],
                "missing": missing,
                "final_p_values_not_computed": True,
            },
        )
        raise ValueError(
            f"{len(missing)} unfinished portfolio/seed records; no reduced denominators"
        )
    rows = []
    for observed in plan["observed"]:
        key = tuple(observed[k] for k in ("portfolio", "representation", "kind", "metric"))
        series = values[key]
        if set(series) != set(range(TOTAL)):
            raise ValueError("Non-exhaustive or duplicated seed set")
        v = np.array([series[i] for i in range(TOTAL)])
        count = int(np.count_nonzero(v <= observed["observed"]))
        ci = binomtest(count, TOTAL).proportion_ci(method="exact")
        rows.append(
            {
                **observed,
                "null_replicates": TOTAL,
                "null_lower_or_equal": count,
                "p_lower": (count + 1) / (TOTAL + 1),
                "null_tail_mc_binomial95": [ci.low, ci.high],
                "null_median": float(np.median(v)),
                "null_range": [float(v.min()), float(v.max())],
            }
        )
    for row, adjusted in zip(rows, holm([r["p_lower"] for r in rows]), strict=True):
        row["p_holm_all_null_comparisons"] = adjusted
    validate_family(rows)
    target = root / "null_refit_statistics_5000.csv"
    with target.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["portfolio", "representation", "kind", "metric", "replicate", "null_statistic"]
        )
        for key in sorted(values):
            for seed, value in sorted(values[key].items()):
                writer.writerow([*key, seed, value])
    atomic(
        root / "null_review_5000.json",
        dict(
            plan_sha256=plan["plan_sha256"],
            complete=True,
            exploratory=True,
            registered_or_preregistered=False,
            observed_analyses_unchanged=True,
            family_size=72,
            null_controls=rows,
            source_statistics=pointer(target),
            completed_utc=datetime.now(UTC).isoformat(),
            caveats=[
                "Fixed-count continuation chosen after reviewing 100 refits; not confirmatory preregistration.",
                "Binomial intervals describe Monte Carlo tail uncertainty, not biological uncertainty.",
                "Label exchangeability and limited independent study count remain unchanged.",
            ],
        ),
    )
    print("NULL_REANALYSIS_COMPLETE", len(rows), TOTAL, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "canary", "worker", "aggregate"])
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--task", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    parser.add_argument("--portfolio", choices=tuple(TRACKS))
    for name in [
        "root",
        "output-root",
        "review",
        "core-evidence",
        "hd-evidence",
        "core-bundle",
        "hd-bundle",
        "hd-release",
    ]:
        parser.add_argument(f"--{name}", type=Path)
    parser.add_argument("--review-sha256")
    args = parser.parse_args()
    required = (
        [
            "root",
            "output_root",
            "review",
            "review_sha256",
            "core_evidence",
            "hd_evidence",
            "core_bundle",
            "hd_bundle",
            "hd_release",
        ]
        if args.mode == "prepare"
        else ["plan"]
    )
    if args.mode == "canary":
        required.append("portfolio")
    for name in required:
        if getattr(args, name) is None:
            parser.error(f"--{name.replace('_', '-')} is required for {args.mode}")
    if args.mode == "prepare":
        prepare(args)
    elif args.mode == "aggregate":
        aggregate(args)
    else:
        work(args, canary=args.mode == "canary")


if __name__ == "__main__":
    main()
