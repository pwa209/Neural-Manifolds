"""Operational/statistical contracts use synthetic fixtures, not study data."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location(
    "null_continuation", Path(__file__).parents[1] / "scripts/extend_null_controls.py"
)
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


def test_array_covers_each_new_portfolio_seed_exactly_once():
    all_seeds = [(p, i) for task in range(392) for p, seeds in [m.chunk_spec(task)] for i in seeds]
    assert len(all_seeds) == 9800
    assert len(set(all_seeds)) == 9800
    assert set(all_seeds) == {(p, i) for p in m.TRACKS for i in range(100, 5000)}
    for task in [-1, 392]:
        with pytest.raises(ValueError):
            m.chunk_spec(task)


def family():
    return [
        dict(zip(("portfolio", "representation", "kind", "metric"), key, strict=True))
        for key in sorted(m.family_keys())
    ]


def test_full_family_cannot_be_selectively_reduced_or_duplicated():
    rows = family()
    m.validate_family(rows)
    for invalid in [rows[:-1], rows + rows[:1], rows[:-1] + rows[:1]]:
        with pytest.raises(ValueError):
            m.validate_family(invalid)


def summary():
    return {
        "models": [
            {"model": "shared_dynamics", "study_group": s, "log_loss": v}
            for s, v in zip("ABC", [0.4, 0.6, 0.8], strict=True)
        ],
        "paired_comparisons": [
            {"baseline": b, "independent_studies": 3, "dynamic_minus_baseline_log_loss": -0.03}
            for b in m.METRICS[1:]
        ],
    }


def test_exact_matched_statistics_and_complete_folds_required():
    compact = m.compact(summary())
    assert compact["absolute_dynamic_log_loss"] == pytest.approx(0.6)
    assert set(compact) == set(m.METRICS)
    for edit in ["missing_study", "duplicate_comparator", "nan"]:
        s = summary()
        if edit == "missing_study":
            s["models"].pop()
        elif edit == "duplicate_comparator":
            s["paired_comparisons"].append(s["paired_comparisons"][0])
        else:
            s["models"][0]["log_loss"] = float("nan")
        with pytest.raises(ValueError):
            m.compact(s)


def test_resolution_and_holm_are_full_family():
    assert min(m.holm([1 / 101] * 72)) == pytest.approx(72 / 101)
    assert min(m.holm([1 / 5001] + [1.0] * 71)) == pytest.approx(72 / 5001)
    assert 72 / 5001 < 0.05
    np.testing.assert_allclose(m.holm([0.03, 0.001, 0.02]), [0.04, 0.003, 0.04])


def test_checkpoint_rejects_mutation_and_fake_completion():
    plan = {"plan_sha256": "synthetic", "policy": {"seed": 12}}
    record = {
        "plan_sha256": "synthetic",
        "portfolio": "core",
        "replicate": 100,
        "seed": 112,
        "status": "partial",
        "rows": [],
    }
    record["record_sha256"] = m.identity(record)
    m.validate_record(record, plan, "core", 100)
    with pytest.raises(ValueError):
        m.validate_record(record, plan, "hd", 100)
    record["status"] = "complete"
    with pytest.raises(ValueError):
        m.validate_record(record, plan, "core", 100)
    record["record_sha256"] = m.identity({k: v for k, v in record.items() if k != "record_sha256"})
    with pytest.raises(ValueError, match="Incomplete"):
        m.validate_record(record, plan, "core", 100)


def test_canary_and_resume_never_replace_completed_cells(tmp_path, monkeypatch):
    stats = {metric: 0.25 for metric in m.METRICS}
    plan = {
        "output_root": str(tmp_path),
        "plan_sha256": "synthetic",
        "policy": {"seed": 12},
        "original_rows": [
            {
                "portfolio": "core",
                "replicate": 0,
                "representation": r,
                "kind": k,
                "statistics": stats,
            }
            for r in m.REPS
            for k in m.KINDS
        ],
    }
    calls = []
    monkeypatch.setattr(m, "fit_cell", lambda *args: calls.append(args) or stats.copy())
    arrays = {r: (None, None) for r in m.REPS}
    path = m.run_seed(plan, "core", 0, arrays, canary=True)
    assert len(calls) == 9
    assert m.read(path)["status"] == "complete"
    m.run_seed(plan, "core", 0, arrays, canary=True)
    assert len(calls) == 9
    plan["original_rows"][0]["statistics"] = {metric: 0.8 for metric in m.METRICS}
    plan["output_root"] = str(tmp_path / "different-canary")
    with pytest.raises(ValueError, match="Canary differs"):
        m.run_seed(plan, "core", 0, arrays, canary=True)


def test_transform_and_nested_fit_arguments_match_original_driver(tmp_path, monkeypatch):
    import copy

    import pandas as pd

    from neural_manifolds.revised import controls, inference

    frame = pd.DataFrame(
        [
            dict(
                study_group="A",
                participant_id="p",
                sleep_stage=2,
                unit_id=f"u{i}",
                experience=i % 2,
            )
            for i in range(6)
        ]
    )
    arrays = {
        unit: dict(
            trajectory=np.arange(24, dtype=float).reshape(8, 3),
            regions={"r": np.arange(16, dtype=float).reshape(8, 2)},
            segments=np.repeat([0, 1], 4),
        )
        for unit in frame.unit_id
    }
    original = copy.deepcopy(arrays)
    calls = []

    def nested(f, a, **kwargs):
        calls.append((f.copy(), copy.deepcopy(a), kwargs))
        return {
            "unavailable": [],
            "predictions": [
                {"unit_id": u, "model": model}
                for u in f.unit_id
                for model in ("shared_dynamics", *m.METRICS[1:])
            ],
        }

    for module in [controls, inference]:
        monkeypatch.setattr(module, "nested_transfer", nested)
        monkeypatch.setattr(module, "summarize_predictions", lambda *a, **k: summary())
    monkeypatch.setattr(controls, "load_measurements", lambda paths: [{"track": "sparse"}])
    monkeypatch.setattr(controls, "measurement_arrays", lambda *a: (frame, arrays))
    policy = {"seed": 13, "profile_dimensions": [1, 2, 3, 4, 5], "regularization": [0.1, 1, 10]}
    controls.run_controls([], tmp_path, policy, replicate=137)
    for kind in m.KINDS:
        m.fit_cell(frame, arrays, policy, 137, kind)
    assert len(calls) == 12
    for old, new in zip(calls[:3], calls[-3:], strict=True):
        pd.testing.assert_frame_equal(old[0], new[0])
        assert old[2] == new[2]
        for unit in arrays:
            np.testing.assert_array_equal(old[1][unit]["trajectory"], new[1][unit]["trajectory"])
            np.testing.assert_array_equal(
                old[1][unit]["regions"]["r"], new[1][unit]["regions"]["r"]
            )
            np.testing.assert_array_equal(arrays[unit]["trajectory"], original[unit]["trajectory"])
    assert frame.experience.tolist() == [0, 1, 0, 1, 0, 1]


def test_aggregation_requires_every_seed_and_keeps_full_family(tmp_path, monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(m, "TOTAL", 3)
    monkeypatch.setattr(m, "ORIGINAL", 1)
    plan = {
        "plan_sha256": "synthetic",
        "sources": [],
        "output_root": str(tmp_path),
        "policy": {"seed": 12},
        "original_rows": [],
        "observed": [{**r, "observed": 0.0, "original_p": 0.5} for r in family()],
    }
    for p in m.TRACKS:
        for r in m.REPS:
            for k in m.KINDS:
                plan["original_rows"].append(
                    dict(
                        portfolio=p,
                        representation=r,
                        kind=k,
                        replicate=0,
                        statistics={s: 0.5 for s in m.METRICS},
                    )
                )
        for seed in [1, 2]:
            record = dict(
                plan_sha256="synthetic",
                portfolio=p,
                replicate=seed,
                seed=12 + seed,
                status="complete",
                rows=[
                    dict(representation=r, kind=k, statistics={s: 0.5 for s in m.METRICS})
                    for r in m.REPS
                    for k in m.KINDS
                ],
            )
            m.save_record(tmp_path / "results" / f"{p}-{seed:04d}.json", record)
    monkeypatch.setattr(m, "load_plan", lambda _: plan)
    args = SimpleNamespace(plan=tmp_path / "plan.json")
    missing = tmp_path / "results" / "hd-0002.json"
    saved = m.read(missing)
    missing.unlink()
    with pytest.raises(ValueError, match="no reduced denominators"):
        m.aggregate(args)
    assert not (tmp_path / "null_review_5000.json").exists()
    assert m.read(tmp_path / "incomplete.json")["final_p_values_not_computed"]
    m.save_record(missing, saved)
    m.aggregate(args)
    report = m.read(tmp_path / "null_review_5000.json")
    assert report["complete"] and report["family_size"] == 72
    assert len(report["null_controls"]) == 72
    assert all(r["p_lower"] == 0.25 and r["null_replicates"] == 3 for r in report["null_controls"])
    assert all(r["p_holm_all_null_comparisons"] == 1 for r in report["null_controls"])


def submission_module():
    spec = importlib.util.spec_from_file_location(
        "null_queue", Path(__file__).parents[1] / "scripts/alliance/queue_null_extension.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def submission_args(tmp_path):
    from types import SimpleNamespace

    plan_path = tmp_path / "plan.json"
    m.atomic(
        plan_path,
        dict(
            plan_sha256="synthetic",
            array_tasks=392,
            max_parallel_cpus=50,
            analysis_release=str(tmp_path),
            output_root=str(tmp_path),
        ),
    )
    return SimpleNamespace(
        plan=plan_path,
        account="synthetic_cpu",
        max_submitted=1000,
        runner=tmp_path / "runner.py",
        wrapper=tmp_path / "wrapper.sh",
    )


def test_queue_graph_has_50_cpu_limit_and_exact_dependencies(tmp_path, monkeypatch):
    q = submission_module()
    args = submission_args(tmp_path)
    calls = []

    def fake_command(argv):
        calls.append(argv)
        return "" if argv[0] == "squeue" else str(1000 + len(calls))

    monkeypatch.setattr(q, "command", fake_command)
    for name in ["NM_NULL_RUNNER", "NM_NULL_PLAN", "NM_ORIGINAL_RELEASE"]:
        monkeypatch.setenv(name, "synthetic")
    q.submit(args)
    sbatch = [c for c in calls if c[0] == "sbatch"]
    assert len(sbatch) == 5
    assert all("--cpus-per-task=1" in c for c in sbatch)
    assert "--array=0-391%50" in sbatch[2] and "--array=0-391%50" in sbatch[3]
    assert "--dependency=afterok:1002:1003" in sbatch[2]
    assert "--dependency=afterany:1004" in sbatch[3]
    assert "--dependency=afterany:1005" in sbatch[4]
    q.submit(args)
    assert len([c for c in calls if c[0] == "sbatch"]) == 5


def test_submission_capacity_failure_does_not_submit_jobs(tmp_path, monkeypatch):
    q = submission_module()
    calls = []
    monkeypatch.setattr(q, "command", lambda args: calls.append(args) or "\n".join(["job"] * 214))
    with pytest.raises(ValueError, match="Insufficient"):
        q.submit(submission_args(tmp_path))
    assert len(calls) == 1 and calls[0][0] == "squeue"


def test_ambiguous_submission_cannot_be_blindly_retried(tmp_path, monkeypatch):
    q = submission_module()
    calls = []

    def fake_command(args):
        calls.append(args)
        return "" if args[0] == "squeue" else "uncertain"

    monkeypatch.setattr(q, "command", fake_command)
    for name in ["NM_NULL_RUNNER", "NM_NULL_PLAN", "NM_ORIGINAL_RELEASE"]:
        monkeypatch.setenv(name, "synthetic")
    args = submission_args(tmp_path)
    with pytest.raises(ValueError, match="Unrecognized"):
        q.submit(args)
    with pytest.raises(ValueError, match="Uncertain previous"):
        q.submit(args)
    assert len([c for c in calls if c[0] == "sbatch"]) == 1
