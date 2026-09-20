"""Exploratory audit of immutable evidence; never overwrites original analyses.

Run on the data host. Exports aggregate results only, not participant records.
Intervals and resampling are conditional on already fitted predictions/features.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def holm(values):
    values = np.asarray(values, float)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite p value")
    order = np.argsort(values)
    adjusted = np.empty(len(values))
    adjusted[order] = np.minimum(
        1, np.maximum.accumulate(values[order] * np.arange(len(values), 0, -1))
    )
    return adjusted.tolist()


def json_finite(value, path="", missing=None):
    """Preserve unavailable original numbers as null, with explicit audit paths."""
    if missing is None:
        missing = []
    if isinstance(value, dict):
        return {k: json_finite(v, f"{path}/{k}", missing) for k, v in value.items()}
    if isinstance(value, list):
        return [json_finite(v, f"{path}/{i}", missing) for i, v in enumerate(value)]
    if isinstance(value, float) and not math.isfinite(value):
        missing.append(path)
        return None
    return value


def empirical_tail(observed, values):
    """Lower is better; plus-one Monte Carlo tail under the specified null."""
    values = np.asarray(values, float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("No finite null replicates")
    return float((1 + np.count_nonzero(values <= observed)) / (len(values) + 1))


def model_loss(summary, model):
    rows = [r["log_loss"] for r in summary.get("models", []) if r["model"] == model]
    return float(np.mean(rows)) if rows else None


def precision(predictions, baseline):
    p = pd.DataFrame(predictions)
    if p.empty:
        return {}
    prob = np.clip(p.probability, 1e-12, 1 - 1e-12)
    p["loss"] = -p.experience * np.log(prob) - (1 - p.experience) * np.log1p(-prob)
    v = p.pivot(index=["study_group", "participant_id", "unit_id"], columns="model", values="loss")
    if baseline not in v or "shared_dynamics" not in v:
        return {}
    d = (v.shared_dynamics - v[baseline]).dropna()
    persons = d.groupby(level=[0, 1]).mean()
    studies = persons.groupby(level=0).mean()
    n = len(studies)
    se = float(studies.std(ddof=1) / np.sqrt(n)) if n > 1 else None
    half = float(stats.t.ppf(0.975, n - 1) * se) if se is not None else None
    return {
        "independent_studies": n,
        "participants": len(persons),
        "observations": len(d),
        "study_mean_deltas": studies.to_dict(),
        "mean_delta": float(studies.mean()),
        "between_study_se": se,
        "study_t95_sensitivity": [float(studies.mean() - half), float(studies.mean() + half)]
        if half is not None
        else None,
        "leave_one_study_out_means": {str(g): float(studies.drop(g).mean()) for g in studies.index}
        if n > 1
        else {},
        "warning": "Only a few studies; folds share training data. Neither this t interval nor original bootstrap captures model-refitting uncertainty. No post-hoc observed power or equivalence claim.",
    }


def prediction_diagnostics(predictions):
    """Training-only constant reference for complete single-stage primary folds."""
    f = pd.DataFrame(predictions)
    f = f[f.model == "shared_dynamics"].copy()
    if f.empty:
        return {}
    persons = f.groupby(["study_group", "participant_id"]).experience.agg(["mean", "nunique"])
    rows = []
    for held, block in f.groupby("study_group"):
        train = persons.drop(held, level=0)
        rate = float(train.groupby(level=0)["mean"].mean().mean())
        prob = np.clip(rate, 1e-12, 1 - 1e-12)
        block = block.copy()
        block["constant_loss"] = -block.experience * np.log(prob) - (
            1 - block.experience
        ) * np.log1p(-prob)
        rows.append(
            dict(
                study_group=held,
                training_only_constant_probability=rate,
                constant_log_loss=float(
                    block.groupby("participant_id").constant_loss.mean().mean()
                ),
            )
        )
    return dict(
        training_constant=rows,
        constant_equal_study_log_loss=float(np.mean([r["constant_log_loss"] for r in rows])),
        participants_with_both_labels=int((persons["nunique"] == 2).sum()),
        participants_total=len(persons),
        warning="Other represented study groups supply training labels; no target-label tuning. Permutable count applies only to single-stage primary analyses.",
    )


def boundary_audit(measurements, draws=19999, seed=20260920):
    f = pd.DataFrame(measurements)
    deltas = {}
    for (context, feature), block in f.groupby(["context", "feature"]):
        wide = block.pivot(index="participant_id", columns="session", values="value")
        if {"01", "02"} <= set(wide):
            deltas[context, feature] = (wide["02"] - wide["01"]).dropna()
    contrasts = [("session_change", c, a, d) for (c, a), d in deltas.items()]
    contrasts += [
        ("context_interaction", c, a, (d - deltas["rest", a]).dropna())
        for (c, a), d in deltas.items()
        if c != "rest" and ("rest", a) in deltas
    ]
    rows = []
    rng = np.random.default_rng(seed)
    for kind, context, feature, d in contrasts:
        x = d.to_numpy(float)
        if len(x) < 2:
            continue
        estimate = float(x.mean())
        count = 0
        for start in range(0, draws, 1000):
            signs = rng.choice([-1.0, 1.0], size=(min(1000, draws - start), len(x)))
            count += int(np.count_nonzero(np.abs(signs @ x / len(x)) >= abs(estimate)))
        se = float(x.std(ddof=1) / np.sqrt(len(x)))
        half = stats.t.ppf(0.975, len(x) - 1) * se
        rows.append(
            dict(
                kind=kind,
                context=context,
                feature=feature,
                participants=len(x),
                estimate=estimate,
                interval_t95=[estimate - half, estimate + half],
                p_sign_flip=(count + 1) / (draws + 1),
                standard_error=se,
            )
        )
    for r, p in zip(rows, holm([r["p_sign_flip"] for r in rows]), strict=True):
        r["p_holm_all_boundary_tests"] = p
    return {
        "contrasts": rows,
        "family_size": len(rows),
        "draws": draws,
        "assumptions": "Participant differences symmetric under null; features and leave-one-participant-out standardization held fixed. Shared training dependence is not resampled. Context/order/eyes-open confounds remain.",
    }


def audit(evidence_paths):
    out = {
        "exploratory": True,
        "registered_or_preregistered": False,
        "sources": [],
        "comparisons": [],
        "null_controls": [],
        "analysis_inventory": [],
        "models": [],
        "prediction_diagnostics": [],
    }
    for label, path in evidence_paths.items():
        content = json.loads(path.read_text())
        out["sources"].append(
            dict(label=label, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        )
        nulls = defaultdict(list)
        observed = {}
        for a in content["evidence"]:
            c = a["content"]
            name = a["artifact"]
            # Verify embedded evidence is consistent with the referenced immutable source.
            source = Path(a["source_path"])
            if hashlib.sha256(source.read_bytes()).hexdigest() != a["sha256"]:
                raise ValueError("Source artifact changed: " + name)
            if name in ("transfer.json", "sensitivities.json", "robustness.json"):
                for key, r in c["analyses"].items():
                    summary = r.get("summary", {})
                    out["analysis_inventory"].append(
                        dict(
                            portfolio=label,
                            artifact=name,
                            analysis=key,
                            status=r.get(
                                "status", "estimated" if summary.get("models") else "no_summary"
                            ),
                            reason=r.get("reason"),
                            unavailable=r.get("unavailable", []),
                            comparisons=summary.get("paired_comparisons", []),
                        )
                    )
                    if name == "transfer.json" and ":within:" not in key and summary.get("models"):
                        observed[key] = summary
                        out["prediction_diagnostics"].append(
                            dict(
                                portfolio=label,
                                analysis=key,
                                **prediction_diagnostics(r["predictions"]),
                            )
                        )
                        out["models"].extend(
                            dict(portfolio=label, analysis=key, **m) for m in summary["models"]
                        )
                        for comp in summary["paired_comparisons"]:
                            out["comparisons"].append(
                                dict(
                                    portfolio=label,
                                    analysis=key,
                                    **comp,
                                    precision=precision(r.get("predictions", []), comp["baseline"]),
                                )
                            )
            elif name == "controls.json":
                for r in c["rows"]:
                    if "summary" in r:
                        nulls[(r["track"] + ":" + r["representation"], r["kind"])].append(r)
            elif name == "boundary.json":
                out["boundary"] = boundary_audit(c["measurements"])
                out["boundary"]["original_limitations"] = c["limitations"]
            elif name == "specificity.json":
                out.setdefault("specificity", []).append(
                    {k: v for k, v in c.items() if k not in ("rows", "recording_audit")}
                )
            elif name in ("axis_recovery.json", "tms.json"):
                out[name.removesuffix(".json")] = c
        for (key, kind), records in nulls.items():
            if key not in observed:
                continue
            summary = observed[key]
            comparisons = {"absolute_dynamic_log_loss": model_loss(summary, "shared_dynamics")}
            comparisons.update(
                {
                    r["baseline"]: r["dynamic_minus_baseline_log_loss"]
                    for r in summary["paired_comparisons"]
                    if r["baseline"]
                    in ("conventional_multivariate", "nonlinear_scalar", "spectral_arousal")
                }
            )
            for metric, actual in comparisons.items():
                vals = []
                for r in records:
                    if metric == "absolute_dynamic_log_loss":
                        v = model_loss(r["summary"], "shared_dynamics")
                    else:
                        v = next(
                            (
                                c["dynamic_minus_baseline_log_loss"]
                                for c in r["summary"]["paired_comparisons"]
                                if c["baseline"] == metric
                            ),
                            None,
                        )
                    if v is not None:
                        vals.append(v)
                out["null_controls"].append(
                    dict(
                        portfolio=label,
                        analysis=key,
                        kind=kind,
                        metric=metric,
                        observed=actual,
                        null_median=float(np.median(vals)),
                        null_range=[min(vals), max(vals)],
                        null_replicates=len(vals),
                        p_lower=empirical_tail(actual, vals),
                    )
                )
    for r, p in zip(
        out["null_controls"], holm([r["p_lower"] for r in out["null_controls"]]), strict=True
    ):
        r["p_holm_all_null_comparisons"] = p
    out["null_cautions"] = [
        "Label exchangeability is assumed within participant and sleep stage, not demonstrated.",
        "Temporal and representation-phase tests target representation structure, not original EEG spectral specificity.",
        "Plus-one one-sided tails; original predictions and null refits use the same statistic. Models share data.",
        "100 replicates give minimum p=1/101; broad multiplicity inference is resolution-limited. No selective extra permutations.",
    ]
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--core", type=Path, required=True)
    p.add_argument("--hd", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = audit({"core": args.core, "hd": args.hd})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # New output required; original evidence is never replaced.
    missing = []
    result = json_finite(result, missing=missing)
    result["nonfinite_values_in_source_or_unavailable_summaries"] = missing
    encoded = json.dumps(result, indent=2, allow_nan=False)
    with args.output.open("x") as f:
        f.write(encoded)
    print("AUDIT_COMPLETE", args.output)
