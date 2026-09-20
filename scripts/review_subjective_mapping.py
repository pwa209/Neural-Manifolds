"""Pinned public ASC11 mapping; between-participant exploratory associations.

No context-specific questionnaire timing is inferred. No score imputation.
All eleven named subscales, nine EEG features and four contexts are reported.
"""

import argparse
import hashlib
import io
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from review_completed_evidence import holm
from scipy import stats

REVISION = "8c5d6b077c63d29796409b622ebc300fdec6aadd"
BASE = "derivatives/phenotype/scored_not_imputed/ses-02/"


def read_public_scores(repository, destination):
    recorded = []
    payload = {}
    for name in ["ASC11_not_imputed.tsv", "ASC11_not_imputed.json", "ses-02_README.txt"]:
        data = subprocess.check_output(
            ["git", "-C", str(repository), "show", f"{REVISION}:{BASE}{name}"]
        )
        if len(data) > 2_000_000 or b"annex/objects" in data[:500]:
            raise ValueError("Expected small regular Git metadata blob")
        target = destination / name
        target.write_bytes(data)
        payload[name] = data
        recorded.append(
            dict(path=BASE + name, sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))
        )
    book = json.loads(payload["ASC11_not_imputed.json"])
    frame = pd.read_csv(
        io.BytesIO(payload["ASC11_not_imputed.tsv"]), sep="\t", na_values=["n/a", "NA"]
    )
    scales = [s for s in book if s.startswith("ASC11_") and "COMPOSITE" not in s]
    if len(scales) != 11 or not set(scales) <= set(frame):
        raise ValueError("Unexpected ASC11 schema")
    if frame.participant_id.duplicated().any():
        raise ValueError("Duplicate participant scale records")
    frame["participant_id"] = frame.participant_id.str.removeprefix("sub-")
    if not frame.participant_id.str.fullmatch(r"PC\d+").all():
        raise ValueError("Participant IDs do not match official EEG IDs")
    return frame, scales, book, recorded


def subjective_audit(measurements, scores, scales, seed=20260920):
    f = pd.DataFrame(measurements)
    rng = np.random.default_rng(seed)
    rows = []
    for (context, feature), block in f.groupby(["context", "feature"]):
        wide = block.pivot(index="participant_id", columns="session", values="value")
        delta = (wide["02"] - wide["01"]).rename("delta").dropna()
        joined = delta.to_frame().join(
            scores.set_index("participant_id")[scales], how="inner", validate="one_to_one"
        )
        for scale in scales:
            pair = joined[["delta", scale]].dropna()
            n = len(pair)
            if n < 10 or pair.nunique().min() < 2:
                rows.append(
                    dict(
                        context=context,
                        feature=feature,
                        scale=scale,
                        participants=n,
                        status="not_estimable",
                    )
                )
                continue
            x, y = pair.to_numpy().T
            rho, p = stats.spearmanr(x, y)
            boot = []
            for _ in range(999):
                take = rng.integers(n, size=n)
                if np.unique(x[take]).size > 1 and np.unique(y[take]).size > 1:
                    boot.append(float(stats.spearmanr(x[take], y[take]).statistic))
            rows.append(
                dict(
                    context=context,
                    feature=feature,
                    scale=scale,
                    participants=n,
                    status="estimated",
                    rho=float(rho),
                    p_asymptotic=float(p),
                    interval_bootstrap95=np.quantile(boot, [0.025, 0.975]).tolist(),
                )
            )
    estimable = [r for r in rows if r["status"] == "estimated"]
    for r, p in zip(estimable, holm([r["p_asymptotic"] for r in estimable]), strict=True):
        r["p_holm"] = p
    return dict(
        rows=rows,
        planned_comparisons=4 * 9 * 11,
        estimable_comparisons=len(estimable),
        limitations=[
            "Exploratory all-subscale screen, not confirmatory replication.",
            "Global post-session ratings are not context-specific or simultaneous EEG ratings.",
            "Association of individual EEG session changes with reported experience, not a drug causal effect.",
            "Unimputed complete pairs; sample varies by scale. Asymptotic rank p-values, Holm across all estimable tests.",
            "Features held fixed; shared cross-fitted normalization uncertainty is not resampled.",
        ],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--boundary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    scores, scales, book, provenance = read_public_scores(args.repository, args.output)
    boundary = json.loads(args.boundary.read_text())
    result = subjective_audit(boundary["measurements"], scores, scales)
    result.update(
        revision=REVISION,
        sources=provenance,
        codebook={s: book[s] for s in scales},
        boundary_sha256=hashlib.sha256(args.boundary.read_bytes()).hexdigest(),
        scale_rows=len(scores),
        join="Exact PC identifier, with optional sub- prefix removed; no row-order join.",
    )
    (args.output / "subjective_review.json").write_text(
        json.dumps(result, indent=2, allow_nan=False)
    )
    print("SUBJECTIVE_COMPLETE", args.output)
