"""Update manuscript Figure 3 and bundle from the final 5,000-refit audit.

The other three figures are copied byte-for-byte from the prior reviewed build.
No model is fitted and no participant-level records are read or exported here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pypdf import PdfReader, PdfWriter


COLORS = {"encoder": "#0F4D92", "sensor": "#42949E", "time_frequency": "#9A4D8E"}
REPS = ("encoder", "sensor", "time_frequency")
PORTS = ("core", "hd")
KINDS = ("label_permutation", "temporal_permutation")
METRICS = ("absolute_dynamic_log_loss", "conventional_multivariate")
ALL_KINDS = (*KINDS, "representation_phase")
ALL_METRICS = (*METRICS, "nonlinear_scalar", "spectral_arousal")
SOURCE_HASH = "75f40c5bf622ef8a05c16a62c218954e013c555e681b7e67f3ca517f54607d13"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_final(report_path: Path, statistics_path: Path):
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not report.get("complete") or report.get("family_size") != 72:
        raise ValueError("The final 72-test review is incomplete")
    if not report.get("observed_analyses_unchanged"):
        raise ValueError("Observed analysis identity was not preserved")
    if digest(statistics_path) != SOURCE_HASH or report["source_statistics"]["sha256"] != SOURCE_HASH:
        raise ValueError("Final statistic source hash mismatch")
    frame = pd.read_csv(statistics_path)
    columns = ["portfolio", "representation", "kind", "metric", "replicate", "null_statistic"]
    if list(frame.columns) != columns or len(frame) != 360_000:
        raise ValueError("Unexpected final null-statistic table")
    if not np.isfinite(frame.null_statistic.to_numpy()).all():
        raise ValueError("Nonfinite null statistic")
    keys = columns[:4]
    expected = {(p, r, k, m) for p in PORTS for r in REPS for k in ALL_KINDS for m in ALL_METRICS}
    grouped = frame.groupby(keys, sort=False)
    if set(grouped.groups) != expected:
        raise ValueError("Missing or unexpected null family")
    rows = {}
    for key, group in grouped:
        if len(group) != 5000 or set(group.replicate) != set(range(5000)):
            raise ValueError(f"Incomplete or duplicate replicate family: {key}")
        rows[key] = np.sort(group.null_statistic.to_numpy(dtype=float))
    records = {}
    for row in report["null_controls"]:
        key = tuple(row[c] for c in keys)
        if key in records or key not in expected or row["null_replicates"] != 5000:
            raise ValueError("Duplicate or unexpected review row")
        values = rows[key]
        tail = int(np.count_nonzero(values <= row["observed"]))
        p = (1 + tail) / 5001
        if tail != row["null_lower_or_equal"] or not np.isclose(p, row["p_lower"], atol=1e-14):
            raise ValueError(f"Reviewed tail does not match 5,000 source refits: {key}")
        if not np.isclose(np.median(values), row["null_median"], atol=1e-13):
            raise ValueError(f"Reviewed median does not match source refits: {key}")
        if not np.allclose([values[0], values[-1]], row["null_range"], atol=1e-13):
            raise ValueError(f"Reviewed range does not match source refits: {key}")
        records[key] = row
    if set(records) != expected:
        raise ValueError("Review does not cover all 72 tests")
    ordered = sorted(records, key=lambda key: records[key]["p_lower"])
    running = 0.0
    for rank, key in enumerate(ordered):
        running = max(running, min(1.0, (72 - rank) * records[key]["p_lower"]))
        if not np.isclose(running, records[key]["p_holm_all_null_comparisons"], atol=1e-12):
            raise ValueError(f"Holm adjustment mismatch: {key}")
    significant = [key for key, row in records.items() if row["p_holm_all_null_comparisons"] < .05]
    expected_sig = {("core", "sensor", "temporal_permutation", metric) for metric in ALL_METRICS}
    if set(significant) != expected_sig:
        raise ValueError("Unexpected corrected-result set")
    return report, frame, rows, records


def draw_figure(rows, records, output: Path):
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8,
        "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 6.7,
        "axes.linewidth": .6, "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.facecolor": "white", "axes.spines.top": False,
        "axes.spines.right": False,
    })
    fig = plt.figure(figsize=(183 / 25.4, 174 / 25.4), facecolor="white")
    fig.text(.028, .953, "a", fontsize=11, fontweight="bold")
    fig.text(.058, .953, "Absolute dynamics loss", fontsize=9, fontweight="bold")
    fig.text(.521, .953, "b", fontsize=11, fontweight="bold")
    fig.text(.551, .953, "Dynamics − conventional loss", fontsize=9, fontweight="bold")
    for midx, metric in enumerate(METRICS):
        all_values = np.concatenate([rows[(p, r, k, metric)] for p in PORTS for r in REPS for k in KINDS])
        observed = [records[(p, r, k, metric)]["observed"] for p in PORTS for r in REPS for k in KINDS]
        lo = min(np.quantile(all_values, .005), min(observed))
        hi = max(np.quantile(all_values, .995), max(observed))
        pad = .05 * (hi - lo)
        for pidx, port in enumerate(PORTS):
            ax = fig.add_axes([.195 + .49 * midx, .555 - .363 * pidx, .277, .295])
            ax.set_xlim(lo - pad, hi + pad)
            ax.set_ylim(-.6, 5.6)
            ax.set_title("Sparse core" if port == "core" else "High-density extension", fontweight="bold", pad=8)
            labels = []
            for ridx, rep in enumerate(REPS):
                for kidx, kind in enumerate(KINDS):
                    y = 5 - 2 * ridx - kidx
                    key = port, rep, kind, metric
                    values, audit = rows[key], records[key]
                    q = np.quantile(values, [.025, .25, .5, .75, .975])
                    color = COLORS[rep]
                    ax.plot([q[0], q[4]], [y, y], color=color, lw=1.4, alpha=.8, zorder=2)
                    ax.plot([q[1], q[3]], [y, y], color=color, lw=5, alpha=.75, solid_capstyle="butt", zorder=3)
                    ax.plot(q[2], y, marker="|", color="#19212B", markersize=10, mew=1.2, zorder=4)
                    ax.plot(audit["observed"], y, marker="D", color="#111111", markersize=4.2, zorder=5)
                    short = {"encoder": "Encoder", "sensor": "Sensor", "time_frequency": "TF"}[rep]
                    control = "label" if kind == "label_permutation" else "temporal"
                    labels.append((y, f"{short} · {control}\np={audit['p_lower']:.4f}; H={audit['p_holm_all_null_comparisons']:.3f}", audit["p_holm_all_null_comparisons"] < .05))
            ax.set_yticks([x[0] for x in labels], [x[1] for x in labels])
            for tick, item in zip(ax.get_yticklabels(), labels):
                if item[2]:
                    tick.set_fontweight("bold")
                    tick.set_color("#16636C")
            ax.tick_params(axis="y", length=0, pad=5)
            ax.tick_params(axis="x", width=.6, length=3)
            ax.grid(axis="x", color="#E4E7EA", linewidth=.5)
            ax.set_axisbelow(True)
            if metric != "absolute_dynamic_log_loss":
                ax.axvline(0, color="#AAB0B5", linewidth=.7, linestyle="--")
            if pidx == 1:
                ax.set_xlabel("Log loss" if midx == 0 else "Difference in log loss")
    fig.text(.04, .045, "Thin line: 2.5–97.5% of 5,000 null refits; thick line: IQR; tick: median; black diamond: observed.", fontsize=6.8)
    fig.text(.04, .022, "Lower is better. p: lower-tail Monte Carlo; H: Holm over all 72 exploratory tests. Bold: H < 0.05.", fontsize=6.8)
    for suffix in ("pdf", "png", "svg"):
        fig.savefig(output / f"fig03_null_controls.{suffix}", dpi=600)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--statistics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Historical exports must be preserved")
    report, _frame, rows, records = load_final(args.report, args.statistics)
    args.output.mkdir(parents=True)
    for stem in ("fig02_prediction", "fig04_context_subjective", "fig05_measurement"):
        for suffix in ("pdf", "png", "svg"):
            shutil.copy2(args.previous / f"{stem}.{suffix}", args.output / f"{stem}.{suffix}")
    draw_figure(rows, records, args.output)
    tables = args.output / "source_tables"
    tables.mkdir()
    for path in (args.previous / "source_tables").glob("*"):
        if path.name not in {"null_controls.csv", "null_refit_statistics.csv"}:
            shutil.copy2(path, tables / path.name)
    shutil.copy2(args.statistics, tables / "null_refit_statistics.csv")
    pd.DataFrame(report["null_controls"]).to_csv(tables / "null_controls.csv", index=False)
    writer = PdfWriter()
    for stem in ("fig02_prediction", "fig03_null_controls", "fig04_context_subjective", "fig05_measurement"):
        writer.append(PdfReader(args.output / f"{stem}.pdf"))
    with (args.output / "Figures_2-5.pdf").open("wb") as stream:
        writer.write(stream)
    manifest = {
        "status": "requires_visual_review", "exploratory": True,
        "registered_or_preregistered": False,
        "final_null_report_sha256": digest(args.report),
        "null_statistics_sha256": digest(args.statistics),
        "prior_reviewed_figure_manifest_sha256": digest(args.previous / "manifest.json"),
        "figure_3_renderer_sha256": digest(Path(__file__)),
        "unchanged_figures": [2, 4, 5],
        "changed_figure": 3,
        "holm_significant": [list(key) for key, row in records.items() if row["p_holm_all_null_comparisons"] < .05],
        "outputs": [
            {"file": str(path.relative_to(args.output)).replace("\\", "/"), "sha256": digest(path)}
            for path in sorted(args.output.rglob("*")) if path.is_file()
        ],
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"figure": str(args.output / "fig03_null_controls.pdf"), "tests": len(records), "corrected": len(manifest["holm_significant"])}))


if __name__ == "__main__":
    main()
