"""Render exploratory manuscript Figures 2-5 from immutable server evidence.

No model fitting, new hypothesis tests, or participant-table export. The only
synthetic example is the explicitly labelled state-sequence schematic. Run on
the data host; figures and aggregate source tables can be transferred for QA.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

REPS = ["encoder", "sensor", "time_frequency"]
REP_NAMES = ["Encoder", "Sensor", "Time-frequency"]
COLORS = ["#0F4D92", "#42949E", "#9A4D8E"]
MARKERS = ["o", "s", "^"]
PORTS = ["core", "hd"]
TRACKS = {"core": "sparse", "hd": "high_density"}
PORT_NAMES = {"core": "Sparse core", "hd": "High-density extension"}
FEATURES = [
    "repertoire",
    "recurrence",
    "exit_entropy",
    "log_dwell",
    "dwell_dispersion",
    "directionality",
    "alignment",
    "reachability",
    "lz",
]
FEATURE_NAMES = [
    "Repertoire",
    "Recurrence",
    "Exit entropy",
    "Log dwell",
    "Dwell dispersion",
    "Directionality",
    "Alignment",
    "Reachability",
    "Lempel-Ziv",
]
ABBR = ["Rep", "Rec", "Ent", "LogD", "Disp", "Dir", "Ali", "Reach", "LZ"]
STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "axes.linewidth": 0.6,
    "lines.linewidth": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "savefig.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.unicode_minus": True,
}
EXPECTED = {
    "review-v3.json": "c5653b1f4790ce1b6fd36c78cd1bd91dd9e1cffe12b84c81457b6f6c70f7c4e6",
    "recovery-geometry-v3.json": "7fb464e533c027168cfe10bbc522406af978437c6be2cdc4e33ce4c160f0f957",
    "subjective/subjective_review.json": "abe98be114785d3f59b82d856ffd1f40a8e1f9783d3899f8a9f52808967fb4c3",
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def unique(rows, **criteria):
    matches = [r for r in rows if all(r.get(k) == v for k, v in criteria.items())]
    if len(matches) != 1:
        raise ValueError(f"Expected one source row for {criteria}; found {len(matches)}")
    return matches[0]


def finite(values):
    result = np.asarray(values, dtype=float)
    if not np.isfinite(result).all():
        raise ValueError("Unavailable/nonfinite plot values must be handled explicitly")
    return result


def load_sources(core, hd, review_root):
    sources, evidence = [], {}
    for name, expected in EXPECTED.items():
        path = review_root / name
        if digest(path) != expected:
            raise ValueError(f"Reviewed artifact changed: {name}")
        evidence[name] = json.loads(path.read_text())
        sources.append({"artifact": name, "sha256": expected})
    review = evidence["review-v3.json"]
    selected, controls, boundary = {}, [], None
    for port, path in [("core", core), ("hd", hd)]:
        bundle = json.loads(path.read_text())
        expected = unique(review["sources"], label=port)["sha256"]
        if digest(path) != expected:
            raise ValueError(f"Synthesis changed: {port}")
        sources.append({"portfolio": port, "artifact": "evidence.json", "sha256": expected})
        for record in bundle["evidence"]:
            name = record["artifact"]
            if name not in {"transfer.json", "controls.json", "boundary.json"}:
                continue
            source = Path(record["source_path"])
            if digest(source) != record["sha256"]:
                raise ValueError(f"Original source changed: {name}")
            original = json.loads(source.read_text())
            # Use hash-verified originals, not potentially divergent embedded copies.
            sources.append({"portfolio": port, "artifact": name, "sha256": record["sha256"]})
            if name == "transfer.json":
                for rep in REPS:
                    key = f"{TRACKS[port]}:{rep}"
                    selected[port, rep] = original["analyses"][key]
            elif name == "controls.json":
                controls.extend(dict(portfolio=port, **r) for r in original["rows"])
            elif name == "boundary.json" and port == "core":
                boundary = original
                if (
                    digest(source)
                    != evidence["subjective/subjective_review.json"]["boundary_sha256"]
                ):
                    raise ValueError("Subjective mapping references a different boundary artifact")
    if boundary is None or len(selected) != 6:
        raise ValueError("Incomplete original evidence")
    return {
        "review": review,
        "recovery": evidence["recovery-geometry-v3.json"],
        "subjective": evidence["subjective/subjective_review.json"],
        "transfer": selected,
        "controls": controls,
        "boundary": boundary,
        "sources": sources,
    }


def get_comparison(data, port, rep, baseline):
    return unique(
        data["review"]["comparisons"],
        portfolio=port,
        analysis=f"{TRACKS[port]}:{rep}",
        baseline=baseline,
    )


def null_values(data, port, rep, kind, metric):
    rows = [
        r
        for r in data["controls"]
        if r["portfolio"] == port
        and r["track"] == TRACKS[port]
        and r["representation"] == rep
        and r["kind"] == kind
        and "summary" in r
    ]
    if len(rows) != 100 or len({r["replicate"] for r in rows}) != 100:
        raise ValueError(f"Incomplete/duplicate null family: {port} {rep} {kind}")
    rows.sort(key=lambda r: r["replicate"])
    values = []
    for row in rows:
        summary = row["summary"]
        if metric == "absolute_dynamic_log_loss":
            values.append(
                np.mean(
                    [r["log_loss"] for r in summary["models"] if r["model"] == "shared_dynamics"]
                )
            )
        else:
            values.append(
                unique(summary["paired_comparisons"], baseline=metric)[
                    "dynamic_minus_baseline_log_loss"
                ]
            )
    values = finite(values)
    audit = unique(
        data["review"]["null_controls"],
        portfolio=port,
        analysis=f"{TRACKS[port]}:{rep}",
        kind=kind,
        metric=metric,
    )
    p = (1 + np.count_nonzero(values <= audit["observed"])) / 101
    if not np.isclose(p, audit["p_lower"], atol=1e-12):
        raise ValueError("Null plot and reviewed test disagree")
    return values, audit


def interaction(data, feature, context):
    original = unique(
        data["boundary"]["context_interactions"], feature=feature, context_minus_rest=context
    )
    audit = unique(
        data["review"]["boundary"]["contrasts"],
        feature=feature,
        context=context,
        kind="context_interaction",
    )
    if original["participants"] != audit["participants"] or not np.isclose(
        original["difference_in_change"], audit["estimate"], atol=1e-12
    ):
        raise ValueError("Boundary original and review disagree")
    return dict(original, p_holm=audit["p_holm_all_boundary_tests"])


def participant_interaction(data, feature, context):
    frame = pd.DataFrame(data["boundary"]["measurements"])
    frame = frame[frame.feature == feature]
    wide = frame.pivot(index="participant_id", columns=["context", "session"], values="value")
    contrast = (
        (wide[context, "02"] - wide[context, "01"]) - (wide["rest", "02"] - wide["rest", "01"])
    ).dropna()
    row = interaction(data, feature, context)
    if len(contrast) != row["participants"] or not np.isclose(
        contrast.mean(), row["difference_in_change"], atol=1e-12
    ):
        raise ValueError("Paired contrast disagrees with original summary")
    return finite(contrast.to_numpy())


def new_figure(number, title, height):
    fig = plt.figure(figsize=(183 / 25.4, height / 25.4), facecolor="white")
    fig.text(0.025, 0.983, f"FIGURE {number}", fontsize=7.5, color="#666666", va="top")
    fig.text(0.025, 0.953, title, fontsize=11, fontweight="bold", va="top")
    fig.text(0.975, 0.983, "EXPLORATORY DRAFT", fontsize=7, ha="right", va="top", color="#777777")
    return fig


def section(fig, letter, text, x, y):
    fig.text(x, y, letter, fontsize=10, weight="bold", va="top")
    fig.text(x + 0.025, y, text, fontsize=8.5, weight="bold", va="top")


def clean(ax, grid="x"):
    ax.tick_params(length=2.5, width=0.6, pad=2)
    ax.set_axisbelow(True)
    ax.grid(axis=grid, color="#E7E7E7", linewidth=0.4)


def interval(ax, y, estimate, endpoints, color, marker="o", size=4):
    lo, hi = finite(endpoints)
    if lo > hi:
        raise ValueError("Reversed interval")
    ax.plot([lo, hi], [y, y], color=color, linewidth=1.3, solid_capstyle="round")
    ax.plot(estimate, y, marker=marker, color=color, markersize=size, linestyle="none")


def fig2(data):
    fig = new_figure(2, "Prediction depends on the comparator", 195)
    section(fig, "a", "Absolute held-out prediction loss", 0.025, 0.902)
    category_labels = ["Constant", "Conv.", "LZ", "Encoder", "Sensor", "Time-freq."]
    category_colors = ["#222222", "#858585", "#4D4D4D", *COLORS]
    category_markers = ["D", "o", "o", *MARKERS]
    loss_axes = []
    all_loss = []
    for j, port in enumerate(PORTS):
        ax = fig.add_axes([0.105 + j * 0.475, 0.674, 0.375, 0.170])
        loss_axes.append(ax)
        models = data["transfer"][port, "encoder"]["summary"]["models"]
        studies = sorted({r["study_group"] for r in models})
        diagnostics = unique(
            data["review"]["prediction_diagnostics"],
            portfolio=port,
            analysis=f"{TRACKS[port]}:encoder",
        )
        vals = []
        for study in studies:
            study_values = [
                unique(diagnostics["training_constant"], study_group=study)["constant_log_loss"]
            ]
            study_values += [
                unique(models, model=m, study_group=study)["log_loss"]
                for m in ["conventional_multivariate", "nonlinear_scalar"]
            ]
            study_values += [
                unique(
                    data["transfer"][port, rep]["summary"]["models"],
                    model="shared_dynamics",
                    study_group=study,
                )["log_loss"]
                for rep in REPS
            ]
            vals.append(finite(study_values))
            ax.plot(
                range(6),
                study_values,
                "o-",
                color="#B5B5B5",
                markersize=2.5,
                linewidth=0.65,
                zorder=1,
            )
        vals = np.asarray(vals)
        all_loss.extend(vals.ravel())
        for k, mean in enumerate(vals.mean(axis=0)):
            ax.plot(k, mean, marker=category_markers[k], color=category_colors[k], markersize=5.5)
        precision = get_comparison(data, port, "encoder", "conventional_multivariate")["precision"]
        ax.set_title(
            f"{PORT_NAMES[port]}\n{precision['participants']} IDs · {precision['observations']} observations · 3 studies",
            pad=8,
        )
        ax.set_xticks(range(6), category_labels, rotation=30, ha="right")
        ax.set_xlim(-0.4, 5.4)
        ax.set_ylabel("Log loss" if j == 0 else "")
        clean(ax, "y")
    bounds = min(all_loss) - 0.06, max(all_loss) + 0.06
    for ax in loss_axes:
        ax.set_ylim(bounds)
    fig.text(
        0.105,
        0.602,
        "Small marks/lines: individual studies. Large marks: equal-study means. Lower loss is better.",
        fontsize=7,
    )

    section(fig, "b", "Against conventional features", 0.025, 0.563)
    section(fig, "c", "Against nonlinear LZ", 0.505, 0.563)
    comps = [
        get_comparison(data, p, r, b)
        for p in PORTS
        for r in REPS
        for b in ["conventional_multivariate", "nonlinear_scalar"]
    ]
    endpoints = [v for c in comps for v in c["interval_95"]]
    endpoints += [v for c in comps for v in c["precision"]["study_mean_deltas"].values()]
    low, high = min([*endpoints, 0]) - 0.03, max([*endpoints, 0]) + 0.03
    for j, baseline in enumerate(["conventional_multivariate", "nonlinear_scalar"]):
        ax = fig.add_axes([0.18 + 0.48 * j, 0.361, 0.295, 0.168])
        labels = []
        for pidx, port in enumerate(PORTS):
            for ridx, rep in enumerate(REPS):
                y = 6 - pidx * 3.7 - ridx
                c = get_comparison(data, port, rep, baseline)
                interval(
                    ax,
                    y,
                    c["dynamic_minus_baseline_log_loss"],
                    c["interval_95"],
                    COLORS[ridx],
                    MARKERS[ridx],
                )
                ax.scatter(
                    list(c["precision"]["study_mean_deltas"].values()),
                    [y - 0.20] * 3,
                    s=8,
                    color="#999999",
                    marker="|",
                    zorder=2,
                )
                labels.append(
                    (
                        y,
                        ("Core " if pidx == 0 else "HD ")
                        + REP_NAMES[ridx].replace("Time-frequency", "Time-freq."),
                    )
                )
        ax.axvline(0, color="#555555", linestyle="--", linewidth=0.7)
        ax.set_yticks([x[0] for x in labels], [x[1] for x in labels])
        ax.set_ylim(-0.4, 6.6)
        ax.set_xlim(low, high)
        ax.set_xlabel("Dynamics \u2212 comparator log loss")
        clean(ax)
    fig.text(
        0.105,
        0.299,
        "Left of zero favours dynamics. Intervals: original 95% bootstrap, conditional on fitted predictions.",
        fontsize=7,
    )

    section(fig, "d", "Discrimination varies across held-out studies", 0.025, 0.266)
    for j, port in enumerate(PORTS):
        ax = fig.add_axes([0.105 + j * 0.475, 0.079, 0.375, 0.145])
        for ridx, rep in enumerate(REPS):
            rows = [
                r
                for r in data["transfer"][port, rep]["summary"]["models"]
                if r["model"] == "shared_dynamics"
            ]
            rows.sort(key=lambda r: r["study_group"])
            for sidx, row in enumerate(rows):
                value = row.get("auroc")
                if value is None or not np.isfinite(value):
                    ax.text(ridx, 0.06 + 0.06 * sidx, "NA", ha="center", fontsize=7)
                else:
                    ax.plot(
                        ridx + (sidx - 1) * 0.15,
                        value,
                        marker=MARKERS[ridx],
                        color=COLORS[ridx],
                        markersize=4,
                        linestyle="none",
                    )
        ax.axhline(0.5, color="#888888", linewidth=0.7, linestyle="--")
        ax.set(ylim=(0, 1), xlim=(-0.5, 2.5), ylabel="AUROC" if j == 0 else "")
        ax.set_xticks(range(3), REP_NAMES)
        ax.set_title(PORT_NAMES[port], pad=5)
        clean(ax, "y")
    fig.text(
        0.025,
        0.018,
        "Overlapping portfolios; not a matched test of channel density. Constant-reference differences are descriptive.",
        fontsize=7,
    )
    return fig


def fig3(data):
    fig = new_figure(3, "Structure controls qualify prediction gains", 210)
    section(fig, "a", "Absolute dynamics loss", 0.025, 0.899)
    section(fig, "b", "Dynamics \u2212 conventional loss", 0.51, 0.899)
    metrics = ["absolute_dynamic_log_loss", "conventional_multivariate"]
    limits = {}
    for metric in metrics:
        vals = []
        for port in PORTS:
            for rep in REPS:
                for kind in ["label_permutation", "temporal_permutation"]:
                    v, audit = null_values(data, port, rep, kind, metric)
                    vals.extend(v)
                    vals.append(audit["observed"])
        lo, hi = min(vals), max(vals)
        limits[metric] = lo - 0.04 * (hi - lo), hi + 0.04 * (hi - lo)
    for pidx, port in enumerate(PORTS):
        for midx, metric in enumerate(metrics):
            ax = fig.add_axes([0.205 + midx * 0.49, 0.54 - pidx * 0.34, 0.275, 0.286])
            labels = []
            for ridx, rep in enumerate(REPS):
                for kidx, kind in enumerate(["label_permutation", "temporal_permutation"]):
                    y = 5 - ridx * 2 - kidx
                    values, audit = null_values(data, port, rep, kind, metric)
                    offsets = (
                        (np.arange(len(values)) * 37) % len(values) / len(values) - 0.5
                    ) * 0.36
                    ax.scatter(
                        values, y + offsets, s=5, color=COLORS[ridx], alpha=0.4, linewidths=0
                    )
                    ax.plot(
                        [audit["observed"]] * 2,
                        [y - 0.30, y + 0.30],
                        color="#111111",
                        linewidth=1.4,
                    )
                    short = REP_NAMES[ridx].replace("Time-frequency", "TF")
                    labels.append(
                        (
                            y,
                            f"{short} · {'label' if kidx == 0 else 'temporal'}\n"
                            f"p={audit['p_lower']:.4f}; H={audit['p_holm_all_null_comparisons']:.3f}",
                        )
                    )
            ax.set_yticks([r[0] for r in labels], [r[1] for r in labels], fontsize=7)
            ax.set_ylim(-0.65, 5.6)
            ax.set_xlim(limits[metric])
            ax.set_title(PORT_NAMES[port], pad=10, fontweight="bold")
            if pidx == 1:
                ax.set_xlabel("Log loss" if midx == 0 else "Difference in log loss")
            if midx:
                ax.axvline(0, color="#BBBBBB", linestyle="--", linewidth=0.6)
            clean(ax)
    section(fig, "c", "How to read the controls", 0.025, 0.128)
    notes = [
        "Coloured dots: all 100 nested null refits per row. Black line: observed statistic. Lower is better.",
        "p: one-sided plus-one Monte Carlo tail. H: Holm-adjusted p across the 72-test exploratory family.",
        "Minimum raw p = 1/101. No adjusted finding; this does not establish equivalence or absence of information.",
        "Permutable participant blocks: 49/80 core; 29/62 extension. Exchangeability within participant/stage assumed.",
        "Temporal controls act on representations, not original-EEG spectra; phase controls are in the source tables.",
    ]
    for i, note in enumerate(notes):
        fig.text(0.025, 0.101 - i * 0.019, note, fontsize=7, va="top")
    return fig


def scale_label(key, codebook):
    label = codebook[key]["Description"].removeprefix("11D-ASC ").removesuffix(" Subscale")
    return label.replace("Impaired Cognition and Control", "Impaired cognition/control").replace(
        "Changing Meaning of Percepts", "Changing percept meaning"
    )


def fig4(data):
    fig = new_figure(4, "Context sensitivity and subjective associations", 245)
    section(fig, "a", "All context-versus-rest differences in session change", 0.025, 0.91)
    fig.text(
        0.025,
        0.884,
        "Eight dynamical features: baseline-training SD units. Lempel-Ziv: native units.",
        fontsize=7,
    )
    contexts = ["movie", "meditation", "music"]
    for idx, feature in enumerate(FEATURES):
        col, row = idx % 3, idx // 3
        ax = fig.add_axes([0.13 + col * 0.314, 0.778 - row * 0.117, 0.19, 0.076])
        ticks = []
        for y, context in zip([2, 1, 0], contexts, strict=True):
            r = interaction(data, feature, context)
            color = "#183F58" if r["p_holm"] < 0.05 else "#7A858A"
            interval(ax, y, r["difference_in_change"], r["interval_95"], color, size=3.5)
            ticks.append(f"{context.title()} ({r['participants']})")
        ax.axvline(0, color="#777777", linestyle="--", linewidth=0.6)
        ax.set_yticks([2, 1, 0], ticks, fontsize=7)
        ax.set_ylim(-0.65, 2.6)
        ax.set_title(FEATURE_NAMES[idx], pad=4)
        if row == 2:
            ax.set_xlabel(
                "Native LZ units" if feature == "lz" else "Baseline-training SD units",
                fontsize=7,
                labelpad=2,
            )
        ax.locator_params(axis="x", nbins=3)
        clean(ax)
    fig.text(
        0.025,
        0.502,
        "Intervals: descriptive 95% participant bootstrap. Dark marks: Holm p < 0.05 across all 63 boundary tests.",
        fontsize=7,
    )
    fig.text(
        0.025,
        0.486,
        "Fixed task order, eyes-open/closed differences and no placebo prevent an isolated causal interpretation.",
        fontsize=7,
    )
    section(
        fig, "b", "Participant variation in the two corrected movie/rest interactions", 0.025, 0.46
    )
    for j, feature in enumerate(["repertoire", "lz"]):
        ax = fig.add_axes([0.12 + 0.48 * j, 0.358, 0.34, 0.062])
        vals = participant_interaction(data, feature, "movie")
        offsets = ((np.arange(len(vals)) * 19) % len(vals) / len(vals) - 0.5) * 0.36
        ax.scatter(vals, offsets, s=10, color="#668995", alpha=0.7, linewidths=0)
        r = interaction(data, feature, "movie")
        interval(ax, 0.38, r["difference_in_change"], r["interval_95"], "#172E3B", "D", 5)
        ax.axvline(0, color="#888888", linestyle="--", linewidth=0.6)
        ax.set_yticks([])
        ax.set_ylim(-0.4, 0.65)
        ax.set_xlabel(
            "Difference in session change (" + ("SD units)" if j == 0 else "native LZ units)"),
            fontsize=7,
        )
        ax.set_title(
            f"{'Repertoire' if j == 0 else 'Lempel-Ziv'} · n={len(vals)} · Holm p={r['p_holm']:.5f}",
            pad=5,
        )
        clean(ax)
    section(fig, "c", "All 396 associations with global post-session ASC11 ratings", 0.025, 0.304)
    rows = data["subjective"]["rows"]
    scales = sorted({r["scale"] for r in rows})
    if len(scales) != 11 or len(rows) != 396:
        raise ValueError("Incomplete subjective screen")
    cmap = LinearSegmentedColormap.from_list("rho", ["#2166AC", "#F5F5F5", "#B45F1B"])
    cmap.set_bad("#CCCCCC")
    labels = [scale_label(s, data["subjective"]["codebook"]) for s in scales]
    # Short display labels; exact public codebook keys are in the source table.
    labels = [
        s.replace("Elementary Imagery", "Elementary imagery")
        .replace("Complex Imagery", "Complex imagery")
        .replace("Impaired Control And Cognition", "Impaired control/cognition")
        .replace("Changed Meaning Of Percepts", "Changed meaning")
        .replace("Experience Of Unity", "Unity")
        .replace("Spiritual Experience", "Spiritual experience")
        .replace("Blissful State", "Blissful state")
        .replace("Audio Visual Synesthesia", "Audiovisual synesthesia")
        for s in labels
    ]
    for j, context in enumerate(["rest", "movie", "meditation", "music"]):
        ax = fig.add_axes([0.265 + j * 0.18, 0.125, 0.16, 0.139])
        matrix = np.array(
            [
                [
                    unique(rows, context=context, feature=f, scale=s).get("rho", np.nan)
                    for f in FEATURES
                ]
                for s in scales
            ]
        )
        if np.nanmax(np.abs(matrix)) > 1:
            raise ValueError("Invalid correlation")
        im = ax.imshow(matrix, vmin=-1, vmax=1, cmap=cmap, aspect="auto", interpolation="none")
        ax.set_title(context.title(), pad=5)
        ax.set_xticks(range(9), ABBR, rotation=90, fontsize=7)
        ax.set_yticks(range(11), labels if j == 0 else [""] * 11, fontsize=7)
        ax.tick_params(length=0, pad=3)
        for spine in ax.spines.values():
            spine.set_visible(False)
        for y, s in enumerate(scales):
            for x, feature in enumerate(FEATURES):
                r = unique(rows, context=context, feature=feature, scale=s)
                if not np.isfinite(matrix[y, x]):
                    ax.add_patch(Rectangle((x - 0.5, y - 0.5), 1, 1, fill=False, hatch="///", lw=0))
                elif r.get("p_holm", 1) < 0.05:
                    ax.add_patch(
                        Rectangle((x - 0.5, y - 0.5), 1, 1, fill=False, edgecolor="black", lw=0.8)
                    )
    cax = fig.add_axes([0.49, 0.076, 0.24, 0.008])
    fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[-1, 0, 1])
    cax.set_xlabel("Spearman rho", fontsize=7, labelpad=1)
    fig.text(
        0.025,
        0.084,
        "Complete pairs: n=47-53.\nNo Holm-corrected association.",
        fontsize=7,
        va="top",
    )
    fig.text(
        0.025,
        0.036,
        "Rep repertoire · Rec recurrence · Ent exit entropy · LogD log dwell · Disp dwell dispersion",
        fontsize=7,
    )
    fig.text(
        0.025,
        0.021,
        "Dir directionality · Ali alignment · Reach reachability · LZ Lempel-Ziv. Ratings are not context-specific.",
        fontsize=7,
    )
    return fig


def fig5(data):
    from neural_manifolds.manifold.metastability import estimate_metastability

    fig = new_figure(5, "Observation history changes the measurement target", 205)
    section(fig, "a", "Same state sequence, different segmentation", 0.025, 0.906)
    ax = fig.add_axes([0.04, 0.741, 0.92, 0.126])
    ax.set_axis_off()
    states = np.tile([0, 1, 0], 4)
    segments = [np.zeros(len(states), int), np.repeat(np.arange(4), 3)]
    shades = ["#C8D8DF", "#E5CBAE"]
    for row in range(2):
        y = 0.61 - row * 0.48
        value = estimate_metastability(states, segment_ids=segments[row]).recurrence_probability
        ax.text(
            0,
            y + 0.10,
            "Continuous" if row == 0 else "Three-sample blocks",
            fontsize=8,
            va="center",
        )
        for i, state in enumerate(states):
            x = 0.25 + i * 0.044
            ax.add_patch(Rectangle((x, y), 0.039, 0.22, facecolor=shades[state], edgecolor="white"))
            ax.text(
                x + 0.0195,
                y + 0.11,
                "A" if state == 0 else "B",
                ha="center",
                va="center",
                fontsize=8,
            )
            if row and i in [3, 6, 9]:
                ax.plot(
                    [x - 0.004, x - 0.004],
                    [y - 0.04, y + 0.26],
                    "--",
                    color="#555555",
                    linewidth=0.8,
                )
        ax.text(0.81, y + 0.11, f"Recurrence = {value:.3f}", va="center", fontsize=8)
    ax.set(xlim=(0, 1), ylim=(0, 1))
    fig.text(
        0.04,
        0.720,
        "Illustrative categorical sequence, not EEG. Values computed with the study estimator; boundaries reset history.",
        fontsize=7,
    )
    section(fig, "b", "Coverage across all 12 simulation settings", 0.025, 0.678)
    axes_names = ["recurrence", "exit_entropy", "dwell_dispersion"]
    methods = ["block3_long_record", "block3_finite_window", "model_based_finite_window"]
    for j, feature in enumerate(axes_names):
        ax = fig.add_axes([0.095 + 0.315 * j, 0.452, 0.247, 0.165])
        rows = [r for r in data["recovery"]["rows"] if r["axis"] == feature]
        if len(rows) != 12:
            raise ValueError("Incomplete simulation settings")
        for k, method in enumerate(methods):
            values = finite([r[method]["coverage"] for r in rows]) * 100
            offsets = np.linspace(-0.17, 0.17, len(rows))
            ax.scatter(
                k + offsets,
                values,
                s=12,
                marker=["o", "s", "^"][k],
                facecolor=["#BBBBBB", "#777777", "#222222"][k],
                edgecolor="white",
                linewidth=0.3,
            )
            ax.plot([k - 0.20, k + 0.20], [np.median(values)] * 2, color="black", linewidth=1)
        ax.axhline(95, color="#666666", linestyle="--", linewidth=0.8)
        ax.set(ylim=(-3, 105), xlim=(-0.4, 2.4), ylabel="Coverage (%)" if j == 0 else "")
        ax.set_xticks(range(3), ["Block /\nlong", "Block /\nfinite", "Model /\nfinite"], fontsize=7)
        ax.set_title(feature.replace("_", " ").title(), pad=7)
        clean(ax, "y")
    fig.text(
        0.04,
        0.398,
        "Each dot: one setting, 200 independent test draws. Bars: medians across settings. Dashed line: nominal 95%.",
        fontsize=7,
    )
    section(fig, "c", "Finite-window targets versus long-record references", 0.025, 0.356)
    setting_colors = ["#385B6B", "#A0B4BD", "#9F6038", "#D7B69D"]
    settings = [(s, m) for s in [0.3, 0.7] for m in [0.0, 0.2]]
    for j, feature in enumerate(axes_names):
        ax = fig.add_axes([0.095 + 0.315 * j, 0.159, 0.247, 0.141])
        for k, (stay, missing) in enumerate(settings):
            rows = sorted(
                [
                    r
                    for r in data["recovery"]["rows"]
                    if r["axis"] == feature
                    and r["stay_probability"] == stay
                    and r["missing_fraction"] == missing
                ],
                key=lambda r: r["n"],
            )
            x = np.array([r["n"] for r in rows], float) * (1 + (k - 1.5) * 0.02)
            ax.scatter(
                x,
                [r["finite_window_target"] for r in rows],
                s=17,
                marker="o",
                color=setting_colors[k],
            )
            ax.scatter(
                x,
                [r["long_record_reference"] for r in rows],
                s=20,
                marker="^",
                facecolors="none",
                edgecolors=setting_colors[k],
                linewidths=0.8,
            )
        ax.set_xscale("log")
        ax.set_xticks([12, 20, 120], ["12", "20", "120"])
        ax.minorticks_off()
        ax.set_xlabel("Sequence length (log scale)")
        ax.set_ylabel("Summary value" if j == 0 else "")
        ax.set_title(feature.replace("_", " ").title(), pad=6)
        clean(ax, "y")
    handles = [
        Line2D(
            [], [], color=c, marker="s", ls="none", markersize=4, label=f"stay {s:g}, missing {m:g}"
        )
        for c, (s, m) in zip(setting_colors, settings, strict=True)
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.061),
        ncol=2,
        frameon=False,
        fontsize=7,
    )
    fig.text(
        0.04,
        0.047,
        "Filled circles: Monte Carlo finite-window targets. Open triangles: simulated long-record references.",
        fontsize=7,
    )
    fig.text(
        0.04,
        0.024,
        "Known three-state model only: not biological EEG calibration and not the prediction bootstrap in Figure 2.",
        fontsize=7,
    )
    return fig


def aggregate_exports(data, output):
    tables = output / "source_tables"
    tables.mkdir()
    review = data["review"]
    for name in ["models", "comparisons", "prediction_diagnostics", "null_controls"]:
        pd.json_normalize(review[name]).to_csv(tables / f"{name}.csv", index=False)
    interactions = [
        interaction(data, f, c) for f in FEATURES for c in ["movie", "meditation", "music"]
    ]
    pd.DataFrame(interactions).to_csv(tables / "context_interactions.csv", index=False)
    pd.DataFrame(data["subjective"]["rows"]).to_csv(tables / "subjective_all_396.csv", index=False)
    codebook = {
        key: value["Description"]
        for key, value in data["subjective"]["codebook"].items()
        if key.startswith("ASC11_") and "COMPOSITE" not in key
    }
    (tables / "subjective_scale_labels.json").write_text(json.dumps(codebook, indent=2))
    pd.json_normalize(data["recovery"]["rows"]).to_csv(
        tables / "simulation_all_settings.csv", index=False
    )
    null_rows = []
    for port in PORTS:
        for rep in REPS:
            for kind in ["label_permutation", "temporal_permutation", "representation_phase"]:
                for metric in [
                    "absolute_dynamic_log_loss",
                    "conventional_multivariate",
                    "nonlinear_scalar",
                    "spectral_arousal",
                ]:
                    matches = [
                        r
                        for r in review["null_controls"]
                        if r["portfolio"] == port
                        and r["analysis"] == f"{TRACKS[port]}:{rep}"
                        and r["kind"] == kind
                        and r["metric"] == metric
                    ]
                    if not matches:
                        continue
                    values, audit = null_values(data, port, rep, kind, metric)
                    for repeat, value in enumerate(values):
                        null_rows.append(
                            dict(
                                portfolio=port,
                                representation=rep,
                                kind=kind,
                                metric=metric,
                                replicate=repeat,
                                null_statistic=value,
                                observed=audit["observed"],
                            )
                        )
    pd.DataFrame(null_rows).to_csv(tables / "null_refit_statistics.csv", index=False)
    # These are aggregate study statistics, never raw predictions or questionnaire records.
    for path in tables.glob("*.csv"):
        columns = pd.read_csv(path, nrows=0).columns
        if any(c in columns for c in ["participant_id", "unit_id", "probability"]):
            raise ValueError("Participant records must remain on the data host")


def submission_artwork(fig, number):
    """Remove editorial scaffolding, not evidence; legends retain the moved notes."""
    remove_prefixes = {
        2: ("Small marks/lines:", "Left of zero favours", "Overlapping portfolios;"),
        3: (
            "How to read",
            "Coloured dots:",
            "p: one-sided",
            "Minimum raw",
            "Permutable",
            "Temporal controls",
        ),
        4: (
            "Eight dynamical",
            "Intervals: descriptive",
            "Fixed task order",
            "Rep repertoire",
            "Dir directionality",
        ),
        5: ("Illustrative categorical", "Each dot:", "Known three-state"),
    }
    for text in list(fig.texts):
        if (
            text.get_position()[1] > 0.93
            or text.get_text().startswith(remove_prefixes[number])
            or (number == 3 and text.get_text() == "c")
        ):
            text.remove()
    # Reclaim header/footnote space without cropping axes, changing values or text size.
    lower, upper, height = {
        2: (0.040, 0.927, 180),
        3: (0.136, 0.930, 174),
        4: (0.047, 0.933, 220),
        5: (0.029, 0.934, 186),
    }[number]
    scale = 0.95 / (upper - lower)
    for ax in fig.axes:
        box = ax.get_position()
        ax.set_position([box.x0, 0.025 + (box.y0 - lower) * scale, box.width, box.height * scale])
    for text in fig.texts:
        x, y = text.get_position()
        text.set_position((x, 0.025 + (y - lower) * scale))
    fig.set_size_inches(183 / 25.4, height / 25.4)
    return fig


def captions(data):
    return """# Draft figure legends

All figures report exploratory, non-preregistered analyses. Numerical panels use original hash-verified outputs; no models or inferential tests were rerun for drawing. Source tables preserve full precision. These figures are not a declaration of completed biological calibration or fully refitted prediction uncertainty.

## Figure 2. Prediction depends on the comparator

(a) Held-out log loss in the sparse core (80 participant identifiers, 386 observations, three study groups) and high-density extension (62 identifiers, 380 observations, three groups). Grey small points/lines identify matched held-out studies across model categories; large marks are equal-study means, with participants weighted equally inside study. Conv., conventional multivariate features; LZ, nonlinear Lempel-Ziv comparator. The constant predictor uses experience rates from the other study groups only; no held-out labels set its probability. Its comparison is descriptive, without newly calculated paired intervals. (b,c) Dynamics-minus-comparator log loss for conventional features and nonlinear LZ. Negative values favour dynamics. Coloured estimates and original hierarchical study/participant 95% bootstrap intervals condition on fitted predictions; small grey ticks show individual study contrasts. These intervals do not include model-refitting uncertainty. (d) AUROC for each held-out study and dynamics representation, with the 0.5 reference. Horizontal offsets separate studies, not another measured variable. The portfolios overlap and differ in cohort composition and report semantics: this is not a paired electrode-density intervention. Benchmark advantages must be read with the null controls in Figure 3.

## Figure 3. Structure controls qualify prediction gains

(a) Absolute dynamics loss and (b) dynamics-minus-conventional loss for all three representations in both portfolios. Each row displays all 100 independently seeded nested null refits of the specified kind; they are algorithmic refits, not independent biological samples. Coloured dots show the actual null statistics; deterministic vertical offsets serve visibility only. Black lines mark observed statistics. Label permutation assumes exchangeability within participant and sleep stage; only 49/80 core and 29/62 extension participant blocks contain both labels. Temporal permutation targets representations, not original-EEG spectral structure. Row annotations give one-sided plus-one Monte Carlo tails p=(1+count(null<=observed))/101 and Holm-adjusted values H across the complete 72-comparison post-results audit family. Complete phase-null and alternative-comparator results are included in the source tables. The extension's conventional-model advantages are not unusual under the displayed nulls. Sparse sensor absolute loss supplies a limited unadjusted lead (p=1/101 against both displayed controls), but no comparison survives the broad Holm adjustment. Limited resolution and shared data constrain inference; non-rejection does not establish equivalence or absence of information. (c) Interpretation and permutation-support notes.

## Figure 4. Context sensitivity and subjective associations

(a) All 27 context-versus-rest differences in session change: (session 02 minus session 01) in each context, minus the same difference at rest. Points are participant means; intervals are the original descriptive participant-bootstrap 95% intervals, not simultaneous intervals. Row labels contain paired participant counts. Eight dynamical features are expressed in baseline-training-participant SD units; LZ remains in its native units. Dark points identify the two interactions surviving the reviewed participant sign-flip test (19,999 draws), with Holm adjustment across all 63 boundary tests (36 session changes plus 27 interactions). Exact raw/adjusted p-values and intervals are in the source table. (b) Every complete participant contrast for the two corrected movie/rest interactions (n=54 each); deterministic vertical offsets encode no variable. Diamonds and bars show means and the same descriptive intervals. These examples are result-selected for display, while the complete interaction set appears in (a). Fixed task order, eyes-open/closed differences, and the absence of placebo preclude an isolated causal context effect. (c) All eleven official non-imputed ASC11 subscales crossed with nine EEG features in four contexts (396 Spearman associations; complete-pair n=47-53). Colour encodes rho on a common -1 to +1 scale, not significance. No association survives Holm correction; no cells are discarded for being non-significant. Ratings describe global post-session experience, not simultaneous context-specific EEG experience. Features/normalization are held fixed during resampling. This operational reanalysis uses the PsiConnect cohort already studied for context-dependent dynamics (Stoliker et al., Nature, 2026; doi:10.1038/s41586-026-10910-z), and is not independent-cohort replication of that finding.

## Figure 5. Observation history changes the measurement target

(a) An explicitly illustrative categorical sequence (not EEG) is assessed continuously or with boundaries every three samples. Recurrence values are calculated using the study estimator; segment boundaries reset available history. (b) Coverage for recurrence, exit entropy and dwell dispersion in each of twelve settings: lengths 12/20/120, stay probabilities 0.3/0.7 and missing fractions 0/0.2. Block/long evaluates the original three-observation block procedure against a 10,000-state simulated reference; block/finite evaluates it against the finite-window target; model/finite uses independent known-model finite-window calibration. Each dot represents 200 independent test draws. Horizontal bars are descriptive medians across settings; the dashed reference is 95%. All cell-level exact binomial intervals are provided in the source tables. Separate samples of 1,000 reference and 1,000 calibration sequences are independent of the test sequences; the block procedure uses 99 resamples per test sequence. (c) Actual recorded finite-window target estimates (filled circles) and long-record references (open triangles), by length and setting. Tiny multiplicative x-offsets separate settings; true lengths are 12, 20 and 120. No fitted curves or target-uncertainty intervals are implied. Targets and references are Monte Carlo quantities, not exact population constants. The known stationary three-state model diagnoses a measurement problem and a model-dependent remedy; it neither validates biological EEG intervals nor repairs all dynamical features. This within-trajectory uncertainty procedure is distinct from the prediction-level bootstrap in Figure 2.
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ["core", "hd", "review-root", "output"]:
        parser.add_argument(f"--{arg}", type=Path, required=True)
    parser.add_argument(
        "--submission", action="store_true", help="Clean artwork; retain interpretation in legends"
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a new output directory; historical figures are preserved")
    data = load_sources(args.core, args.hd, args.review_root)
    args.output.mkdir(parents=True)
    aggregate_exports(data, args.output)
    figures = []
    audits = []
    with plt.rc_context(STYLE):
        for number, (name, draw) in enumerate(
            [
                ("fig02_prediction", fig2),
                ("fig03_null_controls", fig3),
                ("fig04_context_subjective", fig4),
                ("fig05_measurement", fig5),
            ],
            start=2,
        ):
            fig = draw(data)
            if args.submission:
                submission_artwork(fig, number)
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            outside = []
            hidden_tick_labels = set()
            for axes in fig.axes:
                for axis, bounds in [(axes.xaxis, axes.get_xlim()), (axes.yaxis, axes.get_ylim())]:
                    for tick in [*axis.get_major_ticks(), *axis.get_minor_ticks()]:
                        if not min(bounds) <= tick.get_loc() <= max(bounds):
                            hidden_tick_labels.update([id(tick.label1), id(tick.label2)])
            for text in fig.findobj(matplotlib.text.Text):
                if id(text) in hidden_tick_labels or not text.get_visible() or not text.get_text():
                    continue
                box = text.get_window_extent(renderer)
                if (
                    box.x0 < -1
                    or box.y0 < -1
                    or box.x1 > fig.bbox.width + 1
                    or box.y1 > fig.bbox.height + 1
                ):
                    outside.append(text.get_text())
            audits.append(
                {
                    "figure": name,
                    "outside_canvas_text": outside,
                    "size_mm": (fig.get_size_inches() * 25.4).tolist(),
                }
            )
            for extension in ["pdf", "svg", "png"]:
                fig.savefig(
                    args.output / f"{name}.{extension}", dpi=600 if args.submission else 220
                )
            figures.append(fig)
        with PdfPages(
            args.output / ("Figures_2-5.pdf" if args.submission else "Figures_2-5_draft.pdf")
        ) as pdf:
            for fig in figures:
                pdf.savefig(fig)
        for fig in figures:
            plt.close(fig)
    legends = captions(data)
    if args.submission:
        legends = legends.replace("# Draft figure legends", "# Figure legends").replace(
            " (c) Interpretation and permutation-support notes.", ""
        )
        legends += "\nFigure 4 abbreviations: Rep, repertoire; Rec, recurrence; Ent, exit entropy; LogD, log dwell; Disp, dwell dispersion; Dir, directionality; Ali, alignment; Reach, reachability; LZ, Lempel-Ziv.\n"
    (args.output / "CAPTIONS.md").write_text(legends, encoding="utf-8")
    (args.output / "layout_checks.json").write_text(json.dumps(audits, indent=2))
    manifest = {
        "status": "submission_artwork_requires_visual_inspection"
        if args.submission
        else "draft_exports_require_visual_inspection",
        "exploratory": True,
        "registered_or_preregistered": False,
        "source_artifacts": data["sources"],
        "renderer_sha256": digest(Path(__file__)),
        "matplotlib_version": matplotlib.__version__,
        "raw_participant_records_exported": False,
        "panel_sources": {
            "2": "transfer summaries/predictions; review comparisons and training-only constant diagnostics",
            "3": "original control refits matched to reviewed observed statistics and multiplicity",
            "4a": "original boundary.context_interactions + review.boundary.contrasts",
            "4b": "original paired boundary.measurements; participant records remain on host",
            "4c": "subjective_review.rows: all contexts/features/scales, unchanged order within axes",
            "5a": "explicit schematic computed with the study recurrence estimator",
            "5bc": "recovery-geometry-v3.rows: all 12 settings for all three measures",
        },
        "outputs": [
            {"file": str(p.relative_to(args.output)), "sha256": digest(p)}
            for p in sorted(args.output.rglob("*"))
            if p.is_file()
        ],
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("MANUSCRIPT_FIGURES_COMPLETE", args.output)
    print("LAYOUT_CHECKS", json.dumps(audits))


if __name__ == "__main__":
    main()
