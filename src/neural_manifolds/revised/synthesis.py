"""Receipt-derived tables, figures and explicit completion scope."""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from neural_manifolds.provenance import atomic_write_json, sha256_file


def forest(rows, output, name, *, label, value, caption):
    """Plot supplied interval endpoints; do not invent uncertainty."""
    selected = [r for r in rows if r.get(value) is not None]
    paths = []
    for offset in range(0, len(selected), 18):
        page = selected[offset : offset + 18]
        with plt.rc_context({"font.size": 8, "pdf.fonttype": 42, "svg.fonttype": "none"}):
            fig, ax = plt.subplots(
                figsize=(7.1, max(2.2, 0.23 * len(page) + 1.4)), layout="constrained"
            )
            for index, row in enumerate(page):
                interval = row.get("interval_95")
                if interval is not None:
                    ax.plot(interval, [index, index], color="#0072B2", lw=1)
                ax.plot(row[value], index, "o", color="#0072B2", ms=4)
            ax.axvline(0, color=".6", lw=0.6)
            ax.set_yticks(range(len(page)), [str(r[label]).replace("_", " ") for r in page])
            ax.invert_yaxis()
            ax.set_xlabel(value.replace("_", " "))
            ax.spines[["top", "right"]].set_visible(False)
            for suffix in ("pdf", "png"):
                path = output / f"{name}-{offset // 18 + 1}.{suffix}"
                fig.savefig(path, dpi=180)
                paths.append(path)
            plt.close(fig)
    if selected:
        caption_path = output / f"{name}-caption.md"
        caption_path.write_text(caption + "\n", encoding="utf-8")
        paths.append(caption_path)
    return paths


def synthesize(inputs: list[Path], output: Path, *, expected_controls=100) -> list[Path]:
    output.mkdir(parents=True, exist_ok=True)
    evidence, tables, plots, limitations = [], {}, {}, []
    counts = {}
    control_replicates = set()
    for index, path in enumerate(inputs):
        content = json.loads(path.read_text())
        digest = sha256_file(path)
        counts[path.name] = counts.get(path.name, 0) + 1
        evidence.append(
            {"artifact": path.name, "source_path": str(path), "sha256": digest, "content": content}
        )
        limitations.extend(content.get("limitations", []))

        def add(name, rows, producing_digest=digest, **context):
            tables.setdefault(name, []).extend(
                {**r, **context, "producing_artifact_sha256": producing_digest} for r in rows
            )

        if path.name in {"transfer.json", "sensitivities.json", "robustness.json"}:
            for analysis, record in content.get("analyses", {}).items():
                summary = record.get("summary", {})
                add("model_performance", summary.get("models", []), analysis=analysis)
                comparisons = summary.get("paired_comparisons", [])
                add("paired_model_comparisons", comparisons, analysis=analysis)
                add("unavailable", record.get("unavailable", []), analysis=analysis)
                name = re.sub(r"[^a-zA-Z0-9_-]", "_", f"transfer-{index}-{analysis}")
                plots[name] = forest(
                    comparisons,
                    output,
                    name,
                    label="baseline",
                    value="dynamic_minus_baseline_log_loss",
                    caption=f"{analysis}. Shared dynamics minus alternative held-out log loss; negative favours dynamics. "
                    "Points equally weight studies and participants. Lines are 95% hierarchical bootstrap intervals "
                    "conditional on fitted predictions. Few laboratories limit population generality. "
                    f"Source artifact SHA-256: {digest}.",
                )
        elif path.name == "axis_recovery.json":
            add("axis_recovery", content.get("rows", []))
        elif path.name == "controls.json":
            control_replicates.add(content["replicate"])
            add("null_controls", content.get("rows", []))
        elif path.name == "tms.json":
            summaries = []
            for outcome, record in content.get("analyses", {}).items():
                add("tms_held_out_predictions", record.get("predictions", []), outcome=outcome)
                summary = record.get("summary", {})
                if summary:
                    summaries.append({"outcome": outcome, **summary})
            add("tms_incremental_error", summaries)
            for row in summaries:
                name = "tms-" + row["outcome"]
                plots[name] = forest(
                    [row],
                    output,
                    name,
                    label="outcome",
                    value="incremental_squared_error",
                    caption="Condition plus conventional EEG plus dynamics minus condition plus conventional EEG "
                    "held-out squared error, in squared outcome units. Negative favours adding dynamics. "
                    "Intervals resample participants conditional on fitted predictions. This is predictive "
                    f"association, not causal controllability. Artifact: {digest}.",
                )
        elif path.name == "specificity.json":
            add("specificity", content.get("contrasts", []), source_index=index)
            add(
                "specificity_pre_response",
                content.get("pre_response_contrasts", []),
                source_index=index,
            )
            add("confidence", content.get("confidence_associations", []), source_index=index)
            for feature in ("normalized_rms", "spatial_participation", "normalized_roughness"):
                rows = [
                    dict(r, label=f"{r['contrast']} / {r['interval']}")
                    for r in content.get("contrasts", [])
                    if r.get("feature") == feature
                ]
                name = f"specificity-{index}-{feature}"
                plots[name] = forest(
                    rows,
                    output,
                    name,
                    label="label",
                    value="difference",
                    caption="Participant differences matched on physical intensity; 95% participant-bootstrap "
                    "intervals. Independent n is in source tables. Report/no-report is order-confounded; "
                    f"undetected stimuli are not global unconsciousness. Artifact: {digest}.",
                )
        elif path.name == "boundary.json":
            add("boundary_changes", content.get("paired_changes", []))
            add("boundary_interactions", content.get("context_interactions", []))
    paths = [p for group in plots.values() for p in group]
    for name, rows in tables.items():
        path = output / f"{name}.csv"
        data = [
            {
                k: json.dumps(v, sort_keys=True) if isinstance(v, (list, dict)) else v
                for k, v in row.items()
            }
            for row in rows
        ]
        pd.DataFrame(data).to_csv(path, index=False)
        paths.append(path)
    core = (
        all(
            counts.get(n, 0) > 0
            for n in [
                "transfer.json",
                "axis_recovery.json",
                "sensitivities.json",
                "robustness.json",
                "tms.json",
            ]
        )
        and counts.get("specificity.json", 0) >= 2
        and len(control_replicates) >= expected_controls
    )
    result = {
        "evidence": evidence,
        "status": "core_execution_complete" if core else "interim_evidence",
        "scientific_gates": False,
        "registered_or_preregistered": False,
        "core_execution_complete": core,
        "full_study_complete": core and counts.get("boundary.json", 0) > 0,
        "optional_extensions": "clinical_and_fMRI_separate_not_required_for_core",
        "null_replicates_completed": len(control_replicates),
        "null_replicates_expected": expected_controls,
        "limitations": sorted(set(limitations)),
        "visual_qa": "rendered_automatically_not_human_visually_inspected",
        "exports": {p.name: sha256_file(p) for p in paths},
        "interpretation": "Report predictions and their failures; neither establishes consciousness or a causal mechanism.",
    }
    path = output / "evidence.json"
    atomic_write_json(path, result)
    return [path, *paths]
