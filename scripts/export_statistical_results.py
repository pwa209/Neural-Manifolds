"""Package all available aggregate statistical results without participant-level records.

Inputs are the checksum-verified manuscript review, additional-evidence review,
and complete 5,000-refit null continuation. Historical 100-refit null tables are
deliberately excluded. This script performs no new model fitting or hypothesis test.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

from render_null_5000_update import load_final


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def flatten(record: dict, prefix: str = "") -> dict:
    """Flatten aggregate dictionaries; split numeric interval endpoints."""
    result = {}
    for name, value in record.items():
        key = f"{prefix}{name}"
        if isinstance(value, dict):
            result.update(flatten(value, f"{key}_"))
        elif isinstance(value, list) and len(value) == 2 and all(
            isinstance(item, (int, float)) or item is None for item in value
        ):
            result[f"{key}_low"] = value[0]
            result[f"{key}_high"] = value[1]
        elif isinstance(value, list):
            result[key] = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        else:
            result[key] = value
    return result


def write_rows(path: Path, rows: list[dict]) -> int:
    if not rows:
        raise ValueError(f"Refusing empty table: {path.name}")
    if any("participant_id" in row or "paired_values" in row for row in rows):
        raise ValueError(f"Participant-level value present in {path.name}")
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def copy_verified(source: Path, target: Path, expected: str | None = None) -> None:
    if expected and sha256(source) != expected:
        raise ValueError(f"Source checksum mismatch: {source}")
    if target.exists():
        raise FileExistsError(target)
    shutil.copy2(source, target)
    if sha256(source) != sha256(target):
        raise ValueError(f"Copy checksum mismatch: {target}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-tables", type=Path, required=True)
    parser.add_argument("--additional", type=Path, required=True)
    parser.add_argument("--additional-manifest", type=Path, required=True)
    parser.add_argument("--null-report", type=Path, required=True)
    parser.add_argument("--null-statistics", type=Path, required=True)
    parser.add_argument("--figures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a new versioned export directory")

    additional_manifest = json.loads(args.additional_manifest.read_text(encoding="utf-8"))
    expected_additional = next(
        item["sha256"] for item in additional_manifest["outputs"]
        if item["file"] == args.additional.name
    )
    if sha256(args.additional) != expected_additional:
        raise ValueError("Additional aggregate review checksum mismatch")
    additional = json.loads(args.additional.read_text(encoding="utf-8"))
    null_report, _frame, null_rows, null_records = load_final(
        args.null_report, args.null_statistics
    )

    table_dir = args.output / "tables"
    figure_dir = args.output / "figures"
    table_dir.mkdir(parents=True)
    figure_dir.mkdir()
    table_counts = {}
    copies = {
        "prediction_comparisons.csv": "comparisons.csv",
        "prediction_models_by_study.csv": "models.csv",
        "prediction_diagnostics.csv": "prediction_diagnostics.csv",
        "context_interactions_original.csv": "context_interactions.csv",
        "subjective_associations_396.csv": "subjective_all_396.csv",
        "measurement_simulation_36.csv": "simulation_all_settings.csv",
    }
    for target_name, source_name in copies.items():
        source = args.source_tables / source_name
        target = table_dir / target_name
        copy_verified(source, target)
        with target.open(encoding="utf-8", newline="") as stream:
            table_counts[target_name] = sum(1 for _ in csv.DictReader(stream))
    copy_verified(
        args.source_tables / "subjective_scale_labels.json",
        table_dir / "subjective_scale_labels.json",
    )
    copy_verified(
        args.null_statistics,
        table_dir / "null_refit_statistics_5000.csv",
        null_report["source_statistics"]["sha256"],
    )
    table_counts["null_refit_statistics_5000.csv"] = 360_000

    def save(name: str, records: list[dict]) -> None:
        table_counts[name] = write_rows(table_dir / name, [flatten(row) for row in records])

    save("null_tests_72.csv", list(null_report["null_controls"]))
    quantiles = []
    for key, values in sorted(null_rows.items()):
        test = null_records[key]
        quantiles.append({
            "portfolio": key[0], "representation": key[1], "control": key[2],
            "metric": key[3], "refits": len(values), "observed": test["observed"],
            "null_min": float(values[0]), "null_p025": float(np.quantile(values, .025)),
            "null_p25": float(np.quantile(values, .25)),
            "null_median": float(np.median(values)),
            "null_p75": float(np.quantile(values, .75)),
            "null_p975": float(np.quantile(values, .975)),
            "null_max": float(values[-1]),
            "null_at_or_below_observed": test["null_lower_or_equal"],
            "p_lower": test["p_lower"],
            "p_holm_72": test["p_holm_all_null_comparisons"],
        })
    save("null_distribution_summary_72.csv", quantiles)
    save("context_boundary_tests_63.csv", additional["boundary"]["contrasts"])
    save("axis_recovery_256.csv", additional["axis_recovery"]["rows"])

    inventory = []
    inventory_comparisons = []
    for item in additional["analysis_inventory"]:
        inventory.append({
            "portfolio": item["portfolio"], "artifact": item["artifact"],
            "analysis": item["analysis"], "status": item["status"],
            "reason": item.get("reason"), "comparison_count": len(item["comparisons"]),
        })
        for comparison in item["comparisons"]:
            inventory_comparisons.append({
                "portfolio": item["portfolio"], "artifact": item["artifact"],
                "analysis": item["analysis"], "status": item["status"],
                **comparison,
            })
    save("analysis_inventory_186.csv", inventory)
    save("inventory_comparisons.csv", inventory_comparisons)

    report_task, tactile = additional["specificity"]
    save("report_task_contrasts_15.csv", report_task["contrasts"])
    save("tactile_contrasts_30.csv", tactile["contrasts"])
    save("tactile_preresponse_30.csv", tactile["pre_response_contrasts"])
    save("tactile_confidence_15.csv", tactile["confidence_associations"])

    tms_main = []
    tms_condition = []
    tms_changes = []
    for outcome, result in additional["tms"].items():
        tms_main.append({
            "outcome": outcome, "incremental_squared_error": result["incremental_squared_error"],
            "interval_95": result["interval_95"], "participants": result["participants"],
            "uncertainty": result["uncertainty"],
            "within_condition_errors": result["within_condition_errors"],
        })
        for row in result["within_condition_calibration_and_association"]:
            tms_condition.append({"outcome": outcome, **row})
        for row in result["within_participant_changes"]:
            if "paired_values" not in row:
                raise ValueError("Expected participant-level pairs for explicit exclusion")
            tms_changes.append({"outcome": outcome, **{
                key: value for key, value in row.items() if key != "paired_values"
            }})
    save("tms_outcomes_3.csv", tms_main)
    save("tms_condition_calibration.csv", tms_condition)
    save("tms_paired_change_summaries.csv", tms_changes)

    for name in [
        "fig02_prediction", "fig03_null_controls", "fig04_context_subjective", "fig05_measurement"
    ]:
        for suffix in (".pdf", ".png"):
            copy_verified(args.figures / f"{name}{suffix}", figure_dir / f"{name}{suffix}")
    copy_verified(args.figures / "Figures_2-5.pdf", figure_dir / "Figures_2-5.pdf")
    copy_verified(args.figures / "CAPTIONS.md", figure_dir / "CAPTIONS.md")

    manifest = {
        "version": "2026-09-24-final-null-5000",
        "analysis_status": "completed aggregate results; exploratory, not preregistered",
        "source_sha256": {
            "additional_aggregate_results": expected_additional,
            "null_review_5000": sha256(args.null_report),
            "null_refit_statistics_5000": sha256(args.null_statistics),
        },
        "table_rows": table_counts,
        "excluded": [
            "raw EEG, questionnaire records, and participant-level predictions",
            "TMS within_participant_changes.paired_values",
            "historical 100-refit null-control table superseded by complete 5,000-refit audit",
        ],
        "outputs": [
            {"file": path.relative_to(args.output).as_posix(), "sha256": sha256(path)}
            for path in sorted(args.output.rglob("*")) if path.is_file()
        ],
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"tables": len(table_counts), "figures": 9, "rows": table_counts}, indent=2))


if __name__ == "__main__":
    main()
