"""Exact source-to-record linkage and auditable analysis admission.

Numeric report codes alone never establish a construct. Unreviewed sources remain
in the denominator, with explicit reasons. No p-value or predicted effect is read.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from pathlib import PurePosixPath

from neural_manifolds.continuous.audit import safe_member


def normalized_channel(name: str) -> str:
    value = name.strip()
    if value.upper().startswith("EEG "):
        value = value[4:]
    # Remove only documented common references, never arbitrary bipolar leads.
    for suffix in ("-REF", "-A1", "-A2", "-M1", "-M2"):
        if value.upper().endswith(suffix):
            value = value[: -len(suffix)]
            break
    aliases = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}
    return aliases.get(value.upper(), value.upper())


def link_record(record: dict, files: list[dict]) -> dict:
    """Match Records.csv-relative PSG path, not a lossy basename search."""
    filename = record["Filename"].strip().replace("\\", "/").lstrip("/")
    if not filename or not safe_member(filename):
        raise ValueError("unsafe_or_empty_record_filename")
    origin = record["records_origin"]
    origin_path = origin.split("::", 1)[-1]
    expected = str(PurePosixPath(origin_path).parent / "Data/PSG" / filename)
    matches = [item for item in files if item["member"] == expected]
    if len(matches) != 1:
        raise ValueError(f"recording_join_not_unique:{len(matches)}")
    return matches[0]


def build_cohort(audits: dict[str, dict], policy: dict) -> dict:
    rows, exclusions = [], []
    for dataset, audit in sorted(audits.items()):
        identifiers = Counter(r.get("Case ID", "") for r in audit["records"])
        rule = policy["source_policy"].get(dataset)
        if not audit["records"]:
            exclusions.append({"dataset_id": dataset, "reason": "no_supported_report_records"})
        for record in audit["records"]:
            identifier = hashlib.sha256(
                f"{dataset}:{record.get('Case ID', '')}".encode()
            ).hexdigest()[:24]
            base = {"unit_id": identifier, "dataset_id": dataset}
            try:
                if identifiers[record.get("Case ID", "")] != 1:
                    raise ValueError("duplicate_case_identifier")
                if not record.get("Case ID", "").strip():
                    raise ValueError("missing_case_identifier")
                if rule is None:
                    raise ValueError("source_semantics_or_timing_not_adjudicated")
                item = link_record(record, audit["files"])
                if item.get("error"):
                    raise ValueError("source_recording_inventory_error")
                duration = float(record["Duration"])
                if not math.isfinite(duration) or duration < policy["dream_seconds"]:
                    raise ValueError("insufficient_documented_duration")
                header_duration = item.get("duration_seconds")
                if header_duration is None or not 0 <= header_duration - duration <= 1.01:
                    raise ValueError("record_header_duration_mismatch")
                report = record["Experience"].strip()
                stage = record["Last sleep stage"].strip()
                if report not in {"0", "1", "2"}:
                    raise ValueError("missing_or_ambiguous_report_code")
                if stage not in {"0", "1", "2", "3", "5"}:
                    raise ValueError("missing_or_unsupported_sleep_stage")
                if record.get("Subject healthy", "").strip() != "1":
                    raise ValueError("not_documented_healthy")
                if record.get("Treatment group", "").strip() not in {"", "0"}:
                    raise ValueError("treatment_requires_separate_adjudication")
                native_id = record["Subject ID"].strip()
                if not native_id:
                    raise ValueError("missing_participant_identifier")
                participant = hashlib.sha256(
                    f"{rule['participant_namespace']}:{native_id}".encode()
                ).hexdigest()[:24]
                names = item.get("channels", [])
                aliases = rule.get("channel_aliases", {})
                normalized = [normalized_channel(aliases.get(n, n)) for n in names]
                tracks = []
                for track, key in [
                    ("sparse", "sparse_channels"),
                    ("high_density", "high_density_channels"),
                ]:
                    channels = policy[key]
                    if all(normalized.count(n.upper()) == 1 for n in channels):
                        tracks.append(track)
                if not tracks:
                    raise ValueError("no_compatible_observed_channel_track")
                rows.append(
                    {
                        **base,
                        "participant_id": participant,
                        "study_group": rule["study_group"],
                        "release": audit["release"],
                        "recording": item,
                        "completion_marker_sha256": audit["completion_marker_sha256"],
                        "report_code": int(report),
                        "sleep_stage": int(stage),
                        "experience": 1 if report == "2" else (0 if report == "0" else None),
                        "primary_eligible": stage == "2" and report in {"0", "2"},
                        "start_seconds": duration - policy["dream_seconds"],
                        "stop_seconds": duration,
                        "tracks": tracks,
                        "timing_policy": rule["timing"],
                        "source_evidence": rule["source_evidence"],
                        "channel_aliases": aliases,
                        "position_equivalence": rule.get(
                            "position_equivalence", "native_named_channel"
                        ),
                        "position_error_cm": rule.get("position_error_cm", {}),
                        "remarks": record.get("Remarks", ""),
                        "participant_alias_status": "within_oslo_unresolved"
                        if rule["study_group"] == "oslo"
                        else "source_native_identifier",
                    }
                )
            except (ValueError, KeyError, TypeError) as exc:
                exclusions.append({**base, "reason": str(exc)})
    return {
        "units": rows,
        "exclusions": exclusions,
        "scientific_gates": False,
        "scope": "source_adjudicated_before_signal_window_qc",
        "unreviewed_sources_are_not_negative_results": True,
    }
