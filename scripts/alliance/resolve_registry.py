"""Freeze public DREAM metadata and construct the fresh open-only registry.

Acquisition eligibility is not analysis eligibility: channel, stage, age, report
and overlapping-cohort audits happen after download and before inference.
"""

from __future__ import annotations

import argparse
import csv
import io
import re
from pathlib import Path

import yaml

from neural_manifolds.data.http import HttpClient
from neural_manifolds.data.models import DatasetRegistryModel
from neural_manifolds.provenance import atomic_write_json


def eligible_rows(text: str) -> list[dict]:
    rows = list(csv.DictReader(io.StringIO(text)))
    required = {"Set ID", "Accessibility", "Latest amendment", "Revoked", "Data URL"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError("DREAM registry schema changed")
    return [
        r
        for r in rows
        if r["Accessibility"].strip().lower() == "open"
        and r["Latest amendment"].strip().upper() == "TRUE"
        and r["Revoked"].strip().upper() not in {"TRUE", "1", "YES"}
    ]


def run(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    if (output / "datasets.open.yaml").exists():
        raise ValueError("Registry already frozen; choose a new discovery directory")
    client = HttpClient()
    meta = client.get_json("https://api.figshare.com/v2/articles/22133105/versions/9")
    atomic_write_json(output / "dream-registry-v9.json", meta)
    f = next(f for f in meta["files"] if f["name"] == "Datasets.csv")
    path = output / "DREAM-Datasets-v9.csv"
    client.download(
        f["download_url"], path, expected_size=f["size"], expected_hashes={"md5": f["computed_md5"]}
    )
    base = yaml.safe_load(Path("configs/datasets.yaml").read_text())
    policy = yaml.safe_load(Path("configs/revision_v2.yaml").read_text())
    selected = set(policy["initial_datasets"]) - {"dream_tononi_serial_awakenings"}
    datasets = [
        d for d in base["datasets"] if d["id"] in selected and d["access"]["mode"] == "open"
    ]
    decisions = []
    for row in eligible_rows(path.read_text(encoding="utf-8-sig")):
        decision = {
            "set_id": row["Set ID"],
            "name": row["Common name"],
            "study_location": row.get("Study location"),
            "status": "metadata_only",
            "analysis_eligible": False,
            "cohort_overlap": "unresolved",
        }
        match = re.fullmatch(
            r"https://doi.org/(?:10\.6084/m9\.figshare\.|10\.26180/)(\d+)(?:\.v(\d+))?",
            row["Data URL"].strip(),
        )
        if not match:
            decision["status"] = "unsupported_provider_needs_public_route_audit"
            decisions.append(decision)
            continue
        accession, version = match.groups()
        try:
            endpoint = f"https://api.figshare.com/v2/articles/{accession}"
            metadata = client.get_json(endpoint + (f"/versions/{version}" if version else ""))
            version = str(metadata["version"])
            metadata = client.get_json(endpoint + f"/versions/{version}")
            atomic_write_json(output / f"dream-set-{row['Set ID']}-v{version}.json", metadata)
            license_name = metadata.get("license", {}).get("name", "")
            if license_name not in {"CC BY 4.0", "CC0"} or not metadata.get("files"):
                raise ValueError("Open files or supported reuse licence not verified")
            files = metadata["files"]
            if any(not f.get("download_url") or f.get("is_link_only") for f in files):
                raise ValueError("Not all files have direct download links")
            identifier = (
                "dream_tononi_serial_awakenings"
                if row["Set ID"] == "13"
                else f"dream_set_{int(row['Set ID']):02d}"
            )
            datasets.append(
                {
                    "id": identifier,
                    "title": row["Common name"],
                    "role": "dream_candidate_pending_stage_montage_and_cohort_audit",
                    "modalities": [
                        "meg" if row["Set ID"] == "20" else "eeg",
                        "experience_reports",
                    ],
                    "source": {
                        "provider": "figshare",
                        "accession": accession,
                        "version": version,
                        "doi": metadata["doi"],
                        "landing_url": metadata["url_public_html"],
                        "api_url": endpoint + f"/versions/{version}",
                        "mutable_upstream": False,
                    },
                    "license": {
                        "spdx": "CC-BY-4.0" if license_name == "CC BY 4.0" else "CC0-1.0",
                        "status": "verified",
                        "source_url": endpoint + f"/versions/{version}",
                    },
                    "access": {"mode": "open", "terms_url": "https://info.figshare.com/terms/"},
                    "validation": {
                        "minimum_files": len(files),
                        "minimum_bytes": sum(f["size"] for f in files),
                    },
                    "official_sources": [row["Data URL"], endpoint + f"/versions/{version}"],
                }
            )
            decision.update(
                status="open_acquisition_candidate",
                dataset_id=identifier,
                version=version,
                bytes=sum(f["size"] for f in files),
            )
        except Exception as exc:
            decision.update(status="metadata_unresolved", error=str(exc))
        decisions.append(decision)
        atomic_write_json(output / "dream-screening.json", {"rows": decisions})
    registry = {**base, "registry_verified_on": "2026-09-16", "datasets": datasets}
    DatasetRegistryModel.model_validate(registry)
    (output / "datasets.open.yaml").write_text(
        yaml.safe_dump(registry, sort_keys=False), encoding="utf-8"
    )
    atomic_write_json(
        output / "screening-summary.json",
        {
            "datasets": len(datasets),
            "analysis_ready": False,
            "scientific_gates": False,
            "registry_version": 9,
            "excluded": policy["excluded_dataset_ids"],
            "decisions": decisions,
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    run(p.parse_args().output)
