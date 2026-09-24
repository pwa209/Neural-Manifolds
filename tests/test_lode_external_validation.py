from __future__ import annotations

import csv
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import yaml

from neural_manifolds.provenance import sha256_file
from neural_manifolds.validation import lode


def test_lode_metadata_preflight_enforces_source_bad_channels(tmp_path: Path):
    archive = tmp_path / "Data.zip"
    with ZipFile(archive, "w") as z:
        for index in range(3):
            z.writestr(f"Data/PSG/P{index}/record.edf", b"edf")
    records = tmp_path / "Records.csv"
    columns = ["Filename", "Case ID", "Subject ID", "Experience", "Last sleep stage", "Duration", "Remarks"]
    with records.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerow(dict(zip(columns, ["P0/record.edf", "C0", "P0", "2", "2", "140", "Bad channels: 2 6"], strict=True)))
        writer.writerow(dict(zip(columns, ["P1/record.edf", "C1", "P1", "0", "2", "140", "Bad channels: 1 6"], strict=True)))
        writer.writerow(dict(zip(columns, ["P2/record.edf", "C2", "P2", "1", "2", "140", "Bad channels:"], strict=True)))
    policy = yaml.safe_load(Path("configs/lode_external_validation.yaml").read_text())
    policy["source_sha256"] = {"records_csv": sha256_file(records), "data_zip": sha256_file(archive)}
    config = tmp_path / "policy.yaml"
    config.write_text(yaml.safe_dump(policy))
    result = lode.preflight(config, records, archive, tmp_path / "out")
    assert len(result["candidates"]) == 1
    assert result["candidate_counts"] == {2: 1}
    assert result["metadata_exclusions"] == {
        "primary_channel_source_flagged_bad": 1,
        "not_clear_E_or_NE": 1,
    }


def test_lode_constant_uses_only_outer_training_labels(monkeypatch):
    policy = yaml.safe_load(Path("configs/lode_external_validation.yaml").read_text())
    rows = []
    for fold in range(5):
        rows.append(
            {"unit_id": str(fold), "participant_id": f"p{fold}", "study_group": "lode_lucca",
             "validation_fold": fold, "experience": int(fold == 0), "sleep_stage": 2,
             "conventional": {key: 0.1 for key in ["delta", "theta", "alpha", "beta", "spectral_exponent", "lz", "permutation_entropy", "log_rms"]}}
        )
    frame = pd.DataFrame(rows)

    def fake_transfer(data, _arrays, *, families, **_kwargs):
        predictions = []
        for family in families:
            for row in data.itertuples():
                predictions.append({"unit_id": row.unit_id, "participant_id": row.participant_id,
                                    "study_group": row.study_group, "experience": row.experience,
                                    "model": family, "probability": 0.5, "held_group": row.validation_fold})
        return {"predictions": predictions, "unavailable": [], "tuning": []}

    monkeypatch.setattr(lode, "nested_transfer", fake_transfer)
    result = lode._scored(frame, {}, policy, all_models=False)
    constants = {row["unit_id"]: row["probability"] for row in result["predictions"] if row["model"] == "training_constant"}
    assert constants["0"] == 1e-6
    assert all(np.isclose(constants[str(i)], 0.25) for i in range(1, 5))


def test_holm_two_is_monotone():
    assert lode._holm_two([0.04, 0.01]) == [0.04, 0.02]
