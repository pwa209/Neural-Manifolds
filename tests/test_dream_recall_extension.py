from __future__ import annotations

import csv
import io
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import yaml

from neural_manifolds.validation import dream_recall


def _edf_header(channels: list[str]) -> bytes:
    first = bytearray(b" " * 256)
    first[252:256] = f"{len(channels):4d}".encode()
    return bytes(first) + b"".join(ch.encode().ljust(16, b" ") for ch in channels) + b" " * 100


def test_metadata_selects_one_awakening_per_person_before_signal(tmp_path: Path):
    archive = tmp_path / "dream.zip"
    fields = [
        "Filename",
        "Case ID",
        "Subject ID",
        "Experience",
        "Last sleep stage",
        "Duration",
        "Remarks",
    ]
    records = [
        ["B.edf", "B", "person1", "0", "2", "100", ""],
        ["A.edf", "A", "person1", "2", "2", "100", ""],
        ["C.edf", "C", "person2", "0", "2", "100", ""],
        ["D.edf", "D", "person3", "2", "5", "100", ""],
        ["E.edf", "E", "person4", "2", "2", "100", "Bad channels: C3"],
    ]
    stream = io.StringIO()
    writer = csv.writer(stream)
    writer.writerow(fields)
    writer.writerows(records)
    with ZipFile(archive, "w") as z:
        z.writestr("Test/Records.csv", stream.getvalue())
        for name in "ABCDE":
            z.writestr(f"Test/Data/PSG/{name}.edf", _edf_header(["F3", "C3", "O1"]))
    policy = yaml.safe_load(Path("configs/dream_recall_extension.yaml").read_text(encoding="utf-8"))
    source = {
        "selection": "n2_one_record_per_person",
        "laboratory": "test",
        "timing_offset_seconds": 0,
        "timing_caveat": "test",
        "report_construct": "recall",
        "channel_aliases": {},
    }
    rows, exclusions, digest = dream_recall._candidate_rows("test", source, archive, policy)
    assert len(rows) == 2
    assert {row["member"].split("/")[-1] for row in rows} == {"A.edf", "C.edf"}
    assert exclusions == {
        "not_N2": 1,
        "target_channel_flagged_in_source_remarks": 1,
        "additional_awakening_same_person": 1,
    }
    assert len(digest) == 64


def test_training_constant_excludes_held_dataset_labels(monkeypatch):
    policy = yaml.safe_load(Path("configs/dream_recall_extension.yaml").read_text(encoding="utf-8"))
    frame = pd.DataFrame(
        [
            {
                "unit_id": str(i),
                "participant_id": f"p{i}",
                "study_group": f"study{i}",
                "experience": int(i == 0),
            }
            for i in range(3)
        ]
    )

    def fake_transfer(data, _arrays, *, families, **_kwargs):
        return {
            "predictions": [
                {
                    "unit_id": r.unit_id,
                    "participant_id": r.participant_id,
                    "study_group": r.study_group,
                    "experience": r.experience,
                    "model": family,
                    "probability": 0.5,
                    "held_group": r.study_group,
                }
                for family in families
                for r in data.itertuples()
            ],
            "unavailable": [],
            "tuning": [],
        }

    monkeypatch.setattr(dream_recall, "nested_transfer", fake_transfer)
    scored = dream_recall._score(frame, {}, policy, all_models=False)
    constants = {
        r["unit_id"]: r["probability"]
        for r in scored["predictions"]
        if r["model"] == "training_constant"
    }
    assert constants["0"] == 1e-6
    assert np.isclose(constants["1"], 0.5)
    assert np.isclose(constants["2"], 0.5)
