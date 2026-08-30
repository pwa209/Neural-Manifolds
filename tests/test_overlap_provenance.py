from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from neural_manifolds.foundation.overlap import (
    classify_pretraining_overlap,
    ensure_pretraining_overlap_columns,
    summarize_pretraining_overlap,
)


def test_missing_overlap_evidence_is_not_upgraded_to_verified_zero_shot(
    tmp_path: Path,
) -> None:
    models = tmp_path / "models.yaml"
    models.write_text(
        """schema_version: 1
models:
  frozen:
    trainable: false
    pretraining_overlap_audit:
      status: verified_no_overlap
      target_dataset_ids: [target]
      evidence: null
""",
        encoding="utf-8",
    )
    frame = ensure_pretraining_overlap_columns(
        pd.DataFrame({"dataset_id": ["target"], "representation_model_id": ["frozen"]}),
        models_path=models,
    )

    assert frame.loc[0, "pretraining_overlap_configured_status"] == "verified_no_overlap"
    assert frame.loc[0, "pretraining_overlap_status"] == "unresolved"
    assert frame.loc[0, "zero_shot_classification"] == ("unresolved_not_verified_zero_shot")
    assert not bool(frame.loc[0, "zero_shot_verified"])
    assert frame.loc[0, "pretraining_overlap_control"] == ("frozen_weights_no_study_finetuning")


def test_overlap_classification_distinguishes_verified_confirmed_and_unresolved() -> None:
    verified = classify_pretraining_overlap(
        model_id="model",
        dataset_id="verified",
        audit={
            "status": "verified_no_overlap",
            "target_dataset_ids": ["verified"],
            "evidence": {"source": "audited model card", "revision": "abc"},
        },
        trainable=False,
        source="test",
    )
    confirmed = classify_pretraining_overlap(
        model_id="model",
        dataset_id="overlap",
        audit={
            "status": "confirmed_overlap",
            "target_dataset_ids": ["overlap"],
            "evidence": {"source": "pretraining inventory"},
        },
        trainable=False,
        source="test",
    )
    unresolved = classify_pretraining_overlap(
        model_id="model",
        dataset_id="not-covered",
        audit={
            "status": "verified_no_overlap",
            "target_dataset_ids": ["different"],
            "evidence": {"source": "audited model card"},
        },
        trainable=False,
        source="test",
    )

    assert verified["zero_shot_verified"] is True
    assert verified["zero_shot_classification"] == ("verified_zero_shot_no_pretraining_overlap")
    assert json.loads(verified["pretraining_overlap_evidence_json"])["revision"] == "abc"
    assert confirmed["pretraining_overlap_status"] == "confirmed_overlap"
    assert confirmed["zero_shot_classification"] == "confirmed_overlap_not_zero_shot"
    assert confirmed["zero_shot_verified"] is False
    assert unresolved["pretraining_overlap_status"] == "unresolved"
    assert unresolved["zero_shot_verified"] is False


def test_artifact_summary_is_conservative_across_datasets() -> None:
    frame = pd.DataFrame(
        [
            classify_pretraining_overlap(
                model_id="model",
                dataset_id="verified",
                audit={
                    "status": "verified_no_overlap",
                    "target_dataset_ids": ["verified"],
                    "evidence": {"source": "audit"},
                },
                trainable=False,
                source="test",
            ),
            classify_pretraining_overlap(
                model_id="model",
                dataset_id="unknown",
                audit=None,
                trainable=False,
                source="test",
            ),
        ]
    )
    frame["dataset_id"] = ["verified", "unknown"]

    summary = summarize_pretraining_overlap(frame)

    assert summary["pretraining_overlap_status"] == "unresolved"
    assert summary["zero_shot_classification"] == "unresolved_not_verified_zero_shot"
    assert summary["zero_shot_verified"] is False
    assert len(summary["dataset_classifications"]) == 2
    assert {row["dataset_id"] for row in summary["dataset_classifications"]} == {
        "verified",
        "unknown",
    }
