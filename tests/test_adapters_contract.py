"""Contract tests for pre-preprocessing analysis units and label isolation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from neural_manifolds.adapters import (
    AnalysisUnit,
    EncoderInput,
    SchemaError,
    SignalSelector,
    assert_label_free_encoder_payload,
    encoding_view,
    get_adapter,
)


def _unit() -> AnalysisUnit:
    return AnalysisUnit(
        dataset_id="synthetic_dataset",
        unit_id="0123456789abcdef01234567",
        participant_id="sub-01",
        source_file="sub-01/eeg/sub-01_task-test_eeg.vhdr",
        modality="eeg",
        selector=SignalSelector(
            kind="event_epoch",
            event_onset_seconds=12.5,
            epoch_start_offset_seconds=-0.4,
            epoch_stop_offset_seconds=0.8,
        ),
        condition="detected",
        explanatory_target="experienced_content",
        healthy_wake_reference=True,
        clinical_holdout=False,
        report_produced=True,
        task_relevance="relevant",
        content="felt_touch",
        variables={"confidence": 0.9},
    )


def test_encoding_view_contains_selector_but_no_labels() -> None:
    encoded = encoding_view(_unit()).model_dump()
    assert set(encoded) == {"unit_id", "source_file", "modality", "selector"}
    assert encoded["selector"]["event_onset_seconds"] == 12.5
    assert_label_free_encoder_payload(encoded)


def test_encoder_contract_rejects_labels_and_unknown_fields() -> None:
    with pytest.raises(SchemaError, match="label fields are forbidden"):
        assert_label_free_encoder_payload({"unit_id": "x", "condition": "detected"})
    with pytest.raises(SchemaError, match="unknown encoder fields"):
        assert_label_free_encoder_payload({"unit_id": "x", "batch_label": "case"})
    with pytest.raises(ValidationError):
        EncoderInput(
            unit_id="x",
            source_file="x.vhdr",
            modality="eeg",
            selector=SignalSelector(kind="full_recording"),
            condition="detected",  # type: ignore[call-arg]
        )


@pytest.mark.parametrize(
    "selector",
    [
        {"kind": "full_recording", "start_seconds": 0},
        {"kind": "interval_seconds", "start_seconds": 5, "stop_seconds": 4},
        {"kind": "event_epoch", "event_onset_seconds": 1},
        {"kind": "pre_epoched"},
        {"kind": "volume_interval", "volume_start": 3, "volume_stop": 3},
    ],
)
def test_selector_invariants_fail_closed(selector: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        SignalSelector(**selector)  # type: ignore[arg-type]


def test_unknown_dataset_has_no_fallback_adapter() -> None:
    with pytest.raises(SchemaError, match="no audited adapter"):
        get_adapter("unknown_dataset")
