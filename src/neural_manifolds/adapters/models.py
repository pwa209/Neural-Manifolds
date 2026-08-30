"""Strict, label-separated analysis units produced by dataset adapters.

Adapters create participant-condition units before preprocessing.  The signal
encoder receives :class:`EncoderInput` only; contrast and outcome labels remain
in :class:`AnalysisUnit` and are joined to embeddings after encoding by
``unit_id``.
"""

from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class AdapterError(ValueError):
    """Base class for deterministic adapter failures."""


class SchemaError(AdapterError):
    """Raised when native metadata does not match an audited schema."""


class UnresolvedMetadataError(AdapterError):
    """Raised when an upstream release does not yet document a needed label."""


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


SelectorKind = Literal[
    "full_recording",
    "interval_seconds",
    "event_epoch",
    "pre_epoched",
    "volume_interval",
]


class SignalSelector(StrictModel):
    """A label-free selector applied before preprocessing/windowing."""

    kind: SelectorKind
    start_seconds: float | None = Field(default=None, ge=0)
    stop_seconds: float | None = Field(default=None, gt=0)
    event_onset_seconds: float | None = Field(default=None, ge=0)
    event_sample: int | None = Field(default=None, ge=0)
    epoch_start_offset_seconds: float | None = None
    epoch_stop_offset_seconds: float | None = None
    trial_index: int | None = Field(default=None, ge=0)
    volume_start: int | None = Field(default=None, ge=0)
    volume_stop: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_selector(self) -> SignalSelector:
        if self.kind == "full_recording":
            populated = [
                self.start_seconds,
                self.stop_seconds,
                self.event_onset_seconds,
                self.event_sample,
                self.epoch_start_offset_seconds,
                self.epoch_stop_offset_seconds,
                self.trial_index,
                self.volume_start,
                self.volume_stop,
            ]
            if any(value is not None for value in populated):
                raise ValueError("full_recording cannot carry interval/event selectors")
        elif self.kind == "interval_seconds":
            if self.start_seconds is None or self.stop_seconds is None:
                raise ValueError("interval_seconds requires start_seconds and stop_seconds")
            if self.stop_seconds <= self.start_seconds:
                raise ValueError("stop_seconds must exceed start_seconds")
        elif self.kind == "event_epoch":
            if self.event_onset_seconds is None and self.event_sample is None:
                raise ValueError("event_epoch requires an onset or sample")
            if self.epoch_start_offset_seconds is None or self.epoch_stop_offset_seconds is None:
                raise ValueError("event_epoch requires both epoch offsets")
            if self.epoch_stop_offset_seconds <= self.epoch_start_offset_seconds:
                raise ValueError("epoch stop offset must exceed start offset")
        elif self.kind == "pre_epoched":
            if self.trial_index is None:
                raise ValueError("pre_epoched requires trial_index")
        elif self.kind == "volume_interval":
            if self.volume_start is None or self.volume_stop is None:
                raise ValueError("volume_interval requires volume_start and volume_stop")
            if self.volume_stop <= self.volume_start:
                raise ValueError("volume_stop must exceed volume_start")
        return self


ExplanatoryTarget = Literal[
    "conscious_level",
    "experienced_content",
    "report_task_relevance",
    "psychedelic_organisation",
    "clinical_status",
]
MetadataStatus = Literal["verified", "ambiguous", "unresolved"]
TaskRelevance = Literal["relevant", "irrelevant", "not_applicable"]
Scalar = str | int | float | bool | None


class AnalysisUnit(StrictModel):
    """Standardized participant-condition manifest row, including labels."""

    dataset_id: str = Field(pattern=r"^[a-z][a-z0-9_]+$")
    unit_id: str = Field(pattern=r"^[a-f0-9]{24}$")
    participant_id: str = Field(min_length=1)
    session_id: str | None = None
    run_id: str | None = None
    source_file: str = Field(min_length=1)
    modality: Literal["eeg", "meg", "tms-eeg", "psg", "fmri"]
    selector: SignalSelector
    condition: str = Field(min_length=1)
    explanatory_target: ExplanatoryTarget
    healthy_wake_reference: bool | None
    clinical_holdout: bool
    report_produced: bool | None = None
    task_relevance: TaskRelevance | None = None
    content: str | None = None
    metadata_status: MetadataStatus = "verified"
    variables: dict[str, Scalar] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_paths_and_status(self) -> AnalysisUnit:
        normalized = self.source_file.replace("\\", "/")
        if normalized.startswith("/") or ".." in normalized.split("/"):
            raise ValueError("source_file must be a safe path relative to the dataset root")
        if self.metadata_status == "unresolved" and self.healthy_wake_reference is not None:
            # Unresolved content labels are allowed with a verified wake flag.  An
            # unresolved clinical group, however, must use None; adapters enforce
            # that dataset-specific distinction.
            pass
        return self


class EncoderInput(StrictModel):
    """The only manifest view that may be supplied to a foundation encoder."""

    unit_id: str
    source_file: str
    modality: Literal["eeg", "meg", "tms-eeg", "psg", "fmri"]
    selector: SignalSelector


LABEL_FIELDS = frozenset(
    {
        "participant_id",
        "session_id",
        "run_id",
        "condition",
        "explanatory_target",
        "healthy_wake_reference",
        "clinical_holdout",
        "report_produced",
        "task_relevance",
        "content",
        "metadata_status",
        "variables",
    }
)


def encoding_view(unit: AnalysisUnit) -> EncoderInput:
    """Project an analysis unit onto the audited, label-free encoder contract."""

    return EncoderInput(
        unit_id=unit.unit_id,
        source_file=unit.source_file,
        modality=unit.modality,
        selector=unit.selector,
    )


def assert_label_free_encoder_payload(payload: dict[str, object]) -> None:
    leaked = sorted(LABEL_FIELDS.intersection(payload))
    if leaked:
        raise SchemaError(f"label fields are forbidden in encoder payload: {leaked}")
    allowed = set(EncoderInput.model_fields)
    unknown = sorted(set(payload).difference(allowed))
    if unknown:
        raise SchemaError(f"unknown encoder fields: {unknown}")


def make_unit_id(dataset_id: str, *identity: object) -> str:
    serialized = json.dumps([dataset_id, *identity], ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:24]
