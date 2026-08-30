"""Strict configuration loading and stable hashing."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RepresentationConfig(StrictModel):
    primary: str
    weights_frozen: bool
    label_free: bool
    layer: str
    pooling: str
    secondary_pooling: str
    dynamics_rank: int = Field(gt=0)
    harmonised_window_seconds: float = Field(gt=0)
    harmonised_step_seconds: float = Field(gt=0)
    alignment_window_seconds: float = Field(gt=0)
    alignment_step_seconds: float = Field(gt=0)
    native_window_seconds: float = Field(gt=0)
    labram_patch_seconds: float = Field(gt=0)

    @model_validator(mode="after")
    def validate_temporal_tracks(self) -> RepresentationConfig:
        if self.alignment_window_seconds % self.labram_patch_seconds > 1e-9:
            raise ValueError("alignment_window_seconds must contain whole LaBraM patches")
        if self.alignment_step_seconds > self.alignment_window_seconds:
            raise ValueError("alignment_step_seconds cannot exceed its window")
        return self


class PreprocessingConfig(StrictModel):
    target_sampling_hz: int = Field(gt=0)
    highpass_hz: float = Field(ge=0)
    lowpass_hz: float = Field(gt=0)
    sleep_sensitivity_highpass_hz: float = Field(ge=0)
    mains_notch_hz: int | str
    maximum_interpolation_fraction: float = Field(ge=0, le=1)
    minimum_rest_seconds: int = Field(gt=0)
    minimum_event_trials_per_condition: int = Field(gt=0)
    minimum_valid_windows: int = Field(gt=0)
    canonical_channels: list[str]
    minimum_canonical_channels: int = Field(gt=0)
    legacy_channel_map: dict[str, str]

    @model_validator(mode="after")
    def validate_filter_and_channels(self) -> PreprocessingConfig:
        if self.highpass_hz >= self.lowpass_hz:
            raise ValueError("highpass_hz must be lower than lowpass_hz")
        if self.minimum_canonical_channels > len(self.canonical_channels):
            raise ValueError("minimum_canonical_channels exceeds canonical channel count")
        if len(set(self.canonical_channels)) != len(self.canonical_channels):
            raise ValueError("canonical_channels contains duplicates")
        return self


class SamplingConfig(StrictModel):
    equalise_windows: bool
    repeats: int = Field(gt=0)
    reliability_seconds: list[int]


class StatisticsConfig(StrictModel):
    inference_unit: str
    participant_stratified_folds: int = Field(ge=2)
    participant_bootstrap_repetitions: int = Field(gt=0)
    permutation_repetitions: int = Field(gt=0)
    false_discovery_rate: float = Field(gt=0, lt=1)
    continuous_smallest_effect: float = Field(gt=0)
    prediction_smallest_auc_difference: float = Field(gt=0)
    exact_permutation_plus_one: bool

    @model_validator(mode="after")
    def participant_only(self) -> StatisticsConfig:
        if self.inference_unit != "participant":
            raise ValueError("the inferential unit must be participant")
        return self


class ClinicalTransferConfig(StrictModel):
    retrain_representation: bool
    retrain_scaler: bool
    retrain_state_dictionary: bool
    individual_diagnostic_reclassification: bool

    @model_validator(mode="after")
    def locked(self) -> ClinicalTransferConfig:
        forbidden = (
            self.retrain_representation,
            self.retrain_scaler,
            self.retrain_state_dictionary,
            self.individual_diagnostic_reclassification,
        )
        if any(forbidden):
            raise ValueError("clinical transfer must remain locked and non-diagnostic")
        return self


class StudyConfig(StrictModel):
    schema_version: int = Field(ge=1)
    study_id: str
    status: str
    scientific_gates: bool
    random_seeds: list[int] = Field(min_length=1)
    representation: RepresentationConfig
    preprocessing: PreprocessingConfig
    metrics: dict[str, Any]
    sampling: SamplingConfig
    statistics: StatisticsConfig
    clinical_transfer: ClinicalTransferConfig

    @model_validator(mode="after")
    def study_invariants(self) -> StudyConfig:
        if self.status != "exploratory_non_preregistered":
            raise ValueError("project status must remain exploratory_non_preregistered")
        if self.scientific_gates:
            raise ValueError("scientific_gates must be false for this project")
        if len(set(self.random_seeds)) != len(self.random_seeds):
            raise ValueError("random seeds must be unique")
        if not self.representation.weights_frozen or not self.representation.label_free:
            raise ValueError("primary representation must be frozen and label-free")
        return self


def load_yaml(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    with source.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise ValueError(f"configuration root must be a mapping: {source}")
    return data


def load_study(path: str | Path) -> StudyConfig:
    return StudyConfig.model_validate(load_yaml(path))


def canonical_json(value: BaseModel | dict[str, Any]) -> str:
    payload = value.model_dump(mode="json") if isinstance(value, BaseModel) else value
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def config_sha256(value: BaseModel | dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()
