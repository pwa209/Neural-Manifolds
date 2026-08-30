"""Shared contracts for frozen encoders."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

FORBIDDEN_LABEL_FIELDS = frozenset(
    {
        "conscious",
        "consciousness",
        "diagnosis",
        "crs_r",
        "crs-r",
        "drug",
        "condition",
        "report",
        "task",
        "target",
        "response",
        "outcome",
        "group",
        "label",
    }
)


def assert_label_firewall(fields: Iterable[str]) -> None:
    normalised = {str(field).strip().lower() for field in fields}
    forbidden = sorted(normalised.intersection(FORBIDDEN_LABEL_FIELDS))
    if forbidden:
        raise ValueError(f"encoder input metadata includes prohibited label fields: {forbidden}")


def validate_frozen_model(model: Any) -> None:
    parameters = list(model.parameters())
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if trainable:
        raise ValueError(f"foundation model has trainable parameters: {trainable[:5]}")
    if getattr(model, "training", False):
        raise ValueError("foundation model must be in evaluation mode")
    if not parameters:
        raise ValueError("foundation model exposes no parameters")


@dataclass(frozen=True)
class EncodedBatch:
    global_states: np.ndarray
    regional_states: Mapping[str, np.ndarray]
    patch_tokens: np.ndarray
    channel_names: tuple[str, ...]
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.global_states.ndim != 2:
            raise ValueError("global_states must be windows x features")
        if self.patch_tokens.ndim != 4:
            raise ValueError("patch_tokens must be windows x channels x patches x features")
        if self.patch_tokens.shape[0] != self.global_states.shape[0]:
            raise ValueError("window count differs between global states and tokens")
        if self.patch_tokens.shape[1] != len(self.channel_names):
            raise ValueError("channel count differs between names and tokens")
