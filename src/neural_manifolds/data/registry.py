"""Load and query the version-pinned dataset registry."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .models import DatasetRegistryModel, DatasetSpec, assert_no_embedded_secrets


@dataclass(frozen=True)
class DatasetRegistry:
    model: DatasetRegistryModel
    source_path: Path
    sha256: str

    @property
    def datasets(self) -> tuple[DatasetSpec, ...]:
        return tuple(self.model.datasets)

    def get(self, dataset_id: str) -> DatasetSpec:
        for dataset in self.model.datasets:
            if dataset.id == dataset_id:
                return dataset
        choices = ", ".join(sorted(d.id for d in self.model.datasets))
        raise KeyError(f"unknown dataset {dataset_id!r}; choose one of: {choices}")

    def select(self, dataset_ids: Iterable[str] | None = None) -> tuple[DatasetSpec, ...]:
        if dataset_ids is None:
            return self.datasets
        requested = list(dataset_ids)
        if not requested:
            return self.datasets
        return tuple(self.get(identifier) for identifier in requested)


def _canonical_json(value: dict[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "utf-8"
    )


def load_dataset_registry(path: str | Path) -> DatasetRegistry:
    source = Path(path).resolve()
    with source.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, dict):
        raise ValueError(f"dataset registry root must be a mapping: {source}")
    assert_no_embedded_secrets(raw)
    model = DatasetRegistryModel.model_validate(raw)
    return DatasetRegistry(
        model=model,
        source_path=source,
        sha256=hashlib.sha256(_canonical_json(raw)).hexdigest(),
    )
