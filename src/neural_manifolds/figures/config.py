"""Validated figure contracts loaded before any plotting library is used."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class FigureContract:
    """Scientific and export contract for one numbered manuscript figure."""

    figure_id: str
    title: str
    core_conclusion: str
    archetype: str
    final_size_mm: tuple[float, float]
    panel_map: dict[str, str]
    evidence_hierarchy: dict[str, str]
    statistics: str
    source_data: str
    image_integrity: str
    reviewer_risks: tuple[str, ...]
    display: dict[str, Any]


@dataclass(frozen=True)
class FigureConfig:
    """Complete, immutable rendering configuration."""

    source_path: Path
    schema_version: int
    backend: str
    target_journal: str
    export: dict[str, Any]
    integrity: dict[str, Any]
    palette: dict[str, str]
    axis_order: tuple[str, ...]
    axis_labels: dict[str, str]
    figures: dict[str, FigureContract]


def _candidate_paths() -> list[Path]:
    paths: list[Path] = []
    override = os.environ.get("NEURAL_MANIFOLDS_FIGURE_CONFIG")
    if override:
        paths.append(Path(override))
    paths.append(Path.cwd() / "configs" / "figures.yaml")
    paths.append(Path(__file__).resolve().parents[3] / "configs" / "figures.yaml")
    return paths


def _require_mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _parse_contract(figure_id: str, raw_value: object) -> FigureContract:
    raw = _require_mapping(raw_value, figure_id)
    required = {
        "title",
        "core_conclusion",
        "archetype",
        "final_size_mm",
        "panel_map",
        "evidence_hierarchy",
        "statistics",
        "source_data",
        "image_integrity",
        "reviewer_risks",
    }
    missing = sorted(required - raw.keys())
    if missing:
        raise ValueError(f"{figure_id} contract is missing: {', '.join(missing)}")
    size = raw["final_size_mm"]
    if not isinstance(size, list | tuple) or len(size) != 2:
        raise ValueError(f"{figure_id}.final_size_mm must contain width and height")
    width, height = (float(size[0]), float(size[1]))
    if width <= 0 or height <= 0:
        raise ValueError(f"{figure_id}.final_size_mm values must be positive")
    panel_map = dict(_require_mapping(raw["panel_map"], f"{figure_id}.panel_map"))
    if not panel_map:
        raise ValueError(f"{figure_id}.panel_map cannot be empty")
    risks = raw["reviewer_risks"]
    if not isinstance(risks, list) or not risks:
        raise ValueError(f"{figure_id}.reviewer_risks must be a nonempty list")
    display = dict(_require_mapping(raw.get("display", {}), f"{figure_id}.display"))
    for key, value in display.items():
        if key.startswith("max_main_") and (
            not isinstance(value, int) or isinstance(value, bool) or value < 1
        ):
            raise ValueError(f"{figure_id}.display.{key} must be a positive integer")
        if key.endswith("_priority") and not isinstance(value, list | tuple):
            raise ValueError(f"{figure_id}.display.{key} must be a sequence")
    return FigureContract(
        figure_id=figure_id,
        title=str(raw["title"]),
        core_conclusion=str(raw["core_conclusion"]),
        archetype=str(raw["archetype"]),
        final_size_mm=(width, height),
        panel_map={str(key): str(value) for key, value in panel_map.items()},
        evidence_hierarchy={
            str(key): str(value)
            for key, value in _require_mapping(
                raw["evidence_hierarchy"], f"{figure_id}.evidence_hierarchy"
            ).items()
        },
        statistics=str(raw["statistics"]),
        source_data=str(raw["source_data"]),
        image_integrity=str(raw["image_integrity"]),
        reviewer_risks=tuple(str(value) for value in risks),
        display={str(key): value for key, value in display.items()},
    )


def load_figure_config(path: str | Path | None = None) -> FigureConfig:
    """Load and validate all six figure contracts.

    The environment override is useful on the server, while the repository path is
    the normal source-controlled configuration. Loading happens before rendering so
    a malformed contract cannot yield a partially populated export directory.
    """

    if path is None:
        source = next((candidate for candidate in _candidate_paths() if candidate.is_file()), None)
        if source is None:
            searched = ", ".join(str(candidate) for candidate in _candidate_paths())
            raise FileNotFoundError(f"figures.yaml was not found; searched {searched}")
    else:
        source = Path(path)
    with source.open("r", encoding="utf-8") as stream:
        loaded = yaml.safe_load(stream)
    raw = _require_mapping(loaded, "figure configuration")
    if raw.get("backend") != "python-matplotlib":
        raise ValueError("figure backend must remain python-matplotlib")
    figure_values = _require_mapping(raw.get("figures"), "figures")
    expected = {f"figure_{number}" for number in range(1, 7)}
    missing = sorted(expected - figure_values.keys())
    if missing:
        raise ValueError(f"figure contracts are missing: {', '.join(missing)}")
    figures = {
        figure_id: _parse_contract(figure_id, figure_values[figure_id])
        for figure_id in sorted(expected)
    }
    export = dict(_require_mapping(raw.get("export"), "export"))
    integrity = dict(_require_mapping(raw.get("integrity"), "integrity"))
    if int(export.get("tiff_dpi", 0)) != 600:
        raise ValueError("TIFF exports must remain fixed at 600 dpi")
    if int(export.get("pdf_fonttype", 0)) != 42:
        raise ValueError("PDF exports must use TrueType fonttype 42")
    if not bool(export.get("svg_editable_text")):
        raise ValueError("SVG text must remain editable")
    if not bool(integrity.get("production_mock_data_forbidden")):
        raise ValueError("production mock-data fallback must remain forbidden")
    axis_order = tuple(str(value) for value in raw.get("axis_order", ()))
    if axis_order != ("R", "M", "D", "A", "P"):
        raise ValueError("axis_order must be exactly R, M, D, A, P")
    return FigureConfig(
        source_path=source.resolve(),
        schema_version=int(raw.get("schema_version", 0)),
        backend=str(raw["backend"]),
        target_journal=str(raw.get("target_journal", "")),
        export=export,
        integrity=integrity,
        palette={
            str(key): str(value)
            for key, value in _require_mapping(raw.get("palette"), "palette").items()
        },
        axis_order=axis_order,
        axis_labels={
            str(key): str(value)
            for key, value in _require_mapping(raw.get("axis_labels"), "axis_labels").items()
        },
        figures=figures,
    )
