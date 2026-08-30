"""Composition root for the pinned ds006623/BrainLM secondary stage."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from neural_manifolds.config import StudyConfig
from neural_manifolds.foundation.brainlm import OfficialBrainLMEncoder
from neural_manifolds.stages.fmri import run_fmri_triangulation
from neural_manifolds.stages.fmri_manifest import prepare_ds006623_fmri_manifest


def _brainlm_spec(path: str | Path) -> Mapping[str, Any]:
    source = Path(path).resolve(strict=True)
    with source.open("r", encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    if not isinstance(document, dict) or document.get("schema_version") != 1:
        raise ValueError("model configuration must use schema_version 1")
    models = document.get("models")
    if not isinstance(models, dict) or not isinstance(models.get("brainlm"), dict):
        raise ValueError("model configuration has no BrainLM specification")
    specification = models["brainlm"]
    if specification.get("trainable") is not False:
        raise ValueError("BrainLM must remain frozen")
    return specification


def run_ds006623_brainlm_stage(
    *,
    release_root: str | Path,
    models_path: str | Path,
    output_root: str | Path,
    study: StudyConfig,
    timing_index_origin: int | None = None,
    atlas_path: str | Path | None = None,
    coordinates_path: str | Path | None = None,
    device: str = "cuda",
    batch_size: int = 4,
) -> tuple[Path, ...]:
    """Prepare the strict manifest and execute frozen BrainLM triangulation."""

    destination = Path(output_root)
    prepared = prepare_ds006623_fmri_manifest(
        release_root=release_root,
        output_root=destination / "manifest",
        timing_index_origin=timing_index_origin,
        atlas_path=atlas_path,
        coordinates_path=coordinates_path,
    )
    encoder = OfficialBrainLMEncoder.from_environment(
        _brainlm_spec(models_path),
        device=device,
        batch_size=batch_size,
    )
    analysis = run_fmri_triangulation(
        manifest_path=prepared.manifest_path,
        coordinates_path=prepared.coordinates_path,
        atlas_path=prepared.atlas_path,
        output_root=destination / "analysis",
        encoder=encoder,
        study=study,
    )
    return (prepared.manifest_path, prepared.audit_path, *analysis)


__all__ = ["run_ds006623_brainlm_stage"]
