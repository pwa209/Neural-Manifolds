"""Transactional manuscript-figure stage with source-data and hash manifests."""

from __future__ import annotations

import json
import os
import platform
import shutil
import tempfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from neural_manifolds.manifold.clinical_reference import WAKE_REGIME_LLR
from neural_manifolds.provenance import atomic_write_json, git_revision, sha256_file

from .config import FigureConfig, load_figure_config
from .io import (
    FigureInputError,
    SourceBundle,
    atomic_write_csv,
    find_table,
    load_source_bundle,
    public_source_table,
    reject_p_value_stars,
    validate_identifiers,
    validate_numeric,
)
from .renderers import (
    RenderedFigure,
    render_figure_1,
    render_figure_2,
    render_figure_3,
    render_figure_4,
    render_figure_5,
    render_figure_6,
)
from .style import ExportedFigure, export_figure

PROFILE_COLUMNS = {
    "participant_id",
    "dataset_id",
    "condition",
    "R",
    "M",
    "D",
    "A",
    "P",
    "source_artifact_sha256",
}
CONTENT_COLUMNS = {
    "participant_id",
    "dataset_id",
    "contrast",
    "axis",
    "value",
    "positive_conditions",
    "negative_conditions",
    "n_positive_units",
    "n_negative_units",
    "matched_strata",
    "source_artifact_sha256",
}
ROBUSTNESS_COLUMNS = {
    "participant_id",
    "dataset_id",
    "contrast",
    "analysis",
    "family",
    "repeat",
    "seed",
    "metric",
    "value",
    "observed_effect",
    "null_effect",
    "observed_minus_null",
    "signed_effect_survival",
    "positive_conditions",
    "negative_conditions",
    "n_positive_units",
    "n_negative_units",
    "matched_strata",
    "source_artifact_sha256",
}
TMS_PARTICIPANT_COLUMNS = {
    "participant_id",
    "dataset_id",
    "condition",
    "passive_reachability",
    "direct_response",
    "passive_delta",
    "direct_delta",
    "tms_contrast",
    "source_artifact_sha256",
}
TMS_TRAJECTORY_COLUMNS = {
    "participant_id",
    "dataset_id",
    "condition",
    "time_ms",
    "trajectory_value",
    "source_artifact_sha256",
}
CLINICAL_COLUMNS = {
    "participant_id",
    "dataset_id",
    "diagnosis",
    "crs_r_total",
    WAKE_REGIME_LLR,
    "wake_regime_score_status",
    "crs_r_status",
    "R",
    "M",
    "D",
    "A",
    "P",
    "source_artifact_sha256",
}
FMRI_COLUMNS = {
    "participant_id",
    "dataset_id",
    "condition",
    "R",
    "M",
    "D",
    "A",
    "source_artifact_sha256",
}


@dataclass(frozen=True)
class FigureRunResult:
    """Portable references to a completed transactional figure bundle."""

    output_root: Path
    manifest_path: Path
    figure_paths: dict[str, dict[str, Path]]
    source_data_paths: dict[str, dict[str, Path]]
    skipped: tuple[str, ...]


@dataclass(frozen=True)
class _Inputs:
    profiles_bundle: SourceBundle
    models_bundle: SourceBundle
    tms_bundle: SourceBundle
    clinical_bundle: SourceBundle | None
    fmri_bundle: SourceBundle | None
    profiles: pd.DataFrame
    content: pd.DataFrame
    robustness: pd.DataFrame
    tms_participants: pd.DataFrame
    tms_trajectory: pd.DataFrame
    clinical: pd.DataFrame | None
    fmri: pd.DataFrame | None


def _validate_axis_values(frame: pd.DataFrame, *, label: str, axes: set[str]) -> None:
    values = set(frame["axis"].astype(str))
    unexpected = sorted(values - axes)
    missing = sorted(axes - values)
    if unexpected or missing:
        details = []
        if missing:
            details.append(f"missing {missing}")
        if unexpected:
            details.append(f"unexpected {unexpected}")
        raise FigureInputError(f"{label}.axis must cover R, M, D, A, P: {'; '.join(details)}")


def _validate_optional_numeric(frame: pd.DataFrame, column: str, *, label: str) -> None:
    values = pd.to_numeric(frame[column], errors="coerce")
    supplied = frame[column].notna() & frame[column].astype(str).str.strip().ne("")
    invalid = supplied & (values.isna() | ~np.isfinite(values.fillna(0.0).to_numpy()))
    if invalid.any():
        raise FigureInputError(f"{label}.{column} contains an invalid supplied value")
    frame[column] = values.astype(float)


def _validate_effect_survival(frame: pd.DataFrame) -> None:
    observed = frame["observed_effect"].to_numpy(dtype=float)
    null = frame["null_effect"].to_numpy(dtype=float)
    difference = frame["observed_minus_null"].to_numpy(dtype=float)
    survival = frame["signed_effect_survival"].to_numpy(dtype=float)
    value = frame["value"].to_numpy(dtype=float)
    if not np.allclose(difference, observed - null, rtol=1e-10, atol=1e-12):
        raise FigureInputError(
            "robustness.observed_minus_null is inconsistent with observed_effect-null_effect"
        )
    expected_survival = difference * np.sign(observed)
    if not np.allclose(survival, expected_survival, rtol=1e-10, atol=1e-12):
        raise FigureInputError(
            "robustness.signed_effect_survival is inconsistent with its directional contrast"
        )
    if not np.allclose(value, survival, rtol=1e-10, atol=1e-12):
        raise FigureInputError("robustness.value must equal signed_effect_survival for Figure 5")


def _prepare_inputs(
    *,
    profiles_path: str | Path,
    models_path: str | Path,
    tms_path: str | Path,
    clinical_path: str | Path | None,
    fmri_path: str | Path | None,
) -> _Inputs:
    profiles_bundle = load_source_bundle("profiles", profiles_path)
    models_bundle = load_source_bundle("models", models_path)
    tms_bundle = load_source_bundle("tms", tms_path)
    clinical_bundle = (
        load_source_bundle("clinical", clinical_path) if clinical_path is not None else None
    )
    fmri_bundle = load_source_bundle("fmri", fmri_path) if fmri_path is not None else None

    profiles = find_table(
        profiles_bundle,
        ["profiles", "manifold_profiles", "participant_profiles"],
        required_columns=PROFILE_COLUMNS,
    )
    content = find_table(
        models_bundle,
        ["content_report", "content", "report"],
        required_columns=CONTENT_COLUMNS,
    )
    robustness = find_table(
        models_bundle,
        ["robustness", "ablations", "controls"],
        required_columns=ROBUSTNESS_COLUMNS,
    )
    tms_participants = find_table(
        tms_bundle,
        ["participants", "participant", "direct_response", "tms_participants"],
        required_columns=TMS_PARTICIPANT_COLUMNS,
    )
    tms_trajectory = find_table(
        tms_bundle,
        ["trajectory", "timecourse", "time_course", "tms_trajectory"],
        required_columns=TMS_TRAJECTORY_COLUMNS,
    )
    clinical = (
        find_table(
            clinical_bundle,
            ["clinical", "doc", "clinical_profiles"],
            required_columns=CLINICAL_COLUMNS,
        )
        if clinical_bundle is not None
        else None
    )
    fmri = (
        find_table(
            fmri_bundle,
            ["fmri", "fmri_profiles"],
            required_columns=FMRI_COLUMNS,
        )
        if fmri_bundle is not None
        else None
    )

    frames = {
        "profiles": profiles,
        "content_report": content,
        "robustness": robustness,
        "tms_participants": tms_participants,
        "tms_trajectory": tms_trajectory,
        **({"clinical": clinical} if clinical is not None else {}),
        **({"fmri": fmri} if fmri is not None else {}),
    }
    for label, frame in frames.items():
        reject_p_value_stars(frame, label=label)
        validate_identifiers(frame, ["participant_id", "dataset_id"], label=label)
    validate_identifiers(profiles, ["condition"], label="profiles")
    validate_numeric(profiles, ["R", "M", "D", "A", "P"], label="profiles")
    validate_identifiers(
        content,
        ["contrast", "axis", "positive_conditions", "negative_conditions"],
        label="content_report",
    )
    validate_numeric(
        content,
        ["value", "n_positive_units", "n_negative_units", "matched_strata"],
        label="content_report",
    )
    _validate_axis_values(content, label="content_report", axes={"R", "M", "D", "A", "P"})
    validate_identifiers(
        robustness,
        [
            "contrast",
            "analysis",
            "family",
            "metric",
            "positive_conditions",
            "negative_conditions",
        ],
        label="robustness",
    )
    validate_numeric(
        robustness,
        [
            "repeat",
            "seed",
            "value",
            "observed_effect",
            "null_effect",
            "observed_minus_null",
            "signed_effect_survival",
            "n_positive_units",
            "n_negative_units",
            "matched_strata",
        ],
        label="robustness",
    )
    _validate_effect_survival(robustness)
    validate_identifiers(tms_participants, ["condition"], label="tms_participants")
    validate_numeric(
        tms_participants,
        ["passive_reachability", "direct_response", "passive_delta", "direct_delta"],
        label="tms_participants",
    )
    validate_identifiers(tms_participants, ["tms_contrast"], label="tms_participants")
    validate_identifiers(tms_trajectory, ["condition"], label="tms_trajectory")
    validate_numeric(tms_trajectory, ["time_ms", "trajectory_value"], label="tms_trajectory")
    participant_conditions = set(tms_participants["condition"].astype(str))
    trajectory_conditions = set(tms_trajectory["condition"].astype(str))
    missing_trajectories = sorted(participant_conditions - trajectory_conditions)
    if missing_trajectories:
        raise FigureInputError(
            "tms_trajectory is missing participant-table conditions: "
            + ", ".join(missing_trajectories)
        )
    if clinical is not None:
        validate_numeric(
            clinical,
            [WAKE_REGIME_LLR, "R", "M", "D", "A", "P"],
            label="clinical",
        )
        _validate_optional_numeric(clinical, "crs_r_total", label="clinical")
        if clinical.duplicated(["dataset_id", "participant_id"]).any():
            raise FigureInputError(
                "clinical must contain one locked profile per dataset-participant"
            )
    if fmri is not None:
        validate_identifiers(fmri, ["condition"], label="fmri")
        validate_numeric(fmri, ["R", "M", "D", "A"], label="fmri")
    return _Inputs(
        profiles_bundle=profiles_bundle,
        models_bundle=models_bundle,
        tms_bundle=tms_bundle,
        clinical_bundle=clinical_bundle,
        fmri_bundle=fmri_bundle,
        profiles=profiles,
        content=content,
        robustness=robustness,
        tms_participants=tms_participants,
        tms_trajectory=tms_trajectory,
        clinical=clinical,
        fmri=fmri,
    )


def _bundle_manifest(bundle: SourceBundle | None) -> dict[str, Any] | None:
    if bundle is None:
        return None
    return {
        "role": bundle.role,
        "input_name": bundle.root.name,
        "source_table_sha256": dict(sorted(bundle.source_files.items())),
        "producing_artifact_sha256": sorted(
            {
                str(value)
                for frame in bundle.tables.values()
                for value in frame["source_artifact_sha256"].astype(str)
            }
        ),
    }


def _input_manifest(inputs: _Inputs) -> list[dict[str, Any]]:
    return [
        value
        for value in (
            _bundle_manifest(inputs.profiles_bundle),
            _bundle_manifest(inputs.models_bundle),
            _bundle_manifest(inputs.tms_bundle),
            _bundle_manifest(inputs.clinical_bundle),
            _bundle_manifest(inputs.fmri_bundle),
        )
        if value is not None
    ]


def _existing_artifact(destination: Path, item: dict[str, Any], *, label: str) -> Path:
    relative = item.get("path")
    expected_hash = item.get("sha256")
    expected_size = item.get("size_bytes")
    if not isinstance(relative, str) or not isinstance(expected_hash, str):
        raise ValueError(f"existing {label} manifest entry is incomplete")
    path = (destination / relative).resolve(strict=True)
    try:
        path.relative_to(destination)
    except ValueError as error:
        raise ValueError(f"existing {label} path escapes the output root") from error
    if not path.is_file() or path.stat().st_size != expected_size:
        raise ValueError(f"existing {label} file is missing or changed: {path}")
    if sha256_file(path) != expected_hash:
        raise ValueError(f"existing {label} checksum changed: {path}")
    return path


def _implementation_source_hashes() -> dict[str, str]:
    return {
        path.name: sha256_file(path)
        for path in sorted(Path(__file__).resolve().parent.glob("*.py"))
    }


def _reuse_completed_publication(
    *,
    destination: Path,
    config: FigureConfig,
    input_manifest: list[dict[str, Any]],
    stage: str,
    expected_figures: set[str],
) -> FigureRunResult:
    """Recover a publication committed immediately before a queue/receipt crash."""

    manifest_path = destination / "manifest.json"
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != 1 or payload.get("stage") != stage:
            raise ValueError("existing figure manifest has the wrong schema or stage")
        if payload.get("config", {}).get("sha256") != sha256_file(config.source_path):
            raise ValueError("existing figure output used a different figure configuration")
        if payload.get("inputs") != input_manifest:
            raise ValueError("existing figure output used different source tables")
        if (
            payload.get("implementation", {}).get("source_sha256")
            != _implementation_source_hashes()
        ):
            raise ValueError("existing figure output used different renderer source")
        figure_paths: dict[str, dict[str, Path]] = {}
        source_paths: dict[str, dict[str, Path]] = {}
        skipped: list[str] = []
        entries = payload.get("figures")
        if not isinstance(entries, list):
            raise ValueError("existing figure manifest has no figure list")
        for entry in entries:
            if not isinstance(entry, dict) or not isinstance(entry.get("figure_id"), str):
                raise ValueError("existing figure manifest contains an invalid entry")
            figure_id = entry["figure_id"]
            if figure_id in figure_paths or figure_id in skipped:
                raise ValueError(f"existing figure manifest repeats {figure_id}")
            if entry.get("status") == "skipped_optional_input_absent":
                skipped.append(figure_id)
                continue
            if entry.get("status") != "rendered":
                raise ValueError(f"existing {figure_id} has invalid status")
            outputs = entry.get("outputs")
            sources = entry.get("source_data")
            if not isinstance(outputs, dict) or not isinstance(sources, dict):
                raise ValueError(f"existing {figure_id} artifact maps are invalid")
            if set(outputs) != {"svg", "pdf", "tiff"} or not sources:
                raise ValueError(f"existing {figure_id} publication is incomplete")
            figure_paths[figure_id] = {
                extension: _existing_artifact(destination, item, label=f"{figure_id}.{extension}")
                for extension, item in outputs.items()
            }
            source_paths[figure_id] = {
                panel: _existing_artifact(destination, item, label=f"{figure_id}.{panel}")
                for panel, item in sources.items()
            }
        if set(figure_paths).union(skipped) != expected_figures:
            raise ValueError("existing output does not account for every configured figure")
        return FigureRunResult(
            output_root=destination,
            manifest_path=manifest_path,
            figure_paths=figure_paths,
            source_data_paths=source_paths,
            skipped=tuple(sorted(skipped)),
        )
    except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        raise FileExistsError(
            "figure output exists but is not the exact completed publication for the current "
            f"inputs; use a new run-specific output root: {destination} ({error})"
        ) from error


def _reuse_completed_output(
    *, destination: Path, config: FigureConfig, inputs: _Inputs
) -> FigureRunResult:
    return _reuse_completed_publication(
        destination=destination,
        config=config,
        input_manifest=_input_manifest(inputs),
        stage="figures",
        expected_figures=set(config.figures),
    )


def _render_all(inputs: _Inputs, config: FigureConfig) -> dict[str, RenderedFigure | None]:
    contracts = config.figures
    return {
        "figure_1": render_figure_1(inputs.profiles, config, contracts["figure_1"]),
        "figure_2": render_figure_2(inputs.profiles, config, contracts["figure_2"]),
        "figure_3": render_figure_3(inputs.content, config, contracts["figure_3"]),
        "figure_4": render_figure_4(
            inputs.tms_participants,
            inputs.tms_trajectory,
            config,
            contracts["figure_4"],
        ),
        "figure_5": render_figure_5(inputs.robustness, inputs.fmri, config, contracts["figure_5"]),
        "figure_6": (
            render_figure_6(inputs.clinical, config, contracts["figure_6"])
            if inputs.clinical is not None
            else None
        ),
    }


def _export_one(
    *,
    figure_id: str,
    rendered: RenderedFigure,
    staging_root: Path,
    config: FigureConfig,
) -> tuple[ExportedFigure, dict[str, Path], dict[str, str]]:
    contract = config.figures[figure_id]
    exported = export_figure(
        rendered.figure,
        staging_root / "figures" / figure_id,
        size_mm=contract.final_size_mm,
        dpi=int(config.export["tiff_dpi"]),
        tiff_compression=str(config.export["tiff_compression"]),
    )
    source_paths: dict[str, Path] = {}
    source_hashes: dict[str, str] = {}
    for panel, frame in sorted(rendered.source_panels.items()):
        path = staging_root / "source_data" / f"{figure_id}_panel_{panel}.csv"
        atomic_write_csv(public_source_table(frame), path)
        source_paths[panel] = path
        source_hashes[panel] = sha256_file(path)
    return exported, source_paths, source_hashes


def run_figures(
    profiles_path: str | Path,
    models_path: str | Path,
    tms_path: str | Path,
    clinical_path: str | Path | None,
    fmri_path: str | Path | None,
    output_root: str | Path,
) -> FigureRunResult:
    """Render Figures 1-6 from real stage tables and write an auditable bundle.

    Figures 1-5 require profiles, model/content, robustness, and TMS tables. Figure
    6 is explicitly skipped when held-out clinical results are not yet available;
    Figure 5 discloses (rather than fabricates) missing optional fMRI triangulation.
    Every input row must carry ``source_artifact_sha256``. The function has no demo,
    random-data, or placeholder-data branch.
    """

    destination = Path(output_root).resolve()
    config = load_figure_config()
    inputs = _prepare_inputs(
        profiles_path=profiles_path,
        models_path=models_path,
        tms_path=tms_path,
        clinical_path=clinical_path,
        fmri_path=fmri_path,
    )
    if destination.exists():
        return _reuse_completed_output(destination=destination, config=config, inputs=inputs)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.building.", dir=destination.parent)
    )
    try:
        rendered = _render_all(inputs, config)
        figure_entries: list[dict[str, Any]] = []
        staged_figure_paths: dict[str, dict[str, Path]] = {}
        staged_source_paths: dict[str, dict[str, Path]] = {}
        skipped: list[str] = []
        for figure_id in sorted(rendered):
            item = rendered[figure_id]
            contract = config.figures[figure_id]
            if item is None:
                skipped.append(figure_id)
                figure_entries.append(
                    {
                        "figure_id": figure_id,
                        "status": "skipped_optional_input_absent",
                        "contract": asdict(contract),
                        "outputs": {},
                        "source_data": {},
                    }
                )
                continue
            exported, sources, source_hashes = _export_one(
                figure_id=figure_id,
                rendered=item,
                staging_root=staging_root,
                config=config,
            )
            staged_figure_paths[figure_id] = {
                "svg": exported.svg,
                "pdf": exported.pdf,
                "tiff": exported.tiff,
            }
            staged_source_paths[figure_id] = sources
            figure_entries.append(
                {
                    "figure_id": figure_id,
                    "status": "rendered",
                    "contract": asdict(contract),
                    "outputs": {
                        extension: {
                            "path": str(Path("figures") / path.name),
                            "sha256": exported.sha256[extension],
                            "size_bytes": path.stat().st_size,
                        }
                        for extension, path in staged_figure_paths[figure_id].items()
                    },
                    "source_data": {
                        panel: {
                            "path": str(Path("source_data") / path.name),
                            "sha256": source_hashes[panel],
                            "size_bytes": path.stat().st_size,
                        }
                        for panel, path in sources.items()
                    },
                    "qa": exported.qa,
                }
            )
        manifest = {
            "schema_version": 1,
            "stage": "figures",
            "created_at": datetime.now(UTC).isoformat(),
            "backend": config.backend,
            "target_journal": config.target_journal,
            "config": {
                "name": config.source_path.name,
                "sha256": sha256_file(config.source_path),
            },
            "implementation": {
                "source_revision": git_revision(Path.cwd()),
                "python": platform.python_version(),
                "source_sha256": _implementation_source_hashes(),
            },
            "integrity": config.integrity,
            "inputs": _input_manifest(inputs),
            "figures": figure_entries,
            "skipped_optional_figures": skipped,
        }
        manifest_path = staging_root / "manifest.json"
        atomic_write_json(manifest_path, manifest)
        os.replace(staging_root, destination)
    except BaseException:
        shutil.rmtree(staging_root, ignore_errors=True)
        raise

    figure_paths = {
        figure_id: {
            extension: destination / "figures" / staged_path.name
            for extension, staged_path in paths.items()
        }
        for figure_id, paths in staged_figure_paths.items()
    }
    source_paths = {
        figure_id: {
            panel: destination / "source_data" / staged_path.name
            for panel, staged_path in paths.items()
        }
        for figure_id, paths in staged_source_paths.items()
    }
    return FigureRunResult(
        output_root=destination,
        manifest_path=destination / "manifest.json",
        figure_paths=figure_paths,
        source_data_paths=source_paths,
        skipped=tuple(skipped),
    )


def _clinical_supplement_input(
    clinical_path: str | Path,
) -> tuple[SourceBundle, pd.DataFrame]:
    bundle = load_source_bundle("clinical", clinical_path)
    clinical = find_table(
        bundle,
        ["clinical", "doc", "clinical_profiles"],
        required_columns=CLINICAL_COLUMNS,
    )
    reject_p_value_stars(clinical, label="clinical")
    validate_identifiers(clinical, ["participant_id", "dataset_id"], label="clinical")
    validate_numeric(
        clinical,
        [WAKE_REGIME_LLR, "R", "M", "D", "A", "P"],
        label="clinical",
    )
    _validate_optional_numeric(clinical, "crs_r_total", label="clinical")
    if clinical.duplicated(["dataset_id", "participant_id"]).any():
        raise FigureInputError("clinical must contain one locked profile per dataset-participant")
    return bundle, clinical


def _fmri_supplement_inputs(
    *, models_path: str | Path, fmri_path: str | Path
) -> tuple[SourceBundle, SourceBundle, pd.DataFrame, pd.DataFrame]:
    models_bundle = load_source_bundle("models", models_path)
    fmri_bundle = load_source_bundle("fmri", fmri_path)
    robustness = find_table(
        models_bundle,
        ["robustness", "ablations", "controls"],
        required_columns=ROBUSTNESS_COLUMNS,
    )
    fmri = find_table(
        fmri_bundle,
        ["fmri", "fmri_profiles"],
        required_columns=FMRI_COLUMNS,
    )
    reject_p_value_stars(robustness, label="robustness")
    reject_p_value_stars(fmri, label="fmri")
    validate_identifiers(
        robustness,
        [
            "participant_id",
            "dataset_id",
            "contrast",
            "analysis",
            "family",
            "metric",
            "positive_conditions",
            "negative_conditions",
        ],
        label="robustness",
    )
    validate_numeric(
        robustness,
        [
            "repeat",
            "seed",
            "value",
            "observed_effect",
            "null_effect",
            "observed_minus_null",
            "signed_effect_survival",
            "n_positive_units",
            "n_negative_units",
            "matched_strata",
        ],
        label="robustness",
    )
    _validate_effect_survival(robustness)
    validate_identifiers(fmri, ["participant_id", "dataset_id", "condition"], label="fmri")
    validate_numeric(fmri, ["R", "M", "D", "A"], label="fmri")
    return models_bundle, fmri_bundle, robustness, fmri


def _publish_single_figure(
    *,
    figure_id: str,
    rendered: RenderedFigure,
    destination: Path,
    config: FigureConfig,
    input_manifest: list[dict[str, Any]],
    stage: str,
) -> FigureRunResult:
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.building.", dir=destination.parent)
    )
    try:
        exported, sources, source_hashes = _export_one(
            figure_id=figure_id,
            rendered=rendered,
            staging_root=staging_root,
            config=config,
        )
        staged_outputs = {
            "svg": exported.svg,
            "pdf": exported.pdf,
            "tiff": exported.tiff,
        }
        figure_entry = {
            "figure_id": figure_id,
            "status": "rendered",
            "contract": asdict(config.figures[figure_id]),
            "outputs": {
                extension: {
                    "path": str(Path("figures") / path.name),
                    "sha256": exported.sha256[extension],
                    "size_bytes": path.stat().st_size,
                }
                for extension, path in staged_outputs.items()
            },
            "source_data": {
                panel: {
                    "path": str(Path("source_data") / path.name),
                    "sha256": source_hashes[panel],
                    "size_bytes": path.stat().st_size,
                }
                for panel, path in sources.items()
            },
            "qa": exported.qa,
        }
        manifest = {
            "schema_version": 1,
            "stage": stage,
            "created_at": datetime.now(UTC).isoformat(),
            "backend": config.backend,
            "target_journal": config.target_journal,
            "config": {
                "name": config.source_path.name,
                "sha256": sha256_file(config.source_path),
            },
            "implementation": {
                "source_revision": git_revision(Path.cwd()),
                "python": platform.python_version(),
                "source_sha256": _implementation_source_hashes(),
            },
            "integrity": config.integrity,
            "inputs": input_manifest,
            "figures": [figure_entry],
            "skipped_optional_figures": [],
        }
        atomic_write_json(staging_root / "manifest.json", manifest)
        os.replace(staging_root, destination)
    except BaseException:
        shutil.rmtree(staging_root, ignore_errors=True)
        raise
    return FigureRunResult(
        output_root=destination,
        manifest_path=destination / "manifest.json",
        figure_paths={
            figure_id: {
                extension: destination / "figures" / path.name
                for extension, path in staged_outputs.items()
            }
        },
        source_data_paths={
            figure_id: {
                panel: destination / "source_data" / path.name for panel, path in sources.items()
            }
        },
        skipped=(),
    )


def run_clinical_figure_supplement(
    clinical_path: str | Path, output_root: str | Path
) -> FigureRunResult:
    """Publish Figure 6 after the technical lock, with no endpoint substitution."""

    config = load_figure_config()
    bundle, clinical = _clinical_supplement_input(clinical_path)
    bundle_manifest = _bundle_manifest(bundle)
    if bundle_manifest is None:  # pragma: no cover - non-optional by construction
        raise RuntimeError("clinical source bundle manifest is unexpectedly absent")
    input_manifest = [bundle_manifest]
    destination = Path(output_root).resolve()
    if destination.exists():
        return _reuse_completed_publication(
            destination=destination,
            config=config,
            input_manifest=input_manifest,
            stage="clinical_figure_supplement",
            expected_figures={"figure_6"},
        )
    rendered = render_figure_6(clinical, config, config.figures["figure_6"])
    return _publish_single_figure(
        figure_id="figure_6",
        rendered=rendered,
        destination=destination,
        config=config,
        input_manifest=input_manifest,
        stage="clinical_figure_supplement",
    )


def run_fmri_figure_supplement(
    models_path: str | Path,
    fmri_path: str | Path,
    output_root: str | Path,
) -> FigureRunResult:
    """Publish the fMRI-complete Figure 5 after calibrated triangulation exists."""

    config = load_figure_config()
    models_bundle, fmri_bundle, robustness, fmri = _fmri_supplement_inputs(
        models_path=models_path,
        fmri_path=fmri_path,
    )
    models_manifest = _bundle_manifest(models_bundle)
    fmri_manifest = _bundle_manifest(fmri_bundle)
    if models_manifest is None or fmri_manifest is None:  # pragma: no cover
        raise RuntimeError("fMRI supplement source bundle manifest is unexpectedly absent")
    input_manifest = [models_manifest, fmri_manifest]
    destination = Path(output_root).resolve()
    if destination.exists():
        return _reuse_completed_publication(
            destination=destination,
            config=config,
            input_manifest=input_manifest,
            stage="fmri_figure_supplement",
            expected_figures={"figure_5"},
        )
    rendered = render_figure_5(
        robustness,
        fmri,
        config,
        config.figures["figure_5"],
    )
    return _publish_single_figure(
        figure_id="figure_5",
        rendered=rendered,
        destination=destination,
        config=config,
        input_manifest=input_manifest,
        stage="fmri_figure_supplement",
    )


def figure_run_artifacts(result: FigureRunResult) -> tuple[Path, ...]:
    """Return every publication artifact explicitly for durable queue receipts."""

    artifacts = [result.manifest_path]
    for figure_id in sorted(result.figure_paths):
        artifacts.extend(result.figure_paths[figure_id][key] for key in ("svg", "pdf", "tiff"))
    for figure_id in sorted(result.source_data_paths):
        artifacts.extend(
            result.source_data_paths[figure_id][panel]
            for panel in sorted(result.source_data_paths[figure_id])
        )
    return tuple(artifacts)
