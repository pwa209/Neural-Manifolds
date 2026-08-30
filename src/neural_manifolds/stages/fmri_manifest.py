"""Strict ds006623 v1.0.0 manifest preparation for secondary fMRI analysis.

The OpenNeuro release contains fMRIPrep BOLD images, fMRIPrep confounds,
run-aligned propofol effect-site concentrations (ESC), and LOR/ROR timing.
It does not contain the ordered BrainLM A424 atlas/coordinate assets.  This
module binds those independently supplied assets to an immutable acquisition
release and emits the raw-BOLD manifest consumed by ``run_fmri_triangulation``.

No condition label is inferred from filenames.  Labels and volume intervals
are created exclusively by :class:`PropofolFMRIAdapter` after the complete
run-level arrays and the documented timing table have been audited.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from neural_manifolds.adapters.datasets import PropofolFMRIAdapter
from neural_manifolds.data.manifest import MANIFEST_JSON, validate_release
from neural_manifolds.foundation.brainlm import (
    N_PARCELS,
    PARCELLATION,
    load_ukb424_coordinates,
)
from neural_manifolds.provenance import atomic_write_json, sha256_file
from neural_manifolds.stages.fmri import DEFAULT_CONFOUNDS

DATASET_ID = "propofol_fmri"
OPENNEURO_ACCESSION = "ds006623"
RELEASE_VERSION = "1.0.0"
OPENNEURO_GIT_REVISION = "9c36d2c59d58fbbced4af6d0413d22a6ea5c4880"
DATASET_NAME = "Michigan Human Anesthesia fMRI Dataset-1"
DATASET_DOI = "doi:10.18112/openneuro.ds006623.v1.0.0"
FMRIPREP_VERSION = "23.2.1"
OFFICIAL_TR_SECONDS = 0.8
MNI_SPACE = "MNI152NLin2009cAsym"
MNI_RESOLUTION = "04"

ATLAS_ENV = "NEURAL_MANIFOLDS_UKB424_ATLAS"
COORDINATES_ENV = "NEURAL_MANIFOLDS_UKB424_COORDINATES"
TIMING_ORIGIN_ENV = "NEURAL_MANIFOLDS_DS006623_TIMING_INDEX_ORIGIN"

TIMING_COLUMNS = (
    "Subject",
    "LOR time (TR in task2)",
    "ROR time (TR in task3)",
)

REQUIRED_MOTION_CONFOUNDS = (
    "trans_x",
    "trans_y",
    "trans_z",
    "rot_x",
    "rot_y",
    "rot_z",
)

_PARTICIPANT_RE = re.compile(r"^sub-[A-Za-z0-9]+$")
_RAW_BOLD_RE = re.compile(
    r"^(?P<participant>sub-[A-Za-z0-9]+)_task-(?P<task>rest|imagery)_"
    r"run-(?P<run>[1-4])_bold\.nii\.gz$"
)
_PREPROC_BOLD_RE = re.compile(
    r"^(?P<participant>sub-[A-Za-z0-9]+)_task-(?P<task>rest|imagery)_"
    rf"run-(?P<run>[1-4])_space-{MNI_SPACE}_res-{MNI_RESOLUTION}_"
    r"desc-preproc_bold\.nii\.gz$"
)
_ALLOWED_RUNS = {"rest": {1, 2}, "imagery": {1, 2, 3, 4}}


class DS006623ManifestError(RuntimeError):
    """Raised when the pinned ds006623 manifest contract cannot be satisfied."""


@dataclass(frozen=True)
class DS006623ManifestArtifacts:
    """Paths and counts required to invoke ``run_fmri_triangulation``."""

    manifest_path: Path
    audit_path: Path
    atlas_path: Path
    coordinates_path: Path
    analysis_units: int
    participants: int


@dataclass(frozen=True)
class _RunSource:
    participant: str
    task: str
    run: int
    raw_bold: Path
    bold: Path
    bold_sidecar: Path
    confounds: Path
    esc: Path

    @property
    def key(self) -> tuple[str, str, int]:
        return self.participant, self.task, self.run


def _atomic_parquet(frame: pd.DataFrame, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp"
    )
    os.close(descriptor)
    try:
        frame.to_parquet(temporary_name, index=False)
        os.replace(temporary_name, destination)
    except BaseException:
        with suppress(FileNotFoundError):
            os.unlink(temporary_name)
        raise
    return destination


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DS006623ManifestError(f"cannot read valid {label} JSON: {path}") from error
    if not isinstance(value, dict):
        raise DS006623ManifestError(f"{label} must be a JSON object: {path}")
    return value


def _validate_dataset_identity(release: Path) -> dict[str, Any]:
    description = _read_json_object(
        release / "dataset_description.json", label="dataset description"
    )
    expected = {
        "Name": DATASET_NAME,
        "DatasetDOI": DATASET_DOI,
        "DatasetType": "raw",
        "License": "CC0",
    }
    for field, required in expected.items():
        if description.get(field) != required:
            raise DS006623ManifestError(
                f"pinned {OPENNEURO_ACCESSION} identity mismatch for {field}: "
                f"{description.get(field)!r} != {required!r}"
            )

    derivative = _read_json_object(
        release / "derivatives" / "fmriprep_output" / "dataset_description.json",
        label="fMRIPrep derivative description",
    )
    generated = derivative.get("GeneratedBy")
    if not isinstance(generated, list) or not any(
        isinstance(item, dict)
        and item.get("Name") == "fMRIPrep"
        and item.get("Version") == FMRIPREP_VERSION
        for item in generated
    ):
        raise DS006623ManifestError(
            f"ds006623 requires the official fMRIPrep {FMRIPREP_VERSION} derivative"
        )
    return {
        "dataset_name": description["Name"],
        "dataset_doi": description["DatasetDOI"],
        "dataset_license": description["License"],
        "fmriprep_version": FMRIPREP_VERSION,
    }


def _load_release_inventory(release: Path) -> dict[str, dict[str, Any]]:
    manifest = _read_json_object(release / MANIFEST_JSON, label="acquisition manifest")
    entries = manifest.get("files")
    if not isinstance(entries, list):
        raise DS006623ManifestError("acquisition manifest has no file inventory")
    result: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise DS006623ManifestError("malformed acquisition-manifest file entry")
        relative = entry["path"]
        if relative in result:
            raise DS006623ManifestError(f"duplicate acquisition-manifest path: {relative}")
        if not isinstance(entry.get("sha256"), str) or not isinstance(entry.get("size"), int):
            raise DS006623ManifestError(
                f"acquisition-manifest entry lacks SHA-256/size: {relative}"
            )
        result[relative] = entry
    return result


def _inventory_entry(
    release: Path, inventory: Mapping[str, Mapping[str, Any]], path: Path
) -> dict[str, Any]:
    try:
        relative = path.relative_to(release).as_posix()
    except ValueError as error:
        raise DS006623ManifestError(f"release input escapes immutable root: {path}") from error
    entry = inventory.get(relative)
    if entry is None:
        raise DS006623ManifestError(
            f"release input is absent from acquisition manifest: {relative}"
        )
    return {
        "path": relative,
        "size": int(entry["size"]),
        "sha256": str(entry["sha256"]),
    }


def _resolve_required_asset(
    explicit: str | Path | None,
    *,
    environment_name: str,
    description: str,
) -> Path:
    configured = explicit if explicit is not None else os.environ.get(environment_name)
    if configured is None or not str(configured).strip():
        raise DS006623ManifestError(
            f"{description} is not distributed in ds006623; provide it explicitly or set "
            f"{environment_name}"
        )
    try:
        return Path(configured).expanduser().resolve(strict=True)
    except FileNotFoundError as error:
        raise DS006623ManifestError(
            f"configured {description} does not exist: {configured}"
        ) from error


def _resolve_timing_origin(explicit: int | None) -> int:
    configured: object = explicit
    if configured is None:
        configured = os.environ.get(TIMING_ORIGIN_ENV)
    try:
        value = int(configured)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise DS006623ManifestError(
            "ds006623 timing CSV does not document whether TR indices are zero- or one-based; "
            f"pass timing_index_origin=0/1 or set {TIMING_ORIGIN_ENV}"
        ) from error
    if value not in {0, 1} or str(configured).strip() not in {"0", "1"}:
        raise DS006623ManifestError("timing_index_origin must be exactly 0 or 1")
    return value


def _validate_atlas_contract(path: Path) -> None:
    try:
        import nibabel as nib
    except ModuleNotFoundError as error:  # pragma: no cover - deployment dependency
        raise DS006623ManifestError(
            "nibabel is required to validate the UKB_424 atlas; install the fMRI environment"
        ) from error
    try:
        image = nib.load(str(path))
        if len(image.shape) != 3:
            raise DS006623ManifestError(f"UKB_424 atlas must be 3D, got {image.shape}")
        values = np.asarray(image.dataobj)
    except DS006623ManifestError:
        raise
    except Exception as error:
        raise DS006623ManifestError(f"cannot read UKB_424 atlas: {path}") from error
    finite = values[np.isfinite(values)]
    if finite.size != values.size or not np.allclose(finite, np.rint(finite)):
        raise DS006623ManifestError("UKB_424 atlas must contain finite integer labels")
    positive = set(np.unique(np.rint(finite).astype(np.int64)).tolist()) - {0}
    if positive != set(range(1, N_PARCELS + 1)):
        raise DS006623ManifestError(
            "UKB_424 atlas labels must be exactly background 0 and parcels 1..424"
        )


def _validate_coordinates_contract(path: Path) -> None:
    try:
        load_ukb424_coordinates(path)
    except (OSError, ValueError) as error:
        raise DS006623ManifestError(f"invalid ordered UKB_424 coordinates: {path}") from error


def _nifti_run_info(path: Path) -> tuple[int, float]:
    try:
        import nibabel as nib
    except ModuleNotFoundError as error:  # pragma: no cover - deployment dependency
        raise DS006623ManifestError(
            "nibabel is required to audit fMRI run headers; install the fMRI environment"
        ) from error
    try:
        image = nib.load(str(path))
        shape = tuple(int(value) for value in image.shape)
        zooms = image.header.get_zooms()
    except Exception as error:
        raise DS006623ManifestError(f"cannot read preprocessed BOLD header: {path}") from error
    if len(shape) != 4 or shape[3] < 2:
        raise DS006623ManifestError(f"preprocessed BOLD must be nonempty 4D data: {path}")
    if len(zooms) < 4 or not np.isfinite(zooms[3]) or float(zooms[3]) <= 0:
        raise DS006623ManifestError(f"preprocessed BOLD has no valid header TR: {path}")
    return shape[3], float(zooms[3])


def _run_key_from_bold(path: Path, expression: re.Pattern[str]) -> tuple[str, str, int]:
    match = expression.fullmatch(path.name)
    if match is None:
        raise DS006623ManifestError(f"undocumented ds006623 BOLD filename: {path.name}")
    participant = match.group("participant")
    task = match.group("task")
    run = int(match.group("run"))
    if run not in _ALLOWED_RUNS[task]:
        raise DS006623ManifestError(f"undocumented ds006623 {task} run: {run}")
    return participant, task, run


def _discover_runs(release: Path) -> list[_RunSource]:
    raw: dict[tuple[str, str, int], Path] = {}
    for participant_dir in sorted(release.glob("sub-*")):
        if not participant_dir.is_dir() or not _PARTICIPANT_RE.fullmatch(participant_dir.name):
            continue
        functional = participant_dir / "func"
        for path in sorted(functional.glob("*_bold.nii.gz")):
            key = _run_key_from_bold(path, _RAW_BOLD_RE)
            if key[0] != participant_dir.name:
                raise DS006623ManifestError(f"participant/path mismatch: {path}")
            if key in raw:
                raise DS006623ManifestError(f"duplicate raw BOLD run: {key}")
            raw[key] = path

    preprocessed: dict[tuple[str, str, int], Path] = {}
    derivative_root = release / "derivatives" / "fmriprep_output"
    for participant_dir in sorted(derivative_root.glob("sub-*")):
        if not participant_dir.is_dir() or not _PARTICIPANT_RE.fullmatch(participant_dir.name):
            continue
        functional = participant_dir / "func"
        pattern = f"*_space-{MNI_SPACE}_res-{MNI_RESOLUTION}_desc-preproc_bold.nii.gz"
        for path in sorted(functional.glob(pattern)):
            key = _run_key_from_bold(path, _PREPROC_BOLD_RE)
            if key[0] != participant_dir.name:
                raise DS006623ManifestError(f"participant/path mismatch: {path}")
            if key in preprocessed:
                raise DS006623ManifestError(f"duplicate preprocessed BOLD run: {key}")
            preprocessed[key] = path

    if not raw or not preprocessed:
        raise DS006623ManifestError(
            "no audited raw/fMRIPrep ds006623 BOLD runs are available in the immutable release"
        )
    missing_derivative = sorted(set(raw) - set(preprocessed))
    missing_raw = sorted(set(preprocessed) - set(raw))
    if missing_derivative or missing_raw:
        raise DS006623ManifestError(
            "raw/fMRIPrep run inventory mismatch; "
            f"missing_derivative={missing_derivative}, missing_raw={missing_raw}"
        )

    result: list[_RunSource] = []
    for participant, task, run in sorted(raw):
        bold = preprocessed[(participant, task, run)]
        stem = f"{participant}_task-{task}_run-{run}"
        sidecar = Path(str(bold)[: -len(".nii.gz")] + ".json")
        confounds = bold.parent / f"{stem}_desc-confounds_timeseries.tsv"
        esc_label = f"rest{run}" if task == "rest" else f"task{run}"
        esc = (
            release
            / "derivatives"
            / "Propofol_Infusion"
            / participant
            / f"{participant}_{esc_label}_ESC.1D"
        )
        required = {
            "preprocessed BOLD": bold,
            "preprocessed BOLD sidecar": sidecar,
            "fMRIPrep confounds": confounds,
            "effect-site concentration": esc,
        }
        for label, path in required.items():
            if not path.is_file():
                raise DS006623ManifestError(
                    f"missing {label} for {(participant, task, run)}: {path}"
                )
        result.append(
            _RunSource(
                participant=participant,
                task=task,
                run=run,
                raw_bold=raw[(participant, task, run)],
                bold=bold,
                bold_sidecar=sidecar,
                confounds=confounds,
                esc=esc,
            )
        )
    return result


def _read_esc(path: Path) -> np.ndarray:
    values: list[float] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise DS006623ManifestError(
            f"cannot read effect-site concentration array: {path}"
        ) from error
    for line_number, line in enumerate(lines, start=1):
        fields = line.split()
        if len(fields) != 1:
            raise DS006623ManifestError(
                f"ESC must contain exactly one scalar per volume ({path}:{line_number})"
            )
        try:
            value = float(fields[0])
        except ValueError as error:
            raise DS006623ManifestError(f"non-numeric ESC value at {path}:{line_number}") from error
        values.append(value)
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.all(np.isfinite(array)) or np.any(array < 0):
        raise DS006623ManifestError(f"ESC must be a nonempty finite nonnegative array: {path}")
    return array


def _audit_confounds(path: Path, *, n_volumes: int) -> tuple[str, ...]:
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            header_line = stream.readline().rstrip("\r\n")
    except OSError as error:
        raise DS006623ManifestError(f"cannot read fMRIPrep confounds: {path}") from error
    columns = header_line.split("\t") if header_line else []
    if not columns or len(columns) != len(set(columns)):
        raise DS006623ManifestError(f"confounds have an empty or duplicate-column header: {path}")
    missing = [column for column in REQUIRED_MOTION_CONFOUNDS if column not in columns]
    if missing:
        raise DS006623ManifestError(
            f"confounds are missing rigid-body regressors {missing}: {path}"
        )
    selected = tuple(column for column in DEFAULT_CONFOUNDS if column in columns)
    try:
        frame = pd.read_csv(path, sep="\t", usecols=list(selected))
        numeric = frame.apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float64)
    except (OSError, ValueError) as error:
        raise DS006623ManifestError(f"confounds cannot be parsed as numeric TSV: {path}") from error
    if len(frame) != n_volumes:
        raise DS006623ManifestError(
            f"confounds rows ({len(frame)}) do not equal BOLD volumes ({n_volumes}): {path}"
        )
    if np.any(np.isinf(numeric)):
        raise DS006623ManifestError(f"confounds contain infinite values: {path}")
    return selected


def _read_timing_table(path: Path) -> dict[str, tuple[int, int | None]]:
    try:
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    except (OSError, pd.errors.ParserError) as error:
        raise DS006623ManifestError(f"cannot parse official LOR/ROR timing CSV: {path}") from error
    if tuple(frame.columns) != TIMING_COLUMNS:
        raise DS006623ManifestError(
            f"LOR/ROR timing columns must be exactly {TIMING_COLUMNS}, got {tuple(frame.columns)}"
        )
    result: dict[str, tuple[int, int | None]] = {}
    for record in frame.to_dict(orient="records"):
        participant = str(record[TIMING_COLUMNS[0]]).strip()
        if not _PARTICIPANT_RE.fullmatch(participant) or participant in result:
            raise DS006623ManifestError(
                f"timing table has invalid or duplicate participant: {participant!r}"
            )
        lor_text = str(record[TIMING_COLUMNS[1]]).strip()
        ror_text = str(record[TIMING_COLUMNS[2]]).strip()
        try:
            lor = int(lor_text)
        except ValueError as error:
            raise DS006623ManifestError(
                f"LOR timing must be an integer TR for {participant}, got {lor_text!r}"
            ) from error
        if str(lor) != lor_text or lor < 0:
            raise DS006623ManifestError(f"invalid LOR timing for {participant}: {lor_text!r}")
        if ror_text == "N/A":
            ror = None
        else:
            try:
                ror = int(ror_text)
            except ValueError as error:
                raise DS006623ManifestError(
                    f"ROR timing must be an integer TR or N/A for {participant}, got {ror_text!r}"
                ) from error
            if str(ror) != ror_text or ror < 0:
                raise DS006623ManifestError(f"invalid ROR timing for {participant}: {ror_text!r}")
        result[participant] = (lor, ror)
    if not result:
        raise DS006623ManifestError("official LOR/ROR timing table is empty")
    return result


def _sidecar_tr(path: Path) -> float:
    sidecar = _read_json_object(path, label="preprocessed BOLD sidecar")
    try:
        tr_seconds = float(sidecar["RepetitionTime"])
    except (KeyError, TypeError, ValueError) as error:
        raise DS006623ManifestError(
            f"BOLD sidecar has no numeric RepetitionTime: {path}"
        ) from error
    if not np.isfinite(tr_seconds) or tr_seconds <= 0:
        raise DS006623ManifestError(f"BOLD sidecar has invalid RepetitionTime: {path}")
    return tr_seconds


def _assert_complete_segments(units: list[Any], *, n_volumes: int, key: object) -> None:
    intervals = sorted(
        (int(unit.selector.volume_start), int(unit.selector.volume_stop)) for unit in units
    )
    cursor = 0
    for start, stop in intervals:
        if start != cursor or stop <= start:
            raise DS006623ManifestError(
                f"adapter segments do not form a non-overlapping full run for {key}: {intervals}"
            )
        cursor = stop
    if cursor != n_volumes:
        raise DS006623ManifestError(
            f"adapter segments do not cover all {n_volumes} volumes for {key}: {intervals}"
        )


def prepare_ds006623_fmri_manifest(
    *,
    release_root: str | Path,
    output_root: str | Path,
    timing_index_origin: int | None = None,
    atlas_path: str | Path | None = None,
    coordinates_path: str | Path | None = None,
) -> DS006623ManifestArtifacts:
    """Audit the immutable ds006623 release and emit an fMRI stage manifest.

    ``timing_index_origin`` must be 0 or 1 because the official timing CSV names
    values as TR indices but does not declare the index origin.  Omitting the
    argument is allowed only when ``NEURAL_MANIFOLDS_DS006623_TIMING_INDEX_ORIGIN``
    is set.  The selected origin is subtracted from each numeric CSV timing to
    obtain the zero-based half-open volume boundaries supplied to the adapter.

    The A424 atlas and ordered coordinates are resolved from explicit arguments
    first, then ``NEURAL_MANIFOLDS_UKB424_ATLAS`` and
    ``NEURAL_MANIFOLDS_UKB424_COORDINATES``.  Both contracts are validated before
    any output is written.
    """

    try:
        release = Path(release_root).resolve(strict=True)
    except FileNotFoundError as error:
        raise DS006623ManifestError(f"ds006623 release does not exist: {release_root}") from error
    if not release.is_dir():
        raise DS006623ManifestError(f"ds006623 release is not a directory: {release}")

    try:
        release_receipt = validate_release(
            release,
            expected_dataset_id=DATASET_ID,
            expected_release_version=RELEASE_VERSION,
        )
    except Exception as error:
        raise DS006623ManifestError(
            f"immutable ds006623 release validation failed: {error}"
        ) from error
    identity = _validate_dataset_identity(release)
    inventory = _load_release_inventory(release)
    timing_origin = _resolve_timing_origin(timing_index_origin)

    atlas = _resolve_required_asset(
        atlas_path, environment_name=ATLAS_ENV, description="UKB_424 atlas"
    )
    coordinates = _resolve_required_asset(
        coordinates_path,
        environment_name=COORDINATES_ENV,
        description="ordered UKB_424 coordinate table",
    )
    _validate_atlas_contract(atlas)
    _validate_coordinates_contract(coordinates)

    timing_path = release / "derivatives" / "LOR_ROR_Timing.csv"
    if not timing_path.is_file():
        raise DS006623ManifestError(f"missing official LOR/ROR timing table: {timing_path}")
    timing = _read_timing_table(timing_path)
    sources = _discover_runs(release)
    participants = {source.participant for source in sources}
    if participants != set(timing):
        raise DS006623ManifestError(
            "BOLD and timing-table participants differ; "
            f"missing_timing={sorted(participants - set(timing))}, "
            f"missing_bold={sorted(set(timing) - participants)}"
        )

    adapter = PropofolFMRIAdapter()
    rows: list[dict[str, Any]] = []
    run_audit: list[dict[str, Any]] = []
    unit_ids: set[str] = set()
    for source in sources:
        n_volumes, header_tr = _nifti_run_info(source.bold)
        tr_seconds = _sidecar_tr(source.bold_sidecar)
        if not np.isclose(header_tr, tr_seconds, rtol=0.0, atol=1e-6):
            raise DS006623ManifestError(
                f"BOLD header/sidecar TR mismatch for {source.key}: {header_tr} != {tr_seconds}"
            )
        if not np.isclose(tr_seconds, OFFICIAL_TR_SECONDS, rtol=0.0, atol=1e-6):
            raise DS006623ManifestError(
                f"unexpected TR for pinned ds006623 run {source.key}: {tr_seconds}"
            )
        concentration = _read_esc(source.esc)
        if concentration.size != n_volumes:
            raise DS006623ManifestError(
                f"ESC values ({concentration.size}) do not equal BOLD volumes "
                f"({n_volumes}) for {source.key}"
            )
        confounds = _audit_confounds(source.confounds, n_volumes=n_volumes)

        lor_raw, ror_raw = timing[source.participant]
        lor_boundary = (
            lor_raw - timing_origin if source.task == "imagery" and source.run == 2 else None
        )
        ror_boundary = (
            ror_raw - timing_origin
            if source.task == "imagery" and source.run == 3 and ror_raw is not None
            else None
        )
        if lor_boundary is not None and not 0 < lor_boundary < n_volumes:
            raise DS006623ManifestError(
                f"origin-adjusted LOR boundary {lor_boundary} is outside {source.key} "
                f"with {n_volumes} volumes"
            )
        if ror_boundary is not None and not 0 < ror_boundary < n_volumes:
            raise DS006623ManifestError(
                f"origin-adjusted ROR boundary {ror_boundary} is outside {source.key} "
                f"with {n_volumes} volumes"
            )

        logical_source = (
            f"{source.participant}/func/{source.participant}_task-{source.task}_"
            f"run-{source.run}_bold.nii.gz"
        )
        units = adapter.adapt_run(
            participant_id=source.participant,
            task=source.task,
            run=source.run,
            source_file=logical_source,
            effect_site_concentration=concentration,
            lor_volume=lor_boundary,
            ror_volume=ror_boundary,
        )
        _assert_complete_segments(units, n_volumes=n_volumes, key=source.key)
        source_inventory = {
            "raw_bold": _inventory_entry(release, inventory, source.raw_bold),
            "bold": _inventory_entry(release, inventory, source.bold),
            "bold_sidecar": _inventory_entry(release, inventory, source.bold_sidecar),
            "confounds": _inventory_entry(release, inventory, source.confounds),
            "effect_site_concentration": _inventory_entry(release, inventory, source.esc),
        }
        for unit in units:
            if unit.unit_id in unit_ids:
                raise DS006623ManifestError(f"duplicate adapter unit ID: {unit.unit_id}")
            unit_ids.add(unit.unit_id)
            variables = dict(unit.variables)
            rows.append(
                {
                    "unit_id": unit.unit_id,
                    "participant_id": f"{unit.dataset_id}:{unit.participant_id}",
                    "native_participant_id": unit.participant_id,
                    "dataset_id": unit.dataset_id,
                    "modality": unit.modality,
                    "run_id": unit.run_id,
                    "task": source.task,
                    "condition": unit.condition,
                    "explanatory_target": unit.explanatory_target,
                    "healthy_wake_reference": unit.healthy_wake_reference,
                    "clinical_holdout": unit.clinical_holdout,
                    "task_relevance": unit.task_relevance,
                    "content": unit.content,
                    "metadata_status": unit.metadata_status,
                    "source_file": unit.source_file,
                    "source_path": str(source.raw_bold),
                    "bold_path": str(source.bold),
                    "confounds_path": str(source.confounds),
                    "effect_site_concentration_path": str(source.esc),
                    "effect_site_concentration_sha256": source_inventory[
                        "effect_site_concentration"
                    ]["sha256"],
                    "parcellation": PARCELLATION,
                    "space": MNI_SPACE,
                    "resolution_entity": MNI_RESOLUTION,
                    "tr_seconds": tr_seconds,
                    "preprocessed": True,
                    "timeseries_scope": "run",
                    "normalization": "unscaled_denoised",
                    "confound_columns_json": json.dumps(list(confounds), separators=(",", ":")),
                    "volume_start": int(unit.selector.volume_start),
                    "volume_stop": int(unit.selector.volume_stop),
                    "run_volume_count": n_volumes,
                    "selector_json": json.dumps(
                        unit.selector.model_dump(mode="json"),
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    "timing_index_origin": timing_origin,
                    "lor_tr_csv": lor_raw if source.task == "imagery" and source.run == 2 else None,
                    "ror_tr_csv": ror_raw if source.task == "imagery" and source.run == 3 else None,
                    **variables,
                }
            )
        run_audit.append(
            {
                "participant_id": f"{DATASET_ID}:{source.participant}",
                "native_participant_id": source.participant,
                "task": source.task,
                "run": source.run,
                "volumes": n_volumes,
                "tr_seconds": tr_seconds,
                "confound_columns": list(confounds),
                "analysis_units": len(units),
                "lor_tr_csv": lor_raw if source.task == "imagery" and source.run == 2 else None,
                "lor_volume_boundary": lor_boundary,
                "ror_tr_csv": ror_raw if source.task == "imagery" and source.run == 3 else None,
                "ror_volume_boundary": ror_boundary,
                "inputs": source_inventory,
            }
        )

    if not rows:
        raise DS006623ManifestError("no ds006623 fMRI analysis units were produced")
    manifest_frame = pd.DataFrame(rows).sort_values(
        ["native_participant_id", "task", "run_id", "volume_start"], kind="stable"
    )
    destination = Path(output_root).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = _atomic_parquet(
        manifest_frame.reset_index(drop=True), destination / "ds006623-fmri-manifest.parquet"
    )
    audit_path = destination / "ds006623-fmri-manifest-audit.json"
    timing_inventory = _inventory_entry(release, inventory, timing_path)
    atomic_write_json(
        audit_path,
        {
            "schema_version": 1,
            "dataset_id": DATASET_ID,
            "openneuro_accession": OPENNEURO_ACCESSION,
            "release_version": RELEASE_VERSION,
            "openneuro_git_revision": OPENNEURO_GIT_REVISION,
            **identity,
            "immutable_release": release_receipt,
            "timing": {
                **timing_inventory,
                "columns": list(TIMING_COLUMNS),
                "index_origin": timing_origin,
                "boundary_rule": "zero_based_boundary = csv_TR - index_origin",
                "missing_ror_token": "N/A",
            },
            "parcellation": PARCELLATION,
            "atlas": {"path": str(atlas), "sha256": sha256_file(atlas)},
            "coordinates": {
                "path": str(coordinates),
                "sha256": sha256_file(coordinates),
            },
            "manifest": {
                "path": str(manifest_path),
                "sha256": sha256_file(manifest_path),
                "analysis_units": len(manifest_frame),
                "participants": len(participants),
                "runs": len(sources),
            },
            "run_audit": run_audit,
        },
    )
    return DS006623ManifestArtifacts(
        manifest_path=manifest_path,
        audit_path=audit_path,
        atlas_path=atlas,
        coordinates_path=coordinates,
        analysis_units=len(manifest_frame),
        participants=len(participants),
    )


__all__ = [
    "ATLAS_ENV",
    "COORDINATES_ENV",
    "TIMING_ORIGIN_ENV",
    "DS006623ManifestArtifacts",
    "DS006623ManifestError",
    "prepare_ds006623_fmri_manifest",
]
