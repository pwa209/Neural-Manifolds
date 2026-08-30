"""Strict source-table loading and atomic source-data export."""

from __future__ import annotations

import os
import re
import tempfile
from collections.abc import Iterable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from neural_manifolds.provenance import sha256_file

SUPPORTED_SUFFIXES = {".csv", ".tsv", ".parquet", ".feather"}
SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")
STAR_DECORATION = re.compile(r"(^|[\s(])\*{1,4}([\s)]|$)")


class FigureInputError(ValueError):
    """Raised when a production figure input is missing, ambiguous, or unauditable."""


@dataclass(frozen=True)
class SourceBundle:
    """Tables loaded from one explicitly supplied stage artifact."""

    role: str
    root: Path
    tables: dict[str, pd.DataFrame]
    source_files: dict[str, str]


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, keep_default_na=False)
    if path.suffix.lower() == ".tsv":
        return pd.read_csv(path, sep="\t", keep_default_na=False)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".feather":
        return pd.read_feather(path)
    raise FigureInputError(f"unsupported source-table format: {path.suffix}")


def _table_files(path: Path) -> list[Path]:
    if path.is_file():
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            raise FigureInputError(f"input is not a supported source table: {path}")
        return [path]
    if not path.is_dir():
        raise FigureInputError(f"input path does not exist: {path}")
    files = sorted(
        candidate
        for candidate in path.rglob("*")
        if candidate.is_file() and candidate.suffix.lower() in SUPPORTED_SUFFIXES
    )
    if not files:
        raise FigureInputError(f"no CSV, TSV, Parquet, or Feather source table found in {path}")
    return files


def _validate_provenance(frame: pd.DataFrame, *, label: str) -> None:
    provenance_column = "source_artifact_sha256"
    if provenance_column not in frame.columns:
        raise FigureInputError(f"{label} is missing required column {provenance_column}")
    hashes = frame[provenance_column].dropna().astype(str)
    if len(hashes) != len(frame):
        raise FigureInputError(f"{label}.{provenance_column} cannot contain missing values")
    invalid = sorted({value for value in hashes if not SHA256_PATTERN.fullmatch(value)})
    if invalid:
        raise FigureInputError(
            f"{label}.{provenance_column} contains {len(invalid)} invalid SHA-256 value(s)"
        )


def load_source_bundle(role: str, path: str | Path) -> SourceBundle:
    """Load only user-supplied source tables; no synthetic or fallback data exist."""

    root = Path(path).resolve()
    tables: dict[str, pd.DataFrame] = {}
    source_files: dict[str, str] = {}
    for file_index, source in enumerate(_table_files(root)):
        frame = _read_table(source)
        if frame.empty:
            raise FigureInputError(f"source table is empty: {source}")
        _validate_provenance(frame, label=source.name)
        digest = sha256_file(source)
        frame = frame.copy()
        frame["source_table_sha256"] = digest
        name = source.stem.lower().replace("-", "_")
        if "table" in frame.columns:
            for table_name, part in frame.groupby("table", sort=True, dropna=False):
                key = str(table_name).strip().lower().replace("-", "_")
                if not key or key == "nan":
                    raise FigureInputError(f"{source.name}.table contains a missing table name")
                if key in tables:
                    raise FigureInputError(f"duplicate logical table {key!r} in {root}")
                tables[key] = part.drop(columns="table").reset_index(drop=True)
                source_files[key] = digest
        else:
            key = name if len(_table_files(root)) > 1 else role
            if key in tables:
                key = f"{key}_{file_index}"
            tables[key] = frame.reset_index(drop=True)
            source_files[key] = digest
    return SourceBundle(role=role, root=root, tables=tables, source_files=source_files)


def find_table(
    bundle: SourceBundle,
    aliases: Iterable[str],
    *,
    required_columns: Iterable[str],
) -> pd.DataFrame:
    """Resolve a logical table by name or, unambiguously, by its schema."""

    aliases_normalised = {alias.lower().replace("-", "_") for alias in aliases}
    required = set(required_columns)
    named = [
        frame
        for name, frame in bundle.tables.items()
        if name in aliases_normalised or any(alias in name for alias in aliases_normalised)
    ]
    candidates = named or [
        frame for frame in bundle.tables.values() if required.issubset(frame.columns)
    ]
    if len(candidates) != 1:
        available = ", ".join(sorted(bundle.tables))
        raise FigureInputError(
            f"{bundle.role} requires exactly one table matching {sorted(required)}; "
            f"available logical tables: {available}"
        )
    frame = candidates[0].copy()
    validate_columns(frame, required, label=bundle.role)
    return frame


def validate_columns(frame: pd.DataFrame, columns: Iterable[str], *, label: str) -> None:
    required = set(columns)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise FigureInputError(f"{label} is missing required columns: {', '.join(missing)}")


def validate_identifiers(frame: pd.DataFrame, columns: Iterable[str], *, label: str) -> None:
    for column in columns:
        if frame[column].isna().any() or (frame[column].astype(str).str.strip() == "").any():
            raise FigureInputError(f"{label}.{column} must contain nonempty identifiers")


def validate_numeric(frame: pd.DataFrame, columns: Iterable[str], *, label: str) -> None:
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
            raise FigureInputError(f"{label}.{column} must contain only finite numeric values")
        frame[column] = values.astype(float)


def reject_p_value_stars(frame: pd.DataFrame, *, label: str) -> None:
    """Reject star-only significance decorations before any label is rendered."""

    for column in frame.select_dtypes(include=["object", "string"]).columns:
        bad = (
            frame[column]
            .dropna()
            .astype(str)
            .map(lambda value: bool(STAR_DECORATION.search(value)))
        )
        if bad.any():
            raise FigureInputError(
                f"{label}.{column} contains p-value star decoration; provide exact values instead"
            )


def atomic_write_csv(frame: pd.DataFrame, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp"
    )
    os.close(descriptor)
    try:
        frame.to_csv(temporary, index=False, lineterminator="\n")
        os.replace(temporary, destination)
    except BaseException:
        with suppress(FileNotFoundError):
            os.unlink(temporary)
        raise
    return destination


def public_source_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Remove implementation-only columns while retaining both provenance hashes."""

    private_columns = [column for column in frame.columns if column.startswith("__")]
    return frame.drop(columns=private_columns, errors="ignore").copy()
