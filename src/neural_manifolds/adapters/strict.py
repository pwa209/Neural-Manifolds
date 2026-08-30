"""Schema and scalar validators shared by dataset adapters."""

from __future__ import annotations

import math
import re
from collections.abc import Iterable
from pathlib import PurePosixPath

import pandas as pd

from .models import SchemaError

MISSING = frozenset({"", "n/a", "na", "nan", "nd", "none", "null"})


def normalize_relative_path(value: object) -> str:
    text = str(value).strip().replace("\\", "/")
    while text.startswith("./"):
        text = text[2:]
    path = PurePosixPath(text)
    if not text or path.is_absolute() or ".." in path.parts:
        raise SchemaError(f"unsafe relative path: {value!r}")
    return str(path)


def require_exact_columns(table: pd.DataFrame, columns: Iterable[str], *, source: str) -> None:
    expected = list(columns)
    actual = list(table.columns)
    if actual != expected:
        missing = [column for column in expected if column not in actual]
        unknown = [column for column in actual if column not in expected]
        raise SchemaError(
            f"{source} columns do not match audited order; missing={missing}, unknown={unknown}, "
            f"expected={expected}, actual={actual}"
        )


def require_values(values: Iterable[object], allowed: set[str], *, field: str) -> None:
    observed = {str(value).strip() for value in values}
    unknown = sorted(observed.difference(allowed))
    if unknown:
        raise SchemaError(f"{field} contains undocumented values: {unknown}")


def text(value: object, *, field: str, allow_missing: bool = False) -> str | None:
    result = str(value).strip()
    if result.lower() in MISSING:
        if allow_missing:
            return None
        raise SchemaError(f"{field} is missing")
    return result


def number(
    value: object,
    *,
    field: str,
    minimum: float | None = None,
    maximum: float | None = None,
    allow_missing: bool = False,
) -> float | None:
    raw = text(value, field=field, allow_missing=allow_missing)
    if raw is None:
        return None
    try:
        result = float(raw)
    except (TypeError, ValueError) as exc:
        raise SchemaError(f"{field} must be numeric, got {value!r}") from exc
    if not math.isfinite(result):
        raise SchemaError(f"{field} must be finite")
    if minimum is not None and result < minimum:
        raise SchemaError(f"{field} must be >= {minimum}")
    if maximum is not None and result > maximum:
        raise SchemaError(f"{field} must be <= {maximum}")
    return result


def integer(value: object, *, field: str, minimum: int = 0) -> int:
    result = number(value, field=field, minimum=float(minimum))
    assert result is not None
    if not result.is_integer():
        raise SchemaError(f"{field} must be an integer")
    return int(result)


def boolean(value: object, *, field: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise SchemaError(f"{field} must be exactly True or False")


def match(pattern: re.Pattern[str], value: str, *, field: str) -> re.Match[str]:
    result = pattern.fullmatch(value)
    if result is None:
        raise SchemaError(f"{field} does not match audited format: {value!r}")
    return result
