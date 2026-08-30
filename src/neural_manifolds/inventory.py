"""Build a label-blind inventory of acquired electrophysiology recordings."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

PRIMARY_EXTENSIONS = {".vhdr", ".edf", ".bdf", ".set", ".fif"}
ENTITY_PATTERN = re.compile(r"(?:^|_)(sub|ses|task|acq|run|recording)-([^_]+)")


@dataclass(frozen=True)
class Recording:
    recording_id: str
    dataset_id: str
    release_version: str
    participant_id: str
    session: str | None
    task: str | None
    acquisition: str | None
    run: str | None
    source_path: str
    events_path: str | None
    channels_path: str | None
    modality: str


def _entities(path: Path) -> dict[str, str]:
    entities: dict[str, str] = {}
    for component in path.parts:
        for key, value in ENTITY_PATTERN.findall(component):
            entities[key] = value.split(".")[0]
    return entities


def _sidecar(path: Path, suffix: str) -> Path | None:
    stem = path.name
    for ending in (
        "_eeg.vhdr",
        "_eeg.edf",
        "_eeg.bdf",
        "_eeg.set",
        "_eeg.fif",
        ".vhdr",
        ".edf",
        ".bdf",
        ".set",
        ".fif",
    ):
        if stem.lower().endswith(ending):
            prefix = stem[: -len(ending)]
            candidate = path.with_name(f"{prefix}_{suffix}.tsv")
            if candidate.is_file():
                return candidate
    candidates = sorted(path.parent.glob(f"*_{suffix}.tsv"))
    return candidates[0] if len(candidates) == 1 else None


def scan_recordings(raw_root: str | Path) -> list[Recording]:
    root = Path(raw_root).resolve(strict=True)
    records: list[Recording] = []
    for dataset_root in sorted(
        path for path in root.iterdir() if path.is_dir() and not path.name.startswith(".")
    ):
        for release_root in sorted(path for path in dataset_root.iterdir() if path.is_dir()):
            if not (release_root / ".acquisition" / "COMPLETE.json").is_file():
                continue
            for source in sorted(release_root.rglob("*")):
                if not source.is_file() or source.suffix.lower() not in PRIMARY_EXTENSIONS:
                    continue
                entities = _entities(source.relative_to(release_root))
                participant = entities.get("sub")
                if not participant:
                    # Non-BIDS clinical releases are retained for a dataset adapter.
                    participant = source.stem.split("_")[0]
                relative = source.relative_to(release_root).as_posix()
                recording_id = f"{dataset_root.name}:{release_root.name}:{relative}"
                records.append(
                    Recording(
                        recording_id=recording_id,
                        dataset_id=dataset_root.name,
                        release_version=release_root.name,
                        participant_id=f"{dataset_root.name}:sub-{participant}",
                        session=entities.get("ses"),
                        task=entities.get("task"),
                        acquisition=entities.get("acq"),
                        run=entities.get("run"),
                        source_path=str(source),
                        events_path=str(event) if (event := _sidecar(source, "events")) else None,
                        channels_path=str(channels)
                        if (channels := _sidecar(source, "channels"))
                        else None,
                        modality="eeg",
                    )
                )
    identifiers = [record.recording_id for record in records]
    if len(set(identifiers)) != len(identifiers):
        raise RuntimeError("recording inventory contains duplicate identifiers")
    return records


def inventory_frame(records: Iterable[Recording]) -> pd.DataFrame:
    frame = pd.DataFrame(asdict(record) for record in records)
    if not frame.empty:
        frame = frame.sort_values(["dataset_id", "participant_id", "source_path"]).reset_index(
            drop=True
        )
    return frame


def write_inventory(records: Iterable[Recording], destination: str | Path) -> Path:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = inventory_frame(records)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)
    return path
