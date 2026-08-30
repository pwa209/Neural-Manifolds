"""Fail-closed separation of direct TMS-EEG from the passive encoder path."""

from __future__ import annotations

from typing import Final

import pandas as pd

DIRECT_TMS_MODALITY: Final = "tms-eeg"
DIRECT_TMS_ACQUISITION: Final = "tms"
DIRECT_TMS_DATASET: Final = "propofol_tms_eeg"


def direct_tms_mask(frame: pd.DataFrame) -> pd.Series:
    """Identify rows that require the dedicated pulse-aware TMS stage.

    The label-free encoder view retains ``modality`` but deliberately omits the
    dataset and acquisition labels.  Later manifests retain all three.  The
    modality check therefore enforces separation at the earliest boundary,
    while the dataset/acquisition check also catches malformed or legacy rows
    whose modality was downgraded to plain EEG.
    """

    mask = pd.Series(False, index=frame.index, dtype=bool)
    if "modality" in frame:
        modality = frame["modality"].astype("string").str.strip().str.casefold()
        mask |= modality.eq(DIRECT_TMS_MODALITY)
    if {"dataset_id", "acquisition"} <= set(frame.columns):
        dataset = frame["dataset_id"].astype("string").str.strip().str.casefold()
        acquisition = frame["acquisition"].astype("string").str.strip().str.casefold()
        mask |= dataset.eq(DIRECT_TMS_DATASET) & acquisition.eq(DIRECT_TMS_ACQUISITION)
    return mask.fillna(False)


def assert_no_direct_tms(frame: pd.DataFrame, *, stage: str) -> None:
    """Reject direct TMS rows before a passive/general-analysis stage runs."""

    mask = direct_tms_mask(frame)
    if not bool(mask.any()):
        return
    identity_column = next(
        (name for name in ("unit_id", "profile_id", "recording_id") if name in frame),
        None,
    )
    identities = (
        sorted(frame.loc[mask, identity_column].astype(str).unique())
        if identity_column is not None
        else []
    )
    preview = identities[:5]
    suffix = "" if len(identities) <= len(preview) else ", ..."
    detail = f"; unit_ids={preview}{suffix}" if preview else ""
    raise ValueError(
        f"{stage} forbids {int(mask.sum())} direct TMS-EEG row(s){detail}; "
        "retain them in the cohort label/raw-lineage manifest and process them only "
        "with the dedicated pulse-interpolation TMS stage"
    )
