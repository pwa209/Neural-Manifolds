"""Slurm preflight: validate payloads and read every eligible tactile epoch."""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from neural_manifolds.adapters.datasets import TactileDetectionAdapter
from neural_manifolds.provenance import atomic_write_json
from neural_manifolds.revised.specificity import fast_features
from neural_manifolds.revised.tactile_io import (
    IncompleteSourceRecordingError,
    open_tactile_recording,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    participants = pd.read_csv(args.release / "participants.tsv", sep="\t")
    audited, excluded, conditions, timings, recording_exclusions = [], [], Counter(), [], []
    for event_file in sorted(args.release.glob("sub-*/ses-*/eeg/*task-adapt*_events.tsv")):
        relative = event_file.relative_to(args.release).as_posix()
        events = pd.read_csv(event_file, sep="\t")
        units = TactileDetectionAdapter().adapt(
            participants,
            {relative: events},
            missing_responses=excluded,
            non_adaptive_events=excluded,
        )
        assert len(units) == int(events.trial_type.isin(["hit", "miss", "cr", "fa"]).sum())
        conditions.update(u.condition for u in units)
        eligible = []
        for unit in units:
            if (
                unit.variables["first_order_response_onset_seconds"]
                < unit.selector.event_onset_seconds
            ):
                timings.append({"unit_id": unit.unit_id, "reason": "response_precedes_stimulus"})
            else:
                eligible.append(unit)
        if not eligible:
            raise ValueError(f"no timing-valid canary trial: {relative}")
        try:
            raw, payload_audit = open_tactile_recording(args.release / eligible[0].source_file)
        except IncompleteSourceRecordingError as exc:
            recording_exclusions.append(
                {"source_file": eligible[0].source_file, "excluded_units": len(units), **exc.audit}
            )
            print("TACTILE_SOURCE_EXCLUSION", relative, exc.audit, flush=True)
            continue
        try:
            raw.pick("eeg")
            sfreq = float(raw.info["sfreq"])
            read_units = []
            for index, unit in enumerate(eligible):
                onset = unit.selector.event_onset_seconds
                start, stop = round((onset - 0.4) * sfreq), round((onset + 0.8) * sfreq)
                if not 0 <= start < stop <= raw.n_times:
                    excluded.append({"unit_id": unit.unit_id, "reason": "epoch_outside_recording"})
                    continue
                try:
                    epoch = raw.get_data(start=start, stop=stop, reject_by_annotation="NaN")
                except Exception as exc:
                    raise RuntimeError(
                        f"read_failed:{relative}:{unit.unit_id}:{start}:{stop}"
                    ) from exc
                if epoch.shape != (len(raw.ch_names), stop - start):
                    raise ValueError(f"incomplete_epoch:{relative}:{unit.unit_id}")
                read_units.append(unit.unit_id)
                if index in {0, len(eligible) // 2, len(eligible) - 1}:
                    values, _ = fast_features(
                        epoch,
                        np.arange(epoch.shape[1]) / sfreq - 0.4,
                        sfreq,
                        response_seconds=unit.variables["first_order_response_onset_seconds"]
                        - onset,
                    )
                    if not values:
                        raise ValueError(f"no canary features: {relative}")
            audited.append(
                {
                    "source_file": unit.source_file,
                    "channels": len(raw.ch_names),
                    "sfreq": sfreq,
                    "units": len(units),
                    "read_unit_ids": read_units,
                    "payload_audit": payload_audit,
                }
            )
            print("TACTILE_CANARY", relative, len(units), "passed", flush=True)
        finally:
            raw.close()
    if not audited:
        raise ValueError("no adaptive recordings found")
    atomic_write_json(
        args.output,
        {
            "status": "passed",
            "recordings": audited,
            "conditions": dict(conditions),
            "excluded": excluded,
            "timing_exclusions": timings,
            "recording_exclusions": recording_exclusions,
            "scope": "all_eligible_epoch_reads_plus_first_middle_last_features_not_full_inference",
        },
    )


if __name__ == "__main__":
    main()
