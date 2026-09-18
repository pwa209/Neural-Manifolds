"""Small Slurm canary: audit all tactile tables and sample each EEG recording."""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from neural_manifolds.adapters.datasets import TactileDetectionAdapter
from neural_manifolds.provenance import atomic_write_json
from neural_manifolds.revised.specificity import fast_features
from neural_manifolds.stage_processing import read_raw_recording


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    participants = pd.read_csv(args.release / "participants.tsv", sep="\t")
    audited, excluded, conditions, timings = [], [], Counter(), []
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
        unit = eligible[0]
        raw = read_raw_recording(args.release / unit.source_file)
        try:
            raw.pick("eeg")
            sfreq = float(raw.info["sfreq"])
            onset = unit.selector.event_onset_seconds
            start, stop = round((onset - 0.4) * sfreq), round((onset + 0.8) * sfreq)
            if not 0 <= start < stop <= raw.n_times:
                raise ValueError(f"canary epoch outside recording: {relative}")
            epoch = raw.get_data(start=start, stop=stop, reject_by_annotation="NaN")
            values, _ = fast_features(
                epoch,
                np.arange(epoch.shape[1]) / sfreq - 0.4,
                sfreq,
                response_seconds=unit.variables["first_order_response_onset_seconds"] - onset,
            )
            if not values:
                raise ValueError(f"no canary features: {relative}")
            audited.append(
                {
                    "source_file": unit.source_file,
                    "channels": len(raw.ch_names),
                    "sfreq": sfreq,
                    "units": len(units),
                    "sampled_unit": unit.unit_id,
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
            "scope": "one_trial_per_recording_not_full_analysis",
        },
    )


if __name__ == "__main__":
    main()
