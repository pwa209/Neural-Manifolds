"""Independent finite-window calibration of metastability under stated simulation.

This diagnostic does NOT calibrate biological EEG or overwrite original intervals.
Length-matched model-based intervals target expected finite-window summaries, not
an infinite-duration parameter. Simulation parameters are known, not fitted to EEG.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

from neural_manifolds.manifold.metastability import estimate_metastability

AXES = ["recurrence", "exit_entropy", "dwell_dispersion"]


def markov(n, rng, stay):
    x = np.empty(n, int)
    x[0] = rng.integers(3)
    for i in range(1, n):
        x[i] = x[i - 1] if rng.random() < stay else (x[i - 1] + rng.integers(1, 3)) % 3
    return x


def values(x, segments):
    m = estimate_metastability(x, segment_ids=segments)
    return np.array([m.recurrence_probability, m.exit_entropy, m.dwell_dispersion])


def coverage_record(hits):
    k = int(np.count_nonzero(hits))
    n = len(hits)
    ci = stats.binomtest(k, n).proportion_ci(confidence_level=0.95, method="exact")
    return dict(coverage=k / n, test_replicates=n, coverage_binomial95=[ci.low, ci.high])


def run(seed=20260920):
    rng = np.random.default_rng(seed)
    rows = []
    for n in [12, 20, 120]:
        for stay in [0.3, 0.7]:
            for missing in [0.0, 0.2]:

                def sample(n=n, stay=stay, missing=missing):
                    x = markov(n, rng, stay)
                    keep = np.flatnonzero(rng.random(n) >= missing)
                    return x[keep], np.cumsum(np.r_[True, np.diff(keep) != 1])

                reference = np.array([values(*sample()) for _ in range(1000)])
                target = reference.mean(axis=0)
                calibration = np.array([values(*sample()) for _ in range(1000)])
                errors = calibration - target
                qlo, qhi = np.quantile(errors, [0.025, 0.975], axis=0)
                long = values(markov(10000, rng, stay), np.zeros(10000, int))
                parametric = []
                short_boot = []
                long_boot = []
                for _ in range(200):
                    x, segments = sample()
                    v = values(x, segments)
                    # Compare errors directly: subtracting quantile endpoints back
                    # from discrete estimates can lose equality through rounding.
                    test_error = v - target
                    parametric.append((qlo <= test_error) & (test_error <= qhi))
                    boot = []
                    for _ in range(99):
                        starts = rng.integers(0, max(1, len(x) - 2), size=(len(x) + 2) // 3)
                        take = np.concatenate([np.arange(a, min(a + 3, len(x))) for a in starts])[
                            : len(x)
                        ]
                        seg = np.cumsum(
                            np.r_[
                                True,
                                (np.diff(take) != 1) | (segments[take[1:]] != segments[take[:-1]]),
                            ]
                        )
                        boot.append(values(x[take], seg))
                    lo, hi = np.quantile(boot, [0.025, 0.975], axis=0)
                    short_boot.append((lo <= target) & (target <= hi))
                    long_boot.append((lo <= long) & (long <= hi))
                for j, a in enumerate(AXES):
                    rows.append(
                        dict(
                            axis=a,
                            n=n,
                            stay_probability=stay,
                            missing_fraction=missing,
                            finite_window_target=float(target[j]),
                            long_record_reference=float(long[j]),
                            model_based_finite_window=coverage_record(np.array(parametric)[:, j]),
                            block3_finite_window=coverage_record(np.array(short_boot)[:, j]),
                            block3_long_record=coverage_record(np.array(long_boot)[:, j]),
                        )
                    )
    return dict(
        rows=rows,
        seed=seed,
        original_analysis_unchanged=True,
        scope="Known stationary three-state Markov simulation; independent reference, calibration and test samples.",
        limitations=[
            "Does not establish biological calibration, nor repair all eight axes.",
            "Model-based intervals rely on known transition and missingness models; not transferable to EEG without validation.",
            "Finite-sample recurrence is recording-length/segmentation dependent, not a duration-invariant parameter.",
            "Original three-second bootstrap cannot represent long recurrent paths.",
        ],
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    r = run()
    with a.output.open("x") as f:
        json.dump(r, f, indent=2, allow_nan=False)
    print("RECOVERY_GEOMETRY_COMPLETE", a.output)
