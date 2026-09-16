# Revised study execution map

Exploratory, not registered or preregistered. No effect size, direction, P value,
model ranking or journal target controls scheduling. Technical failures and
unestimable analyses remain explicit; they are not negative scientific results.

## Implemented graph

| Phase | Executable code | Dependency and output |
|---|---|---|
| R0 | `scripts/alliance/qualify.sbatch` | Immutable source/environment checks, full tests and GPU check |
| R1–R3 | `continuous/audit.py`, `revised/cohort.py`, `configs/revised_execution.yaml` | Exact source-record joins, report semantics, related-laboratory groups, montage/duration audit and exclusion denominator |
| R2 | `scripts/alliance/acquire.py` | Priority downloads for the primary open portfolio, preserved partial files, sealed verified releases |
| R2 boundary | `scripts/alliance/acquire_psiconnect_eeg.py` | Pinned public FieldTrip EEG and public behaviour subset; explicitly not the whole MRI dataset |
| R4 | `revised/recovery.py` | Observed clean lengths; stable VAR simulations; all implemented dynamics features; bias, RMSE and measured block-bootstrap coverage |
| R5 | `revised/measurement.py`, `foundation/labram.py` | Final pre-awakening windows, physical-unit QC, gap-preserving trajectories, frozen encoder, sensor and time-frequency representations |
| R6 | `revised/inference.py` | Outer laboratory/study holdout, training-only inner fitting, matched conventional alternatives, fixed dimensions, leave-one-axis-out comparisons and separate within-study participant validation |
| R6 sensitivities | `revised/controls.py` | DEWR and REM contrasts kept separate; 100 full-refit null replicates; label exchangeability is explicitly conditional on participant and stage |
| R4–R6 robustness | `revised/robustness.py` | Matched original windows with signal-spectrum-preserving phase perturbations, added noise and channel dropout; frozen re-encoding followed by nested refits |
| R7 | `revised/perturbation.py` | Exact spontaneous/TMS entity matching, blind pulse QC, participant-held-out incremental prediction, within-condition diagnostics and paired state changes |
| R8 perception | `revised/specificity.py` | Fast causal-filter sensor track, intensity-matched perception and task contrasts, catch trials, pre-response intervals and separate confidence associations |
| R8 boundary | `revised/boundary.py` | Public context-specific FieldTrip parsing, held-out-participant baseline fitting and paired context changes |
| R9 | Legacy extension modules, separate commissioning | Optional clinical/fMRI extensions are not dependencies of core synthesis; missing labels/atlas/timing authority are not guessed |
| R10 | `revised/synthesis.py` | Incremental source-linked tables, plots, captions and an evidence bundle; final core execution status requires all required branches and null replicates |

Paths in the table are relative to `src/neural_manifolds` unless they start with
`scripts` or `configs`. Input/output hashes and exact source identities remain in
server-side task receipts. Public Git contains no empirical outputs.

## Meaning of automatic and queued

The controller adds a task when its hashed inputs exist and submits up to two
study jobs at a time, with at most one GPU job. Downstream stages are dependencies
in the durable workflow, not necessarily simultaneous Slurm submissions. It does
not wait for favourable results. Different datasets proceed independently.

Core source downloads and the explicitly bounded PsiConnect subset use separate
locks and storage checks. The subset has a 20 GiB cap and 100 GiB free-space
headroom; it does not silently request the much larger MRI archive. Existing
acquisition retry counters are retained across controller replacement. A failed
software or provenance check needs repair; it is not endlessly retried.
The PsiConnect acquirer configures sparse checkout **before** materializing the
working tree, so unused MRI pointers do not exhaust project file-count quotas.
Check project-ID byte and inode quotas, not just filesystem free space or
default user/group quota output.

After quota inspection, a single additional acquisition pass is scheduled behind
the active worker with a 300 GiB portfolio cap (still 100 GiB per source and
100 GiB free-space headroom). Existing reservations nearly exhaust the initial
180 GiB cap. This is a bounded continuation, not a reset of exhausted automatic
retry counters or unlimited retrying of failed sources.

## Measurement limitations that remain visible

- The Paris source's sampled EDF headers report `n/a` physical units. Values of
  plausible microvolt magnitude do **not** authorize a voltage conversion.
  Physical-amplitude and encoder analyses report undocumented units as unavailable.
- Wisconsin's HydroCel-256 channels 37, 59 and 116 approximate Fp1, C3 and O1.
  The manufacturer's equivalence table reports approximately 1.05, 1.03 and
  1.26 cm discrepancies. These are observed electrodes, not interpolated ones;
  their equivalence is approximate, not exact. The source supplies matching
  HydroCel coordinates. See the manufacturer technical note, *Determination of
  the HydroCel Geodesic Sensor Nets' Average Electrode Positions and Their
  10-10 International Equivalents*, page 8. A sensitivity excluding this cohort
  must retain an unavailable result if too few independent laboratories remain.
- Oslo evening and morning samples share an outer laboratory group. Unresolved
  participant aliases prevent claims of independent subject transfer between them.
- Pisa DREEM bipolar leads are not silently renamed to monopolar scalp electrodes.
- Pretraining participant overlap is unresolved. Frozen-encoder transfer is not
  described as proven unseen-pretraining-cohort generalisation.
- The encoder is slow, second-scale measurement. Fast sensor results do not
  establish subsecond encoder interactions. Upstream zero-phase processing in
  public derivatives limits prospective latency interpretations.
- Representation-spectrum nulls and physical EEG-spectrum nulls are different
  controls and are identified separately. Synthetic VAR references are long
  Monte Carlo estimates, not analytic truth or biological validation.
- PsiConnect uses amplitude-normalized sensors while MAT voltage units remain
  unverified. Public subjective-scale mappings require schema adjudication; no
  participant or scale join is inferred from row order. No simultaneous EEG/fMRI
  claim is made.
- Report/no-report is order-confounded. TMS prediction lacks sham control and
  cannot establish anatomical controllability or a mechanism of consciousness.

## Validation status

Core and branch tests exercise fold separation, exact joins, spectrum preservation,
gap-safe temporal controls, pairing, FieldTrip axes and honest partial synthesis.
The real encoder canary has successfully measured recordings from three cohorts.
The subsequent Wisconsin check measured four additional participant recordings.
The deployed release passed 323 cluster tests and an H100 GPU check; the final
local suite passed 318 tests with five platform/dependency skips. The synthetic
forest-plot test export was visually inspected; empirical figures are not yet
available and have not been visually inspected.
This does not yet validate every acquired source, every full-size job, or journal
readiness. Generated empirical figures require inspection at final display size;
the artifact records automated rendering separately from visual review.

The execution graph does not imply that every design ambition is complete.
Formal independent-sample power/precision simulation, the public subjective-scale
join for PsiConnect and commissioning optional clinical/fMRI extensions remain
separate work. Source-specific unavailable measurements are retained explicitly.

The assistant's recurring monitor remains deleted. Remote execution does not
depend on it or on a continuously connected desktop.
