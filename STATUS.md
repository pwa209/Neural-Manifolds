# Project status

This file records source and orchestration state only. Direct NAS acquisition is in
progress; no external dataset has been analysed. GitHub stores only code,
configuration, tests, checksum metadata, roadmap, and status. All empirical raw and
derived data, participant-level outputs, aggregate analysis tables, figure
source-data tables, caches, and durable logs remain on university storage.

## Current state

- Project status: exploratory; not registered or preregistered.
- Scientific gates: none. Technical validation and provenance checks remain active.
- Local implementation: the complete phase graph and the scientific-validity
  hardening described below are present. Later phases are not yet running and will
  be launched one at a time from a new exact server deployment after their technical
  inputs are present; no result threshold controls that sequence.
- Local verification: 271 repository tests passed, with four documented
  platform-capability skips; Ruff lint, Ruff format, and `git diff --check` are clean.
- GitHub synchronization: the local branch is `main` and tracks `origin/main`.
  Deployment uses an exact pushed commit, never an unrecorded working tree or branch
  tip.
- Server roots: confirmed as `/private_nas/wangpeng/neural-manifolds`,
  `/data1/wangpeng/neural-manifolds-work`, and
  `/data2/wangpeng/neural-manifolds-checkpoints`.
- Server deployment: exact pushed commit
  `9c3ccf71cf474bb1fb00b62318cb5be200f379be` was deployed through the verified
  local-archive fallback at
  `/data1/wangpeng/neural-manifolds-work/source/releases/9c3ccf71cf474bb1fb00b62318cb5be200f379be`.
  The archive SHA-256 and full source manifest were independently verified before
  release activation.
- Server runtime: the lock-addressed Python 3.11 environment at
  `/data1/wangpeng/neural-manifolds-work/envs/e9a37e9acafaead6a6f77d966a9e0b4ef083acd73cf0dc78ac3dd193310ce39e`
  completed with exit code 0. PyTorch `2.13.0+cu126` passed an H100 CUDA matrix
  smoke test. Bootstrap log:
  `/data2/wangpeng/neural-manifolds-checkpoints/logs/bootstrap-runtime-8e382c05491eb0145d9390f15142952c13ee72ad.log`.
- Acquisition queue: running under `acq-20260830-9c3ccf7`. Audit attempt 1
  succeeded with four independently rehashed artifacts. Last verified on 30 August
  2026 at 16:34 UTC, direct NAS acquisition attempt 1 remained active in tmux session
  `neural-manifolds-acq-20260830-9c3ccf7` (pane PID `2525228`; phase PID `2525453`).
  Queue state and log:
  `/data2/wangpeng/neural-manifolds-checkpoints/queue/acq-20260830-9c3ccf7`.
- External-data results: none.

## Integrated execution contract

The final phase order is:

```text
audit -> acquire -> qc -> preprocess -> encode -> metrics -> models -> tms
      -> locked-clinical -> fmri -> figures
```

- The metrics phase includes 100 explicit pre-encoder EEG sensor-row permutation
  repeats, plus repeated contrast-specific equal-window profiles and configured
  reliability-by-duration curves.
- The representation evaluation participants remain untouched by state-dictionary
  fitting and healthy profile calibration; prediction scores them only in
  participant-separated outer folds.
- Clinical raw releases can be acquired and included in the full physical-file
  inventory before the lock, but their signals remain unopened. After the lock
  rehashes healthy success markers and their bound artifacts through TMS,
  `locked-clinical` validates the lock, creates the clinical-only inventory, runs
  label-blind signal QC, then builds the cohort, preprocesses, encodes, and applies
  the frozen transfer.
- `locked-clinical` is a technical provenance boundary only. It is not a
  registration, preregistration, scientific gate, or result-based decision.
- The strict `ds006623` stage joins labels only after frozen encoding and produces
  discovery-calibrated fMRI-compatible R/M/D/A axes; reachability is excluded.
- Final figures run last and require healthy/model, TMS, late clinical, and fMRI
  source bundles.

The server dependency lock SHA-256 is
`e9a37e9acafaead6a6f77d966a9e0b4ef083acd73cf0dc78ac3dd193310ce39e`.

## Unresolved execution inputs

- The exact approved UKB_424 atlas, its ordered 424-row coordinate table, and the
  explicit 0- or 1-based origin of the `ds006623` LOR/ROR timing index.
- Cogitate account approval/download access and a snapshot of its native BIDS event
  schema before an adapter or contrast is enabled.
- Official participant-level clinical labels: the Figshare resting-EEG release has
  no verified stem-to-diagnosis/CRS-R key, and the Mendeley PSG release exposes
  filename diagnosis but no CRS-R/demographic covariates.
- Successful validation of the `ds006623` git-annex content path; its special remote
  may remain unavailable even when the public Git metadata clone succeeds.

## Scientific implementation hardening

The pre-execution audit defects have been corrected and retested. The implementation
now fixes DREAM to official N2/final-20-second DE-versus-NE rows while keeping DEWR
separate; preserves rejected-window gaps and nonoverlapping primary windows; excludes
direct-TMS recordings from the generic encoder; provides a dedicated, provenance-rich
TMS path; supports the sparse observed-channel clinical PSG route; and estimates
repertoire in the untruncated embedding while using a separate projected dynamics
space.

The same hardening adds participant-level fMRI pairing and inference, fold-contained
predictive scaling/tuning and representation-control imputation/scaling, exact
benchmark cell matching, frozen discovery-only representation objects and
microstates, deterministic wSMI, native and harmonised preprocessing receipts,
pretraining-overlap provenance, participant-level clinical bootstrap/permutation/FDR
inference, and the clinical signal-access firewall. QC inventories every physical
recording but opens only healthy signals before the clinical lock; event header names
may be inspected, but only onset/duration values are materialised. Clinical signal QC
starts only after the technical lock validates. These are validity and provenance
protections, not registration, preregistration, or scientific outcome gates.

## Durable run records

Each server run writes a machine-readable record under the configured restart root.
This file should be updated only with high-level state, source commit, run ID, tmux
session/job identifier, and log location. Never paste raw data, secrets, or subject
identifiers here.

- Acquisition/provenance run: `acq-20260830-9c3ccf7`
- Source commit: `9c3ccf71cf474bb1fb00b62318cb5be200f379be`
- tmux session: `neural-manifolds-acq-20260830-9c3ccf7`
- Queue log: `/data2/wangpeng/neural-manifolds-checkpoints/queue/acq-20260830-9c3ccf7/tmux.log`
- Phase log: `/data2/wangpeng/neural-manifolds-checkpoints/queue/acq-20260830-9c3ccf7/logs/acquire.attempt-0001.log`
