# Project status

This file records source and orchestration state only. No external dataset has yet
been downloaded or analysed by this project. Raw data and derived arrays remain on
the university storage system and are never committed here.

## Current state

- Project status: exploratory; not registered or preregistered.
- Scientific gates: none. Technical validation and provenance checks remain active.
- Local implementation: the complete phase graph is present. Audit/acquisition and
  deployment provenance have passed implementation hardening; later scientific
  phases remain under pre-execution audit and are not yet authorized for queueing.
- GitHub synchronization: validated implementation published on `main`; the local
  branch tracks `origin/main`.
- Server roots: confirmed as `/private_nas/wangpeng/neural-manifolds`,
  `/data1/wangpeng/neural-manifolds-work`, and
  `/data2/wangpeng/neural-manifolds-checkpoints`.
- Server deployment: exact pushed commit
  `8e382c05491eb0145d9390f15142952c13ee72ad` was deployed through the verified
  local-archive fallback at
  `/data1/wangpeng/neural-manifolds-work/source/releases/8e382c05491eb0145d9390f15142952c13ee72ad`.
  The archive SHA-256 and full source manifest were independently verified before
  publication.
- Server runtime: the lock-addressed Python 3.11 environment at
  `/data1/wangpeng/neural-manifolds-work/envs/e9a37e9acafaead6a6f77d966a9e0b4ef083acd73cf0dc78ac3dd193310ce39e`
  completed with exit code 0. PyTorch `2.13.0+cu126` passed an H100 CUDA matrix
  smoke test. Bootstrap log:
  `/data2/wangpeng/neural-manifolds-checkpoints/logs/bootstrap-runtime-8e382c05491eb0145d9390f15142952c13ee72ad.log`.
- Acquisition queue: not yet started.
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
- Clinical raw releases can be acquired and file-inventoried before the lock, but
  clinical signals are first parsed, preprocessed, and encoded inside
  `locked-clinical`, after the healthy success markers through TMS are rehashed.
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

## Pre-execution scientific implementation audit

Audit and direct NAS acquisition may run. Preprocessing and later phases remain
paused until confirmed implementation defects are corrected and retested. Current
high-priority items include DREAM N2/final-20-second DE-vs-NE selection, preserving
breaks across rejected signal windows, preventing pseudo-resolution from heavily
overlapping windows, excluding uncleaned direct TMS recordings from the general
encoder path, supporting the low-channel clinical PSG route, and estimating
repertoire in the untruncated embedding space. These are scientific-validity fixes,
not result-contingent gates.

## Durable run records

Each server run writes a machine-readable record under the configured restart root.
This file should be updated only with high-level state, source commit, run ID, tmux
session/job identifier, and log location. Never paste raw data, secrets, or subject
identifiers here.
