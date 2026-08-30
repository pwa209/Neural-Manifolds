# Project status

This file records source and orchestration state only. No external dataset has yet
been downloaded or analysed by this project. When execution begins, raw data and
derived arrays will remain on the university storage system.

## Current state

- Project status: exploratory; not registered or preregistered.
- Scientific gates: none. Technical validation and provenance checks remain active.
- Local implementation: integrated for the complete phase graph and its technical
  boundaries; external-data execution remains unstarted.
- GitHub synchronization: validated implementation published on `main`; the local
  branch tracks `origin/main`.
- Server roots: the exact canonical, work, and checkpoint project directories are
  still unconfirmed.
- Server deployment: not yet started.
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

- User confirmation of all three exact project roots: canonical NAS, active work,
  and checkpoint/log storage.
- The exact approved UKB_424 atlas, its ordered 424-row coordinate table, and the
  explicit 0- or 1-based origin of the `ds006623` LOR/ROR timing index.
- Cogitate account approval/download access and a snapshot of its native BIDS event
  schema before an adapter or contrast is enabled.
- Official participant-level clinical labels: the Figshare resting-EEG release has
  no verified stem-to-diagnosis/CRS-R key, and the Mendeley PSG release exposes
  filename diagnosis but no CRS-R/demographic covariates.
- Successful validation of the `ds006623` git-annex content path; its special remote
  may remain unavailable even when the public Git metadata clone succeeds.

## Durable run records

Each server run writes a machine-readable record under the configured restart root.
This file should be updated only with high-level state, source commit, run ID, tmux
session/job identifier, and log location. Never paste raw data, secrets, or subject
identifiers here.
