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
  be launched one at a time from the exact deployment after their technical inputs
  are present; no result threshold controls that sequence.
- Local verification: 274 repository tests passed, with five documented local
  capability/dependency skips; Ruff lint, Ruff format, and `git diff --check` are
  clean.
- Git source: this tracked change set combines the scientifically hardened pipeline
  with the sanitized operations contract. Active server queues remain pinned to the
  earlier exact releases that created them; this change set does not replace or
  modify those releases.
- Server roots: the canonical NAS, work, and checkpoint roots are confirmed. Public
  documentation denotes them as `<CANONICAL_ROOT>`, `<WORK_ROOT>`, and
  `<CHECKPOINT_ROOT>`. For new runs from the sanitized release, resolved values live
  in an external server-only config selected by `--server-config`, validated against
  explicit roots, and hash-bound to the run.
- Server deployment: the exact pushed scientific commit is active as a
  content-addressed release. Because direct repository access was unavailable, the
  authorized archive transport was used; the archive checksum, embedded commit, and
  deployed manifest were independently reverified before activation. Operational
  hashes and release paths remain in server-only durable records.
- Server runtime: the lock-addressed Python 3.11/CUDA 12.6 environment was
  revalidated and reused for the scientific release. Accelerator/runtime
  compatibility validation passed. The exact hardware inventory, runtime path, lock
  checksum, scheduler/tool inventory, and bootstrap log are retained in server-only
  durable records.
- Acquisition queue: direct-to-NAS acquisition is active and remains pinned to its
  immutable acquisition-safe release. The audit completed successfully, and the
  queue has finalized release content while continuing the current transfer. The
  run, session, process, queue, and log identifiers remain in server-only durable
  records.
- Scientific queue: the scientifically hardened exact release completed its audit
  successfully, including independent rehashing of receipt-bound artifacts. The
  audit session exited normally, and all later phases remain pending. Exact run and
  log identifiers remain in server-only durable records.
- Migration boundary: the external-config interface is not retroactive. Existing
  queues must be monitored with the immutable release/configuration that created
  them. The first queue launched from the sanitized release requires a new run ID.
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

The server dependency lock is checksum-pinned; its verified digest is retained in
the server-only deployment provenance.

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
- Core model materialisation remains pending: the pinned LaBraM and BrainLM source
  checkouts and LaBraM checkpoint are not yet cached. The check-only contract passed,
  but outbound GitHub access timed out while acquisition was active. Retry after
  acquisition or use a separately hash-verified transport before `encode`; do not
  substitute an unpinned model.

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

Each server run writes a machine-readable record under `<CHECKPOINT_ROOT>`. Exact
source commits, run IDs, session/job and process identifiers, operational hashes,
attempt numbers, timestamps, and log paths are server-only durable records and must
not be copied into public status files.

The public durable state is: the scientific commit is deployed, the scientific
audit passed, direct-to-NAS acquisition is active, no external-data results exist,
and all later scientific phases remain pending.
