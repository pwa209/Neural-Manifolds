# Project status

## Larger independent DREAM report-sensitivity extension queued — 24 September 2026

A separate source-only release `df0ed9f0befa` is deployed on Alliance Rorqual
with a preflight-to-final Slurm chain `21742156`–`21742163`, including automatic
checkpointed retry passes for measurement and null tasks. At submission all
jobs were pending because requested nodes were unavailable; no signal-QC or
model result is yet claimed. Six already-downloaded open DREAM cohorts on
personal scratch provide **up to 131 distinct N2 participants before source
and signal exclusions**, with one awakening per participant, four laboratory
clusters, and leave-one-dataset-out nested analysis. The source archives are
read in place, not downloaded again or copied to GitHub. See the
[extension design](docs/DREAM_RECALL_EXTENSION.md).

This is an exploratory validation of *report/recall sensitivity*, not an
enlargement of the strict LODE experience-versus-no-experience sample. Several
sources use no recall as their negative label; children combine remembered
experience and an impression of dreaming; one source may end up to 60 seconds
before awakening. These are stated boundary conditions, not interchangeable
consciousness labels. All QC exclusions and positive, negative or unavailable
results will be retained.

## LODE external validation queued — 24 September 2026

An independent-laboratory, out-of-domain LODE validation of the **method** is
deployed on Alliance Rorqual as source release `757ef9b5e1aa` and queued through
qualification, metadata preflight, signal QC, held-out observed analysis, 999
label and 999 temporal nested null refits, and final two-test Holm synthesis.
The Slurm chain is `21742026`–`21742032`; the final job follows the null array.
These are queued jobs, not completed results. Source data are read from the
already downloaded personal-scratch DREAM archive; no raw data are copied to
GitHub. See the [fixed validation declaration](docs/LODE_EXTERNAL_VALIDATION.md).

This is an exploratory, non-preregistered external method test, **not** a
zero-shot transfer of the original Fp1/C3/O1 model. Its native bipolar pair is
fixed from label-blind channel metadata, and source bad-channel flags are
enforced. Before signal QC, 65 clear-label N2 records from 25 participants
qualify, with eight participants carrying both labels. Model findings, p-values
and any journal claim remain pending.

## Completed-run evidence review — 20 September 2026

The deployed core and supplementary high-density computation, including all
planned control replicates and both synthesis jobs, has completed with verified
output hashes. A separate exploratory review now inventories completed and
unavailable branches, evaluates null comparisons, audits conditional precision,
adds boundary multiplicity correction, maps public non-imputed subjective scales,
and diagnoses observation-length/segmentation effects in a bounded simulation.
Original empirical outputs remain unchanged. See the
[review methods](docs/COMPLETED_EVIDENCE_REVIEW.md) and reproducible scripts.

This is not preregistered research, a scientific stopping gate, or a declaration
of publication readiness. Full empirical calibration, model-refitting uncertainty,
figure QA and final manuscript interpretation are not implied by job completion.
Private result values, participant tables and operational identities are excluded
from this public status record.

## Controller and tactile recovery — 18 September 2026

The quota-interrupted controller has resumed under a source-aware supervisor.
It preserves qualified scientific source identities, checks both byte and file
headroom, and writes independent health records outside the project filesystem.
Capacity failures pause progression until headroom returns; they no longer make
the error-reporting path terminate recovery. This is not protection against a
login-host process termination or a guarantee of future shared quota capacity.

The quota-failed null replicate completed with a new output path; all 100 core
null receipts are verified. Its failed output and job history were preserved.
The repaired tactile analysis has also completed with a verified output receipt.
It explicitly audits
missing response markers and non-adaptive events, uses the source's EEGLAB
recordings, and epochs on stimulus-event timestamps without adding the
trial-relative delay twice. A later-read failure exposed a pinned source payload
shorter than its header declares. The bytes match the published annex checksum;
the entire incomplete recording is now explicitly excluded, not prefix-salvaged
or silently recoded. Raw data are unchanged. An expanded server preflight read
every eligible epoch in all complete recordings before the successful full run.
Successful TMS/report-task analyses and existing null outputs were
not rerun. The independent high-density chain was left running.

Validation: **343 local tests passed, five platform/dependency skips**; 31 targeted
cluster tests and the full-epoch tactile preflight passed. This repairs the
tactile operational blocker; it is not a claim that the full study or independent
high-density control/evidence chain is complete. See
[recovery policy](docs/QUOTA_AND_TACTILE_RECOVERY.md).

## Revised continuous deployment — 16 September 2026

The revised scientific drivers are now written and connected to the durable
controller. They cover audited cohorts, frozen/sensor/time-frequency measurements,
nested study transfer, dimensions/axis ablations, recall/stage sensitivities,
refitted nulls, signal perturbations, TMS, perception/report specificity,
PsiConnect and evidence tables/figures. See the [execution map](docs/REVISED_EXECUTION.md).

Source `b6e8e9240f67` failed cluster qualification: 319 tests passed and two
synthetic-fixture tests inherited production environment settings. The cutover
supervisor correctly left the old controller untouched. Test-only environment
isolation now prevents that contamination without disabling production checks.
Corrected release `0cf0639c40b3` passed **323 cluster tests with no skips** and
the H100 CUDA check. The revised controller is active and healthy; all six
cohort-audit jobs completed, and the first measurement job is submitted.
Four sampled Wisconsin records were successfully measured with the frozen
encoder. These are software/measurement checks, not
inferential findings.
This is not a claim that all analyses have run or that all stages are currently
in Slurm. The full local suite passed **318 tests with five platform/dependency
skips**. Downstream measurements, transfer, recovery, sensitivities, 100 null
replicates, robustness and evidence synthesis are connected through durable
dependencies. TMS, perception/report and context branches become eligible when
their respective downloads finish. They are not all simultaneously submitted.
Seventeen DREAM releases are complete; the TMS release and the separate bounded
PsiConnect EEG/behaviour subset are being acquired.

A PsiConnect full-pointer checkout hit the shared project file-count quota and
interrupted the controller. Only unmaterialized, Git-recoverable MRI pointers
were removed; no acquired data were deleted. The subset acquirer now configures
sparse checkout before checkout and passes its targeted local and cluster test.
It is deployed separately as `b19236dbac7f`. The scientific controller resumed
from the unchanged qualified release and reconciled completed jobs successfully.

Remaining scope boundaries: optional clinical/fMRI extensions are not activated;
the public subjective-scale mapping for PsiConnect has not been adjudicated;
formal sample-structure power/precision analysis and complete empirical source
validation remain outstanding. The executable core is not a claim that every
scientific objective in the design document has already been implemented or run.

The real-data encoder canary exposed and repaired mismatched pretraining versus
fine-tuning construction. Strict backbone loading now retains the pretrained
normalization. Twelve sampled records across three cohorts were measured.
Sampled Paris records have undocumented EDF voltage units; no amplitude-based
guess is used to make them pass. Wisconsin's newly downloaded source has verified
microvolt headers and a documented, explicitly approximate sparse electrode map.

The records below are historical and superseded by this candidate's execution map.

## QC repair deployment — 16 September 2026

Release `ecf87f2f1c1f` passed **305 cluster tests with no skips** and the H100
CUDA check. Local validation passed **300 tests with five platform/dependency
skips**; lint and formatting checks passed. The release is deployed with a healthy
controller after the preceding controller's remaining job finished. Targeted
cluster rechecks successfully inspected the annotation-error recording and all
seven recordings in the previously unsupported RAR dataset, plus all eight
invalid-clock EDFs. Raw files are unchanged; repairs operate on temporary
QC copies or reader options, with explicit provenance. Successful diagnostic
decoding does not establish scientific eligibility or annotation semantics.

Four ZIP members remain unavailable. Fresh SHA-256 and upstream MD5 checks match;
both Python and libarchive fail to decode these members with valid size/CRC. The
latest public metadata still reports the same source versions and hashes. No CRC
checks were bypassed. See [QC repair policy](docs/QC_REPAIRS.md).

These are diagnostic-reader repairs, not completion of the revised study.
Cross-study, TMS, specificity and synthesis integrations remain unfinished. The
assistant monitor remains deleted; remote early-stage progression is independent.

## Continuous execution — 16 September 2026

The updated continuous controller release `5ee3a29b9ac0` passed **298 tests on
Rorqual, with no skips**, plus the H100 CUDA check. Local full-suite validation
passed 292 tests with five platform/dependency skips; the subsequent targeted
scheduling/QC suite passed all ten tests. The new controller has been activated
after stopping the prior scheduler and preserving acquisition retry counters.
It retains the two-CPU-job concurrency limit. Eleven original-source releases
were verified downloaded at the latest check, with another downloading.
Acquisition continues separately without deleting partial data. The earlier
downloader termination cause remains unknown.

The hourly assistant monitor has been deleted at the user's request. The remote
controller remains independent of it. The entire revised scientific pipeline is
not yet implemented or queued.
Current automatic work covers acquisition supervision, per-release metadata/header
inventory, a dependent length-matched AR(1) recovery component, and sampled signal
QC. Analysis-window QC, axis-specific recovery, revised measurement/model
integration, TMS/specificity and final evidence synthesis remain implementation
work. The new signal-QC worker is qualified and deployed; its checkpoints preserve explicit
unavailable records and do not imply cohort admission. The pinned core encoder
source and checkpoint have now been prestaged and checksum-recorded on Rorqual.
See the
[continuous execution runbook](docs/CONTINUOUS_EXECUTION.md) for exact boundaries.

The source audit also identified a report-semantics limitation in DREAM set 1:
numeric 0/2 coding does not establish the strict DE-versus-NE distinction when
experience without recall was not separately elicited. The raw codes are preserved;
`configs/dream_adjudication.yaml` records its sensitivity-only disposition and the
unresolved segment timing. This new adjudication is for the next scientific adapter
integration; the current controller performs no inferential analysis.

The current public documentation/adjudication is newer than the deployed source
snapshot. Deployed controller runtime code matches the tested release. No empirical
study results have been produced, and no result-dependent go/no-go gates are introduced.

## September revision and fresh Compute Canada execution

The [revised scientific design](docs/STUDY_REVISION_20260916.md) and
[Alliance roadmap](docs/ALLIANCE_ROADMAP.md) supersede the execution priorities
and live-status claims below. Earlier acquisition status is historical, not a
current server check. The revised study is exploratory, non-preregistered and
open-data-only, with no scientific result gates.

Rorqual access, account association, modules and storage were checked through
interactive password/Duo authentication. A dedicated fresh project and scratch
namespace have been created. Corrected source release `8b3f69f3d34a` was
transferred and checksum-verified; it reuses the newly built environment
`17359bbe1066` after verifying identical dependency inputs. The source resolver, bounded acquisition worker, source manifest
verification and Slurm qualification script are implemented. New nested
study-transfer and incremental TMS prediction routines passed synthetic tests.
No revised empirical result exists. The full local suite passed **282 tests**
with five platform/dependency skips. Targeted lint and formatting checks passed.
**Cluster qualification passed:** 288 tests, no skips, and an NVIDIA H100
forward/backward CUDA smoke check. The first run exposed Linux directory-move,
MNE API and pandas test-typing issues; these were repaired and the entire suite
rerun successfully. Both source and environment identities are retained remotely.
The DREAM registry has 18 directly downloadable constituent candidates and one
additional public-provider route awaiting audit. They are acquisition candidates,
not yet approved for any particular scientific contrast. Raw acquisition has
resumed from preserved partial downloads. No empirical analysis is running.
The public checkout's status document is newer than the deployed snapshot;
runtime code matches that release. Credentials, account identifiers, private
storage paths, authentication queues, logs and data are excluded from Git.

## Historical August implementation record

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
