# Neural Manifolds implementation roadmap

## Operating premises

This is an exploratory, non-preregistered project. The workflow contains no
scientific go/no-go gate and must preserve negative, mixed, and inconclusive
results just as faithfully as supportive ones. Technical requirements—correct
host, explicit storage roots, source/data hashes, valid output receipts, participant
separation, and successful integrity tests—remain mandatory because they protect
the result rather than select it.

The publication ambition is *Nature* or *Science*. A complete cross-state,
perturbational, and held-out clinical evidence chain is the intended standard;
*Nature Human Behaviour* is a natural destination if the evidence is rigorous but
the cross-modal or clinical reach is weaker. Journal choice never changes queued
analyses after results are seen.

## Current deployment boundary

As of 30 August 2026, the university compute endpoint, accelerator/runtime
compatibility, and required execution tooling have been validated. Exact hardware,
scheduler/tool inventory, account, host, tunnel endpoint, and port details are
retained only in the authenticated server environment. The tracked
`configs/server.yaml` is a schema-valid, non-operational synthetic template. Resolved
identity and root values for new sanitized-release runs live in an external
server-only config selected by `--server-config`, validated against the explicit
roots, and hash-bound to the run. Capability and auxiliary scheduler fields are
provenance; preflight separately validates the actual runtime and tool inventory.
The three project-specific roots are represented publicly by placeholders:

| Role | Confirmed root | Purpose |
|---|---|---|
| Canonical | `<CANONICAL_ROOT>` | immutable raw releases, licences, acquisition manifests, snapshots |
| Work | `<WORK_ROOT>` | source releases, environments, caches, preprocessing, embeddings, metrics, models, figures |
| Checkpoint | `<CHECKPOINT_ROOT>` | queue state, attempt receipts, restart markers, durable logs |

Each root is one direct, project-specific child of its parent. Scripts reject the
parent itself, nested guesses, relative paths, and a root copied from another
project. Keep the resolved config as a regular non-symlink YAML file under an
approved server-only staging location, outside raw storage and the deployed source
tree, for the initial preflight/bootstrap. Export its absolute path as
`NEURAL_MANIFOLDS_SERVER_CONFIG`. After the roots exist, place an unchanged stable
copy under non-raw project metadata storage, update the variable, pass that same path
explicitly with `--server-config` to queue launch/status commands, and pass the exact
configured roots explicitly to every remote script.

This interface begins with the sanitized release and is not retroactive. Active
queues remain bound to the immutable earlier release/configuration that created
them and must be monitored from that release. Use a new run ID for the first queue
launched from the sanitized release; do not migrate an active run in place.

The active scientific source is an exact pushed public commit. The server's outbound
repository probe timed out, so that same commit was deployed through the authorized,
independently hash-verified, unprefixed `git archive` transport. Its embedded commit
and deployed-source manifest were reverified on the server. The pinned Python
3.11/CUDA 12.6 runtime was revalidated and reused. The scientific run completed its
audit successfully; all later phases remain pending. Exact commit, archive hash,
run, release, and log identifiers remain in server-only durable records.

An earlier immutable acquisition-safe release continues only the already-running
direct-to-NAS acquisition. Its process command remains pinned to that release even
though `source/current` now selects the hardened scientific commit. No second
acquisition is started concurrently. Once the original acquisition completes, the
scientific run can revalidate and reuse verified raw releases without copying data
through GitHub or the workstation. Exact acquisition identifiers remain server-only.

## Reproducible storage contract

Given resolved roots `<CANONICAL_ROOT>`, `<WORK_ROOT>`, and `<CHECKPOINT_ROOT>`, the
fixed layout is:

```text
CANONICAL/
  raw/<dataset>/<immutable-release>/       downloaded directly here
  manifests/                              URLs, versions, licences, SHA-256 inventories
  licences/
  snapshots/

WORK/
  source/releases/<git-commit>/            content-addressed source checkout
  source/current -> releases/<git-commit>
  envs/<requirements-lock-sha256>/
  cache/                                   regenerable model/download caches
  runs/<run-id>/
    qc/ preprocess/ embeddings/ metrics/ models/ tms/ clinical/ fmri/ figures/

CHECKPOINT/
  queue/<run-id>/
    queue.lock
    events.jsonl
    clinical_lock.json                     created only before DoC transfer
    logs/<phase>.attempt-NNNN.log
    phases/<phase>/success.json
    phases/<phase>/attempts/NNNN/
```

Raw files never transit through the local workstation or fast work storage. A
dataset downloader writes partial files beside the final NAS target, verifies the
release inventory, then atomically publishes the final release directory. Raw
release files become read-only after their complete manifest validates. Derived
data stay under `<WORK_ROOT>`; only small durable queue records and logs go under
`<CHECKPOINT_ROOT>`.
GitHub contains only code, configuration, tests, checksum metadata, roadmap, and
operational status. No empirical raw or derived data, participant-level outputs,
aggregate analysis tables, figure source-data tables, model caches, durable logs,
or credentials enter GitHub.

## Phase graph

The integrated queue has one final canonical order:

```text
audit -> acquire -> qc -> preprocess -> encode -> metrics -> models -> tms
      -> locked-clinical -> fmri -> figures
```

This keeps costly work restartable and makes the clinical and fMRI panels genuinely
late inputs to the final figures. The order is a technical execution contract, not
a scientific gate.

| Order | Workflow phase | Dispatcher phase | Primary storage | Main completion evidence |
|---:|---|---|---|---|
| 1 | `audit` | `audit` | NAS + work | release/licence/pretraining-overlap audit and hashed inventory |
| 2 | `acquire` | `acquire` | NAS only | immutable release manifests with verified file hashes |
| 3 | `qc` | `qc` | work | full label-free physical inventory and healthy-only signal-QC flow |
| 4 | `preprocess` | `preprocess` | work | primary/native/CSD/sleep derivatives, selectors, availability, and receipts |
| 5 | `encode` | `encode` | work/GPU | deterministic windows, frozen LaBraM embeddings, and label-leakage receipts |
| 6 | `metrics` | `metrics` | work/CPU+GPU | five-axis metrics, benchmarks, representation controls, nulls, and sampling summaries |
| 7 | `models` | `models` | work/CPU | participant-level contrasts, held-out predictions, uncertainty and ablations |
| 8 | `tms` | `tms` | work/CPU+GPU | passive/direct reachability comparison and post-pulse dynamics |
| 9 | `locked-clinical` | `clinical` | work/GPU | post-lock clinical signal QC, parsing, preprocessing, encoding, and frozen transfer |
| 10 | `fmri` | `fmri` | work/GPU | strict-manifest secondary BrainLM propofol-fMRI triangulation |
| 11 | `figures` | `figures` | work | final healthy, perturbational, clinical, and fMRI figures/source data |

Dependencies are technical only. A phase advances after its command exits normally
and every declared receipt artifact matches its recorded size and SHA-256. It does
not inspect effect signs, P values, classification scores, or journal potential.

## Phase-by-phase research plan

### 1. Audit and source freeze

Inputs are `configs/study.yaml`, the dataset catalogue, model-weight catalogue, and
the exact deployed Git commit. Audit every DOI/repository release, licence, access
method, expected participant/recording count, modality, label map, channel metadata,
and known missingness. Record redirect-resolved source URLs, retrieval timestamps,
repository versions, and checksums without reading outcome labels for exclusion.

For LaBraM and BrainLM, record upstream repository commit, weight URL, licence,
weight SHA-256, architecture, selected layer/pooling, and pretraining-corpus audit.
Confirmed or unresolved target-dataset overlap must be represented in the audit
output and handled by the configured representation control; it is not a reason to
silently remove evidence after inspection.

Completion artifacts: a dataset audit table, model audit table, configuration hash,
source manifest hash, and machine-readable issue ledger. Unresolved access/licence
items remain explicit acquisition blockers for that dataset only, not scientific
gates for the project.

### 2. Direct NAS acquisition

Acquire the fixed public releases listed in the study catalogue:

- OpenNeuro `ds005620` propofol repeated awakenings/TMS-EEG;
- DREAM/Tononi Serial Awakenings;
- OpenNeuro `ds001785` near-threshold tactile detection;
- somatosensory report/task-relevance HD-EEG;
- Cogitate EEG/MEG;
- OpenNeuro `ds006110` PsiConnect;
- Figshare resting EEG in prolonged disorders of consciousness;
- Mendeley DoC polysomnography;
- OpenNeuro `ds006623` propofol fMRI.

Each adapter must support `--check-only`, `--dry-run`, bounded retries, partial-file
resume, immutable release names, and a final checksum inventory. Download into
`CANONICAL/raw/.staging/<dataset>/<release>` and publish
`CANONICAL/raw/<dataset>/<release>` only after validation. Never copy raw files to
the Git repository, workstation, `<WORK_ROOT>`, or `<CHECKPOINT_ROOT>`. Capture repository-native
metadata before any conversion. A failed transfer is resumed from its partial
state; an existing verified release is skipped.

Completion artifacts are small NAS manifest files listed in the phase receipt—not
an unverified claim based on downloader exit status. Restricted, withdrawn, or
licence-incompatible material is reported and left untouched rather than bypassed.

### 3. Metadata and signal QC

Build one label-free, recording-level physical-file catalogue across healthy and
clinical releases, covering participants, sessions, task/acquisition identifiers,
modalities, source members, and sidecar availability. Before the technical clinical
lock, signal-level QC opens only the healthy inventory subset; clinical recordings
remain file-inventoried but unopened. Condition/outcome values remain in the separate
cohort label view. Validate BIDS where present and preserve native metadata elsewhere.
Make every QC exclusion blind to the contrast outcome.

The implemented signal-QC reader accepts only a scoped label-free inventory and
rejects condition, diagnosis, target, and outcome fields before opening signal. For
every readable recording it hashes all physical source members (including BrainVision
header, marker, and binary signal companions and external EEGLAB data), then records
deterministic evenly spaced signal samples, channel/montage/reference metadata,
auxiliary EOG/ECG/EMG availability, event-onset integrity, sidecar status, bad-channel
diagnostics, and artefact-window burden. Event TSV handling reads the header first and
then materialises only `onset` and, when present, `duration`; trial type, response,
condition, and outcome values never enter the QC process. Review flags are retained
but are not exclusions. Only unreadable, non-finite, empty/too-short, or non-EEG inputs
receive a technical exclusion, with the reason written to the flow table. The
preprocessing receipts bind the exact scoped inventory and QC-flow hashes, and no
condition/outcome value is consumed or reproduced by QC.

Implement the study's minimum-duration/trial/channel/window rules, flatline and
clipping checks, event synchronization checks, montage recoverability, and
dataset-specific exceptions. Write every inclusion/exclusion and reason to a flow
table. Never delete raw data. Unit of inference is participant; window/trial counts
are precision metadata, not independent sample sizes.

### 4. Preprocessing and harmonisation

Create both the 19-channel harmonised EEG track and native/full-montage sensitivity
track. Apply resampling, filtering/notch choice, reference and CSD variants, bad
channel/interpolation rules, and configured branch selection without using condition
labels to fit a signal transform. Direct-TMS recordings are rejected from this
generic path: pulse-gap interpolation, epoching, and post-pulse processing occur only
inside the dedicated `tms` stage. Deterministic windows, masks, and artifact-window
rejection are generated during `encode`, which preserves segment boundaries so zero
padding and rejected gaps never become signal.

The production implementation now materialises the configured canonical 19-channel,
average-referenced branch as the healthy primary input. Missing and signal-bad
canonical electrodes count together against the configured interpolation fraction;
when the complete harmonised montage is required, they are reconstructed only from
the fixed standard 10-20 coordinates and the final ordered channel list is checked
exactly. The sparse clinical PSG branch remains a declared exception: it retains
only observed electrodes and permits no interpolation.

Each source and selected analysis unit also receives a separately hashed
native/full-montage average-reference sensitivity whenever the signal and channel
metadata permit it. A CSD derivative is emitted only when the configured channel
count and electrode-position fraction are met and MNE completes the transform. A
missing or invalid montage makes CSD explicitly unavailable with a per-unit reason;
it never excludes the primary average-reference derivative. Label-free `psg`
modality membership auditably activates the configured 0.3-Hz high-pass sensitivity,
while other modalities record that branch as not applicable. Auxiliary EOG, ECG,
and EMG inventories and potential EOG/ECG support are preserved, but the fixed
generic policy reports ICA as not performed and records that auxiliary channels were
not used for cleaning. It therefore cannot silently convert channel availability
into a claim that ICA occurred.

The integrated stage writes label-free preprocessed FIF derivatives plus selector,
flow, branch-availability, and receipt manifests on fast work storage. The manifests
retain selector, branch, shape, channel, preprocessing-hash, and raw-input checksum
lineage; labels remain in a separate manifest until after encoding. Synthetic and
small smoke datasets establish technical correctness, not favourable method
selection.

Primary, native-average, native-CSD, and sleep-high-pass files each have an atomic
content receipt binding the complete physical recording inventory, QC-flow hash,
selector, analysis branch, sleep-modality decision, and full preprocessing
configuration hash. Reuse recursively rehashes every available derivative and its
receipt; a changed source companion, configuration, selector, QC decision, branch
status, derivative, or receipt requires a new run directory. Availability counts and
reasons are descriptive execution records only and never scientific result gates.

### 5. Frozen representation

Verify the LaBraM weight hash before GPU use. Keep parameters frozen, disable label
access in the encoding process, use configured patch/mask handling, and write
participant/session/condition trajectories in deterministic chunks. Capture CUDA,
driver, PyTorch, model commit/hash, selected layer, pooling rule, montage, seed, and
input/output shapes. Resume at participant chunks; never recompute validated chunks.

The metrics phase publishes a representation-control availability ledger before any
control is described as an empirical trajectory. The current exact, rehashed branch is
the configured LaBraM-Base final-pre-head/valid-token pooling trajectory. Secondary
pooling, non-primary layers, alternate checkpoint sizes, PCA coordinates, and full
time-frequency coordinates remain explicitly unavailable/not generated until an exact
hash-pinned checkpoint and extraction backend is configured and materialised; scalar
spectral, complexity, connectivity, and wSMI benchmarks are referenced as implemented
comparators but are not relabelled coordinate trajectories.

Dataset-identity decodability is a non-gating confound diagnostic over
participant-condition five-axis cells and the exact complete conventional-benchmark
cells that map back to them. Every fold is participant-separated; median imputation,
scaling, participant/class weighting, and the fixed logistic classifier are fitted only
on training participants. Reports retain balanced accuracy and AUROC only where
defined, participant-cluster bootstrap intervals, participant-label permutation nulls
with plus-one p-values, fold-level disjointness hashes, and structured unavailable
reasons. Participant identifiers, row predictions, raw signals, and coordinate arrays
are not published by this diagnostic.

The healthy-reference split is participant-disjoint: discovery fits the frozen
state/profile objects, validation selects technical state stability, and a third
representation-evaluation partition remains untouched by both. Model prediction
scores that evaluation partition only through participant-separated outer folds;
it is never recycled into representation fitting or profile calibration.

### 6. Manifold metrics, nulls, and benchmarks

Estimate the five properties separately at participant-condition level:

1. structured repertoire—participation ratio, k-NN intrinsic dimension/entropy,
   local anisotropy and surrogate correction;
2. metastability—persistence/revisability, HMM or switching-state stability, and
model-free recurrence replication;
3. directionality—entropy production, arrow-of-time decoding, transition asymmetry,
   and motif recurrence under stationarity controls;
4. alignment—lagged shared predictive/communication subspaces with zero-lag and CSD
   controls;
5. reachability—regularized local dynamics and finite-horizon stochastic Gramian,
   later anchored to TMS.

Implemented conventional comparators are relative spectral power, spectral slope,
permutation entropy, Lempel-Ziv complexity, connectivity summaries, and deterministic
weighted symbolic mutual information. The wSMI implementation fixes ordinal order
three, a 32 ms lag, a 10 Hz zero-phase low-pass, zero weights for identical and
sign-reversed symbols, at least 180 complete symbols, and the median across unique
channel pairs; every row records the realized lag, sample count, pair count, and
availability reason. Microstate prototypes are fit without condition labels only on
representation-discovery participants, with participant-balanced GFP-peak maps and a
fixed channel order, then frozen for validation/evaluation application. Missing split,
channel, or finite-sample prerequisites produce a structured unavailable status, never
per-condition clustering. PCIst remains explicitly unavailable until a validated
backend is supplied. The benchmark ledger retains every manifest unit and hashes each
expected participant-condition cell; a conventional prediction is unavailable unless
its eligible cell-key set exactly equals the corresponding five-axis estimand. Current
technical null machinery records phase randomization,
blockwise temporal permutation, post-encoder latent rotation, and covariance/dwell-
matched state-space simulations with per-repeat error auditing, alongside the separate
true pre-encoder control below. These implementations must pass their technical and
scientific validation before any family is described as an empirical result.
Synthetic recovery tests verify known dimension, dwell time, irreversibility,
communication subspace, and reachability before interpreting empirical estimates.

The integrated metrics phase additionally runs two explicit sensitivity systems:

- 100 true pre-encoder EEG sensor-row permutation repeats. Each repeat permutes
  signal rows while retaining the original channel-name order and passes the
  mismatched input through frozen LaBraM. Labels are opened only after all missing
  repeat encodings complete. This is distinct from the post-encoder latent-rotation
  null.
- Contrast-specific repeated equal-window matching plus reliability-by-duration
  curves. Sampling is deterministic and participant-safe, preserves temporal order
  and trial/segment boundaries, synchronizes coarse trajectory and fine alignment
  spans, and records unavailable/error rows rather than imposing a scientific gate.

### 7. Participant-level biological models

Keep explanatory targets separate: conscious level, experienced content,
report/task relevance, and psychedelic organization. Representation-discovery
participants fit the projection, state dictionary, and healthy profile objects once;
validation selects technical state stability, after which those objects remain
frozen. No participant's windows may cross train/validation/test boundaries. The
primary five-axis classifier fits scaling and tunes its logistic regularization only
inside participant-separated training folds. Median imputation and scaling for the
separate representation-control diagnostic are likewise training-fold-contained;
neither path refits the frozen representation objects. Calibration slope and
intercept are out-of-fold evaluation diagnostics, not a fitted recalibration model.

For DREAM/Tononi, the primary experienced-content contrast is fixed to the final
20 seconds before an awakening from N2: definite experience with recalled specific
content (`DE`, release code 2) versus no experience (`NE`, release code 0). Experience
without recalled content (`DEWR`, code 1) remains a distinct secondary category and
is never merged into DE. An awakening time, recording duration, and last-stage code
must all be present and valid for the primary contrast; absent timing or stage
metadata makes that unit explicitly ineligible rather than invoking a fallback
window or inferred sleep stage.

For categorical contrasts, repeated equal-window profiles are the primary estimand.
All-available participant-condition profiles and conventional-feature comparisons
are separately identified sensitivities and can never be relabelled primary when
matched profiles are absent or invalid. Repeated observations additionally receive
a participant-random-intercept model when at least three participants support that
design; otherwise the mixed-model component is explicitly unavailable.

Report AUROC, AUPRC, balanced accuracy, Brier score, ECE, and out-of-fold calibration
slope/intercept; these are evaluation diagnostics, not a recalibration fit. Axis,
omnibus, and predictive metrics carry 95% participant-cluster
bootstrap intervals. Axis equivalence uses a 90% two-one-sided bootstrap interval
against `statistics.continuous_smallest_effect`; non-significance alone is never
called equivalence. Report pairwise axis correlations and conditioning, explicit
leave-one-property-out metric deltas, and leave-one-dataset-out performance only
when at least two datasets contain both arms and untouched evaluation observations.
Conventional baselines must have exactly the same eligible participant-condition
cell keys and sample counts as their five-axis sensitivity; any missing or extra cell
makes the comparison unavailable rather than silently shrinking either analysis.
Preserve every result and unavailable reason; no threshold controls whether the TMS,
clinical, or fMRI phase runs.

Prediction keeps representation-discovery/validation participants as fixed fitting
rows and scores only untouched evaluation-eligible participants in outer folds.
Participant-condition cells, not windows, receive equal weight.

### 8. Direct perturbational validation

For `ds005620`, align pulse timing, remove/reconstruct the immediate pulse artefact
without leaking post-pulse outcomes, and quantify trajectory spread,
differentiation, propagation, and recovery across wake and propofol. Test whether
passive reachability estimated from pre-pulse/spontaneous activity predicts direct
TMS response at participant-condition level, alongside implemented conventional
baselines. Record pulse counts, rejected epochs, interpolation windows, sensors and
time ranges for every estimate, while retaining an explicit PCIst-unavailable status
until a validated backend exists.

Pulse gaps are interpolated once on continuous EEG before filtering and epoching.
The passive predictor excludes direct-TMS acquisition rows, and associations use
participant-level awake-minus-propofol deltas. PCIst remains explicitly unavailable
unless a validated backend is supplied; its absence is not silently substituted.

### 9. Technically locked clinical transfer

Immediately before clinical execution, the queue creates
`CHECKPOINT/queue/<run-id>/clinical_lock.json`. It hashes the source manifest,
study/dataset/server configurations, and every successful healthy-phase marker
through TMS.
This is an implementation-provenance snapshot only—not a public registration,
preregistration, scientific gate, or claim that healthy findings were favourable.

Clinical raw releases may have been acquired and included in the full physical-file
inventory earlier, but their signals are deliberately excluded from healthy signal
QC, preprocessing, and encoding. Only after the lock validates does
`locked-clinical` create a clinical-only inventory, run the same label-blind signal-QC
contract, parse the two DoC releases, construct their cohort manifests, preprocess
and encode their signals, and apply the frozen healthy objects.

Apply the frozen encoder, projection/state dictionary, and healthy-calibrated profile
estimator to the Figshare resting-EEG and Mendeley PSG DoC resources without
refitting. The Mendeley PSG release follows a dedicated sparse clinical montage
branch (observed frontal, central, and occipital channels; no interpolation) instead
of the healthy `>=15`-channel encoder eligibility rule. Each output records which
five-axis properties are unavailable or limited by the observed montage.

Axis-wise profiles remain the primary clinical transfer outputs. The only composite
similarity summary is the proposal-defined secondary log-likelihood ratio under the
frozen paired healthy wake versus propofol-sedation reference distributions; positive
values are more wake-like and are never interpreted as probabilities of
consciousness. Missing axes or a missing frozen reference make this summary
explicitly unavailable—no axis or endpoint is imputed. Associate the axis-wise
profiles and secondary likelihood ratio with diagnosis and available CRS-R at the
participant level. CRS-R associations use Spearman rank correlation; diagnosis uses
Kruskal-Wallis with epsilon-squared. Both carry participant bootstrap intervals and
plus-one participant-label permutation P values, with Benjamini-Hochberg correction
within each dataset and endpoint. P values and FDR decisions are reported but never
control execution. Do not automatically reclassify individuals. Report discordant
cases. Missing official label keys remain missing; they are never inferred from file
order or signal features.

### 10. Secondary fMRI triangulation

Rehash the already pinned BrainLM source and checkpoint, then require the strict
`ds006623` release/manifest builder: validated release receipt and inventory;
audited BOLD/confound/run pairing; explicit LOR/ROR timing-index origin; explicit
UKB_424 atlas and ordered 424-row coordinate table; run-safe segmentation; and
labels joined only after frozen signal encoding. Discovery participants alone fit
normalization and secondary-axis calibration.

The fMRI output contains modality-compatible, discovery-calibrated R/M/D/A axes:
repertoire, transition-based metastability, directionality, and lagged alignment.
Reachability is explicitly excluded because passive fMRI does not provide direct
perturbational controllability. Treat the stage as secondary cross-modal
triangulation with its own limitations, not an EEG-estimator substitution.

Collapse runs with equal weight inside each participant-condition cell before any
condition comparison. Using only the explicit effect-site concentration and audited
LOR/ROR timing labels, estimate propofol-minus-wake and post-LOR-minus-post-ROR
paired differences separately within discovery, validation, and test partitions.
Bootstrap participant pairs and use participant-level sign-flip permutations with
the plus-one correction; when pairing is insufficient, publish the unavailable
status and issue ledger without substituting windows or runs as observations.
Because the EEG and fMRI samples are unrelated cohorts, describe agreement only as
independent-cohort triangulation. Never calculate participant-level EEG-fMRI
correlations unless a separate, explicit participant mapping has been verified.

### 11. Final figures and source data

Render figures only after healthy modelling, TMS, technically locked clinical
transfer, and fMRI triangulation have completed technically. The final bundle covers
study logic, five-axis healthy profiles, content/report dissociation, direct
perturbation, representation controls, nulls, equal-window/reliability sensitivity,
late clinical transfer, fMRI R/M/D/A triangulation, and ablations.

Every plotted datum must be recoverable from a small source-data table with a
producing artifact hash and script/config version. Display individual participants,
uncertainty, dataset/site structure, unavailable analyses, and discordant outcomes
rather than only group averages. Automated rendering checks cover missing panels,
clipped labels, illegible type, colour-blind distinguishability, and raster/vector
export. No external-data figure exists until the server phases actually run.

## Remote execution sequence

The following is a template; replace angle-bracket values only with user-confirmed
paths and an exact pushed Git commit. Never insert a password into these commands.

```bash
export NEURAL_MANIFOLDS_SERVER_CONFIG=<SERVER_ONLY_CONFIG>

bash scripts/remote/preflight.sh \
  --canonical-root <CANONICAL_ROOT> \
  --work-root <WORK_ROOT> \
  --checkpoint-root <CHECKPOINT_ROOT>

bash scripts/remote/bootstrap.sh \
  --canonical-root <CANONICAL_ROOT> \
  --work-root <WORK_ROOT> \
  --checkpoint-root <CHECKPOINT_ROOT> --check-only
# Review, then repeat with --apply.

bash scripts/remote/deploy_from_git.sh \
  --canonical-root <CANONICAL_ROOT> \
  --work-root <WORK_ROOT> \
  --checkpoint-root <CHECKPOINT_ROOT> \
  --repository https://github.com/pwa209/Neural-Manifolds.git \
  --commit <EXACT_PUSHED_COMMIT> --check-only
# Review, then repeat with --apply.
```

The direct Git check accepts an exact commit only while an approved remote ref
currently advertises that object ID. If it is not advertised, or the approved
repository cannot be reached from the server, create an unprefixed
`git archive --format=tar` for that same reviewed, pushed commit on a trusted
machine, record its SHA-256 independently, transfer it without embedding
credentials, and use `scripts/remote/deploy_from_archive.sh` with `--archive`,
`--archive-sha256`, and the same exact repository/commit. Run `--check-only`, then
`--dry-run`, then `--apply`. This is an explicit transport fallback, never an
automatic downgrade or a substitute for the pushed Git commit.

Build the Python environment only from a complete requirements lock whose SHA-256
has been independently recorded:

```bash
bash scripts/remote/bootstrap_runtime.sh \
  --canonical-root <CANONICAL_ROOT> \
  --work-root <WORK_ROOT> \
  --checkpoint-root <CHECKPOINT_ROOT> \
  --repo-root <WORK_ROOT>/source/releases/<EXACT_PUSHED_COMMIT> \
  --python <ABSOLUTE_PYTHON_3_11> \
  --requirements-lock <ABSOLUTE_REQUIREMENTS_LOCK> \
  --lock-sha256 <REQUIREMENTS_LOCK_SHA256> \
  --check-only
# Review, then repeat with --apply and retain the reported runtime Python path.
```

Materialise the core model cache next. This clones LaBraM and BrainLM source at the
exact Git object IDs in `configs/models.yaml`, but downloads only the LaBraM
checkpoint. The checkpoint is verified against its pinned Git-blob SHA-1 and then
recorded with SHA-256 in `MODEL_MANIFEST.json`.

```bash
bash scripts/remote/bootstrap_models.sh \
  --canonical-root <CANONICAL_ROOT> \
  --work-root <WORK_ROOT> \
  --checkpoint-root <CHECKPOINT_ROOT> \
  --repo-root <DEPLOYED_RELEASE> --python <RUNTIME_PYTHON> \
  --stage core --check-only
# Inspect --dry-run, then repeat with --apply.
```

The launcher automatically sources the generated, non-secret
`WORK/cache/models/model_paths.env`; the queue rehashes the manifest, source
inventories, and checkpoint before model-dependent phases.

The shared hash-pinned runtime supplies dependencies, but it is not the authority
for project source. The launcher replaces (rather than appends to) `PYTHONPATH`
with `<DEPLOYED_RELEASE>/src:<DEPLOYED_RELEASE>`, verifies the imported CLI and
queue file locations, and invokes both queue and phase commands as Python modules
with user-site and unsafe-path injection disabled. Thus an older package copy in a
dependency environment is harmlessly shadowed and cannot select stale phase code.

Use one stable run ID and launch one phase at a time. After each tmux session exits,
inspect `status.sh`, the attempt log, success receipt, artifact hashes, and storage
usage before launching the next phase. This is operational review, not a scientific
gate.

```bash
bash scripts/remote/launch_queue.sh \
  --canonical-root <CANONICAL_ROOT> \
  --work-root <WORK_ROOT> \
  --checkpoint-root <CHECKPOINT_ROOT> \
  --repo-root <DEPLOYED_RELEASE> --python <RUNTIME_PYTHON> \
  --server-config <SERVER_ONLY_CONFIG> \
  --run-id <RUN_ID> --only-phase audit --check-only
# Inspect --dry-run, then repeat with --apply.
```

After the detached phase exits, read status from the same release and config:

```bash
bash scripts/remote/status.sh \
  --canonical-root <CANONICAL_ROOT> \
  --work-root <WORK_ROOT> \
  --checkpoint-root <CHECKPOINT_ROOT> \
  --repo-root <DEPLOYED_RELEASE> \
  --python <RUNTIME_PYTHON> \
  --server-config <SERVER_ONLY_CONFIG> \
  --run-id <RUN_ID> --json
```

Repeat `--only-phase` in this order:

```text
audit
acquire
qc
preprocess
encode
metrics
models
tms
locked-clinical
fmri
figures
```

Immediately before `fmri`, repeat `bootstrap_models.sh` with `--stage fmri`. That
stage re-verifies the exact Hugging Face commit and the already pinned per-file
BrainLM SHA-256 values. BrainLM checkpoint/weight files are used only for this
non-commercial secondary analysis under CC-BY-NC-ND-4.0; they are never redistributed
or downloaded during earlier phases. The pinned BrainLM source may already be
present from the core source-cache bootstrap.

The same immutable run ID may be established by `audit` while the external UKB_424
atlas, ordered coordinates, and timing-index origin remain unresolved. Once these
three inputs have been reviewed, copy `configs/fmri-inputs.template.yaml` to a
small, versioned-by-name metadata location such as
`<CHECKPOINT_ROOT>/metadata/fmri-inputs/<REVIEW_ID>.yaml` (or
`<CANONICAL_ROOT>/metadata/fmri-inputs/<REVIEW_ID>.yaml`), replace every
placeholder, and retain that reviewed file outside the deployed release and raw
data tree. Then use its absolute path for all three fMRI launch steps:

```bash
bash scripts/remote/launch_queue.sh \
  --canonical-root <CANONICAL_ROOT> \
  --work-root <WORK_ROOT> \
  --checkpoint-root <CHECKPOINT_ROOT> \
  --repo-root <DEPLOYED_RELEASE> --python <RUNTIME_PYTHON> \
  --server-config <SERVER_ONLY_CONFIG> \
  --run-id <SAME_RUN_ID> --only-phase fmri \
  --fmri-input-manifest <ABSOLUTE_REVIEWED_MANIFEST> --check-only
# Inspect --dry-run, then repeat with --apply using the identical manifest path.
```

`--check-only` reads and validates the strict YAML/JSON schema, rehashes the
manifest, atlas, coordinate table, and selected model cache, and performs no
writes. On `--apply`, the queue additionally publishes the immutable resolved
record at `<CHECKPOINT_ROOT>/queue/<RUN_ID>/late-inputs/fmri.json` before starting
the phase. A changed manifest, asset, or origin under that run ID is rejected and
requires a new run ID. The base run contract remains limited to the deployed
source, repository configuration, roots, and release; late fMRI inputs never force
earlier phases to use a different run ID. The legacy non-null `fmri_inputs` mapping
in the selected external server config remains supported when no external manifest
is supplied, but mixing the two authorities is rejected. The tracked synthetic
template keeps these fields null.

To let the queue continue across several technically complete phases, replace
`--only-phase` with `--from-phase <NAME> --through-phase <NAME>`. The same run ID
reuses validated successes. A changed commit, configuration, dependency marker, or
artifact hash under that run ID is rejected; issue a new run ID rather than editing
history.

## Restart, failure, and monitoring semantics

- `tmux` owns the queue process; an SSH tunnel loss does not stop it.
- Every attempt has its own append-only log and receipt directory on checkpoint
  storage. Retain its operational identifiers and log/state locations only in the
  server-side durable record.
- The queue holds an advisory run lock and refuses duplicate live execution.
- A killed process leaves a `running` marker with Linux boot/process-start identity.
  A restart will not duplicate it while that process is alive; an interrupted
  attempt gets a new numbered attempt.
- Existing successes are reused only after command/source/config/dependency hashes
  match and every artifact is rehashed. The fMRI phase also requires its immutable
  late-input record to match the manifest and asset hashes already bound to the run.
- A zero exit code without an atomic, schema-valid phase receipt is a failure.
- Acquisition retries are bounded and resume `.part` content; no mutating command
  is blindly replayed after an ambiguous disconnect.
- `status.sh` is read-only. Logs can be tailed through a fresh SSH connection; do not
  attach the only monitoring path to a disposable client session.

## GitHub and release tracking

GitHub is the source-of-truth for code, configuration, tests, checksum metadata,
roadmap, and operational status only. Raw data, derived data, participant-level data,
aggregate analysis tables, figure source-data tables, model caches, and durable logs
remain on approved university storage. Deploy only an exact commit from the
approved repository; never deploy an ambiguous branch tip. The server creates
`SOURCE_PROVENANCE.json` and `SOURCE_MANIFEST.sha256` in a content-addressed release,
and the queue includes that manifest hash in every phase identity.

Before each push, verify that ignore rules cover raw data, partial downloads,
foundation weights, caches, environments, credentials, private keys, window-level
embeddings, large derivatives, logs with host details, and participant-identifying
material. Store only dataset release metadata/checksums—not the datasets
themselves—in Git. Tag coherent analysis snapshots after tests pass, while retaining
the exploratory/non-preregistered project status.

## Remaining prerequisites and dataset-scoped blockers

Server bootstrap, hardened exact-source deployment, accelerator/runtime validation,
the new scientific audit, and the earlier direct-to-NAS acquisition launch have
completed.
Their live identifiers and durable log paths are retained in server-only records.
Repository tests, formatting, source/archive verification, and server
check-only/dry-run/apply contracts all passed for the scientific commit.

The next technical boundary is acquisition completion. Do not launch a concurrent
second acquisition; after the active acquisition run finishes, run `acquire` under
the scientific run so the hardened release revalidates and receipt-binds every
usable raw release before `qc`. The pinned LaBraM/BrainLM source cache and LaBraM
checkpoint must also be materialised before `encode`; their check-only contract
currently reports `would_create`/`would_download`, and server outbound GitHub access
timed out while acquisition was active. Acquisition/model availability are technical
inputs, never scientific outcome gates.

External dataset/phase blockers and limitations remain explicit:

- fMRI requires an approved UKB_424 atlas, its ordered 424-row coordinate table,
  and an explicit 0- or 1-based origin for the `ds006623` LOR/ROR timing index.
- Cogitate requires an approved user account and a snapshot of its native BIDS
  event columns/levels before its adapter or contrasts can be enabled.
- The Figshare resting-EEG DoC release has no verified participant-stem mapping to
  diagnosis or CRS-R. The Mendeley PSG filename supplies diagnosis categories but
  no CRS-R or demographic covariates. Clinical endpoints remain unavailable where
  official labels are absent.
- The large `ds006623` release is git-annex backed. Its public Git metadata may be
  reachable while the annex special remote needed for content is unavailable; the
  acquisition receipt must prove actual file materialization and hashes.
- The somatosensory OSF record still lacks a clear dataset-level licence and its raw
  MAT signal member must be verified after extraction before selection or raw-data
  redistribution.

These are access, metadata, provenance, or infrastructure constraints. They never
become scientific outcome gates.
