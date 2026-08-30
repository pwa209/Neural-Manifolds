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

The compute endpoint has been read-only preflighted as
`wangpeng@kemove-Rack-Server` through local SSH port `1022`. It has an NVIDIA H100
80-GB GPU, `tmux`, and `rsync`, but no detected Slurm installation. The three
project-specific roots have **not** been confirmed. Accordingly,
`configs/server.yaml` deliberately contains null roots and no remote mutation or
queue launch is ready to authorize until the user supplies all three exact values:

| Role | Required parent | Purpose |
|---|---|---|
| Canonical | `/private_nas/wangpeng` | immutable raw releases, licences, acquisition manifests, snapshots |
| Work | `/data1/wangpeng` | source releases, environments, caches, preprocessing, embeddings, metrics, models, figures |
| Checkpoint | `/data2/wangpeng` | queue state, attempt receipts, restart markers, durable logs |

Each root must be one direct, project-specific child of its parent. Scripts reject
the parent itself, nested guesses, relative paths, and a root copied from another
project. Once confirmed, record the exact values in `configs/server.yaml` and pass
the same values explicitly to every remote script.

## Reproducible storage contract

Given confirmed roots `CANONICAL`, `WORK`, and `CHECKPOINT`, the fixed layout is:

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
data stay on `/data1`; only small durable queue records and logs go to `/data2`.
No raw data, model cache, participant-level sensitive arrays, or credentials enter
GitHub.

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
| 3 | `qc` | `qc` | work | recording flow table, metadata validation, blind exclusion log |
| 4 | `preprocess` | `preprocess` | work | harmonised/native windows and participant-condition inventory |
| 5 | `encode` | `encode` | work/GPU | frozen LaBraM embeddings, representation and label-leakage receipts |
| 6 | `metrics` | `metrics` | work/CPU+GPU | five-axis metrics, benchmarks, nulls, seed/reliability summaries |
| 7 | `models` | `models` | work/CPU | participant-level contrasts, held-out predictions, uncertainty and ablations |
| 8 | `tms` | `tms` | work/CPU+GPU | passive/direct reachability comparison and post-pulse dynamics |
| 9 | `locked-clinical` | `clinical` | work/GPU | post-lock parsing, preprocessing, encoding, and frozen transfer for both DoC resources |
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
`CANONICAL/raw/<dataset>/.partial-<release>` and publish
`CANONICAL/raw/<dataset>/<release>` only after validation. Never copy raw files to
the Git repository, workstation, `/data1`, or `/data2`. Capture repository-native
metadata before any conversion. A failed transfer is resumed from its partial
state; an existing verified release is skipped.

Completion artifacts are small NAS manifest files listed in the phase receipt—not
an unverified claim based on downloader exit status. Restricted, withdrawn, or
licence-incompatible material is reported and left untouched rather than bypassed.

### 3. Metadata and signal QC

Build one recording-level catalogue covering participants, sessions, conditions,
modalities, channel names/positions, reference, sampling rate, duration/trials,
events, artefact burden, label availability, and missing covariates. Validate BIDS
where present and preserve native metadata elsewhere. Make exclusions blind to the
contrast outcome whenever file organization permits.

Implement the study's minimum-duration/trial/channel/window rules, flatline and
clipping checks, event synchronization checks, montage recoverability, and
dataset-specific exceptions. Write every inclusion/exclusion and reason to a flow
table. Never delete raw data. Unit of inference is participant; window/trial counts
are precision metadata, not independent sample sizes.

### 4. Preprocessing and harmonisation

Create both the 19-channel harmonised EEG track and native/full-montage sensitivity
track. Apply resampling, filtering/notch choice, reference and CSD variants, bad
channel/interpolation rules, artefact rejection, event locking, sleep-stage
restriction, and TMS pulse handling exactly from configuration. Preserve masks so
zero padding never becomes signal. Preprocessing preserves the ordered segments and
fine/coarse time tracks needed by the post-encoding equal-window and reliability
stage; it does not use condition labels to fit a signal transform.

The integrated stage writes label-free, preprocessed signal objects plus analysis-
unit and flow manifests on fast work storage. The manifests retain shape, channel,
mask/window, preprocessing-hash, and raw-input checksum lineage; labels remain in a
separate manifest until after encoding. Synthetic and small smoke datasets establish
technical correctness, not favourable method selection.

### 5. Frozen representation

Verify the LaBraM weight hash before GPU use. Keep parameters frozen, disable label
access in the encoding process, use configured patch/mask handling, and write
participant/session/condition trajectories in deterministic chunks. Capture CUDA,
driver, PyTorch, model commit/hash, selected layer, pooling rule, montage, seed, and
input/output shapes. Resume at participant chunks; never recompute validated chunks.

Run PCA/time-frequency controls and configured LaBraM layer/size sensitivities as
separate provenance branches. Representation fitting, scaling, and dimensionality
reduction for any downstream prediction must occur inside participant-level training
folds. Dataset-identity decodability is reported as a confound diagnostic.

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
permutation entropy, Lempel-Ziv complexity, and connectivity summaries. Weighted
symbolic mutual information, microstates, and PCIst remain explicit unavailable
placeholders until validated backends are supplied; they are never silently replaced
by another measure. Current technical null machinery records phase randomization,
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
report/task relevance, and psychedelic organization. Use participant random effects
for repeated awakenings/trials and participant-stratified nested validation for
prediction. No participant's windows may cross train/validation/test boundaries.
Scaling, imputation, dimensionality reduction, HMM/state dictionaries, tuning, and
calibration are fitted inside training partitions.

Report AUROC, AUPRC, balanced accuracy, Brier score, calibration slope/ECE,
participant-stratified bootstrap intervals, participant-level permutations with the
plus-one correction, omnibus tests, FDR-controlled axis follow-ups, equivalence
intervals, property redundancy, leave-one-property-out ablations, and matched-size
conventional baselines. Preserve every result; no threshold controls whether the
TMS, clinical, or fMRI phase runs.

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

Clinical raw releases may have been acquired and included in file-level integrity
inventories earlier, but their signals are deliberately excluded from the healthy
preprocess and encode phases. Only after the lock validates does `locked-clinical`
parse the two DoC releases, construct their cohort manifests, preprocess and encode
their signals, and apply the frozen healthy objects.

Apply the frozen encoder, projection/state dictionary, and healthy-calibrated profile
estimator to the Figshare resting-EEG and Mendeley PSG DoC resources without
refitting. Associate preserved regime profiles
with diagnosis and available CRS-R while retaining dataset terms and uncertainty.
Do not automatically reclassify individuals. Document montage-limited properties
for PSG and report discordant cases. Missing official label keys remain missing;
they are never inferred from file order or signal features.

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
bash scripts/remote/preflight.sh \
  --canonical-root <CONFIRMED_CANONICAL_ROOT> \
  --work-root <CONFIRMED_WORK_ROOT> \
  --checkpoint-root <CONFIRMED_CHECKPOINT_ROOT>

bash scripts/remote/bootstrap.sh \
  --canonical-root <CONFIRMED_CANONICAL_ROOT> \
  --work-root <CONFIRMED_WORK_ROOT> \
  --checkpoint-root <CONFIRMED_CHECKPOINT_ROOT> --check-only
# Review, then repeat with --apply.

bash scripts/remote/deploy_from_git.sh \
  --canonical-root <CONFIRMED_CANONICAL_ROOT> \
  --work-root <CONFIRMED_WORK_ROOT> \
  --checkpoint-root <CONFIRMED_CHECKPOINT_ROOT> \
  --repository https://github.com/pwa209/Neural-Manifolds.git \
  --commit <EXACT_PUSHED_COMMIT> --check-only
# Review, then repeat with --apply.
```

Build the Python environment only from a complete requirements lock whose SHA-256
has been independently recorded:

```bash
bash scripts/remote/bootstrap_runtime.sh \
  --canonical-root <CONFIRMED_CANONICAL_ROOT> \
  --work-root <CONFIRMED_WORK_ROOT> \
  --checkpoint-root <CONFIRMED_CHECKPOINT_ROOT> \
  --repo-root <CONFIRMED_WORK_ROOT>/source/releases/<EXACT_PUSHED_COMMIT> \
  --python <ABSOLUTE_PYTHON_3_11> \
  --requirements-lock <ABSOLUTE_REQUIREMENTS_LOCK> \
  --lock-sha256 e9a37e9acafaead6a6f77d966a9e0b4ef083acd73cf0dc78ac3dd193310ce39e \
  --check-only
# Review, then repeat with --apply and retain the reported runtime Python path.
```

Materialise the core model cache next. This clones LaBraM and BrainLM source at the
exact Git object IDs in `configs/models.yaml`, but downloads only the LaBraM
checkpoint. The checkpoint is verified against its pinned Git-blob SHA-1 and then
recorded with SHA-256 in `MODEL_MANIFEST.json`.

```bash
bash scripts/remote/bootstrap_models.sh \
  --canonical-root <CONFIRMED_CANONICAL_ROOT> \
  --work-root <CONFIRMED_WORK_ROOT> \
  --checkpoint-root <CONFIRMED_CHECKPOINT_ROOT> \
  --repo-root <DEPLOYED_RELEASE> --python <RUNTIME_PYTHON> \
  --stage core --check-only
# Inspect --dry-run, then repeat with --apply.
```

The launcher automatically sources the generated, non-secret
`WORK/cache/models/model_paths.env`; the queue rehashes the manifest, source
inventories, and checkpoint before model-dependent phases.

Use one stable run ID and launch one phase at a time. After each tmux session exits,
inspect `status.sh`, the attempt log, success receipt, artifact hashes, and storage
usage before launching the next phase. This is operational review, not a scientific
gate.

```bash
bash scripts/remote/launch_queue.sh \
  --canonical-root <CONFIRMED_CANONICAL_ROOT> \
  --work-root <CONFIRMED_WORK_ROOT> \
  --checkpoint-root <CONFIRMED_CHECKPOINT_ROOT> \
  --repo-root <DEPLOYED_RELEASE> --python <RUNTIME_PYTHON> \
  --run-id <RUN_ID> --only-phase audit --check-only
# Inspect --dry-run, then repeat with --apply.
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
BrainLM SHA-256 values. BrainLM material is used only for this non-commercial
secondary analysis under CC-BY-NC-ND-4.0; it is never redistributed or downloaded
during earlier phases. The fMRI phase still cannot start until the external UKB_424
atlas, ordered coordinates, and timing-index origin are explicitly supplied and
hashed.

To let the queue continue across several technically complete phases, replace
`--only-phase` with `--from-phase <NAME> --through-phase <NAME>`. The same run ID
reuses validated successes. A changed commit, configuration, dependency marker, or
artifact hash under that run ID is rejected; issue a new run ID rather than editing
history.

## Restart, failure, and monitoring semantics

- `tmux` owns the queue process; an SSH tunnel loss does not stop it.
- Every attempt has its own append-only log and receipt directory on checkpoint
  storage. Immediately record the reported tmux session, pane PID, queue log, and
  state root.
- The queue holds an advisory run lock and refuses duplicate live execution.
- A killed process leaves a `running` marker with Linux boot/process-start identity.
  A restart will not duplicate it while that process is alive; an interrupted
  attempt gets a new numbered attempt.
- Existing successes are reused only after command/source/config/dependency hashes
  match and every artifact is rehashed.
- A zero exit code without an atomic, schema-valid phase receipt is a failure.
- Acquisition retries are bounded and resume `.partial` content; no mutating command
  is blindly replayed after an ambiguous disconnect.
- `status.sh` is read-only. Logs can be tailed through a fresh SSH connection; do not
  attach the only monitoring path to a disposable client session.

## GitHub and release tracking

GitHub is the source-of-truth for code, configuration, tests, manifests, roadmap,
and reviewed small results/source-data. Deploy only an exact commit from
`pwa209/Neural-Manifolds`; never deploy an ambiguous branch tip. The server creates
`SOURCE_PROVENANCE.json` and `SOURCE_MANIFEST.sha256` in a content-addressed release
and the queue includes that manifest hash in every phase identity.

Before each push, verify that ignore rules cover raw data, partial downloads,
foundation weights, caches, environments, credentials, private keys, window-level
embeddings, large derivatives, logs with host details, and participant-identifying
material. Store only dataset release metadata/checksums—not the datasets
themselves—in Git. Tag coherent analysis snapshots after tests pass, while retaining
the exploratory/non-preregistered project status.

## Remaining prerequisites and dataset-scoped blockers

No server deployment, queue launch, or real-data acquisition has started. The
reviewed Python 3.11/CUDA 12.6 lock and BrainLM revision/file hashes are already
pinned; they are not remaining prerequisites.

The reviewed initial implementation is now published on GitHub. Remaining global
launch prerequisites are:

1. User confirmation of the exact canonical, work, and checkpoint project roots,
   followed by recording the same three values in `configs/server.yaml`.
2. Selection of the exact published commit for content-addressed deployment after
   the server-root values are recorded.

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
