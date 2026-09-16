# Continuous revised-study execution

The remote controller submits implemented tasks to Slurm and reconciles durable
receipts. The hourly assistant monitor was deleted at the user's request and must
not be recreated without a new request. Remote acquisition and the controller
are separate from that deleted monitor. They do not implement missing analyses.

## Operational contract

- One controller owns its state directory through a process lock.
- At most two CPU tasks (two cores, 8 GiB each) run concurrently. Acquisition is
  single-worker with the existing 100 GiB/source and 180 GiB total limits.
- Each job has a deterministic identity from source, task kind, dataset and input.
  Submission intent is written before `sbatch`. Lost responses are reconciled
  against job records; unknown submissions are not blindly retried.
- Only verified output receipts advance dependencies. A zero process exit without
  the expected artifacts is a technical problem, not successful science.
- Node/preemption failures receive at most three attempts, with backoff. Software,
  metadata and unknown-scheduler failures remain visible for diagnosis.
  Acquisition receives at most three supervised restarts per controller
  generation. Do not reset that budget simply to repeat the same failed operation.
- Null, negative, mixed, inconclusive and positive findings use identical scheduling.
- Data, task specs, job identifiers, logs and result tables stay on the cluster.
  The public repository contains code and reviewed status only.
- Creating `STOP` in the private controller directory stops new scheduling at the
  next tick. It does not cancel running jobs or delete data. Explicit job cancellation
  must target only this study's verified job IDs.

## What is executable now

Acquisition is supervised and resumes partial files. Every completed dataset can
immediately receive its own archive and EDF-header inventory, without waiting for
the entire portfolio. Records.csv fields are preserved verbatim, including blank
and ambiguous experience codes. Unsupported archives and unresolved metadata are
explicit issues. No archive member is extracted by the inventory worker.

Each completed inventory schedules a length-matched AR(1) recovery component.
This is a narrow estimability smoke test, **not** validation of all proposed axes,
uncertainty coverage, measurement invariance, or the biological hypothesis.

The qualified updated release also schedules sampled, label-blind signal QC. ZIP
recordings are materialized one at a time in bounded scratch space, verified
against the inventory size and ZIP integrity, inspected and removed. Per-recording
checkpoints support resumption. Unsupported archive formats and unreadable signals
remain explicit unavailable rows, not admitted cohorts. This is sampled diagnostic
QC, not a replacement for analysis-window QC, channel-type adjudication or cohort
admission. Full-study completion remains false.

The controller writes `state.json`, `heartbeat.json`, `progress.json`, per-task
specifications and hashed receipts. These distinguish task-component completion
from completion of a full scientific phase. Source changes use a new state directory;
inspect the old controller and drain or stop scheduling before replacing it.

## Remaining integration work, in execution order

1. Link each source's official Records.csv rows to exact signal files; verify where
   awakening occurs in the provided segment. Do not apply Tononi's filename or
   segment convention to other cohorts without source evidence. Audit participant
   reuse and laboratory grouping before treating datasets as independent studies.
2. Add safe bounded RAR handling and analysis-window QC. Resolve duration, channel,
   artifact, treatment, age and report exclusions with a recorded denominator.
3. Complete axis-specific observed-length/missingness recovery and uncertainty
   coverage. The existing unrestricted state dictionaries are not automatically
   suitable for 20-second segments.
4. Integrate the slow frozen encoder, separate fast sensor/CSD measurements and
   conventional benchmarks into training-only study splits. Prestage model files
   through the open-data acquisition route and audit pretraining overlap.
5. Connect audited feature tables to `statistics/study_transfer.py`, retaining the
   shared observations, nested tuning and independent participant/study weights.
6. Verify spontaneous/TMS participant-session pairing and add within-condition and
   paired-change evaluations alongside held-out incremental TMS prediction.
7. Implement revised perception/report and psychedelic boundary analyses. Optional
   clinical/fMRI branches must not become prerequisites for core synthesis.
8. Generate final evidence tables, uncertainty, figures and all failure cases.
   A controller progress snapshot must never be presented as final synthesis.

These integrations remain development work, not silently skipped successful stages.
There is no scheduled assistant continuation. Renewed MFA or a material expansion
of authority requires user action; missing code does not become executable merely
because the remote controller is running.

## Continuation checklist

Read this file, STATUS.md and the roadmap. Use the ignored local deployment record
to locate the authenticated bridge, current source and controller directory. Read
remote heartbeat, controller state, acquisition state and exact Slurm receipts.
Reconcile before launching anything. If healthy, advance the earliest remaining
integration above with source-backed adapters and regression tests. Deploy only
after qualification and preserve all prior data and provenance. Update public
status without private paths, credentials or participant-level contents. Notify
only on meaningful change or required action. Do not recreate the deleted monitor.
