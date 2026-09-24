# Fixed-count null-control continuation

The user approved extending every reviewed null distribution from 100 to 5,000
refits on 20 September 2026. This is an exploratory post-results extension, not
preregistration. It must not be stopped or narrowed according to significance.

## Scientific contract

- Keep the entire 72-comparison family: two portfolios, three representations,
  three null mechanisms and four matched statistics.
- Preserve all observed predictions, cohort definitions, transforms, estimator
  code, hyperparameter grids, loss weights and null mechanisms.
- Verify and reuse original replicate indices 0–99; add indices 100–4999 with
  the original seed rule. A failed seed is retried, never replaced or dropped.
- Repeat the complete original nested fit per null cell. The new runner caches
  read-only inputs within a job and saves completed cells; it does not substitute
  fixed-model shuffling for nested refitting.
- Verify one complete original replicate per portfolio before launching new
  seeds. This is an engineering equivalence check, not a scientific outcome gate.
- Compute final tails with denominator 5,001 only when every cell is complete.
  Apply Holm once over all 72 tests, preserving raw tails, all refit statistics,
  and Monte Carlo binomial intervals. These intervals concern simulation error,
  not biological uncertainty.

The attainable minimum raw tail becomes 1/5,001. This removes the original
resolution barrier, but does not guarantee significance or fix exchangeability,
small-study uncertainty, cohort overlap or biological interval calibration.

## Implementation

`scripts/extend_null_controls.py` runs against the original immutable analysis
release and dependency environment. It does not edit the original control
driver or synthesis. `prepare` seals a plan with input, implementation and
observed-statistic checksums. `canary` compares all nine cells of original
replicate 0 with tolerance rtol=1e-7, atol=1e-8. `worker` uses an interleaved
portfolio array: 392 tasks, each covering 25 consecutive replicate indices for
one portfolio. Each cell is atomically checkpointed into a checksummed aggregate
record; no participant predictions are exported. `aggregate` refuses a reduced
denominator and reports missing seeds instead of issuing partial final p-values.

The initial submission used one CPU per task with up to 50 concurrent tasks.
The current operational limit is 250, as recorded below. A dependent second pass
retries the same task indices and skips completed cells. The final aggregation
is queued after that pass. `scripts/alliance/queue_null_extension.py` checks the
association's job-slot headroom and records each submission durably. It refuses
to blindly repeat an ambiguous submission, preventing duplicate arrays. Workers
request 4 GB RAM and five days; engineering checks request one day, and final
aggregation one hour. All CPU jobs use the existing cluster scheduler;
there is no desktop heartbeat or newly installed monitor. Operational receipts
and scheduler IDs are private in `work/` and on the data host. New checkpoints
belong in project-scoped scratch storage to avoid the nearly full project inode
quota; final aggregate tables and the sealed plan belong in project storage.

### Operational amendment: 100 CPUs

On 20 September 2026, at the user's request, the live continuation and dependent
retry arrays were both increased from 50 to 100 concurrent one-CPU tasks using
Slurm's `scontrol update JobId=<array_id> ArrayTaskThrottle=100`. This is a
scheduler-only amendment: existing jobs are not cancelled, restarted or duplicated.
The final aggregation dependency and one-CPU-per-task setting are preserved.
Actual concurrency depends on scheduler availability; 100 is the cap, not a
guarantee of an immediate doubling in throughput.

The immutable scientific plan and deployed runner remain byte-identical, including
the plan's historical initial resource limit of 50. A separate, timestamped
`operational-amendments/` receipt next to the server-side plan records the new
effective limit, before/after Slurm settings and the unchanged plan checksum.
Do not edit or reseal the scientific plan merely to change scheduler concurrency:
existing checkpoint identities must remain valid. The 5,000 refits, 72 tests,
cohorts, seeds, fitting procedure and observations are unchanged. The original
submission helper retains its initial 50-task default; this amendment applies
to the already-submitted arrays, including their automatic retry pass.

### Operational amendment: 250 CPUs

On 21 September 2026, the user requested a concurrency increase into the
200–250 CPU range. The scheduler accepted the preferred upper limit of 250 for
both the continuation and dependent retry arrays through an in-place
`ArrayTaskThrottle=250` update. There was no fallback to 200, cancellation,
restart, resubmission or duplication of work. Each task still requests one CPU
and 4 GB RAM; the two arrays remain dependency-ordered rather than simultaneous.

As with the previous resource amendment, the sealed plan and runner were
verified unchanged. A new timestamped server-side operational receipt records
the transition from 100 to 250, the verified scheduler settings, and the original
plan checksum. The historical submission default and plan value are preserved.
Actual allocation can be below 250 because of scheduler availability and will
also decrease when fewer than 250 tasks remain. No statistical stopping or
scientific criterion is changed by this operational amendment.

### Operational amendment: parallel high-density tail

On 24 September 2026, after 389 of 392 retry-array tasks had completed, the
three remaining 25-seed tasks left a slow one-CPU-per-chunk tail. At the user's
request, `scripts/alliance/parallel_null_tail.py` preserved the original sealed
runner and every existing seed/cell checkpoint, held the pending aggregation
job, submitted one-seed high-density workers for the 51 unfinished seeds at
cutover (up to 50 CPUs concurrently), submitted a dependent checkpoint-aware
retry array for the same seeds, relinked aggregation after that array, and only
then cancelled the three original running elements. The original analysis
release, plan checksum, observed statistics, seeds and 72-test family are
unchanged. An operational manifest and staged submission receipt are stored
beside the project-side plan; scientific checkpoints remain on scratch. The
original retry-array elements were cancelled only after the new aggregation
dependency was verified. The revised Slurm chain is automatic, but the final
aggregation still refuses incomplete seeds or reduced denominators.
The cutover command must run from the sealed analysis release with the same
`NM_PROJECT_ROOT`, `NM_SCRATCH_ROOT` and `NM_ENV_RELEASE` used by the original
qualified runtime; those variables are exported into the new Slurm workers.

## Reporting and manuscript status

The existing 100-refit figures, legends and manuscript wording remain historical
artifacts until all 5,000 refits have been collected and reviewed. Re-export
Figure 3 and update all affected tables, legends, Methods, Results and Discussion
after completion. Report unchanged, strengthened and weakened findings alike.
Do not label the existing manuscript submission-ready while that update is
pending. No positive result is a condition for completing the study.
