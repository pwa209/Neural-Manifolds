# Quota and tactile response recovery

This is an exploratory study, not a preregistered analysis. Recovery uses
technical integrity checks, not scientific-result gates.

## Tactile missing-marker policy

An adaptive stimulus with no subsequent first-order response marker before the
next stimulus (or file end) is unavailable for detection contrasts. It is not
assigned to a miss, correct rejection, or any experience category. Source rows
may still contain response-time/confidence values; these do not substitute for
the absent categorical marker. The adapter records source file, stimulus index,
marker onset, unit ID and exclusion reason. Trial IDs include excluded stimuli,
and each retained unit carries its own matched response onset, avoiding an
ordinal join after exclusions. Unknown markers and orphan responses still fail.
The default adapter remains strict unless an explicit exclusion sink is given.
The public source also contains `stim-thr` events inside an adaptive-task file.
These are separately audited and excluded, never converted into adaptive trials.
Responses without an adaptive stimulus still fail; matching cannot cross such a
marker. Epochs use the BIDS `stim-adapt` event's `onset`. The source describes
`stimon` as latency relative to trial start; it is retained as metadata and must
not be added again to the stimulus marker. A source-table consistency audit
checks that response-marker minus stimulus-marker timing varies inversely with
the trial-relative stimulus delay. This supports the event-label interpretation;
it does not validate millisecond hardware latency. Any genuinely negative
response-relative timing retains the driver's explicit exclusion.
Recording paths use the source's EEGLAB `.set` files with their companion `.fdt`
files, not an assumed BrainVision header. Preserve the logical BIDS path when
opening annex-managed files so companion-file lookup remains valid.

## Source payload integrity and full-epoch preflight

Validate the external float32 payload against the header's original channel and
sample counts before selecting channels or reading epochs. A short file may be
excluded as a source defect only when its bytes match both the size and MD5 in
the pinned acquisition's annex key. Unverified truncation, checksum mismatch,
unexpected layout, or another I/O failure remains an error requiring repair.

Exclude the entire verified incomplete recording, not just its missing tail.
Record the header/payload sizes, checksum evidence, file, participant, and every
affected unit in the result audit. Preserve the raw file unchanged. If no usable
epochs remain, do not emit a successful analysis. This technical exclusion does
not depend on any statistical result and is not a scientific gate.

The tactile Slurm preflight now attempts **every eligible epoch read** for each
complete recording, with feature checks at its beginning, middle and end. The
previous first-trial-only canary could not detect a truncated tail. A preflight
pass is still not a completed inferential analysis.

## Controller recovery

`scripts/alliance/supervise_controller.py` runs from the original qualified
controller release, with that release's source on `PYTHONPATH`. The supervisor
may be supplied by a newer immutable release; both source digests appear in its
health record. Scientific task identities are not rewritten during restart.

The supervisor holds the original controller lock, reloads durable state each
tick, checks project byte and inode headroom, and uses the controller's persisted
submission intent and scheduler reconciliation. It waits through capacity errors
and resumes when headroom returns. Both project and separately located scratch
health writes are attempted independently. Three consecutive non-capacity errors
require inspection rather than an unbounded restart loop. This does not protect
against termination of the login process itself or guarantee future quota space.

Failed jobs require an explicit, recorded retry. Keep failed output/spec paths
unchanged; direct the retry to a new output directory and retain prior job/source
metadata. A repaired scientific branch uses its new source digest and verified
completion receipt. Do not recreate successful TMS, report-task, or null outputs.

No raw data, empirical results, private server addresses, or authentication
material belong in this public source repository.
