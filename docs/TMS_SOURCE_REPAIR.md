# TMS input and flatline-QC repair

Three separate problems were identified after the raw release was published:

1. Resolving a BrainVision header symlink to its annex object breaks the relative
   paths to its companion EEG and marker files. The reader now retains the logical
   BIDS header path; content hashing still follows the storage link.
2. This source's BrainVision markers use recoded stimulus names, whereas the
   corresponding BIDS events tables retain `Response/R128` pulse semantics.
   The repair uses that explicit sidecar authority and requires integer sample
   positions, onset/sample agreement, unique ordering and exact correspondence
   to recorded annotations. It never assumes an arbitrary stimulus code is TMS.
   The sidecar hash and authority are retained in the epoch manifest.
3. The pointwise flatness rule misclassifies isolated equal adjacent samples in
   quantized high-rate EEG as flatlining. For the TMS branch only, flat time now
   counts sustained runs lasting at least 100 ms. This is a post-inspection,
   exploratory preprocessing correction. The five-percent flat-time limit,
   robust outlier criteria and maximum interpolation fraction are unchanged.
   Other analysis branches retain their previous default behavior.

All-record failure now writes its individual exclusion reasons before raising an
error. Synthetic tests distinguish quantized varying signals from genuinely dead
channels and reject missing or inconsistent event authority. Real-record canaries
must distinguish technical correction from genuine remaining QC exclusions.
No favourable condition effect, prediction result or journal preference controls
these checks. The original failed outputs are retained for provenance.

## Execution status

The initial source-path/event repair was deployed and tested on real recordings.
Those recordings then failed the original channel-quality rule. The duration-aware
flatline correction passed 18 targeted server tests. The fixed three-record
canary still fails the remaining channel-quality criteria. The full cohort is
therefore being evaluated with those criteria preserved, to record its actual
eligible denominator rather than treating three exclusions as a software failure
or silently relaxing thresholds. The repaired full job is tracked by the existing
controller with its distinct source digest and output path; original failed
outputs are retained. It is not yet a verified successful full TMS analysis.

Perception acquisition was moved back to the internet-connected login host after
compute-node requests timed out. Missing annex transfers resumed. Source hashes,
budgets, publication checks and explicit failures remain in force.
