# Bounded OpenNeuro acquisition repair

The TMS acquisition stopped making progress while DataLad's recursive `get`
process remained alive. Existing downloaded annex objects were retained.

OpenNeuro acquisition now enumerates missing annex objects and retrieves them
directly in batches of eight, with two transfer workers, a 20-minute timeout per
attempt and at most two attempts per batch. Timeout handling terminates the
subprocess group on POSIX, so an orphaned transfer cannot keep holding the batch.
Persistent failure is reported as incomplete; the acquisition loop can continue
to subsequent datasets. Partial downloads are retained for a later retry.

Pinned source revisions, missing-object checks, provenance manifests and sealed
completion checks remain mandatory. Repositories containing subdatasets require
an explicit recursive adapter rather than silently receiving an incomplete seal.

## Deployment verification — 17 September 2026

The repaired worker passed 14 targeted tests on the cluster and demonstrably
downloaded 152 of the 224 objects missing at restart. The existing scientific
controller remained healthy and its completed outputs were not replaced.
This is evidence of resumed acquisition, not a claim that TMS acquisition or the
entire study is complete. The supplementary high-density repair is documented
separately in `HIGH_DENSITY_EXTENSION.md`.
