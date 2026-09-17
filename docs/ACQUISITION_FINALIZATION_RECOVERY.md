# Interrupted publication recovery

The resumed TMS acquisition downloaded every missing annex object and wrote its
completion metadata, but its worker disappeared before publishing the staging
directory or reconciling acquisition state. The available logs do not establish
why that worker terminated. A staging completion marker alone is not evidence
that a dataset has been published and made available to downstream workers.

`scripts/alliance/recover_acquisition.py` performs recovery on an allocated
compute node. It requires existing completion metadata, verifies registry
identity, takes the acquisition-state lock, and calls the existing transactional
acquisition manager. That manager rehashes the complete dataset, seals file
permissions and publishes it. Only a successful return marks acquisition
complete. Failures remain explicit and preserve downloaded content. The script
does not download data or bypass content validation.

`scripts/alliance/acquire.py --only-datasets ...` supports targeted continuation
with the original registry identity and storage budgets. Global completion now
requires completion of every registry dataset, not merely all currently recorded
state entries. This prevents a selected-source run from declaring the entire
acquisition complete.

The TMS recovery is scheduled separately from existing scientific work. The next
two core acquisitions depend on successful recovery and execute in a scheduled
compute job. The existing scientific controller discovers published releases
and registers downstream analyses automatically; its completed outputs and
original scientific policy are unchanged. No desktop monitor is introduced.
