# Data acquisition phases

These scripts materialise raw data directly at the absolute path supplied in
`NEURAL_MANIFOLDS_RAW_ROOT`. Use the project's approved NAS root; the scripts do
not invent or register a server path.

1. `phase_00_check.sh` checks endpoints and required tools without writing data.
2. `phase_01_open_data.sh` acquires the eight currently machine-accessible
   releases. Each download is restartable and published only after validation
   and SHA-256 manifest creation.
3. `phase_02_validate_raw.sh` independently rehashes every published file.

The Cogitate M-EEG release is intentionally absent from phase 1. Its official
download route requires a Cogitate Data User Account and acceptance of its
terms. Run `datasets.py check --dataset cogitate_meeg` for the current access
instructions. Authenticated acquisition and an audited import step remain
blocked until an account is approved and the official client workflow can be
tested. Never save credentials in this repository, a command argument, an
environment variable, or a log.

Raw releases live at `<root>/<dataset-id>/<version>/`. A completed directory is
immutable by policy: reruns validate it and never update it in place. Interrupted
downloads remain under `<root>/.staging/` and resume on the next run.

The OSF somatosensory project is public but is a mutable project rather than a
registration/release, and its API exposes no named dataset licence. The first
successful acquisition freezes its complete remote inventory and hashes under
the local `doi-2025` release label. Do not redistribute those raw files until
the rights holder clarifies the licence.
