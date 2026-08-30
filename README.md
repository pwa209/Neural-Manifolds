# Neural Manifolds

This repository implements the study **A conserved manifold regime separates
consciousness from responsiveness**. It is an exploratory, non-preregistered
research project aimed at *Nature* or *Science*, with *Nature Human Behaviour*
as a realistic destination if the complete evidence chain is weaker.

The central empirical object is a five-axis participant-condition profile:

1. structured repertoire;
2. metastable persistence with revisability;
3. directed transition architecture;
4. cross-module alignment; and
5. perturbational reachability, directly anchored to TMS-EEG where available.

The foundation model supplies a frozen coordinate transform. It is never trained
on consciousness, diagnosis, drug, report, or task labels. Claims are made about
prespecified geometric and dynamical measurements, not unrestricted embedding
classification.

## Repository policy

- Raw data are never committed and are acquired directly onto canonical NAS storage.
- Source, configuration, checksum metadata, manifests, roadmap, and operational
  status are versioned in GitHub. Raw data, derived data, participant-level or
  aggregate analysis tables, model caches, and durable logs stay on approved
  university storage; credentials are never committed.
- Every phase is restartable and writes an atomic completion record only after its
  declared outputs validate.
- Participant identifiers, never windows or trials, define train/test boundaries.
- Clinical outcomes remain a separate, late transfer target. The healthy pipeline
  is protected by a technical provenance lock, never a scientific go/no-go gate.

## Implemented workflow

The integrated local phase order is:

```text
audit -> acquire -> qc -> preprocess -> encode -> metrics -> models -> tms
      -> locked-clinical -> fmri -> figures
```

Several boundaries are deliberately explicit:

- The metrics phase runs 100 true pre-encoder EEG channel-permutation repeats:
  sensor rows are permuted while the original channel-name order is retained, and
  each repeat passes through the frozen encoder. This is separate from the
  post-encoder latent-rotation null.
- Contrast-specific repeated equal-window profiles and reliability-by-duration
  curves preserve temporal order, trial/segment boundaries, and synchronized
  coarse/fine analysis spans. Unavailable cases are audited rather than gated.
- Healthy-reference participants are split into representation discovery,
  validation, and untouched evaluation partitions. The evaluation partition does
  not fit the state dictionary or profile calibration and is scored only through
  participant-separated outer prediction folds.
- Clinical releases may be acquired and file-inventoried earlier, but their signals
  are not parsed, preprocessed, or encoded until `locked-clinical`, after the queue
  has hashed the completed healthy workflow through TMS. The lock is technical
  provenance only; the study remains exploratory and non-preregistered.
- The secondary `ds006623` path uses a strict release/manifest contract and frozen
  BrainLM coordinates. It reports discovery-calibrated fMRI-compatible R/M/D/A
  axes and explicitly excludes passive-fMRI reachability.

## Quick start

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
neural-manifolds validate-config --study configs/study.yaml
pytest
```

Server deployment, storage layout, acquisition, and phase mapping are documented in
`docs/IMPLEMENTATION_ROADMAP.md`. Run `scripts/remote/preflight.sh` before any remote
mutation. The tracked `configs/server.yaml` is a non-operational synthetic template;
resolved infrastructure values live in an external server-only config. Export its
absolute path as `NEURAL_MANIFOLDS_SERVER_CONFIG` for bootstrap/deployment scripts,
and pass the same path with `--server-config` to queue launch/status commands. The
queue hash-binds that file to the run.

This external-config contract starts with the sanitized release that introduces it;
it is not retroactive. Keep every already-active queue on the exact earlier
release/configuration that created it. Use a new run ID for the first launch from the
sanitized release.

The pinned Python 3.11/CUDA 12.6 server lock is
`requirements/server-py311-cu126.lock`. Its verified checksum is retained in the
server-only deployment provenance.

## Status

The server roots, pinned runtime, accelerator compatibility, and acquisition-safe
source deployment are confirmed, with exact infrastructure details retained only in
server-side durable records. Direct-to-NAS acquisition is running; no external
dataset has been analysed. The scientifically hardened commit has passed local
verification, is deployed as an exact content-addressed server release, and completed
its audit phase. Acquisition, model-cache, dataset-, and fMRI-specific prerequisites
remain explicit; see `STATUS.md` for the current phase boundary.
