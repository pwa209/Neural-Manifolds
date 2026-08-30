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
mutation and keep the explicit roots in `configs/server.yaml` aligned with the paths
approved for this project.

The pinned Python 3.11/CUDA 12.6 server lock is
`requirements/server-py311-cu126.lock`, SHA-256
`e9a37e9acafaead6a6f77d966a9e0b4ef083acd73cf0dc78ac3dd193310ce39e`.

## Status

The exact server roots, pinned runtime, and acquisition-safe source deployment are
confirmed. Direct-to-NAS acquisition is running; no external dataset has been
analysed. The scientifically hardened execution tree has passed local verification
and is being frozen for exact server deployment before later phases are queued.
Dataset- and fMRI-specific prerequisites remain explicit; see `STATUS.md` for the
current boundary.
