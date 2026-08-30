#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${NEURAL_MANIFOLDS_RAW_ROOT:-}" ]]; then
  echo "NEURAL_MANIFOLDS_RAW_ROOT must be an absolute NAS path" >&2
  exit 2
fi

REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${NEURAL_MANIFOLDS_PYTHON:-python}"
DATASETS=(
  propofol_tms_eeg
  dream_tononi_serial_awakenings
  tactile_detection
  somatosensory_report_task
  psiconnect
  doc_resting_eeg
  doc_polysomnography
  propofol_fmri
)

cd "${REPOSITORY_ROOT}"
for dataset_id in "${DATASETS[@]}"; do
  "${PYTHON_BIN}" scripts/acquire/datasets.py \
    --config configs/datasets.yaml \
    acquire \
    --root "${NEURAL_MANIFOLDS_RAW_ROOT}" \
    --dataset "${dataset_id}"
done
