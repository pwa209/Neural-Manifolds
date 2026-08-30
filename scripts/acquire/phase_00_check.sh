#!/usr/bin/env bash
set -euo pipefail

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
arguments=()
for dataset_id in "${DATASETS[@]}"; do
  arguments+=(--dataset "${dataset_id}")
done
"${PYTHON_BIN}" scripts/acquire/datasets.py \
  --config configs/datasets.yaml \
  check \
  "${arguments[@]}"

# Cogitate is an expected, documented account gate, not a failed endpoint check.
if "${PYTHON_BIN}" scripts/acquire/datasets.py \
  --config configs/datasets.yaml \
  check \
  --dataset cogitate_meeg; then
  :
else
  status=$?
  if [[ "${status}" -ne 3 ]]; then
    exit "${status}"
  fi
fi
