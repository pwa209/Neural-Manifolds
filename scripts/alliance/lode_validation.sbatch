#!/usr/bin/env bash
# Slurm resources, array range, dependencies and output are supplied by the launcher.
set -euo pipefail
source scripts/alliance/runtime.sh
stage="$1"
output="$2"
source_root="$3"
args=("$stage" --config configs/lode_external_validation.yaml --output "$output")
if [[ "$stage" == preflight ]]; then
  args+=(--records-csv "$source_root/39373085-Records.csv" --archive "$source_root/39373088-Data.zip")
fi
if [[ "$stage" == measure || "$stage" == null ]]; then
  args+=(--index "${SLURM_ARRAY_TASK_ID:?array index required}")
fi
python -m neural_manifolds.validation.lode "${args[@]}"
