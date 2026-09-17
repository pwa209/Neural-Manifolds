#!/usr/bin/env bash
# Resource requests and dependencies are supplied explicitly by the deployment.
set -euo pipefail
source scripts/alliance/runtime.sh
phase="$1"
root="$2"
shift 2
python -m neural_manifolds.revised.high_density_extension "$phase" --root "$root" --replicate "${SLURM_ARRAY_TASK_ID:-0}" "$@"
