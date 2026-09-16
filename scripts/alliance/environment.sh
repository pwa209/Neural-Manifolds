#!/usr/bin/env bash
set -euo pipefail
module load StdEnv/2023 python/3.11.5 cuda/12.6 arrow/21.0.0 git-annex/10.20231129
: "${NM_PROJECT_ROOT:?personal project directory required}"
: "${NM_SCRATCH_ROOT:?personal scratch directory required}"
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export MPLBACKEND=Agg
export MPLCONFIGDIR="$NM_SCRATCH_ROOT/cache/matplotlib"
export PIP_CACHE_DIR="$NM_SCRATCH_ROOT/cache/pip"
export HF_HOME="$NM_SCRATCH_ROOT/cache/huggingface"
export GIT_TERMINAL_PROMPT=0
