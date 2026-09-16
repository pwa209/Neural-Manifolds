#!/usr/bin/env bash
#SBATCH --job-name=nm-v2-qualify
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:h100:1
set -euo pipefail
source scripts/alliance/environment.sh
digest=$(python scripts/alliance/verify_release.py)
source "$NM_PROJECT_ROOT/environments/$digest/bin/activate"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
receipt="$NM_PROJECT_ROOT/provenance/qualification/$SLURM_JOB_ID"
mkdir -p "$receipt"
python -m pip check
python -m pytest -q -p no:cacheprovider --junitxml="$receipt/tests.xml"
python - <<'PY'
import torch
assert torch.cuda.is_available()
x=torch.randn(128,128,device='cuda',requires_grad=True)
(x@x.T).square().mean().backward()
assert torch.isfinite(x.grad).all()
print('CUDA_SMOKE_PASSED',torch.cuda.get_device_name(0),torch.__version__)
PY
printf 'QUALIFICATION_PASSED\n' > "$receipt/PASSED"
