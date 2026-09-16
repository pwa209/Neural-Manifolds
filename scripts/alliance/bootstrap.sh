#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/alliance/environment.sh
digest=$(python scripts/alliance/verify_release.py)
export NM_ENV_ROOT="$NM_PROJECT_ROOT/environments/$digest"
mkdir -p "$NM_PROJECT_ROOT/provenance/environments" "$PIP_CACHE_DIR" "$MPLCONFIGDIR"
if [[ ! -x "$NM_ENV_ROOT/bin/python" ]]; then virtualenv --no-download "$NM_ENV_ROOT"; fi
source "$NM_ENV_ROOT/bin/activate"
python -m pip install --no-index 'torch==2.6.0' 'torchvision==0.21.0'
python -m pip install -c scripts/alliance/constraints.txt -e '.[eeg,foundation,dynamics,figures,fmri,workflow,dev]'
python -m pip check
python -m pip freeze > "$NM_PROJECT_ROOT/provenance/environments/$digest.freeze.txt"
module -t list 2> "$NM_PROJECT_ROOT/provenance/environments/$digest.modules.txt"
printf 'NM_BOOTSTRAP_COMPLETE %s\n' "$digest"
