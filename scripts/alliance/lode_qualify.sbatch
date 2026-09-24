#!/usr/bin/env bash
set -euo pipefail
source scripts/alliance/runtime.sh
python -m pytest -q tests/test_lode_external_validation.py tests/test_revised_branches.py tests/test_revised_execution.py
python - <<'PY'
import mne
from neural_manifolds.validation.lode import preflight
print('LODE_RUNTIME_IMPORTS_OK', mne.__version__, preflight.__name__, flush=True)
PY
