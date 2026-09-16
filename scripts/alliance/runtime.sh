#!/usr/bin/env bash
# Reuse a fresh Alliance environment only when dependency inputs are identical.
set -euo pipefail
source scripts/alliance/environment.sh
digest=$(python scripts/alliance/verify_release.py)
environment_digest="${NM_ENV_RELEASE:-$digest}"
export NM_SOURCE_RELEASE="$digest" NM_ENV_RELEASE="$environment_digest"
python - <<'PY'
import json, os
from pathlib import Path
root = Path(os.environ['NM_PROJECT_ROOT'])
current = json.loads(Path('release_manifest.json').read_text())
original = json.loads((root/'releases'/os.environ['NM_ENV_RELEASE']/'release_manifest.json').read_text())
for path in ['pyproject.toml', 'scripts/alliance/constraints.txt', 'scripts/alliance/environment.sh']:
    if current['files'][path] != original['files'][path]:
        raise RuntimeError(f'Environment rebuild required: {path}')
PY
source "$NM_PROJECT_ROOT/environments/$environment_digest/bin/activate"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python - <<'PY'
from pathlib import Path
import neural_manifolds
assert Path(neural_manifolds.__file__).resolve().is_relative_to(Path('src').resolve())
PY
