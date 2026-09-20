#!/usr/bin/env bash
# Fixed-count null continuation; resource limits and dependencies set at submission.
set -euo pipefail
: "${NM_ORIGINAL_RELEASE:?original immutable analysis release required}"
: "${NM_NULL_RUNNER:?sealed continuation runner required}"
: "${NM_NULL_PLAN:?sealed continuation plan required}"
cd "$NM_ORIGINAL_RELEASE"
source scripts/alliance/runtime.sh
python "$NM_NULL_RUNNER" "$@" --plan "$NM_NULL_PLAN"
