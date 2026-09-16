#!/usr/bin/env bash
set -euo pipefail
source scripts/alliance/runtime.sh
python -m neural_manifolds.continuous.worker --spec "$1"
