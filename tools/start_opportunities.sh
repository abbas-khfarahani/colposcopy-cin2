#!/usr/bin/env bash
# Dataset-opportunity audit + experiments (separate from ml.protocol).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate colpo
fi
mkdir -p ml/runs/opportunities/LOGS
exec python -u -m ml.opportunities run --all --device cuda --num-workers 4 \
  2>&1 | tee ml/runs/opportunities/LOGS/opportunities.console.log
