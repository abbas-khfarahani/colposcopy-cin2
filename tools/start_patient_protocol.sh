#!/usr/bin/env bash
# Launch the patient-level grouped-CV protocol on CUDA.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate colpo
fi
mkdir -p ml/runs/protocol/LOGS
exec python -u -m ml.protocol run --all --device cuda --num-workers 4 \
  2>&1 | tee ml/runs/protocol/LOGS/protocol.console.log
