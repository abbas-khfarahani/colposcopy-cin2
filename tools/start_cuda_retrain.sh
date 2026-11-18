#!/usr/bin/env bash
# Start full model retrain on CUDA (run this in a NORMAL WSL / Windows Terminal,
# NOT inside a Cursor agent sandbox — that environment has no /dev/dxg).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source /home/admin1/miniforge3/etc/profile.d/conda.sh
conda activate colpo
export PYTHONUNBUFFERED=1

echo "=== GPU check ==="
nvidia-smi | head -20
python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA not available in this shell — open Ubuntu/WSL Terminal, not Cursor agent"
print("OK:", torch.cuda.get_device_name(0))
PY

# Stop any old CPU retrain
pkill -f "retrain_updated_dataset.py" 2>/dev/null || true
pkill -f "ml/train.py" 2>/dev/null || true
sleep 2

# Enlarge shm if possible (helps num_workers)
sudo mount -o remount,size=2G /dev/shm 2>/dev/null || true
df -h /dev/shm

mkdir -p ml/runs/retrain_logs
LOG="ml/runs/retrain_logs/retrain_cuda.console.log"
echo "Logging to $LOG"
echo "Status: ml/runs/retrain_status.json"

nohup python -u ml/scripts/retrain_updated_dataset.py --device cuda --num-workers 4 \
  > "$LOG" 2>&1 &
echo "Started PID=$!"
sleep 5
tail -30 "$LOG"
echo
echo "Follow with:  tail -f \"$LOG\""
