#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/ham10000_stage1.yaml}"
VENV_DIR="${VENV_DIR:-.venv}"
LOG_DIR="${LOG_DIR:-server-logs}"
mkdir -p "$LOG_DIR"

source "$VENV_DIR/bin/activate"

echo "=== GPU ==="
nvidia-smi || true

echo "=== Python / Torch ==="
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda available", torch.cuda.is_available())
print("cuda runtime", torch.version.cuda)
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
PY

echo "=== Config/Data Smoke ==="
python - <<'PY'
from src.config import load_config
from src.datasets import make_dataloaders
cfg = load_config("configs/ham10000_stage1.yaml", [])
bundle = make_dataloaders(cfg)
print("classes", bundle.class_to_idx)
print("counts", bundle.class_counts)
batch = next(iter(bundle.loaders["train"]))
print("batch image", tuple(batch["image"].shape))
print("batch target", tuple(batch["target"].shape))
PY

echo "=== Smoke Training: 1 epoch ==="
python -m src.train --config "$CONFIG" \
  experiment.name=smoke_gpu_pipeline \
  training.epochs=1 \
  training.early_stopping_patience=1 \
  runtime.compile=false \
  runtime.amp=bf16

echo "=== Starting stage1 matrix under nohup ==="
RUN_LOG="$LOG_DIR/stage1_$(date +%Y%m%d-%H%M%S).log"
nohup bash scripts/run_stage1_matrix.sh "$CONFIG" > "$RUN_LOG" 2>&1 &
echo "$!" > "$LOG_DIR/stage1.pid"
echo "PID: $(cat "$LOG_DIR/stage1.pid")"
echo "Log: $RUN_LOG"
echo "Follow with: tail -f $RUN_LOG"
