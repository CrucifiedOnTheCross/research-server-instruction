#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/srv/research/projects/default}"
WORKDIR="$PROJECT_ROOT/research-ml"
USER_SPEC="${USER_SPEC:-1000:1006}"
GROUP_ID="${GROUP_ID:-1006}"
mkdir -p "$WORKDIR/server-logs" "$WORKDIR/.cache/huggingface" "$WORKDIR/.cache/torch"

docker rm -f research-stage12-diagnostics >/dev/null 2>&1 || true
docker run -d \
  --name research-stage12-diagnostics \
  --gpus all \
  --ipc=host \
  --shm-size=24g \
  --user "$USER_SPEC" \
  --group-add "$GROUP_ID" \
  -e PYTHONUNBUFFERED=1 \
  -e "HF_HOME=$WORKDIR/.cache/huggingface" \
  -e "TORCH_HOME=$WORKDIR/.cache/torch" \
  -e "OMP_NUM_THREADS=24" \
  -e "MKL_NUM_THREADS=24" \
  -v "$PROJECT_ROOT:$PROJECT_ROOT" \
  -w "$WORKDIR" \
  local/research-cuda-notebook:latest \
  bash -lc "source .venv/bin/activate && set -o pipefail && bash scripts/run_stage12_diagnostics.sh 2>&1 | tee server-logs/stage12_$(date +%Y%m%d-%H%M%S).log"
