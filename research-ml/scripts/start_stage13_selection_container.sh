#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/srv/research/projects/default}"
WORKDIR="$PROJECT_ROOT/research-ml"
USER_SPEC="${USER_SPEC:-1000:1006}"
GROUP_ID="${GROUP_ID:-1006}"
mkdir -p "$WORKDIR/server-logs" "$WORKDIR/.cache/huggingface" "$WORKDIR/.cache/torch"

docker rm -f research-stage13-selection >/dev/null 2>&1 || true
docker run -d \
  --name research-stage13-selection \
  --shm-size=24g \
  --gpus all \
  --user "$USER_SPEC" \
  --group-add "$GROUP_ID" \
  -v "$PROJECT_ROOT:$PROJECT_ROOT" \
  -w "$WORKDIR" \
  -e "HF_HOME=$WORKDIR/.cache/huggingface" \
  -e "TORCH_HOME=$WORKDIR/.cache/torch" \
  -e "OMP_NUM_THREADS=24" \
  -e "MKL_NUM_THREADS=24" \
  local/research-cuda-notebook:latest \
  bash -lc "source .venv/bin/activate && set -o pipefail && bash scripts/run_stage13_selection.sh 2>&1 | tee server-logs/stage13_selection_$(date +%Y%m%d-%H%M%S).log"

