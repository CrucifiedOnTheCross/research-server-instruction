#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/srv/research/projects/default}"
WORKDIR="$PROJECT_ROOT/research-ml"
USER_SPEC="${USER_SPEC:-1000:1006}"
GROUP_ID="${GROUP_ID:-1006}"
mkdir -p "$WORKDIR/server-logs" "$WORKDIR/.cache/huggingface" "$WORKDIR/.cache/torch"

docker rm -f research-stage16g-generator-smoke >/dev/null 2>&1 || true
docker run -d \
  --name research-stage16g-generator-smoke \
  --shm-size=24g \
  --gpus all \
  --user "$USER_SPEC" \
  --group-add "$GROUP_ID" \
  -v "$PROJECT_ROOT:$PROJECT_ROOT" \
  -w "$WORKDIR" \
  -e "HF_HOME=$WORKDIR/.cache/huggingface" \
  -e "TORCH_HOME=$WORKDIR/.cache/torch" \
  local/research-cuda-notebook:latest \
  bash -lc "set -o pipefail && bash scripts/run_stage16g_generator_smoke.sh 2>&1 | tee server-logs/stage16g_generator_smoke_$(date +%Y%m%d-%H%M%S).log"

