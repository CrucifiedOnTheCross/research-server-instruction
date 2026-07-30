#!/usr/bin/env bash
set -euo pipefail

NAME="${NAME:-research-stage16g-segmentation}"
IMAGE="${IMAGE:-local/research-cuda-notebook:latest}"
PROJECT_ROOT="${PROJECT_ROOT:-/srv/research/projects/default}"
WORKDIR="$PROJECT_ROOT/research-ml"
USER_SPEC="${USER_SPEC:-1000:1006}"
GROUP_ID="${GROUP_ID:-1006}"

docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d \
  --name "$NAME" \
  --gpus all \
  --shm-size 16g \
  --user "$USER_SPEC" \
  --group-add "$GROUP_ID" \
  -v "$PROJECT_ROOT:$PROJECT_ROOT" \
  -w "$WORKDIR" \
  -e OMP_NUM_THREADS=24 \
  -e MKL_NUM_THREADS=24 \
  "$IMAGE" \
  bash -lc "source .venv/bin/activate && python tools/train_stage16g_segmenter.py"

echo "Container: $NAME"
