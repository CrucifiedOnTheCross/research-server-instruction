#!/usr/bin/env bash
set -euo pipefail

NAME="${NAME:-research-stage16g-segdata}"
IMAGE="${IMAGE:-local/research-cuda-notebook:latest}"
PROJECT_ROOT="${PROJECT_ROOT:-/srv/research/projects/default}"
WORKDIR="$PROJECT_ROOT/research-ml"
USER_SPEC="${USER_SPEC:-1000:1006}"
GROUP_ID="${GROUP_ID:-1006}"

docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d \
  --name "$NAME" \
  --user "$USER_SPEC" \
  --group-add "$GROUP_ID" \
  -v "$PROJECT_ROOT:$PROJECT_ROOT" \
  -w "$WORKDIR" \
  "$IMAGE" \
  bash -lc "source .venv/bin/activate && python tools/prepare_stage16g_segmentation_data.py"

echo "Container: $NAME"
