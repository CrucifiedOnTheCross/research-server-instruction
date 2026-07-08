#!/usr/bin/env bash
set -euo pipefail

NAME="${NAME:-research-ml-stage1}"
CONFIG="${CONFIG:-configs/ham10000_stage1.yaml}"
IMAGE="${IMAGE:-local/research-cuda-notebook:latest}"
PROJECT_ROOT="${PROJECT_ROOT:-/srv/research/projects/default}"
WORKDIR="$PROJECT_ROOT/research-ml"
USER_SPEC="${USER_SPEC:-1000:1006}"
GROUP_ID="${GROUP_ID:-1006}"
SHM_SIZE="${SHM_SIZE:-16g}"
STAMP="$(date +%Y%m%d-%H%M%S)"

mkdir -p "$WORKDIR/server-logs"
docker rm -f "$NAME" >/dev/null 2>&1 || true

docker run -d \
  --name "$NAME" \
  --shm-size="$SHM_SIZE" \
  --gpus all \
  --user "$USER_SPEC" \
  --group-add "$GROUP_ID" \
  -v "$PROJECT_ROOT:$PROJECT_ROOT" \
  -w "$WORKDIR" \
  "$IMAGE" \
  bash -lc "source .venv/bin/activate && set -o pipefail && bash scripts/run_stage1_matrix.sh '$CONFIG' 2>&1 | tee 'server-logs/stage1_$STAMP.log'"

echo "Container: $NAME"
echo "Log file: $WORKDIR/server-logs/stage1_$STAMP.log"
echo "Follow: docker logs -f $NAME"

