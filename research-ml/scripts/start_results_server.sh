#!/usr/bin/env bash
set -euo pipefail

NAME="${NAME:-research-ml-results}"
PORT="${PORT:-8010}"
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
  -p "0.0.0.0:$PORT:$PORT" \
  -v "$PROJECT_ROOT:$PROJECT_ROOT" \
  -w "$WORKDIR" \
  "$IMAGE" \
  bash -lc "source .venv/bin/activate && while true; do python tools/build_results_index.py --outputs outputs --title 'Research ML Results'; sleep 30; done & python -m http.server '$PORT' --directory outputs"

echo "Results URL: http://10.200.1.180:$PORT/"

