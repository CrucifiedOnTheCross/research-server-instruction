#!/usr/bin/env bash
set -euo pipefail

NAME="${NAME:-research-ml-results-jupyter}"
PORT="${PORT:-8011}"
IMAGE="${IMAGE:-local/research-cuda-notebook:latest}"
PROJECT_ROOT="${PROJECT_ROOT:-/srv/research/projects/default}"
WORKDIR="$PROJECT_ROOT/research-ml"
USER_SPEC="${USER_SPEC:-1000:1006}"
GROUP_ID="${GROUP_ID:-1006}"
TOKEN="${TOKEN:-results}"

docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d \
  --name "$NAME" \
  --shm-size=2g \
  --user "$USER_SPEC" \
  --group-add "$GROUP_ID" \
  -p "0.0.0.0:$PORT:$PORT" \
  -v "$PROJECT_ROOT:$PROJECT_ROOT" \
  -w "$WORKDIR" \
  "$IMAGE" \
  bash -lc "source .venv/bin/activate 2>/dev/null || true; while true; do python3 tools/build_results_index.py --outputs outputs; sleep 30; done & jupyter lab --no-browser --ip=0.0.0.0 --port=$PORT --ServerApp.root_dir=$WORKDIR/outputs --ServerApp.token=$TOKEN --ServerApp.password= --ServerApp.allow_remote_access=True --ServerApp.disable_check_xsrf=True"

echo "Dashboard: http://10.200.1.180:$PORT/files/index.html?token=$TOKEN"
echo "Files: http://10.200.1.180:$PORT/lab/tree/?token=$TOKEN"

