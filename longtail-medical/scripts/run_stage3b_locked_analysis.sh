#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/srv/research/projects/default}"
WORKDIR="$PROJECT_ROOT/longtail-medical"
RESEARCH_VENV="${RESEARCH_VENV:-$PROJECT_ROOT/research-ml/.venv}"
IMAGE="${IMAGE:-local/research-cuda-notebook:latest}"
NAME="${NAME:-longtail-stage3b-locked-evaluation}"
USER_SPEC="${USER_SPEC:-1000:1006}"
GROUP_ID="${GROUP_ID:-1006}"

docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run --rm \
  --name "$NAME" --shm-size=24g --gpus all \
  --user "$USER_SPEC" --group-add "$GROUP_ID" \
  -v "$PROJECT_ROOT:$PROJECT_ROOT" -w "$WORKDIR" \
  -e "PYTHONPATH=$WORKDIR/src:$WORKDIR" \
  -e "CODE_COMMIT=$(cat "$WORKDIR/.code-version")" \
  -e "OMP_NUM_THREADS=24" -e "MKL_NUM_THREADS=24" \
  "$IMAGE" bash -lc "source '$RESEARCH_VENV/bin/activate' && python tools/check_stage3b_readiness.py && python tools/evaluate_stage3b_locked.py && python tools/analyze_stage3b.py --bootstrap-repeats 10000 --workers 4"
