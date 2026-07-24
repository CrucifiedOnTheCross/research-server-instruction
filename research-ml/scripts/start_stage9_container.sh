#!/usr/bin/env bash
set -euo pipefail

NAME="${NAME:-research-stage9-controls}"
IMAGE="${IMAGE:-local/research-cuda-notebook:latest}"
PROJECT_ROOT="${PROJECT_ROOT:-/srv/research/projects/default}"
WORKDIR="$PROJECT_ROOT/research-ml"
USER_SPEC="${USER_SPEC:-1000:1006}"
GROUP_ID="${GROUP_ID:-1006}"
SHM_SIZE="${SHM_SIZE:-24g}"
STAMP="$(date +%Y%m%d-%H%M%S)"

mkdir -p "$WORKDIR/server-logs"
mkdir -p "$WORKDIR/.cache/huggingface" "$WORKDIR/.cache/torch"
docker rm -f "$NAME" >/dev/null 2>&1 || true

docker run -d \
  --name "$NAME" \
  --shm-size="$SHM_SIZE" \
  --gpus all \
  --user "$USER_SPEC" \
  --group-add "$GROUP_ID" \
  -v "$PROJECT_ROOT:$PROJECT_ROOT" \
  -w "$WORKDIR" \
  -e "HF_HOME=$WORKDIR/.cache/huggingface" \
  -e "TORCH_HOME=$WORKDIR/.cache/torch" \
  -e "OMP_NUM_THREADS=24" \
  -e "MKL_NUM_THREADS=24" \
  "$IMAGE" \
  bash -lc "source .venv/bin/activate && set -o pipefail && bash scripts/run_stage9_controls.sh 2>&1 | tee 'server-logs/stage9_$STAMP.log'"

echo "Container: $NAME"
echo "Structured results: $WORKDIR/outputs/stage9_*"
echo "Summary: $WORKDIR/outputs/reports/stage9_multiseed"
