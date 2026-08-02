#!/usr/bin/env bash
set -euo pipefail

NAME="${NAME:-longtail-stage2-lesion-audit}"
IMAGE="${IMAGE:-local/research-cuda-notebook:latest}"
PROJECT_ROOT="${PROJECT_ROOT:-/srv/research/projects/default}"
WORKDIR="$PROJECT_ROOT/longtail-medical"
RESEARCH_VENV="${RESEARCH_VENV:-$PROJECT_ROOT/research-ml/.venv}"
USER_SPEC="${USER_SPEC:-1000:1006}"
GROUP_ID="${GROUP_ID:-1006}"
STAMP="$(date +%Y%m%d-%H%M%S)"

mkdir -p "$WORKDIR/server-logs" "$WORKDIR/.cache/torch"
docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d \
  --name "$NAME" \
  --shm-size=24g \
  --gpus all \
  --user "$USER_SPEC" \
  --group-add "$GROUP_ID" \
  -v "$PROJECT_ROOT:$PROJECT_ROOT" \
  -w "$WORKDIR" \
  -e "PYTHONPATH=$WORKDIR/src:$WORKDIR" \
  -e "TORCH_HOME=$WORKDIR/.cache/torch" \
  -e "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" \
  -e "OMP_NUM_THREADS=24" \
  -e "MKL_NUM_THREADS=24" \
  "$IMAGE" \
  bash -lc "source '$RESEARCH_VENV/bin/activate' && set -o pipefail && python tools/run_stage2_training_matrix.py 2>&1 | tee 'server-logs/stage2_$STAMP.log'"

echo "Container: $NAME"
echo "Structured results: $WORKDIR/outputs/stage2"
echo "MLflow: http://10.200.1.180:5000"
