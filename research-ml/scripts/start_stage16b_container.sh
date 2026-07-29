#!/usr/bin/env bash
set -euo pipefail

NAME="${NAME:-research-stage16b-isic2019}"
IMAGE="${IMAGE:-local/research-cuda-notebook:latest}"
PROJECT_ROOT="${PROJECT_ROOT:-/srv/research/projects/default}"
WORKDIR="$PROJECT_ROOT/research-ml"
USER_SPEC="${USER_SPEC:-1000:1006}"
GROUP_ID="${GROUP_ID:-1006}"
SHM_SIZE="${SHM_SIZE:-24g}"
STAMP="$(date +%Y%m%d-%H%M%S)"

mkdir -p "$WORKDIR/server-logs" "$WORKDIR/.cache/huggingface" "$WORKDIR/.cache/torch"

docker run --rm \
  --user "$USER_SPEC" \
  --group-add "$GROUP_ID" \
  -v "$PROJECT_ROOT:$PROJECT_ROOT" \
  -w "$WORKDIR" \
  "$IMAGE" \
  bash -lc "source .venv/bin/activate && python tools/check_stage16a_gate.py --check-files"

docker run --rm \
  --gpus all \
  --user "$USER_SPEC" \
  --group-add "$GROUP_ID" \
  -v "$PROJECT_ROOT:$PROJECT_ROOT" \
  -w "$WORKDIR" \
  -e "HF_HOME=$WORKDIR/.cache/huggingface" \
  "$IMAGE" \
  bash -lc "source .venv/bin/activate && python -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'"

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
  -e "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" \
  "$IMAGE" \
  bash -lc "source .venv/bin/activate && set -o pipefail && bash scripts/run_stage16b_isic2019_baselines.sh 2>&1 | tee 'server-logs/stage16b_$STAMP.log'"

echo "Container: $NAME"
echo "Structured results: $WORKDIR/outputs/stage16_isic2019_*"
echo "MLflow: http://10.200.1.180:5000"
