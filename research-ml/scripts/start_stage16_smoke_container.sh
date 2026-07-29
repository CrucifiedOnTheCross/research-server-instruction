#!/usr/bin/env bash
set -euo pipefail

NAME="${NAME:-research-stage16-smoke}"
IMAGE="${IMAGE:-local/research-cuda-notebook:latest}"
PROJECT_ROOT="${PROJECT_ROOT:-/srv/research/projects/default}"
WORKDIR="$PROJECT_ROOT/research-ml"
USER_SPEC="${USER_SPEC:-1000:1006}"
GROUP_ID="${GROUP_ID:-1006}"

docker run --rm \
  --user "$USER_SPEC" \
  --group-add "$GROUP_ID" \
  -v "$PROJECT_ROOT:$PROJECT_ROOT" \
  -w "$WORKDIR" \
  "$IMAGE" \
  bash -lc "source .venv/bin/activate && python tools/check_stage16a_gate.py --check-files"

docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d \
  --name "$NAME" \
  --shm-size=24g \
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
  bash -lc "source .venv/bin/activate && python -m src.train \
    --config configs/isic2019_stage16_real_ce_natural_384.yaml \
    experiment.name=smoke_stage16_isic2019_convnext_small_384 \
    data.train_csv=splits/smoke_train.csv \
    data.val_csv=splits/smoke_val.csv \
    data.test_csv=splits/smoke_locked_test.csv \
    training.epochs=1 \
    training.early_stopping_patience=1 \
    tracking.experiment=ISIC2019_Stage16_Smoke \
    evaluation.run_test=false"

echo "Container: $NAME"
echo "Structured results: $WORKDIR/outputs/smoke_stage16_isic2019_convnext_small_384"
