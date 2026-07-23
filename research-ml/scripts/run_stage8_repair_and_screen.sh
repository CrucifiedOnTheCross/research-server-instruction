#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/srv/research/projects/default/ham10000}"
PROJECT_ROOT="${PROJECT_ROOT:-/srv/research/projects/default/research-ml}"
GENERATION_CONFIG="${GENERATION_CONFIG:-configs/stage8_generation_corrected.yaml}"
SEEDS="${SEEDS:-42 43 44 45 46}"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

python tools/make_stage8_splits.py \
  --data-root "$DATA_ROOT" \
  --out-dir splits/stage8 \
  --locked-test-size 0.15 \
  --val-size 0.15 \
  --seed 20260723

python tools/generate_synthetic_img2img.py \
  --config "$GENERATION_CONFIG" \
  --data-root "$DATA_ROOT" \
  --train-csv splits/stage8/train_real.csv

python tools/audit_synthetic_artifacts.py \
  --data-root "$DATA_ROOT" \
  --real-csv splits/stage8/train_real.csv \
  --synthetic-csv synthetic/ham10000_stage8_corrected_img2img/synthetic_manifest.csv \
  --out-dir "$PROJECT_ROOT/outputs/reports/stage8_artifact_audit"

python tools/make_stage8_splits.py \
  --data-root "$DATA_ROOT" \
  --synthetic-csv synthetic/ham10000_stage8_corrected_img2img/synthetic_manifest.csv \
  --out-dir splits/stage8 \
  --locked-test-size 0.15 \
  --val-size 0.15 \
  --seed 20260723

python tools/stage6_feature_geometry.py \
  --data-config configs/ham10000_stage8_real_ce_weighted_384.yaml \
  --encoder-model vit_base_patch14_dinov2.lvd142m \
  --synthetic-csv splits/stage8/synthetic_candidates_train.csv \
  --out-dir "$PROJECT_ROOT/outputs/reports/stage8_dino_geometry" \
  --split-out-dir "$DATA_ROOT/splits/stage8" \
  --target-classes mel,akiec,bkl \
  --select-classes mel,akiec,bkl \
  --top-k-per-class 80 \
  --real-k 5 \
  --min-feature-margin 0.02 \
  --max-real-distance-quantile 0.80 \
  --diversity-min-distance 0.025 \
  --train-name train_stage8_dino_selected.csv \
  --selected-name selected_synthetic_dino.csv \
  --sample-name synthetic_dino_scores.csv \
  --batch-size 48 \
  --num-workers 12 \
  --seed 20260723

configs=(
  "ham10000_stage8_real_ce_natural_384.yaml"
  "ham10000_stage8_real_ce_weighted_384.yaml"
  "ham10000_stage8_real_logit_adjust_natural_384.yaml"
  "ham10000_stage8_synthetic_dino_384.yaml"
)

for config_name in "${configs[@]}"; do
  experiment_name="${config_name#ham10000_}"
  experiment_name="${experiment_name%.yaml}"
  for seed in $SEEDS; do
    python -m src.train \
      --config "configs/$config_name" \
      "experiment.name=$experiment_name" \
      "runtime.seed=$seed"
    python tools/summarize_multiseed.py \
      --outputs outputs \
      --prefix stage8_ \
      --out-dir outputs/reports/stage8_multiseed
  done
done
