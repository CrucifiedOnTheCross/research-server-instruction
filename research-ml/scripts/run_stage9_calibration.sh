#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/srv/research/projects/default/research-ml}"
DATA_ROOT="${DATA_ROOT:-/srv/research/projects/default/ham10000}"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

for experiment in \
  stage8_real_ce_natural_384 \
  stage8_real_ce_weighted_384 \
  stage8_real_logit_adjust_natural_384 \
  stage8_synthetic_dino_384 \
  stage9_real_ce_undersample_384 \
  stage9_real_balanced_softmax_natural_384 \
  stage9_source_replay_weighted_384 \
  stage9_crt_weighted_384
do
  [[ -d "outputs/$experiment" ]] || continue
  while IFS= read -r run_dir; do
    [[ -f "$run_dir/summary.json" ]] || continue
    python tools/calibrate_stage9_predictions.py \
      --run-dir "$run_dir" \
      --split-csv "$DATA_ROOT/splits/stage8/val_real.csv" \
      --target-class mel \
      --seed 20260724
  done < <(find "outputs/$experiment" -mindepth 1 -maxdepth 1 -type d | sort)
done
