#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/srv/research/projects/default/research-ml}"
OUTPUTS="${OUTPUTS:-$PROJECT_ROOT/outputs}"
SPLIT_CSV="${SPLIT_CSV:-/srv/research/projects/default/ham10000/splits/stage8/val_real.csv}"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$PROJECT_ROOT"

experiments=(
  stage8_real_ce_natural_384
  stage11_real_convnext_base_regularized_384
  stage11_real_convnext_small_regularized_384
  stage11_real_dinov2_base_linear_392
  stage11_real_dinov2_base_finetune_392
)

for experiment in "${experiments[@]}"; do
  for seed in 42 43 44; do
    run_dir="$(find "$OUTPUTS/$experiment" -mindepth 1 -maxdepth 1 -type d -name "*_${seed}" | sort | tail -1)"
    python tools/calibrate_stage9_predictions.py \
      --run-dir "$run_dir" \
      --split-csv "$SPLIT_CSV" \
      --seed "$seed" \
      --out-dir "$run_dir/stage11_calibration_diagnostic" \
      >/dev/null
  done
done

python tools/analyze_stage11_results.py \
  --outputs "$OUTPUTS" \
  --split-csv "$SPLIT_CSV" \
  --out-dir "$OUTPUTS/reports/stage11_analysis" \
  --bootstrap-replicates 5000
