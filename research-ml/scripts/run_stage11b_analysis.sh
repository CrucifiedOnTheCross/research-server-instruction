#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/srv/research/projects/default/research-ml}"
OUTPUTS="${OUTPUTS:-$PROJECT_ROOT/outputs}"
SPLIT_CSV="${SPLIT_CSV:-/srv/research/projects/default/ham10000/splits/stage8/val_real.csv}"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$PROJECT_ROOT"

experiments=(
  stage11b_replay_strict_id_convnext_small_384
  stage11b_synthetic_strict_id_convnext_small_384
)

for experiment in "${experiments[@]}"; do
  for seed in 42 43 44; do
    run_dir="$(find "$OUTPUTS/$experiment" -mindepth 1 -maxdepth 1 -type d -name "*_${seed}" | sort | tail -1)"
    python tools/calibrate_stage9_predictions.py \
      --run-dir "$run_dir" \
      --split-csv "$SPLIT_CSV" \
      --seed "$seed" \
      --out-dir "$run_dir/calibration_diagnostic" \
      >/dev/null
  done
done

python tools/analyze_stage11b_results.py \
  --project-root "$PROJECT_ROOT" \
  --data-root /srv/research/projects/default/ham10000 \
  --out-dir "$OUTPUTS/reports/stage11b_analysis" \
  --bootstrap-replicates 5000

python tools/sync_mlflow_runs.py \
  --outputs "$OUTPUTS" \
  --tracking-uri http://10.200.1.180:5000 \
  --experiment "HAM10000 Historical" \
  --include-reports
