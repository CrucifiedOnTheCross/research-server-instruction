#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/srv/research/projects/default/research-ml}"
OUTPUTS="${OUTPUTS:-$PROJECT_ROOT/outputs}"
SPLIT_CSV="${SPLIT_CSV:-/srv/research/projects/default/ham10000/splits/stage8/val_real.csv}"
MLFLOW_URI="${MLFLOW_URI:-http://10.200.1.180:5000}"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$PROJECT_ROOT"

experiments=(
  stage13b_replay_coverage_convnext_small_384
  stage13b_synthetic_coverage_convnext_small_384
)

for experiment in "${experiments[@]}"; do
  for seed in 42 43 44; do
    run_dir="$(find "$OUTPUTS/$experiment" -mindepth 1 -maxdepth 1 -type d -name "*_${seed}" | sort | tail -1)"
    test -f "$run_dir/summary.json"
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
  --out-dir "$OUTPUTS/reports/stage13b_analysis" \
  --stage stage13b \
  --stratum coverage \
  --replay-experiment stage13b_replay_coverage_convnext_small_384 \
  --synthetic-experiment stage13b_synthetic_coverage_convnext_small_384 \
  --bootstrap-replicates 5000 \
  --seed 20260729

python tools/sync_mlflow_runs.py \
  --outputs "$OUTPUTS" \
  --tracking-uri "$MLFLOW_URI" \
  --experiment "HAM10000 Historical" \
  --include-reports
