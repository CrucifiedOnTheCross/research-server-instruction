#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/srv/research/projects/default/research-ml}"
OUTPUTS="${OUTPUTS:-$PROJECT_ROOT/outputs}"
SPLIT_CSV="${SPLIT_CSV:-/srv/research/projects/default/ham10000/splits/stage8/val_real.csv}"
MLFLOW_URI="${MLFLOW_URI:-http://10.200.1.180:5000}"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$PROJECT_ROOT"

experiments=(
  stage15a_original_replay_convnext_small_384
  stage15a_offline_crop_convnext_small_384
  stage15a_vae_roundtrip_convnext_small_384
  stage15a_img2img_strength05_convnext_small_384
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

comparisons=(
  "offline_crop:stage15a_offline_crop_convnext_small_384"
  "vae_roundtrip:stage15a_vae_roundtrip_convnext_small_384"
  "img2img_strength05:stage15a_img2img_strength05_convnext_small_384"
)

for item in "${comparisons[@]}"; do
  label="${item%%:*}"
  experiment="${item#*:}"
  python tools/analyze_stage11b_results.py \
    --project-root "$PROJECT_ROOT" \
    --data-root /srv/research/projects/default/ham10000 \
    --out-dir "$OUTPUTS/reports/stage15a_analysis/$label" \
    --stage stage15a \
    --stratum "$label" \
    --replay-experiment stage15a_original_replay_convnext_small_384 \
    --synthetic-experiment "$experiment" \
    --bootstrap-replicates 5000 \
    --seed 20260729
done

python tools/sync_mlflow_runs.py \
  --outputs "$OUTPUTS" \
  --tracking-uri "$MLFLOW_URI" \
  --experiment "HAM10000 Historical" \
  --include-reports
