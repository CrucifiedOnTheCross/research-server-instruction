#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/srv/research/projects/default/ham10000}"
PROJECT_ROOT="${PROJECT_ROOT:-/srv/research/projects/default/research-ml}"
SEEDS="${SEEDS:-42 43 44}"
STRATA="${STRATA:-strict_id aid_radial ood_far random_remaining}"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

completed_run_exists() {
  local experiment="$1"
  local seed="$2"
  compgen -G "outputs/$experiment/*_${seed}/summary.json" >/dev/null
}

run_once() {
  local config="$1"
  local experiment="$2"
  local train_csv="$3"
  local seed="$4"
  if completed_run_exists "$experiment" "$seed"; then
    echo "Skipping completed $experiment seed=$seed"
    return
  fi
  python -m src.train \
    --config "$config" \
    "experiment.name=$experiment" \
    "data.train_csv=$train_csv" \
    "runtime.seed=$seed"
  python tools/summarize_multiseed.py \
    --outputs outputs \
    --prefix stage10_ \
    --out-dir outputs/reports/stage10_multiseed
}

python tools/make_stage10_geometry_strata.py \
  --data-root "$DATA_ROOT" \
  --train-csv splits/stage8/train_real.csv \
  --scores-csv "$PROJECT_ROOT/outputs/reports/stage8_dino_geometry/synthetic_dino_scores.csv" \
  --out-dir splits/stage10 \
  --target-classes mel,akiec,bkl \
  --dose-per-class 30 \
  --sample-weight 0.5 \
  --max-source-reuse-per-stratum 2 \
  --seed 20260727

for stratum in $STRATA; do
  for seed in $SEEDS; do
    run_once \
      configs/ham10000_stage10_synthetic_geometry_384.yaml \
      "stage10_synthetic_${stratum}_384" \
      "splits/stage10/train_synthetic_${stratum}.csv" \
      "$seed"
    run_once \
      configs/ham10000_stage10_source_replay_384.yaml \
      "stage10_replay_${stratum}_384" \
      "splits/stage10/train_source_replay_${stratum}.csv" \
      "$seed"
  done
done
