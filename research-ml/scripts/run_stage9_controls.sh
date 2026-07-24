#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/srv/research/projects/default/ham10000}"
PROJECT_ROOT="${PROJECT_ROOT:-/srv/research/projects/default/research-ml}"
SEEDS="${SEEDS:-42 43 44}"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

completed_run_exists() {
  local experiment="$1"
  local seed="$2"
  compgen -G "outputs/$experiment/*_${seed}/summary.json" >/dev/null
}

run_config() {
  local config_name="$1"
  local experiment_name="${config_name#ham10000_}"
  experiment_name="${experiment_name%.yaml}"
  for seed in $SEEDS; do
    if completed_run_exists "$experiment_name" "$seed"; then
      echo "Skipping completed $experiment_name seed=$seed"
      continue
    fi
    python -m src.train \
      --config "configs/$config_name" \
      "experiment.name=$experiment_name" \
      "runtime.seed=$seed"
    python tools/summarize_multiseed.py \
      --outputs outputs \
      --prefix stage9_ \
      --out-dir outputs/reports/stage9_multiseed
  done
}

python tools/make_source_matched_replay.py \
  --data-root "$DATA_ROOT" \
  --train-csv splits/stage8/train_real.csv \
  --synthetic-scores-csv "$PROJECT_ROOT/outputs/reports/stage8_dino_geometry/synthetic_dino_scores.csv" \
  --out-train-csv splits/stage9/train_source_matched_replay.csv \
  --out-replay-csv splits/stage9/source_matched_replay_rows.csv \
  --sample-weight 0.5 \
  --expected-selected 240

run_config "ham10000_stage9_real_ce_undersample_384.yaml"
run_config "ham10000_stage9_real_balanced_softmax_natural_384.yaml"
run_config "ham10000_stage9_source_replay_weighted_384.yaml"

for seed in $SEEDS; do
  experiment_name="stage9_crt_weighted_384"
  if completed_run_exists "$experiment_name" "$seed"; then
    echo "Skipping completed $experiment_name seed=$seed"
    continue
  fi
  checkpoint="$(
    find outputs/stage8_real_ce_natural_384 -mindepth 2 -maxdepth 2 \
      -type f -path "*_${seed}/best.pt" -printf '%T@ %p\n' \
      | sort -n \
      | tail -n 1 \
      | cut -d' ' -f2-
  )"
  if [[ -z "$checkpoint" ]]; then
    echo "No Stage 8 natural CE checkpoint found for seed=$seed" >&2
    exit 1
  fi
  python -m src.train \
    --config configs/ham10000_stage9_crt_weighted_384.yaml \
    "experiment.name=$experiment_name" \
    "runtime.seed=$seed" \
    "model.initial_checkpoint=$checkpoint"
  python tools/summarize_multiseed.py \
    --outputs outputs \
    --prefix stage9_ \
    --out-dir outputs/reports/stage9_multiseed
done
