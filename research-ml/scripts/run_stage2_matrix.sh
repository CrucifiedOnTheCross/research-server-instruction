#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/ham10000_stage2_synthetic.yaml}"
TRAIN_CSV="${TRAIN_CSV:-splits/stage2/train_stage2_selected.csv}"
EXPERIMENT_SUFFIX="${EXPERIMENT_SUFFIX:-}"

python -m src.train --config "$CONFIG" \
  experiment.name="stage2_synthetic${EXPERIMENT_SUFFIX}_ce_weighted" \
  data.train_csv="$TRAIN_CSV" \
  imbalance.loss=cross_entropy \
  imbalance.sampler=weighted

python -m src.train --config "$CONFIG" \
  experiment.name="stage2_synthetic${EXPERIMENT_SUFFIX}_balanced_softmax_none" \
  data.train_csv="$TRAIN_CSV" \
  imbalance.loss=balanced_softmax \
  imbalance.sampler=none
