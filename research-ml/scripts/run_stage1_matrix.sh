#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/ham10000_stage1.yaml}"

declare -a LOSSES=(
  "cross_entropy"
  "focal"
  "balanced_softmax"
)

declare -a SAMPLERS=(
  "none"
  "weighted"
)

for loss in "${LOSSES[@]}"; do
  for sampler in "${SAMPLERS[@]}"; do
    python -m src.train --config "$CONFIG" \
      experiment.name="stage1_${loss}_${sampler}" \
      imbalance.loss="$loss" \
      imbalance.sampler="$sampler"
  done
done

