#!/usr/bin/env bash
set -euo pipefail

python tools/stage6_visualize_embeddings.py \
  --encoder-run-dir outputs/stage1_cross_entropy_weighted/20260708-060820_42 \
  --pool stage3_utility:splits/stage3/selected_synthetic_utility.csv \
  --pool stage4_mel_akiec:splits/stage4/train_stage4_utility_mel_akiec.csv \
  --pool stage6_geometry:splits/stage6/selected_synthetic_geometry.csv \
  --out-dir /srv/research/projects/default/ham10000/reports/stage6_embedding_visuals \
  --target-classes mel,akiec,bkl \
  --max-real-per-class 250 \
  --max-points-tsne 1800
