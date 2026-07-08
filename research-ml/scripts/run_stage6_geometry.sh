#!/usr/bin/env bash
set -euo pipefail

python tools/stage6_feature_geometry.py \
  --encoder-run-dir outputs/stage1_cross_entropy_weighted/20260708-060820_42 \
  --synthetic-csv synthetic/ham10000_mel_boundary_img2img_v1/synthetic_manifest.csv \
  --out-dir /srv/research/projects/default/ham10000/reports/stage6_feature_geometry \
  --split-out-dir /srv/research/projects/default/ham10000/splits/stage6 \
  --target-classes mel,akiec,bkl \
  --select-classes mel,akiec \
  --top-k-per-class 80 \
  --real-k 5 \
  --min-feature-margin 0.02 \
  --max-real-distance-quantile 0.80 \
  --diversity-min-distance 0.035

python -m src.train --config configs/ham10000_stage6_geometry_mel_akiec.yaml \
  experiment.name="stage6_geometry_mel_akiec_ce_weighted"
