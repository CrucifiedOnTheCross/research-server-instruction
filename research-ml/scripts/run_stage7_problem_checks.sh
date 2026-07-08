#!/usr/bin/env bash
set -euo pipefail

python tools/make_synthetic_class_ablation.py \
  --real-train-csv /srv/research/projects/default/ham10000/splits/train.csv \
  --synthetic-csv /srv/research/projects/default/ham10000/splits/stage6/selected_synthetic_geometry.csv \
  --out-csv /srv/research/projects/default/ham10000/splits/stage7/train_stage7_geometry_mel_only.csv \
  --classes mel \
  --seed 42

python -m src.train --config configs/ham10000_stage7_geometry_mel_only.yaml \
  experiment.name="stage7_geometry_mel_only_ce_weighted"

python -m src.train --config configs/ham10000_stage7_real_logit_adjust_tau1.yaml \
  experiment.name="stage7_real_logit_adjust_tau1_weighted"
