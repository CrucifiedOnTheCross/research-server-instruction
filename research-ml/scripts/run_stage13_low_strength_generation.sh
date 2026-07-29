#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/srv/research/projects/default/research-ml}"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$PROJECT_ROOT"

python tools/generate_synthetic_img2img.py \
  --config configs/stage13_generation_low_strength.yaml \
  --data-root /srv/research/projects/default/ham10000 \
  --train-csv splits/stage8/train_real.csv

python tools/select_stage13_multiencoder_coverage.py \
  --project-root "$PROJECT_ROOT" \
  --data-root /srv/research/projects/default/ham10000 \
  --pool-csv synthetic/ham10000_stage13_low_strength_img2img/synthetic_manifest.csv \
  --strict-csv splits/stage10/selected_synthetic_strict_id.csv \
  --out-dir outputs/reports/stage13_low_strength_selection \
  --split-out-dir splits/stage13_low_strength \
  --expected-pool-size 960
