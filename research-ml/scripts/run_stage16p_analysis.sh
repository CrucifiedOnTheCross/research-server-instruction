#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/srv/research/projects/default/research-ml}"
cd "$PROJECT_ROOT"
python tools/analyze_stage16p_results.py \
  --project-root "$PROJECT_ROOT" \
  --data-root /srv/research/projects/default/isic2019 \
  --split /srv/research/projects/default/isic2019/splits/val.csv \
  --out-dir outputs/reports/stage16p_analysis \
  --bootstrap-replicates 5000 \
  --seed 20260730
