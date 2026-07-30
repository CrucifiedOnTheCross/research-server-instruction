#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/srv/research/projects/default/research-ml}"
SPLIT="${ISIC2019_VAL_SPLIT:-/srv/research/projects/default/isic2019/splits/val.csv}"

cd "${PROJECT_ROOT}"
python tools/analyze_stage16b_results.py \
  --project-root "${PROJECT_ROOT}" \
  --split "${SPLIT}" \
  --out-dir outputs/reports/stage16b_analysis \
  --bootstrap-replicates 5000 \
  --seed 20260730
