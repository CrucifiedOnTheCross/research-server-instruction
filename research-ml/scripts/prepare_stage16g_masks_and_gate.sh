#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/srv/research/projects/default}"
WORKDIR="${WORKDIR:-$PROJECT_ROOT/research-ml}"
DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/isic2019}"

cd "$WORKDIR"
source .venv/bin/activate

python tools/prepare_stage16g_masks.py --data-root "$DATA_ROOT"
python tools/check_stage16g_readiness.py \
  --config configs/stage16g_generator_qualification.yaml \
  --report outputs/reports/stage16g_readiness.json
