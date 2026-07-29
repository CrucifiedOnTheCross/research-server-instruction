#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/srv/research/projects/default/research-ml}"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$PROJECT_ROOT"

python tools/select_stage13_multiencoder_coverage.py \
  --project-root "$PROJECT_ROOT" \
  --data-root /srv/research/projects/default/ham10000 \
  --batch-size 32 \
  --num-workers 12 \
  --prdc-k 5

