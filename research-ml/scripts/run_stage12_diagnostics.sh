#!/usr/bin/env bash
set -euo pipefail

cd /srv/research/projects/default/research-ml
export PYTHONPATH="/srv/research/projects/default/research-ml${PYTHONPATH:+:$PYTHONPATH}"

python tools/analyze_stage12_failure_modes.py \
  --project-root /srv/research/projects/default/research-ml \
  --data-root /srv/research/projects/default/ham10000 \
  --batch-size 32 \
  --num-workers 12 \
  --prdc-k 5 \
  --bootstrap-replicates 5000
