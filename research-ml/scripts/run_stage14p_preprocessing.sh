#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/srv/research/projects/default/research-ml}"
SEEDS="${SEEDS:-42 43 44}"
MLFLOW_URI="${MLFLOW_URI:-http://10.200.1.180:5000}"
CONFIG="configs/ham10000_stage14p_real_aspect_pad_convnext_small_384.yaml"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$PROJECT_ROOT"

python tools/check_stage14p_gate.py
experiment="$(
  python -c "from src.config import load_config; print(load_config('$CONFIG')['experiment']['name'])"
)"

for seed in $SEEDS; do
  if compgen -G "outputs/$experiment/*_${seed}/summary.json" >/dev/null; then
    echo "Skipping completed $experiment seed=$seed"
    continue
  fi
  python -m src.train --config "$CONFIG" "runtime.seed=$seed"
  python tools/summarize_multiseed.py \
    --outputs outputs \
    --prefix stage14p_ \
    --out-dir outputs/reports/stage14p_multiseed
  python tools/sync_mlflow_runs.py \
    --outputs outputs \
    --tracking-uri "$MLFLOW_URI" \
    --experiment "HAM10000 Historical"
done

