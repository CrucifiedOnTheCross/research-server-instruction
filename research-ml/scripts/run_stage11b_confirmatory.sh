#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/srv/research/projects/default/research-ml}"
SEEDS="${SEEDS:-42 43 44}"
MLFLOW_URI="${MLFLOW_URI:-http://10.200.1.180:5000}"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

cd "$PROJECT_ROOT"

mapfile -t CONFIGS < <(
  python tools/check_stage11b_gate.py \
    --decision configs/stage11_decision.yaml \
    --outputs outputs \
    --data-root /srv/research/projects/default/ham10000 \
    --report outputs/reports/stage11b_gate.json \
    --print-configs
)

completed_run_exists() {
  local experiment="$1"
  local seed="$2"
  compgen -G "outputs/$experiment/*_${seed}/summary.json" >/dev/null
}

for seed in $SEEDS; do
  for config in "${CONFIGS[@]}"; do
    experiment="$(
      python -c "from src.config import load_config; print(load_config('$config')['experiment']['name'])"
    )"
    if completed_run_exists "$experiment" "$seed"; then
      echo "Skipping completed $experiment seed=$seed"
      continue
    fi
    python -m src.train \
      --config "$config" \
      "runtime.seed=$seed"
    python tools/summarize_multiseed.py \
      --outputs outputs \
      --prefix stage11b_ \
      --out-dir outputs/reports/stage11b_multiseed
    python tools/sync_mlflow_runs.py \
      --outputs outputs \
      --tracking-uri "$MLFLOW_URI" \
      --experiment "HAM10000 Historical"
  done
done
