#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/srv/research/projects/default/research-ml}"
SEEDS="${SEEDS:-43 44}"
MLFLOW_URI="${MLFLOW_URI:-http://10.200.1.180:5000}"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$PROJECT_ROOT"

python tools/check_stage16a_gate.py --check-files

configs=(
  configs/isic2019_stage16_real_ce_natural_384.yaml
  configs/isic2019_stage16_real_balanced_softmax_384.yaml
)

completed_run_exists() {
  local experiment="$1"
  local seed="$2"
  compgen -G "outputs/$experiment/*_${seed}/summary.json" >/dev/null
}

for seed in $SEEDS; do
  for config in "${configs[@]}"; do
    experiment="$(
      python -c "from src.config import load_config; print(load_config('$config')['experiment']['name'])"
    )"
    if completed_run_exists "$experiment" "$seed"; then
      echo "Skipping completed $experiment seed=$seed"
      continue
    fi
    python -m src.train --config "$config" "runtime.seed=$seed"
    python tools/sync_mlflow_runs.py \
      --outputs outputs \
      --tracking-uri "$MLFLOW_URI" \
      --experiment "ISIC2019 Stage16"
  done
done
