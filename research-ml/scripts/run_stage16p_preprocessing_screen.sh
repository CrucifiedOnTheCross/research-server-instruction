#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/srv/research/projects/default/research-ml}"
SEED="${SEED:-42}"
MLFLOW_URI="${MLFLOW_URI:-http://10.200.1.180:5000}"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$PROJECT_ROOT"

python tools/check_stage16a_gate.py --check-files

configs=(
  configs/isic2019_stage16p_real_aspect_pad_natural_384.yaml
  configs/isic2019_stage16p_real_dark_fov_pad_natural_384.yaml
)
for config in "${configs[@]}"; do
  experiment="$(
    python -c "from src.config import load_config; print(load_config('$config')['experiment']['name'])"
  )"
  if compgen -G "outputs/$experiment/*_${SEED}/summary.json" >/dev/null; then
    echo "Skipping completed $experiment seed=$SEED"
    continue
  fi
  python -m src.train --config "$config" "runtime.seed=$SEED"
  python tools/sync_mlflow_runs.py \
    --outputs outputs \
    --tracking-uri "$MLFLOW_URI" \
    --experiment "ISIC2019 Stage16P Preprocessing"
done
