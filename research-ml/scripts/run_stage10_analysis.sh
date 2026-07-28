#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/srv/research/projects/default/research-ml}"
DATA_ROOT="${DATA_ROOT:-/srv/research/projects/default/ham10000}"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

cd "$PROJECT_ROOT"

for stratum in strict_id aid_radial ood_far random_remaining; do
  for arm in synthetic replay; do
    experiment="stage10_${arm}_${stratum}_384"
    for seed in 42 43 44; do
      run_dir="$(find "outputs/$experiment" -mindepth 1 -maxdepth 1 -type d -name "*_${seed}" | sort | tail -1)"
      if [[ -z "$run_dir" ]]; then
        echo "Missing run: $experiment seed=$seed" >&2
        exit 1
      fi
      python tools/calibrate_stage9_predictions.py \
        --run-dir "$run_dir" \
        --split-csv "$DATA_ROOT/splits/stage8/val_real.csv" \
        --seed "$seed"
    done
  done
done

python tools/analyze_stage10_results.py \
  --project-root "$PROJECT_ROOT" \
  --data-root "$DATA_ROOT" \
  --out-dir outputs/reports/stage10_analysis \
  --bootstrap-replicates 5000
