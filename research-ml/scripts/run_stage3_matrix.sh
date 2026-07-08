#!/usr/bin/env bash
set -euo pipefail

python -m src.train --config configs/ham10000_stage3_utility_weight025.yaml \
  experiment.name="stage3_utility_weight025_ce_weighted"

python -m src.train --config configs/ham10000_stage3_utility_weight05.yaml \
  experiment.name="stage3_utility_weight05_ce_weighted"

