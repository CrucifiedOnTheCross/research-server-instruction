#!/usr/bin/env bash
set -euo pipefail

python -m src.train --config configs/ham10000_stage4_utility_mel_akiec_weight05.yaml \
  experiment.name="stage4_utility_mel_akiec_weight05_ce_weighted"

python -m src.train --config configs/ham10000_stage4_utility_mel_only_weight05.yaml \
  experiment.name="stage4_utility_mel_only_weight05_ce_weighted"

