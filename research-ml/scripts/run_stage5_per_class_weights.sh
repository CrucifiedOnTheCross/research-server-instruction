#!/usr/bin/env bash
set -euo pipefail

python -m src.train --config configs/ham10000_stage5_mel075_akiec05_bkl0.yaml \
  experiment.name="stage5_mel075_akiec05_bkl0_ce_weighted"

python -m src.train --config configs/ham10000_stage5_mel05_akiec025_bkl0.yaml \
  experiment.name="stage5_mel05_akiec025_bkl0_ce_weighted"

python -m src.train --config configs/ham10000_stage5_mel05_akiec05_bkl01.yaml \
  experiment.name="stage5_mel05_akiec05_bkl01_ce_weighted"

