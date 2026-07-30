#!/usr/bin/env bash
set -euo pipefail

source .venv/bin/activate
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
python tools/prepare_stage16g_p1_domain_data.py
python tools/train_stage16g_p1_domain_lora.py
python tools/generate_stage16g_generator_smoke.py \
  --config configs/stage16g_p1_generator_smoke.yaml \
  --arm domain_lora_img2img
python tools/generate_stage16g_generator_smoke.py \
  --config configs/stage16g_p1_generator_smoke.yaml \
  --arm domain_lora_mask_inpaint
python tools/analyze_stage16g_generator_smoke.py \
  --config configs/stage16g_p1_generator_smoke.yaml
