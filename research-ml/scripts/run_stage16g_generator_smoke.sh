#!/usr/bin/env bash
set -euo pipefail

source .venv/bin/activate
python tools/prepare_stage16g_generator_smoke.py
python tools/generate_stage16g_generator_smoke.py --arm historical_sd15_img2img
python tools/generate_stage16g_generator_smoke.py --arm generic_sd15_mask_inpaint

