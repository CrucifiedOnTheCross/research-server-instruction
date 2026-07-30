#!/usr/bin/env bash
set -euo pipefail

source .venv/bin/activate
python tools/analyze_stage16g_generator_smoke.py

