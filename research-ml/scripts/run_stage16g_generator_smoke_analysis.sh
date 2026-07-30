#!/usr/bin/env bash
set -euo pipefail

source .venv/bin/activate
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
python tools/analyze_stage16g_generator_smoke.py
