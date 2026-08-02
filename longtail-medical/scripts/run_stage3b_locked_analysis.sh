#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/srv/research/projects/default}"
WORKDIR="$PROJECT_ROOT/longtail-medical"
RESEARCH_VENV="${RESEARCH_VENV:-$PROJECT_ROOT/research-ml/.venv}"

cd "$WORKDIR"
source "$RESEARCH_VENV/bin/activate"
export PYTHONPATH="$WORKDIR/src:$WORKDIR"
python tools/check_stage3b_readiness.py
python tools/evaluate_stage3b_locked.py
python tools/analyze_stage3b.py --bootstrap-repeats 10000 --workers 4
