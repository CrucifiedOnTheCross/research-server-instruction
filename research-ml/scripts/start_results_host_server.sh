#!/usr/bin/env bash
set -euo pipefail

PORT="${PORT:-8010}"
ROOT="${ROOT:-/srv/research/projects/default/research-ml}"
LOG_DIR="$ROOT/server-logs"
mkdir -p "$LOG_DIR"

pkill -f "python3 -m http.server $PORT --directory outputs" >/dev/null 2>&1 || true
pkill -f "tools/build_results_index.py --outputs outputs" >/dev/null 2>&1 || true

cd "$ROOT"
nohup bash -lc '
  while true; do
    python3 tools/build_results_index.py --outputs outputs --title "Research ML Results"
    sleep 30
  done &
  python3 -m http.server '"$PORT"' --directory outputs
' > "$LOG_DIR/results-server.log" 2>&1 &

echo "$!" > "$LOG_DIR/results-server.pid"
echo "Results URL: http://10.200.1.180:$PORT/"
echo "PID: $(cat "$LOG_DIR/results-server.pid")"

