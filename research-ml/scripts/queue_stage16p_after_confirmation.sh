#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/srv/research/projects/default/research-ml}"
PID_FILE="$PROJECT_ROOT/server-logs/stage16p_queue.pid"
QUEUE_LOG="$PROJECT_ROOT/server-logs/stage16p_queue.log"
mkdir -p "$PROJECT_ROOT/server-logs"

if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "Stage 16P queue already active: PID $(cat "$PID_FILE")"
  exit 0
fi

nohup bash -c "
  set -euo pipefail
  while docker ps -q --filter name=research-stage16b-confirmation | grep -q .; do
    sleep 60
  done
  cd '$PROJECT_ROOT'
  bash scripts/start_stage16p_container.sh
" >"$QUEUE_LOG" 2>&1 &
echo "$!" >"$PID_FILE"

echo "Stage 16P queued after Stage 16B confirmation"
echo "Queue PID: $(cat "$PID_FILE")"
echo "Queue log: $QUEUE_LOG"
