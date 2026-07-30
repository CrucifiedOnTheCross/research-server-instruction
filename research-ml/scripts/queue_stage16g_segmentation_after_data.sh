#!/usr/bin/env bash
set -euo pipefail

WORKDIR="${WORKDIR:-/srv/research/projects/default/research-ml}"
LOG="$WORKDIR/server-logs/stage16g_segmentation_queue.log"
PID_FILE="$WORKDIR/server-logs/stage16g_segmentation_queue.pid"

mkdir -p "$WORKDIR/server-logs"
nohup bash -lc "
  while docker inspect research-stage16g-segdata >/dev/null 2>&1; do
    state=\$(docker inspect research-stage16g-segdata --format '{{.State.Status}} {{.State.ExitCode}}')
    status=\${state%% *}
    code=\${state##* }
    if [[ \"\$status\" == exited ]]; then
      if [[ \"\$code\" != 0 ]]; then
        echo \"Stage16G segmentation-data preparation failed: \$state\"
        exit 1
      fi
      break
    fi
    sleep 30
  done
  cd '$WORKDIR'
  scripts/start_stage16g_segmentation_container.sh
" >"$LOG" 2>&1 &
echo "$!" >"$PID_FILE"
echo "Queue PID: $(cat "$PID_FILE")"
