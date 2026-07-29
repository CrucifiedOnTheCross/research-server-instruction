#!/usr/bin/env bash
set -euo pipefail

NAME="${NAME:-research-stage16a-isic-data}"
IMAGE="${IMAGE:-local/research-cuda-notebook:latest}"
PROJECT_ROOT="${PROJECT_ROOT:-/srv/research/projects/default}"
WORKDIR="$PROJECT_ROOT/research-ml"
DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/isic2019}"
USER_SPEC="${USER_SPEC:-1000:1006}"
GROUP_ID="${GROUP_ID:-1006}"
STAMP="$(date +%Y%m%d-%H%M%S)"

mkdir -p "$WORKDIR/server-logs" "$DATA_ROOT"
docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d \
  --name "$NAME" \
  --user "$USER_SPEC" \
  --group-add "$GROUP_ID" \
  -v "$PROJECT_ROOT:$PROJECT_ROOT" \
  -w "$WORKDIR" \
  -e "OMP_NUM_THREADS=24" \
  -e "MKL_NUM_THREADS=24" \
  "$IMAGE" \
  bash -lc "source .venv/bin/activate && set -o pipefail && python tools/prepare_isic2019.py --root '$DATA_ROOT' && python tools/check_stage16a_gate.py --data-root '$DATA_ROOT' --check-files 2>&1 | tee 'server-logs/stage16a_data_$STAMP.log'"

echo "Container: $NAME"
echo "Dataset root: $DATA_ROOT"
echo "Gate report: $WORKDIR/outputs/reports/stage16a_data_gate.json"
