#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/srv/research/projects/default/research-ml}"
TRACKING_DATA_ROOT="${TRACKING_DATA_ROOT:-$PROJECT_ROOT/tracking-data}"
export PROJECT_ROOT TRACKING_DATA_ROOT
cd "$PROJECT_ROOT"

mkdir -p "$TRACKING_DATA_ROOT/mlflow/artifacts"
mkdir -p "$TRACKING_DATA_ROOT/fiftyone/mongo" "$TRACKING_DATA_ROOT/fiftyone/datasets"
docker compose -f tracking/docker-compose.yml up -d --build mlflow mongo

docker compose -f tracking/docker-compose.yml run --rm fiftyone \
  python tools/sync_fiftyone_synthetic.py --replace
docker compose -f tracking/docker-compose.yml up -d fiftyone

MLFLOW_READY=false
for _ in $(seq 1 60); do
  if python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:5000/health')" >/dev/null 2>&1; then
    MLFLOW_READY=true
    break
  fi
  sleep 2
done
if [[ "$MLFLOW_READY" != "true" ]]; then
  echo "MLflow did not become healthy in time" >&2
  exit 1
fi

docker compose -f tracking/docker-compose.yml run --rm mlflow \
  python tools/sync_mlflow_runs.py \
    --outputs outputs \
    --tracking-uri http://mlflow:5000 \
    --include-reports

echo "MLflow: http://10.200.1.180:5000"
echo "FiftyOne: http://10.200.1.180:5151"
