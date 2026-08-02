#!/usr/bin/env python3
"""Idempotently sync Stage 2 grouping tags and locked-test scalar metrics."""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path


def post(base: str, endpoint: str, payload: dict) -> dict:
    request = urllib.request.Request(
        f"{base.rstrip('/')}{endpoint}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        body = response.read()
    return json.loads(body) if body else {}


def policy_and_protocol(name: str) -> tuple[str, str]:
    if "exposure_matched" in name:
        return "decontaminated_exposure_matched", "stage2_retrospective_contamination_v2_exposure_matched"
    if "decontaminated_unmatched" in name:
        return "decontaminated_unmatched", "stage2_retrospective_contamination_v2_unmatched"
    if "lesion_disjoint" in name:
        return "lesion_disjoint", "isic2019_lt_ir100_lesion_disjoint_v1"
    return "original_contaminated", "monica_isic2019_lt_ir100_pinned_3dd808d6"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracking-uri", default="http://10.200.1.180:5000")
    parser.add_argument("--experiment-id", default="9")
    parser.add_argument(
        "--metrics", type=Path,
        default=Path("outputs/stage2_locked_test/all_run_metrics.json"),
    )
    args = parser.parse_args()
    metric_rows = json.loads(args.metrics.read_text(encoding="utf-8"))
    metrics_by_key = {
        (row["arm"], int(row["seed"]), row["checkpoint"]): row for row in metric_rows
    }
    runs = post(args.tracking_uri, "/api/2.0/mlflow/runs/search", {
        "experiment_ids": [args.experiment_id], "max_results": 1000,
    }).get("runs", [])
    timestamp = int(time.time() * 1000)
    updated = 0
    for run in runs:
        name = run["info"]["run_name"].split("/", 1)[0]
        if not name.startswith("stage2_"):
            continue
        params = {item["key"]: item["value"] for item in run.get("data", {}).get("params", [])}
        seed = int(params["seed"])
        policy, protocol = policy_and_protocol(name)
        tags = {
            "project": "longtail-medical",
            "stage": "stage2_confirmatory",
            "dataset": "ISIC2019",
            "task": "multiclass_skin_lesion_classification",
            "experiment_arm": name.removeprefix("stage2_"),
            "contamination_policy": policy,
            "protocol_version": protocol,
            "seed": str(seed),
            "model": params.get("model", "resnet50"),
            "checkpoint_policy": "last",
            "test_evaluated": "true",
            "training_run_test_evaluated": "false",
            "test_evaluation_source": "central_one_shot",
            "inference_git_commit": "d0824eaebabd5f699f49236df8534a2864b61b49",
            "analysis_git_commit": "89ac89d07d84e9e3dd2d7bb345462a00cbfb45ad",
        }
        logged_metrics = []
        for checkpoint, prefix in (("last.pt", "test/"), ("best.pt", "test_best/")):
            row = metrics_by_key[(name, seed, checkpoint)]
            for key, value in row.items():
                if key not in {"arm", "seed", "checkpoint"} and isinstance(value, (int, float)):
                    logged_metrics.append({
                        "key": f"{prefix}{key}", "value": float(value),
                        "timestamp": timestamp, "step": 0,
                    })
        post(args.tracking_uri, "/api/2.0/mlflow/runs/log-batch", {
            "run_id": run["info"]["run_id"],
            "metrics": logged_metrics,
            "params": [],
            "tags": [{"key": key, "value": value} for key, value in tags.items()],
        })
        updated += 1
    if updated != 12:
        raise RuntimeError(f"Expected 12 Stage 2 MLflow runs, updated {updated}")
    print(json.dumps({"updated_runs": updated, "checkpoints_uploaded": False}, indent=2))


if __name__ == "__main__":
    main()
