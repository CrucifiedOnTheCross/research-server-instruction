#!/usr/bin/env python3
"""Idempotently add grouping tags to existing longtail-medical MLflow runs."""

from __future__ import annotations

import argparse

import mlflow
from mlflow import MlflowClient

from longtail_medical.tracking import contamination_policy, experiment_arm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracking-uri", default="http://10.200.1.180:5000")
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--dataset", default="ISIC2019")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mlflow.set_tracking_uri(args.tracking_uri)
    client = MlflowClient()
    experiment = client.get_experiment_by_name(args.experiment)
    if experiment is None:
        raise SystemExit(f"MLflow experiment not found: {args.experiment}")

    runs = client.search_runs([experiment.experiment_id], max_results=5000)
    updated = 0
    for run in runs:
        run_name = run.data.tags.get("mlflow.runName", "")
        name = run_name.split("/", 1)[0]
        if not name.startswith(("stage1_", "stage2_")):
            continue
        tags = {
            "project": "longtail-medical",
            "stage": args.stage,
            "dataset": args.dataset,
            "task": "multiclass_skin_lesion_classification",
            "experiment_arm": experiment_arm(name),
            "contamination_policy": contamination_policy(name),
            "seed": str(run.data.params.get("seed", "unknown")),
            "model": str(run.data.params.get("model", "unknown")),
            "checkpoint_policy": "last" if args.stage == "stage2_confirmatory" else "best_validation",
            "test_evaluated": "false",
        }
        for key, value in tags.items():
            client.set_tag(run.info.run_id, key, value)
        updated += 1
    print(f"updated_runs={updated}")


if __name__ == "__main__":
    main()
