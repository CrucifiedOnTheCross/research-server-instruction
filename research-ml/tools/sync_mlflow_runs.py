from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
import yaml
from mlflow import MlflowClient
from mlflow.entities import Metric

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.tracking import flatten_mapping, flatten_numeric


STRUCTURED_SUFFIXES = {".json", ".jsonl", ".yaml", ".yml", ".csv", ".png", ".jpg", ".jpeg"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill structured experiment artifacts into MLflow.")
    parser.add_argument("--outputs", default="outputs")
    parser.add_argument("--tracking-uri", default="http://mlflow:5000")
    parser.add_argument("--experiment", default="HAM10000 Historical")
    parser.add_argument("--reports-experiment", default="HAM10000 Reports")
    parser.add_argument("--include-reports", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def existing_source_dirs(client: MlflowClient, experiment_id: str) -> set[str]:
    frame = mlflow.search_runs(
        experiment_ids=[experiment_id],
        output_format="pandas",
        max_results=50000,
    )
    if frame.empty or "tags.source_run_dir" not in frame:
        return set()
    return set(frame["tags.source_run_dir"].dropna().astype(str))


def log_metric_history(client: MlflowClient, run_id: str, metrics_path: Path) -> None:
    if not metrics_path.exists():
        return
    frame = pd.read_csv(metrics_path)
    timestamp = int(time.time() * 1000)
    batch: list[Metric] = []
    for row in frame.to_dict(orient="records"):
        step = int(row["epoch"])
        for key, value in row.items():
            if key == "epoch" or pd.isna(value):
                continue
            if isinstance(value, (int, float)):
                batch.append(Metric(str(key), float(value), timestamp, step))
        if len(batch) >= 800:
            client.log_batch(run_id, metrics=batch)
            batch = []
    if batch:
        client.log_batch(run_id, metrics=batch)


def log_structured_artifacts(run_dir: Path) -> None:
    for path in sorted(run_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in STRUCTURED_SUFFIXES:
            mlflow.log_artifact(str(path), artifact_path="structured")
    checkpoints = [str(path) for path in (run_dir / "best.pt", run_dir / "last.pt") if path.exists()]
    if checkpoints:
        mlflow.log_text("\n".join(checkpoints), "checkpoint_locations.txt")


def import_runs(outputs: Path, experiment_name: str, force: bool) -> dict[str, int]:
    experiment = mlflow.set_experiment(experiment_name)
    client = MlflowClient()
    imported = 0
    skipped = 0
    known = set() if force else existing_source_dirs(client, experiment.experiment_id)
    for summary_path in sorted(outputs.glob("*/*/summary.json")):
        run_dir = summary_path.parent
        source = str(run_dir.resolve())
        if source in known:
            skipped += 1
            continue
        config_path = run_dir / "config.resolved.yaml"
        metrics_path = run_dir / "metrics.csv"
        if not config_path.exists() or not metrics_path.exists():
            continue
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        summary = read_json(summary_path)
        experiment_name_from_config = str(config["experiment"]["name"])
        tags = {
            "source_run_dir": source,
            "experiment_name": experiment_name_from_config,
            "seed": str(config["runtime"]["seed"]),
            "imported": "true",
            "test_evaluated": str(bool(summary.get("test_evaluated", False))).lower(),
        }
        with mlflow.start_run(run_name=f"{experiment_name_from_config}/{run_dir.name}", tags=tags) as run:
            params = {
                key: value if len(str(value)) <= 5000 else str(value)[:4997] + "..."
                for key, value in flatten_mapping(config).items()
                if not key.startswith("tracking.")
            }
            mlflow.log_params(params)
            log_metric_history(client, run.info.run_id, metrics_path)
            mlflow.log_metrics(flatten_numeric(summary, "summary"))
            val_path = run_dir / "val_metrics_best.json"
            if val_path.exists():
                mlflow.log_metrics(flatten_numeric(read_json(val_path), "best.val"))
            test_path = run_dir / "test_metrics.json"
            if test_path.exists() and bool(summary.get("test_evaluated", False)):
                mlflow.log_metrics(flatten_numeric(read_json(test_path), "test"))
            log_structured_artifacts(run_dir)
        imported += 1
    return {"imported": imported, "skipped": skipped}


def import_reports(outputs: Path, experiment_name: str, force: bool) -> dict[str, int]:
    reports_root = outputs / "reports"
    if not reports_root.exists():
        return {"imported": 0, "skipped": 0}
    experiment = mlflow.set_experiment(experiment_name)
    client = MlflowClient()
    known = set() if force else existing_source_dirs(client, experiment.experiment_id)
    imported = 0
    skipped = 0
    for report_dir in sorted(path for path in reports_root.iterdir() if path.is_dir()):
        source = str(report_dir.resolve())
        if source in known:
            skipped += 1
            continue
        files = [
            path
            for path in report_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in STRUCTURED_SUFFIXES
        ]
        if not files:
            continue
        with mlflow.start_run(
            run_name=report_dir.name,
            tags={"source_run_dir": source, "artifact_type": "analysis_report", "imported": "true"},
        ):
            for path in files:
                artifact_path = Path("report") / path.parent.relative_to(report_dir)
                mlflow.log_artifact(str(path), artifact_path=artifact_path.as_posix())
        imported += 1
    return {"imported": imported, "skipped": skipped}


def main() -> None:
    args = parse_args()
    mlflow.set_tracking_uri(args.tracking_uri)
    outputs = Path(args.outputs)
    result = {"runs": import_runs(outputs, args.experiment, args.force)}
    if args.include_reports:
        result["reports"] = import_reports(outputs, args.reports_experiment, args.force)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
