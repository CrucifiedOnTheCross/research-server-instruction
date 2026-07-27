from __future__ import annotations

import atexit
import json
import os
from pathlib import Path
from typing import Any


def flatten_mapping(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in data.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flattened.update(flatten_mapping(value, name))
        elif isinstance(value, (list, tuple)):
            flattened[name] = json.dumps(value, ensure_ascii=False)
        elif value is None:
            flattened[name] = "null"
        else:
            flattened[name] = value
    return flattened


def flatten_numeric(data: dict[str, Any], prefix: str = "") -> dict[str, float]:
    flattened: dict[str, float] = {}
    for key, value in data.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flattened.update(flatten_numeric(value, name))
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            flattened[name] = float(value)
    return flattened


class MlflowTracker:
    def __init__(self, config: dict[str, Any], run_dir: Path, logger: Any) -> None:
        self.config = config
        self.run_dir = run_dir
        self.logger = logger
        self.settings = config.get("tracking", {})
        self.enabled = bool(self.settings.get("enabled", False))
        self.fail_on_error = bool(self.settings.get("fail_on_error", False))
        self.mlflow: Any | None = None
        self.active = False
        self.finished = False

    def _handle_error(self, action: str, error: Exception) -> None:
        self.logger.warning("MLflow %s failed: %s", action, error)
        if self.fail_on_error:
            raise error

    def start(self) -> None:
        if not self.enabled:
            return
        try:
            import mlflow

            self.mlflow = mlflow
            mlflow.set_tracking_uri(str(self.settings["uri"]))
            mlflow.set_experiment(str(self.settings.get("experiment", "HAM10000")))
            tags = {
                "experiment_name": str(self.config["experiment"]["name"]),
                "seed": str(self.config["runtime"]["seed"]),
                "source_run_dir": str(self.run_dir),
                "test_evaluated": str(bool(self.config["evaluation"].get("run_test", True))).lower(),
                "code_commit": os.environ.get("RESEARCH_CODE_COMMIT", "unknown"),
            }
            mlflow.start_run(
                run_name=f"{self.config['experiment']['name']}/{self.run_dir.name}",
                tags=tags,
                log_system_metrics=bool(self.settings.get("log_system_metrics", True)),
            )
            self.active = True
            atexit.register(self._close_unfinished)
            params = {
                key: value if len(str(value)) <= 5000 else str(value)[:4997] + "..."
                for key, value in flatten_mapping(self.config).items()
                if not key.startswith("tracking.")
            }
            mlflow.log_params(params)
            self.logger.info("MLflow run: %s", mlflow.active_run().info.run_id)
        except Exception as error:
            self._handle_error("start", error)

    def log_epoch(self, metrics: dict[str, Any]) -> None:
        if not self.active or self.mlflow is None:
            return
        try:
            step = int(metrics["epoch"])
            values = {
                str(key): float(value)
                for key, value in metrics.items()
                if key != "epoch" and isinstance(value, (int, float))
            }
            self.mlflow.log_metrics(values, step=step)
        except Exception as error:
            self._handle_error("epoch logging", error)

    def finish(self) -> None:
        if not self.active or self.mlflow is None:
            return
        try:
            summary_path = self.run_dir / "summary.json"
            if summary_path.exists():
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                self.mlflow.log_metrics(flatten_numeric(summary, "summary"))
            for path in sorted(self.run_dir.iterdir()):
                if not path.is_file():
                    continue
                if path.suffix.lower() not in {".json", ".jsonl", ".yaml", ".yml", ".csv", ".png", ".jpg", ".jpeg"}:
                    continue
                self.mlflow.log_artifact(str(path), artifact_path="structured")
            checkpoint_locations = [
                str(path)
                for path in (self.run_dir / "best.pt", self.run_dir / "last.pt")
                if path.exists()
            ]
            if checkpoint_locations:
                self.mlflow.log_text("\n".join(checkpoint_locations), "checkpoint_locations.txt")
            self.mlflow.end_run(status="FINISHED")
            self.finished = True
            self.active = False
        except Exception as error:
            self._handle_error("finish", error)

    def _close_unfinished(self) -> None:
        if self.active and not self.finished and self.mlflow is not None:
            try:
                self.mlflow.end_run(status="KILLED")
            except Exception:
                pass
