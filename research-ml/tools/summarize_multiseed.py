from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml


METRICS = ("macro_f1", "balanced_accuracy", "mcc", "ece", "worst_class_recall")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate completed runs from structured validation artifacts.")
    parser.add_argument("--outputs", default="outputs")
    parser.add_argument("--prefix", default="stage8_")
    parser.add_argument("--out-dir", default="outputs/reports/stage8_multiseed")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def mean_std(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    return float(array.mean()), float(array.std(ddof=1)) if len(array) > 1 else 0.0


def main() -> None:
    args = parse_args()
    outputs = Path(args.outputs)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for experiment_dir in sorted(outputs.glob(f"{args.prefix}*")):
        if not experiment_dir.is_dir():
            continue
        for run_dir in sorted(experiment_dir.iterdir()):
            metrics_path = run_dir / "val_metrics_best.json"
            config_path = run_dir / "config.resolved.yaml"
            summary_path = run_dir / "summary.json"
            if not metrics_path.exists() or not config_path.exists() or not summary_path.exists():
                continue
            metrics = read_json(metrics_path)
            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            grouped[experiment_dir.name].append(
                {
                    "run_dir": str(run_dir),
                    "seed": int(config["runtime"]["seed"]),
                    "metrics": metrics,
                }
            )

    summaries: list[dict[str, Any]] = []
    for experiment, runs in sorted(grouped.items()):
        item: dict[str, Any] = {
            "experiment": experiment,
            "n": len(runs),
            "seeds": [run["seed"] for run in runs],
            "runs": runs,
        }
        for metric in METRICS:
            values = [float(run["metrics"][metric]) for run in runs]
            mean, std = mean_std(values)
            item[f"{metric}_mean"] = mean
            item[f"{metric}_std"] = std
        for label in ("mel", "akiec", "bkl"):
            values = [float(run["metrics"]["per_class"][label]["recall"]) for run in runs]
            mean, std = mean_std(values)
            item[f"{label}_recall_mean"] = mean
            item[f"{label}_recall_std"] = std
        summaries.append(item)

    report = {"prefix": args.prefix, "experiments": summaries}
    (out_dir / "multiseed_summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    flat_fields = [
        key
        for key in summaries[0]
        if key != "runs"
    ] if summaries else []
    with (out_dir / "multiseed_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=flat_fields)
        writer.writeheader()
        writer.writerows([{key: row.get(key, "") for key in flat_fields} for row in summaries])

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
