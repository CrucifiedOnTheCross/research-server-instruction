from __future__ import annotations

import argparse
import csv
import html
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

    table_rows = []
    for row in summaries:
        table_rows.append(
            "<tr>"
            f"<td>{html.escape(row['experiment'])}</td>"
            f"<td>{row['n']}</td>"
            f"<td>{row['macro_f1_mean']:.4f} ± {row['macro_f1_std']:.4f}</td>"
            f"<td>{row['balanced_accuracy_mean']:.4f} ± {row['balanced_accuracy_std']:.4f}</td>"
            f"<td>{row['mcc_mean']:.4f} ± {row['mcc_std']:.4f}</td>"
            f"<td>{row['ece_mean']:.4f} ± {row['ece_std']:.4f}</td>"
            f"<td>{row['mel_recall_mean']:.4f}</td>"
            f"<td>{row['akiec_recall_mean']:.4f}</td>"
            f"<td>{row['bkl_recall_mean']:.4f}</td>"
            "</tr>"
        )
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta http-equiv="refresh" content="30">
<title>Stage 8 multi-seed summary</title>
<style>body{{font-family:system-ui,sans-serif;margin:24px}}table{{border-collapse:collapse;width:100%}}
td,th{{border-bottom:1px solid #ddd;padding:8px;text-align:left}}th{{background:#f6f8fa}}</style></head>
<body><h1>Stage 8 multi-seed validation</h1>
<table><thead><tr><th>Experiment</th><th>N</th><th>Macro F1</th><th>Bal acc</th><th>MCC</th>
<th>ECE</th><th>mel R</th><th>akiec R</th><th>bkl R</th></tr></thead>
<tbody>{''.join(table_rows)}</tbody></table>
<p><a href="multiseed_summary.json">JSON</a> | <a href="multiseed_summary.csv">CSV</a></p>
</body></html>"""
    (out_dir / "index.html").write_text(page, encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
