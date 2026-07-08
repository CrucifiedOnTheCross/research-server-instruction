from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path
from typing import Any


ARTIFACTS = [
    "summary.json",
    "test_metrics.json",
    "val_metrics_best.json",
    "metrics.csv",
    "run.log",
    "config.resolved.yaml",
    "class_counts.json",
    "environment.json",
    "feature_space_analysis.json",
    "test_predictions.csv",
    "val_predictions_best.csv",
    "best.pt",
    "last.pt",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a static HTML index for experiment outputs.")
    parser.add_argument("--outputs", default="outputs")
    parser.add_argument("--title", default="Research ML Results")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def read_last_metric_row(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return rows[-1] if rows else {}


def discover_runs(outputs: Path) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for metrics_path in sorted(outputs.glob("*/*/metrics.csv")):
        run_dir = metrics_path.parent
        experiment = run_dir.parent.name
        summary = read_json(run_dir / "summary.json")
        test = read_json(run_dir / "test_metrics.json") or summary.get("test", {})
        val = read_json(run_dir / "val_metrics_best.json")
        last = read_last_metric_row(metrics_path)
        runs.append(
            {
                "experiment": experiment,
                "run": run_dir.name,
                "path": run_dir.relative_to(outputs).as_posix(),
                "mtime": run_dir.stat().st_mtime,
                "epoch": last.get("epoch", ""),
                "val_macro_f1": val.get("macro_f1", ""),
                "test_macro_f1": test.get("macro_f1", ""),
                "test_balanced_accuracy": test.get("balanced_accuracy", ""),
                "test_mcc": test.get("mcc", ""),
                "test_ece": test.get("ece", ""),
                "has_summary": (run_dir / "summary.json").exists(),
            }
        )
    return sorted(runs, key=lambda row: row["mtime"], reverse=True)


def fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, str):
        try:
            return f"{float(value):.4f}"
        except ValueError:
            return html.escape(value)
    return html.escape(str(value))


def artifact_links(outputs: Path, run_path: str) -> str:
    run_dir = outputs / run_path
    links = []
    for name in ARTIFACTS:
        if (run_dir / name).exists():
            links.append(f'<a href="{html.escape(run_path)}/{html.escape(name)}">{html.escape(name)}</a>')
    return " ".join(links)


def build_html(outputs: Path, title: str) -> str:
    runs = discover_runs(outputs)
    rows = []
    for run in runs:
        rows.append(
            "<tr>"
            f"<td>{html.escape(run['experiment'])}</td>"
            f"<td><a href=\"{html.escape(run['path'])}/\">{html.escape(run['run'])}</a></td>"
            f"<td>{html.escape(str(run['epoch']))}</td>"
            f"<td>{fmt(run['val_macro_f1'])}</td>"
            f"<td>{fmt(run['test_macro_f1'])}</td>"
            f"<td>{fmt(run['test_balanced_accuracy'])}</td>"
            f"<td>{fmt(run['test_mcc'])}</td>"
            f"<td>{fmt(run['test_ece'])}</td>"
            f"<td>{artifact_links(outputs, run['path'])}</td>"
            "</tr>"
        )
    return f"""<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta http-equiv="refresh" content="60">
  <title>{html.escape(title)}</title>
  <style>
    body {{ font-family: system-ui, -apple-system, Segoe UI, sans-serif; margin: 24px; color: #17202a; }}
    h1 {{ font-size: 24px; margin-bottom: 4px; }}
    .sub {{ color: #586069; margin-bottom: 18px; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 14px; }}
    th, td {{ border-bottom: 1px solid #d8dee4; padding: 8px; vertical-align: top; text-align: left; }}
    th {{ position: sticky; top: 0; background: #f6f8fa; }}
    a {{ color: #0969da; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    td:last-child a {{ display: inline-block; margin-right: 8px; margin-bottom: 4px; }}
  </style>
</head>
<body>
  <h1>{html.escape(title)}</h1>
  <div class="sub">Автообновление раз в 60 секунд. Runs: {len(runs)}.</div>
  <table>
    <thead>
      <tr>
        <th>Experiment</th><th>Run</th><th>Epoch</th><th>Val macro F1</th>
        <th>Test macro F1</th><th>Test bal acc</th><th>Test MCC</th><th>Test ECE</th><th>Artifacts</th>
      </tr>
    </thead>
    <tbody>
      {''.join(rows)}
    </tbody>
  </table>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    outputs = Path(args.outputs)
    html_text = build_html(outputs, args.title)
    (outputs / "index.html").write_text(html_text, encoding="utf-8")
    print(outputs / "index.html")


if __name__ == "__main__":
    main()
