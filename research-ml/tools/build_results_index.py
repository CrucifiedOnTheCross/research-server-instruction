from __future__ import annotations

import argparse
import csv
import html
import json
from datetime import datetime
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
    "synthetic_weight_by_class.json",
    "best.pt",
    "last.pt",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a static HTML index for experiment outputs.")
    parser.add_argument("--outputs", default="outputs")
    parser.add_argument("--title", default="Research ML Results")
    parser.add_argument("--tensorboard-url", default="http://10.200.1.180:8012")
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


def best_epoch(metrics_path: Path, monitor: str = "val/macro_f1") -> str:
    if not metrics_path.exists():
        return ""
    try:
        with metrics_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except Exception:
        return ""
    if not rows or monitor not in rows[0]:
        return ""
    try:
        best = max(rows, key=lambda row: float(row.get(monitor, "-inf") or "-inf"))
    except ValueError:
        return ""
    return str(best.get("epoch", ""))


def discover_runs(outputs: Path) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for metrics_path in sorted(outputs.glob("*/*/metrics.csv")):
        run_dir = metrics_path.parent
        experiment = run_dir.parent.name
        summary = read_json(run_dir / "summary.json")
        test = read_json(run_dir / "test_metrics.json") or summary.get("test", {})
        val = read_json(run_dir / "val_metrics_best.json")
        last = read_last_metric_row(metrics_path)
        per_class = test.get("per_class", {}) if isinstance(test, dict) else {}
        runs.append(
            {
                "experiment": experiment,
                "run": run_dir.name,
                "path": run_dir.relative_to(outputs).as_posix(),
                "mtime": run_dir.stat().st_mtime,
                "status": "finished" if (run_dir / "summary.json").exists() else "running",
                "epoch": last.get("epoch", ""),
                "best_epoch": best_epoch(metrics_path),
                "current_val_macro_f1": last.get("val/macro_f1", ""),
                "val_macro_f1": val.get("macro_f1", ""),
                "test_macro_f1": test.get("macro_f1", ""),
                "test_balanced_accuracy": test.get("balanced_accuracy", ""),
                "test_mcc": test.get("mcc", ""),
                "test_ece": test.get("ece", ""),
                "akiec_recall": per_class.get("akiec", {}).get("recall", ""),
                "bkl_recall": per_class.get("bkl", {}).get("recall", ""),
                "mel_recall": per_class.get("mel", {}).get("recall", ""),
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


def report_links(outputs: Path, tensorboard_url: str) -> str:
    links = [f'<a class="pill primary" href="{html.escape(tensorboard_url)}">TensorBoard</a>']
    candidates = [
        ("PCA/t-SNE", outputs / "reports" / "stage6_embedding_visuals" / "index.html"),
        ("Feature geometry", outputs / "reports" / "stage6_feature_geometry" / "index.html"),
    ]
    for label, path in candidates:
        if path.exists():
            links.append(f'<a class="pill" href="{html.escape(path.relative_to(outputs).as_posix())}">{html.escape(label)}</a>')
    return "".join(links)


def build_html(outputs: Path, title: str, tensorboard_url: str) -> str:
    runs = discover_runs(outputs)
    rows = []
    for run in runs:
        status_class = "status-done" if run["status"] == "finished" else "status-run"
        rows.append(
            f"<tr class=\"{html.escape(run['status'])}\">"
            f"<td>{html.escape(run['experiment'])}</td>"
            f"<td><a href=\"{html.escape(run['path'])}/\">{html.escape(run['run'])}</a></td>"
            f"<td><span class=\"status {status_class}\">{html.escape(run['status'])}</span></td>"
            f"<td>{html.escape(str(run['epoch']))}</td>"
            f"<td>{html.escape(str(run['best_epoch']))}</td>"
            f"<td>{fmt(run['current_val_macro_f1'])}</td>"
            f"<td>{fmt(run['val_macro_f1'])}</td>"
            f"<td>{fmt(run['test_macro_f1'])}</td>"
            f"<td>{fmt(run['test_balanced_accuracy'])}</td>"
            f"<td>{fmt(run['test_mcc'])}</td>"
            f"<td>{fmt(run['test_ece'])}</td>"
            f"<td>{fmt(run['mel_recall'])}</td>"
            f"<td>{fmt(run['akiec_recall'])}</td>"
            f"<td>{fmt(run['bkl_recall'])}</td>"
            f"<td>{artifact_links(outputs, run['path'])}</td>"
            "</tr>"
        )
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"""<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta http-equiv="refresh" content="10">
  <title>{html.escape(title)}</title>
  <style>
    :root {{ color-scheme: light; }}
    body {{ font-family: system-ui, -apple-system, Segoe UI, sans-serif; margin: 20px; color: #17202a; background: #fff; }}
    h1 {{ font-size: 24px; margin: 0 0 4px; }}
    .topbar {{ display: flex; align-items: flex-start; justify-content: space-between; gap: 16px; margin-bottom: 14px; }}
    .sub {{ color: #586069; margin-top: 4px; line-height: 1.45; }}
    .quick {{ display: flex; gap: 8px; flex-wrap: wrap; justify-content: flex-end; }}
    .pill {{ display: inline-block; border: 1px solid #d0d7de; border-radius: 999px; padding: 6px 10px; background: #f6f8fa; font-size: 13px; }}
    .pill.primary {{ color: #fff; background: #0969da; border-color: #0969da; font-weight: 650; }}
    .table-wrap {{ overflow-x: auto; border: 1px solid #d8dee4; border-radius: 8px; }}
    table {{ border-collapse: collapse; width: 100%; min-width: 1280px; font-size: 13px; }}
    th, td {{ border-bottom: 1px solid #d8dee4; padding: 8px; vertical-align: top; text-align: left; }}
    th {{ position: sticky; top: 0; background: #f6f8fa; z-index: 1; }}
    tr.running {{ background: #fff8e6; }}
    tr.finished:hover, tr.running:hover {{ background: #f6f8fa; }}
    .status {{ display: inline-block; min-width: 58px; text-align: center; border-radius: 999px; padding: 2px 8px; font-weight: 650; font-size: 12px; }}
    .status-run {{ color: #7a4d00; background: #fff0b3; }}
    .status-done {{ color: #0f5132; background: #d1e7dd; }}
    a {{ color: #0969da; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    td:last-child a {{ display: inline-block; margin-right: 8px; margin-bottom: 4px; }}
    .muted {{ color: #6e7781; }}
  </style>
</head>
<body>
  <div class="topbar">
    <div>
      <h1>{html.escape(title)}</h1>
      <div class="sub">
        Автообновление страницы раз в 10 секунд. Индекс пересобирается фоновым процессом.
        Runs: {len(runs)}. <span class="muted">Generated: {html.escape(generated_at)}</span>
      </div>
    </div>
    <div class="quick">{report_links(outputs, tensorboard_url)}</div>
  </div>
  <div class="table-wrap">
    <table>
      <thead>
        <tr>
          <th>Experiment</th><th>Run</th><th>Status</th><th>Epoch</th><th>Best epoch</th>
          <th>Current val F1</th><th>Best val F1</th><th>Test macro F1</th>
          <th>Test bal acc</th><th>Test MCC</th><th>Test ECE</th>
          <th>mel R</th><th>akiec R</th><th>bkl R</th><th>Artifacts</th>
        </tr>
      </thead>
      <tbody>
        {''.join(rows)}
      </tbody>
    </table>
  </div>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    outputs = Path(args.outputs)
    html_text = build_html(outputs, args.title, args.tensorboard_url)
    (outputs / "index.html").write_text(html_text, encoding="utf-8")
    print(outputs / "index.html")


if __name__ == "__main__":
    main()
