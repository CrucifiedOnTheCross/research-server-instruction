from __future__ import annotations

import argparse
import csv
import html
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import confusion_matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two runs on the same prediction CSV.")
    parser.add_argument("--base-run-dir", required=True)
    parser.add_argument("--candidate-run-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--name", default="")
    parser.add_argument("--split", default="test")
    parser.add_argument("--top-examples", type=int, default=30)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def load_predictions(run_dir: Path, split: str) -> pd.DataFrame:
    path = run_dir / f"{split}_predictions.csv"
    frame = pd.read_csv(path)
    required = {"path", "target", "prediction", "confidence"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")
    return frame


def format_float(value: Any) -> str:
    if value is None or value == "":
        return ""
    return f"{float(value):.4f}"


def metric_table(base: dict[str, Any], candidate: dict[str, Any]) -> str:
    keys = ["macro_f1", "balanced_accuracy", "mcc", "ece", "worst_class_recall", "accuracy"]
    rows = ["| Metric | Base | Candidate | Delta |", "|---|---:|---:|---:|"]
    for key in keys:
        b = base.get(key)
        c = candidate.get(key)
        delta = None if b is None or c is None else float(c) - float(b)
        rows.append(f"| `{key}` | {format_float(b)} | {format_float(c)} | {format_float(delta)} |")
    return "\n".join(rows)


def per_class_table(base: dict[str, Any], candidate: dict[str, Any]) -> str:
    labels = sorted(set(base.get("per_class", {})) | set(candidate.get("per_class", {})))
    rows = ["| Class | Recall base | Recall candidate | Delta recall | F1 base | F1 candidate | Delta F1 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for label in labels:
        b = base.get("per_class", {}).get(label, {})
        c = candidate.get("per_class", {}).get(label, {})
        br = b.get("recall")
        cr = c.get("recall")
        bf = b.get("f1")
        cf = c.get("f1")
        rows.append(
            f"| `{label}` | {format_float(br)} | {format_float(cr)} | {format_float(None if br is None or cr is None else cr - br)} "
            f"| {format_float(bf)} | {format_float(cf)} | {format_float(None if bf is None or cf is None else cf - bf)} |"
        )
    return "\n".join(rows)


def class_status_table(class_status: pd.DataFrame) -> str:
    statuses = ["both_correct", "candidate_only", "base_only", "both_wrong"]
    rows = ["| Class | Both correct | Candidate fixed | Candidate broke | Both wrong |", "|---|---:|---:|---:|---:|"]
    for label, row in class_status.sort_index().iterrows():
        rows.append(
            f"| `{label}` | {int(row.get('both_correct', 0))} | {int(row.get('candidate_only', 0))} "
            f"| {int(row.get('base_only', 0))} | {int(row.get('both_wrong', 0))} |"
        )
    if len(rows) == 2:
        rows.append("| none | 0 | 0 | 0 | 0 |")
    return "\n".join(rows)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def render_html(markdown_text: str) -> str:
    escaped = html.escape(markdown_text)
    return f"""<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Run comparison</title>
  <style>
    body {{ font-family: system-ui, -apple-system, Segoe UI, sans-serif; margin: 24px; color: #17202a; }}
    pre {{ white-space: pre-wrap; background: #f6f8fa; padding: 16px; border-radius: 8px; }}
  </style>
</head>
<body>
  <pre>{escaped}</pre>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    base_run = Path(args.base_run_dir)
    candidate_run = Path(args.candidate_run_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = args.name or f"{base_run.parent.name}_vs_{candidate_run.parent.name}"

    base_pred = load_predictions(base_run, args.split).add_prefix("base_")
    cand_pred = load_predictions(candidate_run, args.split).add_prefix("candidate_")
    merged = base_pred.merge(cand_pred, left_on="base_path", right_on="candidate_path", how="inner")
    if len(merged) != len(base_pred) or len(merged) != len(cand_pred):
        raise ValueError(f"Prediction rows do not align: base={len(base_pred)} candidate={len(cand_pred)} merged={len(merged)}")

    labels = sorted(set(merged["base_target"].astype(str)) | set(merged["base_prediction"].astype(str)) | set(merged["candidate_prediction"].astype(str)))
    base_cm = confusion_matrix(merged["base_target"], merged["base_prediction"], labels=labels)
    cand_cm = confusion_matrix(merged["base_target"], merged["candidate_prediction"], labels=labels)
    delta_cm = cand_cm - base_cm

    rows: list[dict[str, Any]] = []
    for _, row in merged.iterrows():
        target = row["base_target"]
        base_ok = row["base_prediction"] == target
        cand_ok = row["candidate_prediction"] == target
        if base_ok and cand_ok:
            status = "both_correct"
        elif base_ok and not cand_ok:
            status = "base_only"
        elif not base_ok and cand_ok:
            status = "candidate_only"
        else:
            status = "both_wrong"
        rows.append(
            {
                "path": row["base_path"],
                "target": target,
                "base_prediction": row["base_prediction"],
                "candidate_prediction": row["candidate_prediction"],
                "base_confidence": row["base_confidence"],
                "candidate_confidence": row["candidate_confidence"],
                "status": status,
            }
        )
    write_csv(out_dir / f"{name}_prediction_delta.csv", rows)

    status_counts = Counter(row["status"] for row in rows)
    class_status = pd.DataFrame(rows).groupby(["target", "status"]).size().unstack(fill_value=0)
    class_status.to_csv(out_dir / f"{name}_class_status.csv")

    base_metrics = read_json(base_run / f"{args.split}_metrics.json")
    cand_metrics = read_json(candidate_run / f"{args.split}_metrics.json")
    summary = {
        "name": name,
        "base_run_dir": str(base_run),
        "candidate_run_dir": str(candidate_run),
        "split": args.split,
        "n": int(len(merged)),
        "status_counts": dict(status_counts),
        "labels": labels,
        "base_confusion_matrix": base_cm.tolist(),
        "candidate_confusion_matrix": cand_cm.tolist(),
        "delta_confusion_matrix": delta_cm.tolist(),
    }
    (out_dir / f"{name}_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    improved = [row for row in rows if row["status"] == "candidate_only"][: args.top_examples]
    worsened = [row for row in rows if row["status"] == "base_only"][: args.top_examples]
    def example_lines(items: list[dict[str, Any]]) -> str:
        if not items:
            return "нет примеров"
        return "\n".join(
            f"- `{Path(item['path']).name}` target=`{item['target']}` base=`{item['base_prediction']}` candidate=`{item['candidate_prediction']}`"
            for item in items
        )

    md = f"""# Run Comparison: {name}

Base: `{base_run}`

Candidate: `{candidate_run}`

Split: `{args.split}`

Rows compared: `{len(merged)}`

## Metrics

{metric_table(base_metrics, cand_metrics)}

## Per-Class Recall/F1

{per_class_table(base_metrics, cand_metrics)}

## Prediction Status Counts

| Status | Count |
|---|---:|
| both correct | {status_counts.get('both_correct', 0)} |
| candidate fixed base error | {status_counts.get('candidate_only', 0)} |
| candidate broke base correct | {status_counts.get('base_only', 0)} |
| both wrong | {status_counts.get('both_wrong', 0)} |

## Class x Status

{class_status_table(class_status)}

## Examples Candidate Fixed

{example_lines(improved)}

## Examples Candidate Broke

{example_lines(worsened)}

## Artifacts

- `{name}_prediction_delta.csv`
- `{name}_class_status.csv`
- `{name}_summary.json`
"""
    (out_dir / f"{name}.md").write_text(md, encoding="utf-8")
    (out_dir / f"{name}.html").write_text(render_html(md), encoding="utf-8")
    print(out_dir / f"{name}.md")


if __name__ == "__main__":
    main()
