from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


CURRENT_REQUIRED_ARTIFACTS = (
    "summary.json",
    "config.resolved.yaml",
    "sampling_plan.json",
    "model_initialization.json",
    "class_counts.json",
    "val_metrics_best.json",
    "val_predictions_best.csv",
    "best.pt",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit completed experiments using structured artifacts only."
    )
    parser.add_argument("--outputs", default="outputs")
    parser.add_argument("--out-dir", default="outputs/reports/history_audit")
    parser.add_argument("--modern-stage-min", type=int, default=9)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def stage_number(experiment: str) -> int | None:
    if not experiment.startswith("stage"):
        return None
    digits = []
    for char in experiment[5:]:
        if char.isdigit():
            digits.append(char)
        else:
            break
    return int("".join(digits)) if digits else None


def seed_from_run(run_name: str) -> int | None:
    try:
        return int(run_name.rsplit("_", 1)[1])
    except (IndexError, ValueError):
        return None


def audit(outputs: Path, modern_stage_min: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for summary_path in sorted(outputs.glob("*/*/summary.json")):
        run_dir = summary_path.parent
        experiment = run_dir.parent.name
        summary = read_json(summary_path)
        val_path = run_dir / "val_metrics_best.json"
        val = read_json(val_path) if val_path.is_file() else {}
        stage = stage_number(experiment)
        required = CURRENT_REQUIRED_ARTIFACTS if stage is not None and stage >= modern_stage_min else ()
        missing = [name for name in required if not (run_dir / name).is_file()]
        records.append(
            {
                "experiment": experiment,
                "run": run_dir.name,
                "seed": summary.get("seed", seed_from_run(run_dir.name)),
                "stage": stage,
                "test_evaluated": summary.get("test_evaluated"),
                "best_epoch": summary.get("best_epoch"),
                "macro_f1": val.get("macro_f1"),
                "mcc": val.get("mcc"),
                "balanced_accuracy": val.get("balanced_accuracy"),
                "ece": val.get("ece"),
                "artifacts_complete": not missing,
                "missing_artifacts": ";".join(missing),
            }
        )

    duplicate_keys = [
        {"experiment": experiment, "seed": seed, "count": count}
        for (experiment, seed), count in Counter(
            (record["experiment"], record["seed"]) for record in records
        ).items()
        if count > 1
    ]
    modern = [
        record
        for record in records
        if record["stage"] is not None and record["stage"] >= modern_stage_min
    ]
    violations = [
        record
        for record in modern
        if record["test_evaluated"] is not False or not record["artifacts_complete"]
    ]
    report = {
        "status": "pass" if not duplicate_keys and not violations else "fail",
        "completed_runs": len(records),
        "experiments": len({record["experiment"] for record in records}),
        "modern_stage_min": modern_stage_min,
        "modern_runs": len(modern),
        "duplicate_experiment_seed": duplicate_keys,
        "modern_protocol_violations": violations,
        "logs_read": False,
    }
    return records, report


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    outputs = Path(args.outputs).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    records, report = audit(outputs, args.modern_stage_min)
    write_csv(out_dir / "runs.csv", records)
    (out_dir / "audit_summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
