from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


REQUIRED_ARTIFACTS = (
    "summary.json",
    "config.resolved.yaml",
    "model_initialization.json",
    "sampling_plan.json",
    "class_counts.json",
    "val_metrics_best.json",
    "val_predictions_best.csv",
    "best.pt",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decision", default="configs/stage11_decision.yaml")
    parser.add_argument("--outputs", default="outputs")
    parser.add_argument("--report", default="outputs/reports/stage11_gate.json")
    parser.add_argument("--print-configs", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def expected_stage10(decision: dict[str, Any]) -> list[tuple[str, int]]:
    required = decision["required_stage10"]
    return [
        (f"stage10_{arm}_{stratum}_384", int(seed))
        for stratum in required["strata"]
        for arm in required["arms"]
        for seed in required["seeds"]
    ]


def validate_stage10(outputs: Path, decision: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    runs: list[dict[str, Any]] = []
    errors: list[str] = []
    for experiment, seed in expected_stage10(decision):
        candidates = sorted((outputs / experiment).glob(f"*_{seed}/summary.json"))
        if len(candidates) != 1:
            errors.append(f"{experiment} seed={seed}: expected one summary.json, found {len(candidates)}")
            continue
        run_dir = candidates[0].parent
        missing = [name for name in REQUIRED_ARTIFACTS if not (run_dir / name).is_file()]
        if missing:
            errors.append(f"{experiment} seed={seed}: missing {', '.join(missing)}")
            continue
        summary = read_json(run_dir / "summary.json")
        if summary.get("test_evaluated") is not False:
            errors.append(f"{experiment} seed={seed}: locked test was evaluated")
            continue
        sampling_plan = read_json(run_dir / "sampling_plan.json")
        initialization = read_json(run_dir / "model_initialization.json")
        if not sampling_plan:
            errors.append(f"{experiment} seed={seed}: empty sampling_plan.json")
            continue
        if int(initialization.get("total_parameter_count", 0)) <= 0:
            errors.append(f"{experiment} seed={seed}: invalid model_initialization.json")
            continue
        runs.append(
            {
                "experiment": experiment,
                "seed": seed,
                "run_dir": str(run_dir),
                "best_epoch": summary.get("best_epoch"),
                "best_metric": summary.get("best_metric"),
            }
        )
    return runs, errors


def main() -> None:
    args = parse_args()
    decision_path = Path(args.decision)
    with decision_path.open("r", encoding="utf-8") as handle:
        decision = yaml.safe_load(handle)

    runs, errors = validate_stage10(Path(args.outputs), decision)
    approved = (
        decision.get("status") == "approved"
        and decision.get("stage10_analysis_reviewed") is True
    )
    if not approved:
        errors.append("Stage 11 decision is not approved after reviewed Stage 10 analysis")

    enabled_configs = [str(path) for path in decision["stage11"]["enabled_configs"]]
    missing_configs = [path for path in enabled_configs if not Path(path).is_file()]
    errors.extend(f"Missing Stage 11 config: {path}" for path in missing_configs)

    report = {
        "gate_open": not errors,
        "decision_status": decision.get("status"),
        "stage10_analysis_reviewed": decision.get("stage10_analysis_reviewed"),
        "validated_stage10_runs": len(runs),
        "expected_stage10_runs": len(expected_stage10(decision)),
        "enabled_configs": enabled_configs,
        "runs": runs,
        "errors": errors,
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if errors:
        raise SystemExit("Stage 11 gate closed: " + "; ".join(errors))
    if args.print_configs:
        print("\n".join(enabled_configs))
    else:
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
