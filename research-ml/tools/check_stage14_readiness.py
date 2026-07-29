from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Choose the predeclared Stage 14 branch from Stage 13B artifacts."
    )
    parser.add_argument("--decision", default="configs/stage14_decision.yaml")
    parser.add_argument("--analysis-dir", default="outputs/reports/stage13b_analysis")
    parser.add_argument("--report", default="outputs/reports/stage14_readiness.json")
    return parser.parse_args()


def metric_rows(frame: pd.DataFrame) -> dict[str, pd.Series]:
    return {str(row["metric"]): row for _, row in frame.iterrows()}


def choose_branch(
    paired: pd.DataFrame,
    bootstrap: pd.DataFrame,
    thresholds: dict[str, Any],
) -> tuple[str, dict[str, bool]]:
    metrics = metric_rows(paired)
    boot = metric_rows(bootstrap)

    def delta(metric: str) -> float:
        return float(metrics[metric]["mean_difference_synthetic_minus_replay"])

    def wins(metric: str) -> int:
        return int(metrics[metric]["synthetic_wins"])

    minimum_wins = int(thresholds["minimum_seed_wins"])
    global_positive = all(delta(metric) > 0 for metric in ("macro_f1", "mcc"))
    ranking_positive = delta("auprc_ovr_macro") > 0
    stable_f1 = wins("macro_f1") >= minimum_wins
    stable_ranking = wins("auprc_ovr_macro") >= minimum_wins
    inferential_signal = any(
        float(boot[metric]["ci95_low"]) > 0 for metric in ("macro_f1", "mcc")
    )
    melanoma_guardrail = (
        delta("mel_recall") >= float(thresholds["melanoma_recall_margin"])
        and delta("mel_auprc") >= float(thresholds["melanoma_auprc_margin"])
    )
    calibration_guardrail = delta("ece") <= float(
        thresholds["ece_worsening_margin"]
    )
    checks = {
        "global_positive": global_positive,
        "ranking_positive": ranking_positive,
        "stable_macro_f1": stable_f1,
        "stable_macro_auprc": stable_ranking,
        "lesion_bootstrap_signal": inferential_signal,
        "melanoma_guardrail": melanoma_guardrail,
        "calibration_guardrail": calibration_guardrail,
    }
    if all(checks.values()):
        return "A_external_robustness", checks
    if (global_positive and stable_f1) or (ranking_positive and stable_ranking):
        return "B_generator_pilot", checks
    return "C_preprocessing_qualification", checks


def main() -> None:
    args = parse_args()
    decision_path = Path(args.decision)
    analysis_dir = Path(args.analysis_dir)
    decision = yaml.safe_load(decision_path.read_text(encoding="utf-8"))
    summary = json.loads(
        (analysis_dir / "analysis_summary.json").read_text(encoding="utf-8")
    )
    if summary.get("status") != "complete" or summary.get("runs") != 6:
        raise SystemExit("Stage 13B analysis is not complete with six runs")
    if summary.get("locked_test_evaluated") is not False:
        raise SystemExit("Stage 13B opened the locked test")
    paired = pd.read_csv(analysis_dir / "paired_seed_comparisons.csv")
    bootstrap = pd.read_csv(analysis_dir / "hierarchical_lesion_bootstrap.csv")
    branch, checks = choose_branch(paired, bootstrap, decision["thresholds"])
    report = {
        "status": "open",
        "source_stage": "stage13b",
        "locked_test_evaluated": False,
        "selected_branch": branch,
        "checks": checks,
        "objective": decision["branches"][branch]["objective"],
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
