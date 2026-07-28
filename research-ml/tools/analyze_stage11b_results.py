from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

import tools.analyze_stage10_results as paired_analysis


STRATUM = "strict_id"
SEEDS = (42, 43, 44)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze the Stage 11B paired confirmation.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--data-root", default="/srv/research/projects/default/ham10000")
    parser.add_argument("--out-dir", default="outputs/reports/stage11b_analysis")
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260728)
    return parser.parse_args()


def experiment_name(arm: str, stratum: str) -> str:
    if stratum != STRATUM:
        raise ValueError(f"Stage 11B only permits {STRATUM}")
    return f"stage11b_{arm}_strict_id_convnext_small_384"


def decision_summary(
    paired: pd.DataFrame, bootstrap: pd.DataFrame
) -> dict[str, object]:
    by_metric = paired.set_index("metric")
    boot = bootstrap.set_index("metric")

    def delta(metric: str) -> float:
        return float(by_metric.loc[metric, "mean_difference_synthetic_minus_replay"])

    def wins(metric: str) -> int:
        return int(by_metric.loc[metric, "synthetic_wins"])

    global_direction = (
        delta("macro_f1") > 0
        and delta("mcc") > 0
        and delta("balanced_accuracy") > 0
    )
    stable_global = wins("macro_f1") == 3 and wins("mcc") == 3
    inferential_signal = (
        float(boot.loc["macro_f1", "ci95_low"]) > 0
        or float(boot.loc["mcc", "ci95_low"]) > 0
    )
    mel_guardrail = delta("mel_recall") >= -0.05 and wins("mel_auprc") >= 1

    if global_direction and stable_global and inferential_signal and mel_guardrail:
        classification = "strong_positive"
    elif global_direction and stable_global:
        classification = "mixed_positive"
    elif global_direction:
        classification = "weak_or_unstable_positive"
    else:
        classification = "null_or_negative"
    return {
        "classification": classification,
        "global_direction_positive": global_direction,
        "macro_f1_and_mcc_positive_3_of_3": stable_global,
        "macro_f1_or_mcc_lesion_ci_above_zero": inferential_signal,
        "melanoma_guardrail_passed": mel_guardrail,
        "predeclared_melanoma_recall_margin": -0.05,
        "note": (
            "This is an internal validation confirmation. It does not authorize opening "
            "the locked test without a separate documented decision."
        ),
    }


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    data_root = Path(args.data_root).resolve()
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = project_root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    paired_analysis.STRATA = (STRATUM,)
    paired_analysis.SEEDS = SEEDS
    paired_analysis.experiment_name = experiment_name

    frame, predictions, checks = paired_analysis.load_runs(project_root)
    split = pd.read_csv(data_root / "splits/stage8/val_real.csv")
    alignment = paired_analysis.validate_alignment(split, predictions)
    method_summary = paired_analysis.summarize_runs(frame)
    paired = paired_analysis.paired_seed_comparisons(frame)
    per_class = paired_analysis.per_class_discrimination(predictions)
    bootstrap = paired_analysis.hierarchical_bootstrap(
        split,
        predictions,
        STRATUM,
        args.bootstrap_replicates,
        args.seed,
    )
    calibration, fixed = paired_analysis.load_calibration(project_root)
    decision = decision_summary(paired, bootstrap)

    frame.to_csv(out_dir / "seed_metrics.csv", index=False)
    method_summary.to_csv(out_dir / "method_summary.csv", index=False)
    paired.to_csv(out_dir / "paired_seed_comparisons.csv", index=False)
    bootstrap.to_csv(out_dir / "hierarchical_lesion_bootstrap.csv", index=False)
    per_class.to_csv(out_dir / "per_class_discrimination.csv", index=False)
    calibration.to_csv(out_dir / "calibration_summary.csv", index=False)
    fixed.to_csv(out_dir / "fixed_specificity_diagnostics.csv", index=False)
    checks.to_csv(out_dir / "artifact_integrity.csv", index=False)
    alignment.to_csv(out_dir / "prediction_alignment.csv", index=False)
    paired_analysis.plot_deltas(paired, out_dir / "paired_validation_deltas.png")

    summary = {
        "status": "complete",
        "stage": "stage11b",
        "runs": int(len(frame)),
        "seeds": list(SEEDS),
        "locked_test_evaluated": False,
        "validation_rows": int(len(split)),
        "validation_groups": int(split["group_id"].nunique()),
        "bootstrap_replicates": int(args.bootstrap_replicates),
        "metric_recompute_max_abs_error": float(
            checks["metric_recompute_max_abs_error"].max()
        ),
        "prediction_alignment_passed": True,
        "decision": decision,
        "calibration_limitation": (
            "Exploratory group-held-out diagnostic after validation-based early stopping."
        ),
    }
    (out_dir / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
