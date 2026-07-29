from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config


BASELINE_EXPERIMENT = "stage11_real_convnext_small_regularized_384"
SEEDS = (42, 43, 44)
REQUIRED_ARTIFACTS = (
    "summary.json",
    "config.resolved.yaml",
    "sampling_plan.json",
    "model_initialization.json",
    "class_counts.json",
    "val_metrics_best.json",
    "val_predictions_best.csv",
    "best.pt",
)
MATCH_PATHS = (
    ("runtime", "deterministic"),
    ("data", "train_csv"),
    ("data", "val_csv"),
    ("data", "test_csv"),
    ("data", "image_size"),
    ("data", "val_size"),
    ("model",),
    ("training",),
    ("imbalance",),
    ("evaluation", "run_test"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Stage 14P Branch C.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--data-root", default="/srv/research/projects/default/ham10000")
    parser.add_argument("--readiness", default="outputs/reports/stage14_readiness.json")
    parser.add_argument(
        "--baseline-config",
        default="configs/ham10000_stage11_convnext_small_regularized_384.yaml",
    )
    parser.add_argument(
        "--candidate-config",
        default="configs/ham10000_stage14p_real_aspect_pad_convnext_small_384.yaml",
    )
    parser.add_argument("--report", default="outputs/reports/stage14p_gate.json")
    return parser.parse_args()


def nested(config: dict[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = config
    for key in path:
        value = value[key]
    return value


def main() -> None:
    args = parse_args()
    root = Path(args.project_root).resolve()
    data_root = Path(args.data_root).resolve()
    readiness = json.loads((root / args.readiness).read_text(encoding="utf-8"))
    baseline = load_config(root / args.baseline_config)
    candidate = load_config(root / args.candidate_config)
    errors: list[str] = []

    if readiness.get("selected_branch") != "C_preprocessing_qualification":
        errors.append("Stage 14 readiness did not select Branch C")
    if readiness.get("locked_test_evaluated") is not False:
        errors.append("Stage 14 readiness indicates locked-test use")
    for path in MATCH_PATHS:
        if nested(baseline, path) != nested(candidate, path):
            errors.append(f"Candidate changed non-preprocessing recipe: {'.'.join(path)}")

    train_aug = candidate["augmentation"]["train"]
    eval_aug = candidate["augmentation"]["eval"]
    if train_aug.get("aspect_preserving_pad") is not True:
        errors.append("Training aspect-preserving pad is not enabled")
    if train_aug.get("random_resized_crop") is not False:
        errors.append("Random crop must be disabled for the full-frame arm")
    if eval_aug.get("aspect_preserving_pad") is not True:
        errors.append("Evaluation aspect-preserving pad is not enabled")
    if eval_aug.get("center_crop") is not False:
        errors.append("Evaluation center crop must be disabled")
    if candidate["evaluation"]["run_test"] is not False:
        errors.append("Locked test evaluation must remain disabled")

    train = pd.read_csv(data_root / candidate["data"]["train_csv"])
    validation = pd.read_csv(data_root / candidate["data"]["val_csv"])
    test = pd.read_csv(data_root / candidate["data"]["test_csv"])
    if train["is_synthetic"].astype(int).any():
        errors.append("Stage 14P candidate must be real-only")
    train_groups = set(train["group_id"].astype(str))
    eval_groups = set(validation["group_id"].astype(str)) | set(test["group_id"].astype(str))
    if train_groups & eval_groups:
        errors.append("Training lesion groups overlap validation/test")

    baseline_runs: list[dict[str, Any]] = []
    for seed in SEEDS:
        matches = sorted((root / "outputs" / BASELINE_EXPERIMENT).glob(f"*_{seed}"))
        completed = [run for run in matches if (run / "summary.json").is_file()]
        if len(completed) != 1:
            errors.append(f"Expected one completed baseline run for seed {seed}")
            continue
        run = completed[0]
        summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
        missing = [name for name in REQUIRED_ARTIFACTS if not (run / name).is_file()]
        if summary.get("test_evaluated") is not False:
            errors.append(f"Baseline seed {seed} opened locked test")
        if missing:
            errors.append(f"Baseline seed {seed} missing artifacts: {missing}")
        baseline_runs.append({"seed": seed, "run": str(run), "missing": missing})

    report = {
        "protocol": "stage14p_real_only_preprocessing_qualification",
        "gate_open": not errors,
        "selected_branch": readiness.get("selected_branch"),
        "candidate_config": args.candidate_config,
        "historical_control": BASELINE_EXPERIMENT,
        "seeds": list(SEEDS),
        "locked_test_used": False,
        "train_rows": int(len(train)),
        "train_groups": int(train["group_id"].nunique()),
        "baseline_runs": baseline_runs,
        "errors": errors,
    }
    report_path = root / args.report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
