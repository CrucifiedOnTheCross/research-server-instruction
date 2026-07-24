from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.metrics import compute_metrics, softmax


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exploratory group-held-out calibration diagnostic for a completed run."
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--split-csv", required=True)
    parser.add_argument("--target-class", default="mel")
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--ece-bins", type=int, default=15)
    parser.add_argument("--out-dir", default=None)
    return parser.parse_args()


def load_aligned(
    run_dir: Path,
    split_csv: Path,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[str]]:
    predictions = pd.read_csv(run_dir / "val_predictions_best.csv")
    split = pd.read_csv(split_csv)
    predictions["image_id"] = predictions["path"].map(lambda value: Path(value).stem)
    required_split = {"image_id", "group_id", "label"}
    missing = required_split - set(split.columns)
    if missing:
        raise ValueError(f"Split CSV is missing columns: {sorted(missing)}")
    merged = predictions.merge(
        split[["image_id", "group_id", "label"]],
        on="image_id",
        how="inner",
        validate="one_to_one",
        suffixes=("_prediction", "_split"),
    )
    if len(merged) != len(predictions):
        raise ValueError("Prediction and split image sets are not identical")
    classes = sorted(
        column.removeprefix("logit_")
        for column in merged.columns
        if column.startswith("logit_")
    )
    if not classes:
        classes = sorted(
            column.removeprefix("prob_")
            for column in merged.columns
            if column.startswith("prob_")
        )
        probabilities = merged[[f"prob_{name}" for name in classes]].to_numpy(float)
        logits = np.log(np.clip(probabilities, 1e-12, 1.0))
    else:
        logits = merged[[f"logit_{name}" for name in classes]].to_numpy(float)
    class_to_idx = {name: index for index, name in enumerate(classes)}
    target_names = merged["target"].astype(str)
    if not target_names.isin(class_to_idx).all():
        raise ValueError("Prediction targets contain unknown classes")
    targets = target_names.map(class_to_idx).to_numpy(int)
    if not (merged["label"].astype(str).to_numpy() == target_names.to_numpy()).all():
        raise ValueError("Split labels do not match prediction targets")
    return merged, logits, targets, classes


def group_calibration_split(
    frame: pd.DataFrame,
    targets: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    splitter = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=seed)
    calibration, evaluation = next(
        splitter.split(
            np.zeros(len(frame)),
            targets,
            groups=frame["group_id"].astype(str).to_numpy(),
        )
    )
    calibration_groups = set(frame.iloc[calibration]["group_id"].astype(str))
    evaluation_groups = set(frame.iloc[evaluation]["group_id"].astype(str))
    if calibration_groups & evaluation_groups:
        raise RuntimeError("Calibration/evaluation group leakage detected")
    return calibration, evaluation


def fit_temperature(logits: np.ndarray, targets: np.ndarray) -> float:
    labels = np.arange(logits.shape[1])

    def objective(log_temperature: float) -> float:
        temperature = float(np.exp(log_temperature))
        return float(log_loss(targets, softmax(logits / temperature), labels=labels))

    result = minimize_scalar(objective, bounds=(-4.0, 4.0), method="bounded")
    if not result.success:
        raise RuntimeError(f"Temperature optimization failed: {result.message}")
    return float(np.exp(result.x))


def choose_offsets(
    logits: np.ndarray,
    targets: np.ndarray,
    target_index: int,
    idx_to_class: dict[int, str],
    ece_bins: int,
) -> dict[str, dict[str, float]]:
    candidates: list[tuple[float, dict[str, Any]]] = []
    for offset in np.linspace(-5.0, 5.0, 1001):
        adjusted = logits.copy()
        adjusted[:, target_index] += offset
        metrics = compute_metrics(adjusted, targets, idx_to_class, ece_bins)
        candidates.append((float(offset), metrics))
    objectives = {
        "macro_f1": lambda metrics: float(metrics["macro_f1"]),
        "mcc": lambda metrics: float(metrics["mcc"]),
        "target_f1": lambda metrics: float(
            metrics["per_class"][idx_to_class[target_index]]["f1"]
        ),
    }
    selected = {}
    for name, key in objectives.items():
        offset, metrics = max(candidates, key=lambda item: (key(item[1]), -abs(item[0])))
        selected[name] = {
            "offset": offset,
            "calibration_objective_value": key(metrics),
        }
    return selected


def binary_operating_point(
    calibration_scores: np.ndarray,
    calibration_target: np.ndarray,
    evaluation_scores: np.ndarray,
    evaluation_target: np.ndarray,
    minimum_specificity: float,
) -> dict[str, float]:
    thresholds = np.unique(
        np.concatenate(
            [
                np.asarray([0.0, 1.0]),
                calibration_scores,
            ]
        )
    )
    candidates = []
    for threshold in thresholds:
        prediction = calibration_scores >= threshold
        negative = calibration_target == 0
        positive = calibration_target == 1
        specificity = float((~prediction[negative]).mean())
        sensitivity = float(prediction[positive].mean())
        if specificity >= minimum_specificity:
            candidates.append((sensitivity, specificity, float(threshold)))
    if not candidates:
        raise RuntimeError("No threshold satisfies the requested specificity")
    _, calibration_specificity, threshold = max(
        candidates,
        key=lambda item: (item[0], item[1], -item[2]),
    )
    prediction = evaluation_scores >= threshold
    negative = evaluation_target == 0
    positive = evaluation_target == 1
    tp = int((prediction & positive).sum())
    fp = int((prediction & negative).sum())
    return {
        "minimum_calibration_specificity": minimum_specificity,
        "threshold": threshold,
        "calibration_specificity": calibration_specificity,
        "evaluation_specificity": float((~prediction[negative]).mean()),
        "evaluation_sensitivity": float(prediction[positive].mean()),
        "evaluation_precision": float(tp / (tp + fp)) if tp + fp else 0.0,
    }


def compact_metrics(metrics: dict[str, Any], target_class: str) -> dict[str, Any]:
    return {
        key: metrics[key]
        for key in (
            "macro_f1",
            "balanced_accuracy",
            "mcc",
            "nll",
            "brier",
            "ece",
            "worst_class_recall",
            "auroc_ovr_macro",
            "auprc_ovr_macro",
        )
        if key in metrics
    } | {target_class: metrics["per_class"][target_class]}


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    split_csv = Path(args.split_csv)
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "calibration_diagnostic"
    out_dir.mkdir(parents=True, exist_ok=True)

    frame, logits, targets, classes = load_aligned(run_dir, split_csv)
    if args.target_class not in classes:
        raise ValueError(f"Unknown target class: {args.target_class}")
    class_to_idx = {name: index for index, name in enumerate(classes)}
    idx_to_class = {index: name for name, index in class_to_idx.items()}
    target_index = class_to_idx[args.target_class]
    calibration, evaluation = group_calibration_split(frame, targets, args.seed)
    temperature = fit_temperature(logits[calibration], targets[calibration])
    calibrated_logits = logits / temperature
    offsets = choose_offsets(
        calibrated_logits[calibration],
        targets[calibration],
        target_index,
        idx_to_class,
        args.ece_bins,
    )

    result: dict[str, Any] = {
        "status": "exploratory",
        "limitation": (
            "The full validation split was used for early stopping. This group-held-out "
            "calibration diagnostic is not confirmatory."
        ),
        "run_dir": str(run_dir),
        "split_csv": str(split_csv),
        "classes": classes,
        "target_class": args.target_class,
        "seed": args.seed,
        "temperature": temperature,
        "calibration_rows": int(len(calibration)),
        "evaluation_rows": int(len(evaluation)),
        "calibration_groups": int(frame.iloc[calibration]["group_id"].nunique()),
        "evaluation_groups": int(frame.iloc[evaluation]["group_id"].nunique()),
    }
    result["evaluation_uncalibrated"] = compact_metrics(
        compute_metrics(
            logits[evaluation],
            targets[evaluation],
            idx_to_class,
            args.ece_bins,
        ),
        args.target_class,
    )
    result["evaluation_temperature_scaled"] = compact_metrics(
        compute_metrics(
            calibrated_logits[evaluation],
            targets[evaluation],
            idx_to_class,
            args.ece_bins,
        ),
        args.target_class,
    )

    result["offsets"] = {}
    for objective, selection in offsets.items():
        adjusted = calibrated_logits[evaluation].copy()
        adjusted[:, target_index] += selection["offset"]
        metrics = compute_metrics(adjusted, targets[evaluation], idx_to_class, args.ece_bins)
        result["offsets"][objective] = {
            **selection,
            "evaluation": compact_metrics(metrics, args.target_class),
        }

    probabilities = softmax(calibrated_logits)
    binary_target = (targets == target_index).astype(int)
    result["fixed_specificity"] = [
        binary_operating_point(
            probabilities[calibration, target_index],
            binary_target[calibration],
            probabilities[evaluation, target_index],
            binary_target[evaluation],
            specificity,
        )
        for specificity in (0.90, 0.95)
    ]

    assignments = frame[["image_id", "group_id", "target"]].copy()
    assignments["calibration_role"] = "evaluation"
    assignments.loc[calibration, "calibration_role"] = "calibration"
    assignments.to_csv(out_dir / "assignments.csv", index=False)
    (out_dir / "calibration_diagnostic.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
