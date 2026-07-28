from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


METHODS = {
    "stage8_real_ce_natural_384": "Stage 8 ConvNeXt-B anchor",
    "stage11_real_convnext_base_regularized_384": "Stage 11 ConvNeXt-B",
    "stage11_real_convnext_small_regularized_384": "Stage 11 ConvNeXt-S",
    "stage11_real_dinov2_base_linear_392": "DINOv2-B linear",
    "stage11_real_dinov2_base_finetune_392": "DINOv2-B full",
}
SEEDS = (42, 43, 44)
CLASSES = ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc")
CORE_METRICS = (
    "macro_f1",
    "mcc",
    "balanced_accuracy",
    "ece",
    "worst_class_recall",
    "auroc_ovr_macro",
    "auprc_ovr_macro",
    "mel_precision",
    "mel_recall",
    "mel_f1",
    "mel_auroc",
    "mel_auprc",
)
REQUIRED_COMMON = (
    "summary.json",
    "config.resolved.yaml",
    "class_counts.json",
    "val_metrics_best.json",
    "val_predictions_best.csv",
    "best.pt",
)
REQUIRED_STAGE11 = REQUIRED_COMMON + ("sampling_plan.json", "model_initialization.json")
COMPARISONS = (
    ("stage11_real_convnext_base_regularized_384", "stage8_real_ce_natural_384"),
    ("stage11_real_convnext_small_regularized_384", "stage8_real_ce_natural_384"),
    ("stage11_real_convnext_small_regularized_384", "stage11_real_convnext_base_regularized_384"),
    ("stage11_real_dinov2_base_linear_392", "stage8_real_ce_natural_384"),
    ("stage11_real_dinov2_base_finetune_392", "stage8_real_ce_natural_384"),
    ("stage11_real_dinov2_base_finetune_392", "stage11_real_convnext_small_regularized_384"),
)


def metric_bundle_arrays(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    class_count = len(CLASSES)
    matrix = np.bincount(
        target * class_count + prediction, minlength=class_count * class_count
    ).reshape(class_count, class_count)
    true_count = matrix.sum(axis=1)
    predicted_count = matrix.sum(axis=0)
    diagonal = np.diag(matrix)
    precision = np.divide(
        diagonal,
        predicted_count,
        out=np.zeros(class_count, dtype=float),
        where=predicted_count != 0,
    )
    recall = np.divide(
        diagonal,
        true_count,
        out=np.zeros(class_count, dtype=float),
        where=true_count != 0,
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros(class_count, dtype=float),
        where=(precision + recall) != 0,
    )
    sample_count = matrix.sum()
    numerator = diagonal.sum() * sample_count - np.dot(true_count, predicted_count)
    denominator = np.sqrt(
        (sample_count**2 - np.dot(predicted_count, predicted_count))
        * (sample_count**2 - np.dot(true_count, true_count))
    )
    mel_index = CLASSES.index("mel")
    return {
        "macro_f1": float(f1.mean()),
        "balanced_accuracy": float(recall.mean()),
        "mcc": float(numerator / denominator) if denominator else 0.0,
        "mel_precision": float(precision[mel_index]),
        "mel_recall": float(recall[mel_index]),
        "mel_f1": float(f1[mel_index]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Stage 11 baseline qualification.")
    parser.add_argument("--outputs", default="outputs")
    parser.add_argument(
        "--split-csv",
        default="/srv/research/projects/default/ham10000/splits/stage8/val_real.csv",
    )
    parser.add_argument("--out-dir", default="outputs/reports/stage11_analysis")
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260728)
    return parser.parse_args()


def find_run(outputs: Path, experiment: str, seed: int) -> Path:
    matches = [
        path.parent
        for path in (outputs / experiment).glob(f"*_{seed}/summary.json")
        if "invalid" not in str(path)
    ]
    if len(matches) != 1:
        raise ValueError(f"{experiment} seed={seed}: expected one completed run, found {len(matches)}")
    return matches[0]


def load_data(
    outputs: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[int, pd.DataFrame]]]:
    rows: list[dict[str, Any]] = []
    integrity: list[dict[str, Any]] = []
    predictions: dict[str, dict[int, pd.DataFrame]] = {}
    for experiment, method in METHODS.items():
        predictions[experiment] = {}
        for seed in SEEDS:
            run_dir = find_run(outputs, experiment, seed)
            required = (
                REQUIRED_COMMON
                if experiment == "stage8_real_ce_natural_384"
                else REQUIRED_STAGE11
            )
            missing = [name for name in required if not (run_dir / name).is_file()]
            summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
            metrics = json.loads((run_dir / "val_metrics_best.json").read_text(encoding="utf-8"))
            if summary.get("test_evaluated") is not False or summary.get("test"):
                raise ValueError(f"{run_dir}: locked test was evaluated")
            prediction = pd.read_csv(run_dir / "val_predictions_best.csv")
            prediction["image_id"] = prediction["path"].map(lambda value: Path(value).stem)
            if len(prediction) != 1280 or prediction["image_id"].nunique() != 1280:
                raise ValueError(f"{run_dir}: invalid validation population")
            predictions[experiment][seed] = prediction
            initialization_path = run_dir / "model_initialization.json"
            initialization = (
                json.loads(initialization_path.read_text(encoding="utf-8"))
                if initialization_path.is_file()
                else {}
            )
            if "best_epoch" in summary:
                best_epoch = int(summary["best_epoch"])
            else:
                history = pd.read_csv(run_dir / "metrics.csv")
                best_epoch = int(
                    history.loc[history["val/macro_f1"].idxmax(), "epoch"]
                )
            row: dict[str, Any] = {
                "experiment": experiment,
                "method": method,
                "seed": seed,
                "run_dir": str(run_dir),
                "best_epoch": best_epoch,
                "elapsed_seconds": float(summary["elapsed_seconds"]),
                "total_parameter_count": initialization.get("total_parameter_count"),
                "trainable_parameter_count": initialization.get("trainable_parameter_count"),
            }
            for metric in (
                "macro_f1",
                "mcc",
                "balanced_accuracy",
                "ece",
                "worst_class_recall",
                "auroc_ovr_macro",
                "auprc_ovr_macro",
            ):
                row[metric] = float(metrics[metric])
            for class_name in CLASSES:
                for metric in ("precision", "recall", "f1", "auroc", "auprc"):
                    class_metrics = metrics["per_class"][class_name]
                    if metric in class_metrics:
                        value = float(class_metrics[metric])
                    else:
                        target = (prediction["target"].astype(str) == class_name).astype(int)
                        score = prediction[f"prob_{class_name}"].to_numpy(float)
                        value = float(
                            roc_auc_score(target, score)
                            if metric == "auroc"
                            else average_precision_score(target, score)
                        )
                    row[f"{class_name}_{metric}"] = value
            rows.append(row)
            integrity.append(
                {
                    "experiment": experiment,
                    "seed": seed,
                    "run_dir": str(run_dir),
                    "missing_artifacts": json.dumps(missing),
                    "artifact_integrity": not missing,
                    "test_evaluated": summary.get("test_evaluated"),
                    "validation_rows": len(prediction),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(integrity), predictions


def align_predictions(
    split_csv: Path,
    predictions: dict[str, dict[int, pd.DataFrame]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    split = pd.read_csv(split_csv)[["image_id", "group_id", "label"]]
    if split["group_id"].isna().any() or split["image_id"].duplicated().any():
        raise ValueError("Invalid validation split groups")
    expected = set(split["image_id"])
    checks = []
    reference: dict[str, str] | None = None
    for experiment, by_seed in predictions.items():
        for seed, frame in by_seed.items():
            targets = dict(zip(frame["image_id"], frame["target"], strict=True))
            checks.append(
                {
                    "experiment": experiment,
                    "seed": seed,
                    "image_set_matches": set(frame["image_id"]) == expected,
                    "targets_match": reference is None or targets == reference,
                }
            )
            reference = targets if reference is None else reference
    checks_frame = pd.DataFrame(checks)
    if not checks_frame[["image_set_matches", "targets_match"]].all().all():
        raise ValueError("Prediction alignment failed")
    return split, checks_frame


def summarize_methods(seed_metrics: pd.DataFrame) -> pd.DataFrame:
    metrics = list(CORE_METRICS) + ["best_epoch", "elapsed_seconds"]
    rows = []
    for (experiment, method), frame in seed_metrics.groupby(["experiment", "method"]):
        row: dict[str, Any] = {"experiment": experiment, "method": method, "seeds": len(frame)}
        for metric in metrics:
            row[f"{metric}_mean"] = float(frame[metric].mean())
            row[f"{metric}_std"] = float(frame[metric].std(ddof=1))
        row["total_parameter_count"] = frame["total_parameter_count"].iloc[0]
        row["trainable_parameter_count"] = frame["trainable_parameter_count"].iloc[0]
        rows.append(row)
    return pd.DataFrame(rows)


def paired_seed_comparisons(seed_metrics: pd.DataFrame) -> pd.DataFrame:
    indexed = seed_metrics.set_index(["experiment", "seed"])
    rows = []
    for method, baseline in COMPARISONS:
        for metric in CORE_METRICS:
            differences = np.asarray(
                [
                    indexed.loc[(method, seed), metric]
                    - indexed.loc[(baseline, seed), metric]
                    for seed in SEEDS
                ]
            )
            rows.append(
                {
                    "method": METHODS[method],
                    "baseline": METHODS[baseline],
                    "metric": metric,
                    "mean_difference": float(differences.mean()),
                    "sd_difference": float(differences.std(ddof=1)),
                    "wins": int((differences > 0).sum()),
                    "differences_by_seed": json.dumps(
                        dict(zip(map(str, SEEDS), map(float, differences))), sort_keys=True
                    ),
                }
            )
    return pd.DataFrame(rows)


def hierarchical_bootstrap(
    split: pd.DataFrame,
    predictions: dict[str, dict[int, pd.DataFrame]],
    replicates: int,
    random_seed: int,
) -> pd.DataFrame:
    groups = split["group_id"].drop_duplicates().to_numpy()
    group_indices = {
        group: np.flatnonzero(split["group_id"].to_numpy() == group) for group in groups
    }
    prepared: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]] = {}
    for experiment, by_seed in predictions.items():
        prepared[experiment] = {}
        for seed, frame in by_seed.items():
            aligned = frame.set_index("image_id").loc[split["image_id"]]
            prepared[experiment][seed] = (
                pd.Categorical(aligned["target"], categories=CLASSES).codes,
                pd.Categorical(aligned["prediction"], categories=CLASSES).codes,
            )
    rng = np.random.default_rng(random_seed)
    metrics = ("macro_f1", "mcc", "balanced_accuracy", "mel_precision", "mel_recall", "mel_f1")
    samples: dict[tuple[str, str, str], list[float]] = {
        (method, baseline, metric): []
        for method, baseline in COMPARISONS
        for metric in metrics
    }
    for _ in range(replicates):
        sampled_seeds = rng.choice(SEEDS, size=len(SEEDS), replace=True)
        sampled_groups = rng.choice(groups, size=len(groups), replace=True)
        indices = np.concatenate([group_indices[group] for group in sampled_groups])
        for method, baseline in COMPARISONS:
            seed_differences = {metric: [] for metric in metrics}
            for seed in sampled_seeds:
                bundles = {}
                for experiment in (method, baseline):
                    target, prediction = prepared[experiment][int(seed)]
                    bundles[experiment] = metric_bundle_arrays(
                        target[indices], prediction[indices]
                    )
                for metric in metrics:
                    seed_differences[metric].append(
                        bundles[method][metric] - bundles[baseline][metric]
                    )
            for metric in metrics:
                samples[(method, baseline, metric)].append(
                    float(np.mean(seed_differences[metric]))
                )
    rows = []
    for (method, baseline, metric), values in samples.items():
        array = np.asarray(values)
        rows.append(
            {
                "method": METHODS[method],
                "baseline": METHODS[baseline],
                "metric": metric,
                "mean_difference": float(array.mean()),
                "ci95_low": float(np.quantile(array, 0.025)),
                "ci95_high": float(np.quantile(array, 0.975)),
                "probability_difference_gt_zero": float((array > 0).mean()),
                "replicates": replicates,
            }
        )
    return pd.DataFrame(rows)


def per_class_summary(seed_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (experiment, method), frame in seed_metrics.groupby(["experiment", "method"]):
        for class_name in CLASSES:
            row: dict[str, Any] = {
                "experiment": experiment,
                "method": method,
                "class": class_name,
            }
            for metric in ("precision", "recall", "f1", "auroc", "auprc"):
                values = frame[f"{class_name}_{metric}"]
                row[f"{metric}_mean"] = float(values.mean())
                row[f"{metric}_std"] = float(values.std(ddof=1))
            rows.append(row)
    return pd.DataFrame(rows)


def load_calibration(outputs: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    calibration_rows = []
    fixed_rows = []
    for experiment, method in METHODS.items():
        for seed in SEEDS:
            run_dir = find_run(outputs, experiment, seed)
            path = run_dir / "stage11_calibration_diagnostic" / "calibration_diagnostic.json"
            report = json.loads(path.read_text(encoding="utf-8"))
            raw = report["evaluation_uncalibrated"]
            scaled = report["evaluation_temperature_scaled"]
            calibration_rows.append(
                {
                    "experiment": experiment,
                    "method": method,
                    "seed": seed,
                    "temperature": report["temperature"],
                    "ece_before": raw["ece"],
                    "ece_after": scaled["ece"],
                    "nll_before": raw["nll"],
                    "nll_after": scaled["nll"],
                    "brier_before": raw["brier"],
                    "brier_after": scaled["brier"],
                }
            )
            for fixed in report["fixed_specificity"]:
                fixed_rows.append(
                    {
                        "experiment": experiment,
                        "method": method,
                        "seed": seed,
                        **fixed,
                    }
                )
    return pd.DataFrame(calibration_rows), pd.DataFrame(fixed_rows)


def plot_methods(seed_metrics: pd.DataFrame, out_path: Path) -> None:
    metrics = ("macro_f1", "mcc", "balanced_accuracy", "auprc_ovr_macro", "mel_f1", "mel_auprc")
    figure, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for axis, metric in zip(axes.flat, metrics):
        for method, frame in seed_metrics.groupby("method", sort=False):
            frame = frame.sort_values("seed")
            axis.plot(frame["seed"], frame[metric], marker="o", label=method)
        axis.set_title(metric.replace("_", " "))
        axis.set_xticks(SEEDS)
        axis.grid(alpha=0.25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
    figure.suptitle("Stage 11 baseline qualification across matched seeds")
    figure.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    outputs = Path(args.outputs)
    split_csv = Path(args.split_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_metrics, integrity, predictions = load_data(outputs)
    split, alignment = align_predictions(split_csv, predictions)
    method_summary = summarize_methods(seed_metrics)
    paired = paired_seed_comparisons(seed_metrics)
    bootstrap = hierarchical_bootstrap(
        split, predictions, args.bootstrap_replicates, args.seed
    )
    per_class = per_class_summary(seed_metrics)
    calibration, fixed = load_calibration(outputs)
    seed_metrics.to_csv(out_dir / "seed_metrics.csv", index=False)
    integrity.to_csv(out_dir / "artifact_integrity.csv", index=False)
    alignment.to_csv(out_dir / "prediction_alignment.csv", index=False)
    method_summary.to_csv(out_dir / "method_summary.csv", index=False)
    paired.to_csv(out_dir / "paired_seed_comparisons.csv", index=False)
    bootstrap.to_csv(out_dir / "hierarchical_lesion_bootstrap.csv", index=False)
    per_class.to_csv(out_dir / "per_class_metrics.csv", index=False)
    calibration.to_csv(out_dir / "calibration_summary.csv", index=False)
    fixed.to_csv(out_dir / "fixed_specificity_diagnostics.csv", index=False)
    plot_methods(seed_metrics, out_dir / "seed_metric_comparison.png")
    summary = {
        "status": "complete",
        "methods": METHODS,
        "seeds": list(SEEDS),
        "runs": len(seed_metrics),
        "validation_rows": len(split),
        "validation_groups": int(split["group_id"].nunique()),
        "bootstrap_replicates": args.bootstrap_replicates,
        "locked_test_evaluated": False,
        "artifact_integrity_passed": bool(integrity["artifact_integrity"].all()),
        "prediction_alignment_passed": bool(
            alignment[["image_set_matches", "targets_match"]].all().all()
        ),
    }
    (out_dir / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
