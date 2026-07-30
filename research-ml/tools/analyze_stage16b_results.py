from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)

from src.metrics import expected_calibration_error


SEEDS = (42, 43, 44)
ARMS = {
    "natural": "stage16_isic2019_real_ce_natural_384",
    "balanced_softmax": "stage16_isic2019_real_balanced_softmax_384",
}
REQUIRED_ARTIFACTS = (
    "summary.json",
    "config.resolved.yaml",
    "sampling_plan.json",
    "model_initialization.json",
    "class_counts.json",
    "environment.json",
    "val_metrics_best.json",
    "val_predictions_best.csv",
    "metrics.csv",
    "best.pt",
)
CORE_METRICS = (
    "macro_auprc",
    "macro_auroc",
    "macro_f1",
    "mcc",
    "balanced_accuracy",
    "ece",
    "worst_class_recall",
    "mel_precision",
    "mel_recall",
    "mel_f1",
    "mel_auprc",
    "scc_precision",
    "scc_recall",
    "scc_f1",
    "scc_auprc",
)
BOOTSTRAP_METRICS = (
    "macro_auprc",
    "macro_f1",
    "mcc",
    "balanced_accuracy",
    "ece",
    "worst_class_recall",
    "mel_auprc",
    "scc_auprc",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Stage 16B ISIC 2019 confirmation.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument(
        "--split",
        default="/srv/research/projects/default/isic2019/splits/val.csv",
    )
    parser.add_argument("--out-dir", default="outputs/reports/stage16b_analysis")
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260730)
    return parser.parse_args()


def latest_run_by_seed(experiment_dir: Path) -> dict[int, Path]:
    runs: dict[int, Path] = {}
    for path in sorted(experiment_dir.glob("*_*")):
        suffix = path.name.rsplit("_", 1)[-1]
        if path.is_dir() and suffix.isdigit():
            runs[int(suffix)] = path
    return runs


def class_names(frame: pd.DataFrame) -> list[str]:
    return [column.removeprefix("prob_") for column in frame if column.startswith("prob_")]


def metric_bundle(
    frame: pd.DataFrame,
    classes: list[str],
    indices: np.ndarray | None = None,
) -> dict[str, float]:
    if indices is not None:
        frame = frame.iloc[indices]
    probabilities = frame[[f"prob_{name}" for name in classes]].to_numpy(float)
    target = pd.Categorical(frame["target"], categories=classes).codes
    prediction = pd.Categorical(frame["prediction"], categories=classes).codes
    if (target < 0).any() or (prediction < 0).any():
        raise ValueError("Predictions contain labels outside the probability columns")

    _, recall, _, _ = precision_recall_fscore_support(
        target, prediction, labels=range(len(classes)), zero_division=0
    )
    one_hot = np.eye(len(classes))[target]
    result = {
        "macro_auprc": float(average_precision_score(one_hot, probabilities, average="macro")),
        "macro_auroc": float(
            roc_auc_score(target, probabilities, multi_class="ovr", average="macro")
        ),
        "macro_f1": float(f1_score(target, prediction, average="macro", zero_division=0)),
        "mcc": float(matthews_corrcoef(target, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(target, prediction)),
        "ece": expected_calibration_error(probabilities, target, n_bins=15),
        "worst_class_recall": float(recall.min()),
    }
    for name in ("mel", "scc"):
        index = classes.index(name)
        binary = (target == index).astype(int)
        precision, class_recall, class_f1, _ = precision_recall_fscore_support(
            binary,
            prediction == index,
            average="binary",
            zero_division=0,
        )
        result.update(
            {
                f"{name}_precision": float(precision),
                f"{name}_recall": float(class_recall),
                f"{name}_f1": float(class_f1),
                f"{name}_auprc": float(
                    average_precision_score(binary, probabilities[:, index])
                ),
            }
        )
    return result


def load_runs(
    project_root: Path,
) -> tuple[pd.DataFrame, dict[str, dict[int, pd.DataFrame]], pd.DataFrame, list[str]]:
    rows: list[dict[str, object]] = []
    checks: list[dict[str, object]] = []
    predictions: dict[str, dict[int, pd.DataFrame]] = {arm: {} for arm in ARMS}
    expected_classes: list[str] | None = None

    for arm, experiment in ARMS.items():
        by_seed = latest_run_by_seed(project_root / "outputs" / experiment)
        if not set(SEEDS).issubset(by_seed):
            raise ValueError(f"{experiment}: expected seeds {SEEDS}, found {sorted(by_seed)}")
        for seed in SEEDS:
            run_dir = by_seed[seed]
            missing = [name for name in REQUIRED_ARTIFACTS if not (run_dir / name).is_file()]
            if missing:
                raise ValueError(f"{run_dir}: missing artifacts {missing}")
            summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
            if summary.get("test_evaluated") is not False or summary.get("test"):
                raise ValueError(f"{run_dir}: locked test was evaluated")
            config = yaml.safe_load((run_dir / "config.resolved.yaml").read_text(encoding="utf-8"))
            if config["training"]["monitor"] != "val/auprc_ovr_macro":
                raise ValueError(f"{run_dir}: unexpected monitor {config['training']['monitor']}")
            if int(config["data"]["batch_size"]) != 48:
                raise ValueError(f"{run_dir}: unexpected batch size")

            frame = pd.read_csv(run_dir / "val_predictions_best.csv")
            if len(frame) != 3799 or int(frame["is_synthetic"].astype(int).sum()) != 0:
                raise ValueError(f"{run_dir}: invalid validation population")
            classes = class_names(frame)
            if expected_classes is None:
                expected_classes = classes
            elif classes != expected_classes:
                raise ValueError(f"{run_dir}: class order changed")
            frame["image_id"] = frame["path"].map(lambda value: Path(value).stem)
            metrics = metric_bundle(frame, classes)
            stored = json.loads((run_dir / "val_metrics_best.json").read_text(encoding="utf-8"))
            comparisons = {
                "macro_auprc": stored["auprc_ovr_macro"],
                "macro_auroc": stored["auroc_ovr_macro"],
                "macro_f1": stored["macro_f1"],
                "mcc": stored["mcc"],
                "balanced_accuracy": stored["balanced_accuracy"],
                "ece": stored["ece"],
                "worst_class_recall": stored["worst_class_recall"],
                **{
                    f"{name}_{metric}": stored["per_class"][name][metric]
                    for name in ("mel", "scc")
                    for metric in ("precision", "recall", "f1", "auprc")
                },
            }
            max_error = max(abs(metrics[key] - float(value)) for key, value in comparisons.items())
            if max_error > 5e-7:
                raise ValueError(f"{run_dir}: metric recomputation error {max_error}")

            predictions[arm][seed] = frame
            rows.append(
                {
                    "arm": arm,
                    "experiment": experiment,
                    "seed": seed,
                    "run": run_dir.name,
                    "best_epoch": int(summary["best_epoch"]),
                    "elapsed_seconds": float(summary["elapsed_seconds"]),
                    **metrics,
                }
            )
            checks.append(
                {
                    "arm": arm,
                    "seed": seed,
                    "run": run_dir.name,
                    "test_evaluated": False,
                    "validation_rows": len(frame),
                    "missing_artifacts": "",
                    "metric_recompute_max_abs_error": max_error,
                    "monitor": config["training"]["monitor"],
                    "batch_size": int(config["data"]["batch_size"]),
                }
            )
    assert expected_classes is not None
    return pd.DataFrame(rows), predictions, pd.DataFrame(checks), expected_classes


def align_predictions(
    split: pd.DataFrame,
    predictions: dict[str, dict[int, pd.DataFrame]],
) -> pd.DataFrame:
    if split["group_id"].isna().any() or split["image_id"].duplicated().any():
        raise ValueError("Validation split has invalid group or image IDs")
    expected = split["image_id"].astype(str).tolist()
    reference: dict[str, str] | None = None
    rows = []
    for arm in ARMS:
        for seed in SEEDS:
            frame = predictions[arm][seed]
            target_map = dict(zip(frame["image_id"], frame["target"], strict=True))
            row = {
                "arm": arm,
                "seed": seed,
                "image_set_matches": set(frame["image_id"]) == set(expected),
                "targets_match": reference is None or target_map == reference,
            }
            rows.append(row)
            if reference is None:
                reference = target_map
            predictions[arm][seed] = frame.set_index("image_id").loc[expected].reset_index()
    result = pd.DataFrame(rows)
    if not result[["image_set_matches", "targets_match"]].all().all():
        raise ValueError("Prediction populations are not aligned")
    return result


def summarize(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    for arm, subset in frame.groupby("arm", sort=False):
        row: dict[str, object] = {"arm": arm, "seeds": len(subset)}
        for metric in CORE_METRICS:
            row[f"{metric}_mean"] = float(subset[metric].mean())
            row[f"{metric}_sd"] = float(subset[metric].std(ddof=1))
        row["best_epoch_mean"] = float(subset["best_epoch"].mean())
        row["elapsed_seconds_mean"] = float(subset["elapsed_seconds"].mean())
        summary_rows.append(row)

    indexed = frame.set_index(["arm", "seed"])
    paired_rows = []
    for metric in CORE_METRICS:
        differences = np.asarray(
            [
                indexed.loc[("balanced_softmax", seed), metric]
                - indexed.loc[("natural", seed), metric]
                for seed in SEEDS
            ]
        )
        interval = stats.t.interval(
            0.95,
            len(differences) - 1,
            loc=differences.mean(),
            scale=stats.sem(differences),
        )
        paired_rows.append(
            {
                "metric": metric,
                "mean_difference_balanced_minus_natural": float(differences.mean()),
                "sd_difference": float(differences.std(ddof=1)),
                "ci95_low_seed_t": float(interval[0]),
                "ci95_high_seed_t": float(interval[1]),
                "balanced_wins": int((differences > 0).sum()),
                "differences_by_seed": json.dumps(
                    dict(zip(map(str, SEEDS), differences.tolist(), strict=True))
                ),
            }
        )
    return pd.DataFrame(summary_rows), pd.DataFrame(paired_rows)


def hierarchical_bootstrap(
    split: pd.DataFrame,
    predictions: dict[str, dict[int, pd.DataFrame]],
    classes: list[str],
    replicates: int,
    random_seed: int,
) -> pd.DataFrame:
    groups = split["group_id"].astype(str).drop_duplicates().to_numpy()
    group_values = split["group_id"].astype(str).to_numpy()
    group_indices = {group: np.flatnonzero(group_values == group) for group in groups}
    rng = np.random.default_rng(random_seed)
    values = {metric: [] for metric in BOOTSTRAP_METRICS}

    for _ in range(replicates):
        sampled_seeds = rng.choice(SEEDS, size=len(SEEDS), replace=True)
        sampled_groups = rng.choice(groups, size=len(groups), replace=True)
        indices = np.concatenate([group_indices[group] for group in sampled_groups])
        replicate = {metric: [] for metric in BOOTSTRAP_METRICS}
        for seed in sampled_seeds:
            bundles = {
                arm: metric_bundle(predictions[arm][int(seed)], classes, indices)
                for arm in ARMS
            }
            for metric in BOOTSTRAP_METRICS:
                replicate[metric].append(
                    bundles["balanced_softmax"][metric] - bundles["natural"][metric]
                )
        for metric in BOOTSTRAP_METRICS:
            values[metric].append(float(np.mean(replicate[metric])))

    rows = []
    for metric, samples in values.items():
        array = np.asarray(samples)
        rows.append(
            {
                "metric": metric,
                "mean_difference": float(array.mean()),
                "ci95_low": float(np.quantile(array, 0.025)),
                "ci95_high": float(np.quantile(array, 0.975)),
                "probability_difference_gt_zero": float((array > 0).mean()),
                "replicates": replicates,
            }
        )
    return pd.DataFrame(rows)


def per_class_and_fixed_specificity(
    predictions: dict[str, dict[int, pd.DataFrame]],
    classes: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_class_rows = []
    fixed_rows = []
    for arm in ARMS:
        for seed in SEEDS:
            frame = predictions[arm][seed]
            target = pd.Categorical(frame["target"], categories=classes).codes
            for index, name in enumerate(classes):
                binary = (target == index).astype(int)
                scores = frame[f"prob_{name}"].to_numpy(float)
                per_class_rows.append(
                    {
                        "arm": arm,
                        "seed": seed,
                        "class": name,
                        "support": int(binary.sum()),
                        "auroc": float(roc_auc_score(binary, scores)),
                        "auprc": float(average_precision_score(binary, scores)),
                    }
                )
                if name not in {"mel", "scc"}:
                    continue
                false_positive_rate, true_positive_rate, thresholds = roc_curve(binary, scores)
                specificity = 1.0 - false_positive_rate
                for target_specificity in (0.90, 0.95):
                    eligible = np.flatnonzero(specificity >= target_specificity)
                    best = eligible[np.argmax(true_positive_rate[eligible])]
                    fixed_rows.append(
                        {
                            "arm": arm,
                            "seed": seed,
                            "class": name,
                            "target_specificity": target_specificity,
                            "specificity": float(specificity[best]),
                            "sensitivity": float(true_positive_rate[best]),
                            "threshold": float(thresholds[best]),
                        }
                    )
    return pd.DataFrame(per_class_rows), pd.DataFrame(fixed_rows)


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    split_path = Path(args.split)
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = project_root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    frame, predictions, checks, classes = load_runs(project_root)
    split = pd.read_csv(split_path)
    alignment = align_predictions(split, predictions)
    method_summary, paired = summarize(frame)
    bootstrap = hierarchical_bootstrap(
        split,
        predictions,
        classes,
        args.bootstrap_replicates,
        args.seed,
    )
    per_class, fixed = per_class_and_fixed_specificity(predictions, classes)

    frame.to_csv(out_dir / "seed_metrics.csv", index=False)
    method_summary.to_csv(out_dir / "method_summary.csv", index=False)
    paired.to_csv(out_dir / "paired_seed_comparisons.csv", index=False)
    bootstrap.to_csv(out_dir / "hierarchical_lesion_bootstrap.csv", index=False)
    per_class.to_csv(out_dir / "per_class_discrimination.csv", index=False)
    fixed.to_csv(out_dir / "fixed_specificity_diagnostics.csv", index=False)
    checks.to_csv(out_dir / "artifact_integrity.csv", index=False)
    alignment.to_csv(out_dir / "prediction_alignment.csv", index=False)

    summary = {
        "status": "complete",
        "runs": len(frame),
        "seeds": list(SEEDS),
        "classes": classes,
        "locked_test_evaluated": False,
        "validation_rows": len(split),
        "validation_groups": int(split["group_id"].nunique()),
        "bootstrap_replicates": args.bootstrap_replicates,
        "metric_recompute_max_abs_error": float(
            checks["metric_recompute_max_abs_error"].max()
        ),
        "prediction_alignment_passed": True,
        "contrast": "balanced_softmax_minus_natural",
    }
    (out_dir / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
