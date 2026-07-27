from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score


CLASSES = ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc")
SEEDS = (42, 43, 44)
EXPERIMENTS = {
    "stage8_real_ce_natural_384": "Real CE natural",
    "stage8_real_ce_weighted_384": "Real CE oversampling",
    "stage8_real_logit_adjust_natural_384": "Real Logit Adjustment",
    "stage8_synthetic_dino_384": "Synthetic + DINO",
    "stage9_real_ce_undersample_384": "Real CE undersampling",
    "stage9_real_balanced_softmax_natural_384": "Real Balanced Softmax",
    "stage9_source_replay_weighted_384": "Source-matched replay",
    "stage9_crt_weighted_384": "cRT classifier retraining",
}
STAGE9_EXPERIMENTS = tuple(name for name in EXPERIMENTS if name.startswith("stage9_"))
METRICS = (
    "macro_f1",
    "balanced_accuracy",
    "mcc",
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
COMPARISONS = (
    ("stage8_synthetic_dino_384", "stage9_source_replay_weighted_384"),
    ("stage8_synthetic_dino_384", "stage8_real_ce_weighted_384"),
    ("stage8_synthetic_dino_384", "stage9_real_ce_undersample_384"),
    ("stage8_synthetic_dino_384", "stage9_real_balanced_softmax_natural_384"),
    ("stage9_source_replay_weighted_384", "stage8_real_ce_weighted_384"),
    ("stage9_crt_weighted_384", "stage8_real_ce_weighted_384"),
)
BOOTSTRAP_METRICS = (
    "macro_f1",
    "balanced_accuracy",
    "mcc",
    "mel_precision",
    "mel_recall",
    "mel_f1",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and compare the completed Stage 8/9 screening runs."
    )
    parser.add_argument(
        "--artifact-root",
        default="local_artifacts/stage9_structured_20260727",
    )
    parser.add_argument(
        "--out-dir",
        default="local_artifacts/stage9_structured_20260727/analysis",
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260727)
    return parser.parse_args()


def expected_calibration_error(
    probabilities: np.ndarray, target: np.ndarray, bins: int = 15
) -> float:
    confidence = probabilities.max(axis=1)
    prediction = probabilities.argmax(axis=1)
    edges = np.linspace(0.0, 1.0, bins + 1)
    value = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        selected = (confidence > lower) & (confidence <= upper)
        if selected.any():
            value += selected.mean() * abs(
                (prediction[selected] == target[selected]).mean() - confidence[selected].mean()
            )
    return float(value)


def metric_bundle(
    target: np.ndarray,
    prediction: np.ndarray,
    probabilities: np.ndarray | None = None,
) -> dict[str, float]:
    class_count = len(CLASSES)
    matrix = np.bincount(
        target * class_count + prediction, minlength=class_count * class_count
    ).reshape(class_count, class_count)
    true_count = matrix.sum(axis=1)
    predicted_count = matrix.sum(axis=0)
    diagonal = np.diag(matrix)
    precision = np.divide(
        diagonal, predicted_count, out=np.zeros(class_count, dtype=float), where=predicted_count != 0
    )
    recall = np.divide(
        diagonal, true_count, out=np.zeros(class_count, dtype=float), where=true_count != 0
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
    result = {
        "macro_f1": float(f1.mean()),
        "balanced_accuracy": float(recall.mean()),
        "mcc": float(numerator / denominator) if denominator else 0.0,
        "worst_class_recall": float(recall.min()),
        "mel_precision": float(precision[mel_index]),
        "mel_recall": float(recall[mel_index]),
        "mel_f1": float(f1[mel_index]),
    }
    if probabilities is not None:
        one_hot = np.eye(class_count, dtype=int)[target]
        result.update(
            {
                "ece": expected_calibration_error(probabilities, target),
                "auroc_ovr_macro": float(
                    roc_auc_score(one_hot, probabilities, average="macro", multi_class="ovr")
                ),
                "auprc_ovr_macro": float(
                    average_precision_score(one_hot, probabilities, average="macro")
                ),
                "mel_auroc": float(roc_auc_score(one_hot[:, mel_index], probabilities[:, mel_index])),
                "mel_auprc": float(
                    average_precision_score(one_hot[:, mel_index], probabilities[:, mel_index])
                ),
            }
        )
    return result


def load_runs(
    root: Path,
) -> tuple[pd.DataFrame, dict[str, dict[int, pd.DataFrame]], pd.DataFrame]:
    rows: list[dict[str, object]] = []
    predictions: dict[str, dict[int, pd.DataFrame]] = {}
    artifact_checks: list[dict[str, object]] = []
    outputs = root / "research-ml" / "outputs"
    required_stage9 = (
        "summary.json",
        "val_metrics_best.json",
        "val_predictions_best.csv",
        "metrics.csv",
        "config.resolved.yaml",
        "environment.json",
        "sampling_plan.json",
        "model_initialization.json",
        "trainable_parameters.json",
    )
    probability_columns = [f"prob_{name}" for name in CLASSES]

    for experiment, method in EXPERIMENTS.items():
        predictions[experiment] = {}
        run_dirs = sorted((outputs / experiment).glob("*_*"))
        by_seed = {int(path.name.rsplit("_", 1)[-1]): path for path in run_dirs}
        missing_seeds = set(SEEDS) - set(by_seed)
        if missing_seeds:
            raise ValueError(f"{experiment}: missing seeds {sorted(missing_seeds)}")
        for seed in SEEDS:
            run_dir = by_seed[seed]
            metrics = json.loads((run_dir / "val_metrics_best.json").read_text(encoding="utf-8"))
            summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
            if summary.get("test_evaluated") is not False or summary.get("test"):
                raise ValueError(f"{run_dir}: locked test was evaluated")

            frame = pd.read_csv(run_dir / "val_predictions_best.csv")
            if len(frame) != 1280 or frame["is_synthetic"].astype(int).sum() != 0:
                raise ValueError(f"{run_dir}: invalid validation prediction population")
            frame["image_id"] = frame["path"].map(lambda value: Path(value).stem)
            predictions[experiment][seed] = frame

            target = pd.Categorical(frame["target"], categories=CLASSES).codes
            prediction = pd.Categorical(frame["prediction"], categories=CLASSES).codes
            probabilities = frame[probability_columns].to_numpy(dtype=float)
            recalculated = metric_bundle(target, prediction, probabilities)
            expected = {key: metrics[key] for key in METRICS if key in metrics}
            for key in ("precision", "recall", "f1", "auroc", "auprc"):
                if key in metrics["per_class"]["mel"]:
                    expected[f"mel_{key}"] = metrics["per_class"]["mel"][key]
            max_error = max(
                abs(recalculated[key] - float(expected[key])) for key in expected
            )
            if max_error > 5e-7:
                raise ValueError(f"{run_dir}: metric recomputation error {max_error}")

            history = pd.read_csv(run_dir / "metrics.csv")
            best_row = history.loc[history["val/macro_f1"].idxmax()]
            if not np.isclose(
                float(best_row["val/macro_f1"]),
                float(summary["best_metric"]),
                rtol=0.0,
                atol=1e-10,
            ):
                raise ValueError(f"{run_dir}: best checkpoint does not match history")
            rows.append(
                {
                    "experiment": experiment,
                    "method": method,
                    "seed": seed,
                    "run": run_dir.name,
                    "best_epoch": int(summary.get("best_epoch", best_row["epoch"])),
                    "elapsed_seconds": float(summary.get("elapsed_seconds", np.nan)),
                    "metric_recompute_max_abs_error": max_error,
                    **recalculated,
                }
            )

            if experiment in STAGE9_EXPERIMENTS:
                missing = [name for name in required_stage9 if not (run_dir / name).is_file()]
                calibration = run_dir / "calibration_diagnostic" / "calibration_diagnostic.json"
                artifact_checks.append(
                    {
                        "experiment": experiment,
                        "seed": seed,
                        "run": run_dir.name,
                        "missing_required_artifacts": ";".join(missing),
                        "calibration_diagnostic_present": calibration.is_file(),
                        "test_evaluated": summary["test_evaluated"],
                        "validation_rows": len(frame),
                        "validation_synthetic_rows": int(frame["is_synthetic"].sum()),
                        "metric_recompute_max_abs_error": max_error,
                    }
                )
                if missing or not calibration.is_file():
                    raise ValueError(f"{run_dir}: incomplete Stage 9 artifacts")

    return (
        pd.DataFrame(rows).sort_values(["experiment", "seed"]).reset_index(drop=True),
        predictions,
        pd.DataFrame(artifact_checks),
    )


def validate_alignment(
    root: Path, predictions: dict[str, dict[int, pd.DataFrame]]
) -> pd.DataFrame:
    split = pd.read_csv(root / "ham10000" / "splits" / "stage8" / "val_real.csv")
    if split["group_id"].isna().any() or split["image_id"].duplicated().any():
        raise ValueError("Validation split has missing lesion groups or duplicate image IDs")
    expected = set(split["image_id"])
    reference_targets: dict[str, str] | None = None
    rows = []
    for experiment, by_seed in predictions.items():
        for seed, frame in by_seed.items():
            target_map = dict(zip(frame["image_id"], frame["target"], strict=True))
            rows.append(
                {
                    "experiment": experiment,
                    "seed": seed,
                    "rows": len(frame),
                    "unique_images": frame["image_id"].nunique(),
                    "image_set_matches_split": set(frame["image_id"]) == expected,
                    "targets_match_all_runs": reference_targets is None
                    or target_map == reference_targets,
                }
            )
            if reference_targets is None:
                reference_targets = target_map
    checks = pd.DataFrame(rows)
    if not checks[["image_set_matches_split", "targets_match_all_runs"]].all().all():
        raise ValueError("Prediction populations are not aligned")
    return checks


def summarize_methods(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (experiment, method), subset in frame.groupby(["experiment", "method"], sort=False):
        row: dict[str, object] = {
            "experiment": experiment,
            "method": method,
            "seeds": len(subset),
        }
        for metric in METRICS:
            row[f"{metric}_mean"] = subset[metric].mean()
            row[f"{metric}_sd"] = subset[metric].std(ddof=1)
        rows.append(row)
    return pd.DataFrame(rows)


def per_class_discrimination(
    predictions: dict[str, dict[int, pd.DataFrame]],
) -> pd.DataFrame:
    rows = []
    for experiment, by_seed in predictions.items():
        for seed, frame in by_seed.items():
            target = pd.Categorical(frame["target"], categories=CLASSES).codes
            for class_index, class_name in enumerate(CLASSES):
                binary_target = (target == class_index).astype(int)
                probability = frame[f"prob_{class_name}"].to_numpy(dtype=float)
                rows.append(
                    {
                        "experiment": experiment,
                        "method": EXPERIMENTS[experiment],
                        "seed": seed,
                        "class": class_name,
                        "support": int(binary_target.sum()),
                        "auroc": roc_auc_score(binary_target, probability),
                        "auprc": average_precision_score(binary_target, probability),
                    }
                )
    return pd.DataFrame(rows)


def exact_sign_flip_pvalue(differences: np.ndarray) -> float:
    observed = abs(float(differences.mean()))
    permuted = [
        abs(float(np.mean(differences * np.asarray(signs))))
        for signs in itertools.product((-1.0, 1.0), repeat=len(differences))
    ]
    return float(np.mean(np.asarray(permuted) >= observed - 1e-12))


def paired_comparisons(frame: pd.DataFrame) -> pd.DataFrame:
    indexed = frame.set_index(["experiment", "seed"])
    rows = []
    for method, baseline in COMPARISONS:
        for metric in METRICS:
            differences = np.asarray(
                [
                    indexed.loc[(method, seed), metric]
                    - indexed.loc[(baseline, seed), metric]
                    for seed in SEEDS
                ]
            )
            sem = stats.sem(differences)
            interval = stats.t.interval(
                0.95, len(differences) - 1, loc=differences.mean(), scale=sem
            )
            rows.append(
                {
                    "method": EXPERIMENTS[method],
                    "baseline": EXPERIMENTS[baseline],
                    "metric": metric,
                    "mean_difference": differences.mean(),
                    "sd_difference": differences.std(ddof=1),
                    "ci95_low_seed_t": interval[0],
                    "ci95_high_seed_t": interval[1],
                    "wins": int((differences > 0).sum()),
                    "exact_sign_flip_p_two_sided": exact_sign_flip_pvalue(differences),
                    "differences_by_seed": json.dumps(
                        dict(zip(map(str, SEEDS), differences.tolist(), strict=True))
                    ),
                }
            )
    return pd.DataFrame(rows)


def hierarchical_bootstrap(
    root: Path,
    predictions: dict[str, dict[int, pd.DataFrame]],
    method: str,
    baseline: str,
    replicates: int,
    random_seed: int,
) -> pd.DataFrame:
    split = pd.read_csv(
        root / "ham10000" / "splits" / "stage8" / "val_real.csv"
    )[["image_id", "group_id"]]
    groups = split["group_id"].drop_duplicates().to_numpy()
    group_indices = {
        group_id: np.flatnonzero(split["group_id"].to_numpy() == group_id)
        for group_id in groups
    }
    prepared: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]] = {
        method: {},
        baseline: {},
    }
    for experiment in (method, baseline):
        for seed in SEEDS:
            aligned = predictions[experiment][seed].set_index("image_id").loc[split["image_id"]]
            prepared[experiment][seed] = (
                pd.Categorical(aligned["target"], categories=CLASSES).codes,
                pd.Categorical(aligned["prediction"], categories=CLASSES).codes,
            )

    rng = np.random.default_rng(random_seed)
    samples = {metric: [] for metric in BOOTSTRAP_METRICS}
    for _ in range(replicates):
        sampled_seeds = rng.choice(SEEDS, size=len(SEEDS), replace=True)
        sampled_groups = rng.choice(groups, size=len(groups), replace=True)
        indices = np.concatenate([group_indices[group] for group in sampled_groups])
        differences = {metric: [] for metric in BOOTSTRAP_METRICS}
        for seed in sampled_seeds:
            bundles = {}
            for experiment in (method, baseline):
                target, prediction = prepared[experiment][int(seed)]
                bundles[experiment] = metric_bundle(target[indices], prediction[indices])
            for metric in BOOTSTRAP_METRICS:
                differences[metric].append(
                    bundles[method][metric] - bundles[baseline][metric]
                )
        for metric in BOOTSTRAP_METRICS:
            samples[metric].append(float(np.mean(differences[metric])))

    rows = []
    for metric, values in samples.items():
        array = np.asarray(values)
        rows.append(
            {
                "method": EXPERIMENTS[method],
                "baseline": EXPERIMENTS[baseline],
                "metric": metric,
                "mean_difference": array.mean(),
                "ci95_low_hierarchical_bootstrap": np.quantile(array, 0.025),
                "ci95_high_hierarchical_bootstrap": np.quantile(array, 0.975),
                "probability_difference_gt_zero": (array > 0).mean(),
                "replicates": replicates,
            }
        )
    return pd.DataFrame(rows)


def load_calibration(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    outputs = root / "research-ml" / "outputs"
    rows = []
    specificity_rows = []
    offset_rows = []
    for experiment in EXPERIMENTS:
        run_dirs = {
            int(path.name.rsplit("_", 1)[-1]): path
            for path in (outputs / experiment).glob("*_*")
        }
        for seed in SEEDS:
            report = json.loads(
                (
                    run_dirs[seed]
                    / "calibration_diagnostic"
                    / "calibration_diagnostic.json"
                ).read_text(encoding="utf-8")
            )
            uncalibrated = report["evaluation_uncalibrated"]
            scaled = report["evaluation_temperature_scaled"]
            rows.append(
                {
                    "experiment": experiment,
                    "method": EXPERIMENTS[experiment],
                    "seed": seed,
                    "temperature": report["temperature"],
                    "calibration_groups": report["calibration_groups"],
                    "evaluation_groups": report["evaluation_groups"],
                    "ece_before": uncalibrated["ece"],
                    "ece_after_temperature": scaled["ece"],
                    "nll_before": uncalibrated["nll"],
                    "nll_after_temperature": scaled["nll"],
                    "brier_before": uncalibrated["brier"],
                    "brier_after_temperature": scaled["brier"],
                    "macro_f1_evaluation_subset": scaled["macro_f1"],
                    "mel_auroc_evaluation_subset": scaled["mel"]["auroc"],
                    "mel_auprc_evaluation_subset": scaled["mel"]["auprc"],
                }
            )
            for fixed in report["fixed_specificity"]:
                specificity_rows.append(
                    {
                        "experiment": experiment,
                        "method": EXPERIMENTS[experiment],
                        "seed": seed,
                        **fixed,
                    }
                )
            for objective, outcome in report["offsets"].items():
                evaluation = outcome["evaluation"]
                offset_rows.append(
                    {
                        "experiment": experiment,
                        "method": EXPERIMENTS[experiment],
                        "seed": seed,
                        "selection_objective": objective,
                        "mel_logit_offset": outcome["offset"],
                        "calibration_objective_value": outcome[
                            "calibration_objective_value"
                        ],
                        "evaluation_macro_f1": evaluation["macro_f1"],
                        "evaluation_mcc": evaluation["mcc"],
                        "evaluation_ece": evaluation["ece"],
                        "evaluation_mel_precision": evaluation["mel"]["precision"],
                        "evaluation_mel_recall": evaluation["mel"]["recall"],
                        "evaluation_mel_f1": evaluation["mel"]["f1"],
                    }
                )
    return (
        pd.DataFrame(rows),
        pd.DataFrame(specificity_rows),
        pd.DataFrame(offset_rows),
    )


def plot_summary(frame: pd.DataFrame, out_path: Path) -> None:
    metrics = ("macro_f1", "mcc", "mel_precision", "mel_recall")
    order = list(EXPERIMENTS.values())
    short_labels = (
        "CE natural",
        "CE oversample",
        "Logit adjust",
        "Synthetic+DINO",
        "CE undersample",
        "Balanced Softmax",
        "Source replay",
        "cRT",
    )
    figure, axes = plt.subplots(1, 4, figsize=(16, 6.5), constrained_layout=True)
    colors = ["#3b6fb6", "#2f855a", "#8a5a44", "#c2413b", "#6b7280", "#9c6b30", "#2a7f89", "#7251a3"]
    for axis, metric in zip(axes, metrics):
        for index, method in enumerate(order):
            values = frame.loc[frame["method"] == method, metric].to_numpy()
            axis.scatter(
                np.full(len(values), index),
                values,
                color=colors[index],
                s=34,
                alpha=0.85,
                zorder=2,
            )
            axis.errorbar(
                index,
                values.mean(),
                yerr=values.std(ddof=1),
                marker="D",
                color="black",
                capsize=3,
                markersize=5,
                zorder=3,
            )
        axis.set_title(metric.replace("_", " "))
        axis.set_xticks(range(len(order)), short_labels, rotation=35, ha="right")
        axis.tick_params(axis="x", labelsize=8)
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("Stage 8/9 matched-seed validation results (mean +/- SD, n=3)")
    figure.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    root = Path(args.artifact_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    frame, predictions, artifact_checks = load_runs(root)
    alignment = validate_alignment(root, predictions)
    method_summary = summarize_methods(frame)
    per_class = per_class_discrimination(predictions)
    paired = paired_comparisons(frame)
    bootstrap = pd.concat(
        [
            hierarchical_bootstrap(
                root,
                predictions,
                method,
                baseline,
                args.bootstrap_replicates,
                args.seed + index,
            )
            for index, (method, baseline) in enumerate(COMPARISONS)
        ],
        ignore_index=True,
    )
    calibration, fixed_specificity, calibration_offsets = load_calibration(root)

    frame.to_csv(out_dir / "seed_metrics.csv", index=False)
    method_summary.to_csv(out_dir / "method_summary.csv", index=False)
    per_class.to_csv(out_dir / "per_class_discrimination.csv", index=False)
    paired.to_csv(out_dir / "paired_seed_comparisons.csv", index=False)
    bootstrap.to_csv(out_dir / "hierarchical_lesion_bootstrap.csv", index=False)
    calibration.to_csv(out_dir / "calibration_summary.csv", index=False)
    fixed_specificity.to_csv(out_dir / "fixed_specificity_diagnostics.csv", index=False)
    calibration_offsets.to_csv(out_dir / "calibration_offset_outcomes.csv", index=False)
    artifact_checks.to_csv(out_dir / "stage9_artifact_integrity.csv", index=False)
    alignment.to_csv(out_dir / "prediction_alignment.csv", index=False)
    plot_summary(frame, out_dir / "stage8_stage9_comparison.png")

    summary = {
        "artifact_root": str(Path(args.artifact_root)),
        "experiments": list(EXPERIMENTS),
        "stage9_runs": len(artifact_checks),
        "seeds": list(SEEDS),
        "validation_rows": 1280,
        "validation_groups": int(
            pd.read_csv(root / "ham10000" / "splits" / "stage8" / "val_real.csv")[
                "group_id"
            ].nunique()
        ),
        "locked_test_evaluated": False,
        "metric_recompute_max_abs_error": float(
            frame["metric_recompute_max_abs_error"].max()
        ),
        "prediction_alignment_passed": True,
        "stage9_artifact_integrity_passed": True,
        "bootstrap_replicates": args.bootstrap_replicates,
        "calibration_status": "exploratory; group-held-out diagnostic after validation-based early stopping",
    }
    (out_dir / "analysis_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
