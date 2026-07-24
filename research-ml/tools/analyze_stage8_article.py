from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


EXPERIMENTS = {
    "stage8_real_ce_natural_384": "Real CE natural",
    "stage8_real_ce_weighted_384": "Real CE weighted",
    "stage8_real_logit_adjust_natural_384": "Real Logit Adjustment",
    "stage8_synthetic_dino_384": "Synthetic + DINO",
}
PRIMARY_METRICS = (
    "macro_f1",
    "balanced_accuracy",
    "mcc",
    "ece",
    "worst_class_recall",
)
CLASSES = ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and statistically compare the completed Stage 8 screening runs."
    )
    parser.add_argument(
        "--artifact-root",
        default="local_artifacts/stage8_article_analysis",
        help="Directory containing the downloaded server output tree.",
    )
    parser.add_argument(
        "--out-dir",
        default="local_artifacts/stage8_article_analysis/analysis",
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260724)
    return parser.parse_args()


def load_runs(root: Path) -> tuple[pd.DataFrame, dict[str, dict[int, pd.DataFrame]]]:
    rows: list[dict[str, object]] = []
    predictions: dict[str, dict[int, pd.DataFrame]] = {}
    outputs = root / "outputs"
    for experiment, label in EXPERIMENTS.items():
        predictions[experiment] = {}
        run_dirs = sorted((outputs / experiment).glob("*_*"))
        if len(run_dirs) != 5:
            raise ValueError(f"{experiment}: expected 5 runs, found {len(run_dirs)}")
        for run_dir in run_dirs:
            seed = int(run_dir.name.rsplit("_", 1)[-1])
            metrics = json.loads((run_dir / "val_metrics_best.json").read_text(encoding="utf-8"))
            summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
            if summary.get("test_evaluated") is not False or summary.get("test"):
                raise ValueError(f"{run_dir}: locked test was evaluated")
            prediction = pd.read_csv(run_dir / "val_predictions_best.csv")
            if len(prediction) != 1280:
                raise ValueError(f"{run_dir}: expected 1280 validation rows, found {len(prediction)}")
            if prediction["is_synthetic"].astype(int).sum() != 0:
                raise ValueError(f"{run_dir}: synthetic image found in validation predictions")
            prediction["image_id"] = prediction["path"].map(lambda value: Path(value).stem)
            predictions[experiment][seed] = prediction

            history = pd.read_csv(run_dir / "metrics.csv")
            best_row = history.loc[history["val/macro_f1"].idxmax()]
            if not np.isclose(
                float(best_row["val/macro_f1"]),
                float(summary["best_metric"]),
                rtol=0.0,
                atol=1e-10,
            ):
                raise ValueError(f"{run_dir}: best checkpoint metric does not match history")
            row: dict[str, object] = {
                "experiment": experiment,
                "method": label,
                "seed": seed,
                "best_epoch": int(best_row["epoch"]),
            }
            for metric in PRIMARY_METRICS:
                row[metric] = float(metrics[metric])
            for class_name in CLASSES:
                class_metrics = metrics["per_class"][class_name]
                for metric in ("precision", "recall", "f1"):
                    row[f"{class_name}_{metric}"] = float(class_metrics[metric])
            rows.append(row)
    frame = pd.DataFrame(rows).sort_values(["experiment", "seed"]).reset_index(drop=True)
    if sorted(frame["seed"].unique().tolist()) != [42, 43, 44, 45, 46]:
        raise ValueError("Stage 8 seed set is incomplete")
    return frame, predictions


def validate_prediction_alignment(
    root: Path, predictions: dict[str, dict[int, pd.DataFrame]]
) -> pd.DataFrame:
    split = pd.read_csv(root / "val_real.csv")
    if split["group_id"].isna().any() or split["image_id"].duplicated().any():
        raise ValueError("Validation split has missing groups or duplicate image IDs")
    expected = set(split["image_id"])
    reference_targets: dict[str, str] | None = None
    checks = []
    for experiment, by_seed in predictions.items():
        for seed, frame in by_seed.items():
            target_map = dict(zip(frame["image_id"], frame["target"], strict=True))
            checks.append(
                {
                    "experiment": experiment,
                    "seed": seed,
                    "rows": len(frame),
                    "unique_images": frame["image_id"].nunique(),
                    "image_set_matches_split": set(frame["image_id"]) == expected,
                    "targets_match_other_runs": reference_targets is None
                    or target_map == reference_targets,
                }
            )
            if reference_targets is None:
                reference_targets = target_map
    result = pd.DataFrame(checks)
    boolean_columns = ("image_set_matches_split", "targets_match_other_runs")
    if not result[list(boolean_columns)].all().all():
        raise ValueError("Prediction files are not aligned across methods and seeds")
    return result


def exact_sign_flip_pvalue(differences: np.ndarray) -> float:
    observed = abs(float(np.mean(differences)))
    means = []
    for signs in itertools.product((-1.0, 1.0), repeat=len(differences)):
        means.append(abs(float(np.mean(differences * np.asarray(signs)))))
    return float(np.mean(np.asarray(means) >= observed - 1e-12))


def paired_seed_comparisons(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metrics = PRIMARY_METRICS + (
        "mel_precision",
        "mel_recall",
        "mel_f1",
        "akiec_recall",
        "bkl_recall",
    )
    pivot = frame.set_index(["experiment", "seed"])
    comparisons = (
        ("stage8_synthetic_dino_384", "stage8_real_ce_weighted_384"),
        ("stage8_synthetic_dino_384", "stage8_real_ce_natural_384"),
        ("stage8_real_ce_weighted_384", "stage8_real_ce_natural_384"),
        ("stage8_real_logit_adjust_natural_384", "stage8_real_ce_natural_384"),
    )
    for method, baseline in comparisons:
        for metric in metrics:
            differences = np.asarray(
                [
                    pivot.loc[(method, seed), metric] - pivot.loc[(baseline, seed), metric]
                    for seed in (42, 43, 44, 45, 46)
                ],
                dtype=float,
            )
            sem = stats.sem(differences)
            interval = stats.t.interval(0.95, len(differences) - 1, loc=differences.mean(), scale=sem)
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
                    "ties": int((differences == 0).sum()),
                    "exact_sign_flip_p_two_sided": exact_sign_flip_pvalue(differences),
                    "differences_by_seed": json.dumps(
                        {str(seed): value for seed, value in zip((42, 43, 44, 45, 46), differences)}
                    ),
                }
            )
    return pd.DataFrame(rows)


def metric_bundle_arrays(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
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
    return {
        "macro_f1": float(f1.mean()),
        "balanced_accuracy": float(recall.mean()),
        "mcc": float(numerator / denominator) if denominator else 0.0,
        "mel_precision": float(precision[mel_index]),
        "mel_recall": float(recall[mel_index]),
        "mel_f1": float(f1[mel_index]),
    }


def hierarchical_bootstrap(
    root: Path,
    predictions: dict[str, dict[int, pd.DataFrame]],
    method: str,
    baseline: str,
    replicates: int,
    random_seed: int,
) -> pd.DataFrame:
    split = pd.read_csv(root / "val_real.csv")[["image_id", "group_id"]]
    groups = split["group_id"].drop_duplicates().to_numpy()
    group_indices = {
        group_id: np.flatnonzero(split["group_id"].to_numpy() == group_id)
        for group_id in groups
    }
    seeds = np.asarray([42, 43, 44, 45, 46])
    prepared: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]] = {method: {}, baseline: {}}
    for experiment in (method, baseline):
        for seed in seeds:
            source = predictions[experiment][int(seed)].set_index("image_id")
            aligned = source.loc[split["image_id"]]
            prepared[experiment][int(seed)] = (
                pd.Categorical(aligned["target"], categories=CLASSES).codes,
                pd.Categorical(aligned["prediction"], categories=CLASSES).codes,
            )
    rng = np.random.default_rng(random_seed)
    values: dict[str, list[float]] = {
        metric: [] for metric in ("macro_f1", "balanced_accuracy", "mcc", "mel_precision", "mel_recall", "mel_f1")
    }
    for _ in range(replicates):
        sampled_seeds = rng.choice(seeds, size=len(seeds), replace=True)
        sampled_groups = rng.choice(groups, size=len(groups), replace=True)
        sampled_indices = np.concatenate([group_indices[group_id] for group_id in sampled_groups])
        seed_differences = {metric: [] for metric in values}
        for seed in sampled_seeds:
            metrics_by_method = {}
            for experiment in (method, baseline):
                target, prediction = prepared[experiment][int(seed)]
                metrics_by_method[experiment] = metric_bundle_arrays(
                    target[sampled_indices], prediction[sampled_indices]
                )
            for metric in values:
                seed_differences[metric].append(
                    metrics_by_method[method][metric] - metrics_by_method[baseline][metric]
                )
        for metric in values:
            values[metric].append(float(np.mean(seed_differences[metric])))

    rows = []
    for metric, samples in values.items():
        array = np.asarray(samples)
        rows.append(
            {
                "method": EXPERIMENTS[method],
                "baseline": EXPERIMENTS[baseline],
                "metric": metric,
                "mean_difference": float(array.mean()),
                "ci95_low_hierarchical_bootstrap": float(np.quantile(array, 0.025)),
                "ci95_high_hierarchical_bootstrap": float(np.quantile(array, 0.975)),
                "probability_difference_gt_zero": float((array > 0).mean()),
                "replicates": replicates,
            }
        )
    return pd.DataFrame(rows)


def mel_offset_frontier(
    predictions: dict[str, dict[int, pd.DataFrame]],
) -> pd.DataFrame:
    rows = []
    probability_columns = [f"prob_{class_name}" for class_name in CLASSES]
    offsets = np.linspace(-30.0, 30.0, 6001)
    synthetic_experiment = "stage8_synthetic_dino_384"
    for baseline in ("stage8_real_ce_weighted_384", "stage8_real_ce_natural_384"):
        for seed in (42, 43, 44, 45, 46):
            synthetic = predictions[synthetic_experiment][seed]
            target = pd.Categorical(synthetic["target"], categories=CLASSES).codes
            synthetic_prediction = pd.Categorical(
                synthetic["prediction"], categories=CLASSES
            ).codes
            synthetic_metrics = metric_bundle_arrays(target, synthetic_prediction)

            baseline_frame = predictions[baseline][seed]
            probabilities = baseline_frame[probability_columns].to_numpy(dtype=float)
            log_probabilities = np.log(np.clip(probabilities, 1e-12, 1.0))
            candidates = []
            for offset in offsets:
                adjusted = log_probabilities.copy()
                adjusted[:, CLASSES.index("mel")] += offset
                prediction = adjusted.argmax(axis=1)
                metrics = metric_bundle_arrays(target, prediction)
                if metrics["mel_recall"] + 1e-12 >= synthetic_metrics["mel_recall"]:
                    candidates.append((float(offset), metrics))
            if not candidates:
                raise ValueError(f"No mel-offset candidate reached synthetic recall for seed {seed}")

            for objective in ("macro_f1", "mcc", "mel_f1"):
                offset, metrics = max(candidates, key=lambda item: item[1][objective])
                rows.append(
                    {
                        "baseline": EXPERIMENTS[baseline],
                        "seed": seed,
                        "selection_objective": objective,
                        "mel_logit_offset": offset,
                        "synthetic_mel_recall": synthetic_metrics["mel_recall"],
                        "adjusted_mel_recall": metrics["mel_recall"],
                        "synthetic_mel_precision": synthetic_metrics["mel_precision"],
                        "adjusted_mel_precision": metrics["mel_precision"],
                        "synthetic_mel_f1": synthetic_metrics["mel_f1"],
                        "adjusted_mel_f1": metrics["mel_f1"],
                        "synthetic_macro_f1": synthetic_metrics["macro_f1"],
                        "adjusted_macro_f1": metrics["macro_f1"],
                        "synthetic_mcc": synthetic_metrics["mcc"],
                        "adjusted_mcc": metrics["mcc"],
                        "adjusted_dominates_synthetic": bool(
                            metrics["mel_recall"] >= synthetic_metrics["mel_recall"]
                            and metrics["mel_precision"] >= synthetic_metrics["mel_precision"]
                            and metrics["macro_f1"] >= synthetic_metrics["macro_f1"]
                            and metrics["mcc"] >= synthetic_metrics["mcc"]
                        ),
                    }
                )
    return pd.DataFrame(rows)


def plot_seed_metrics(frame: pd.DataFrame, out_path: Path) -> None:
    metrics = ("macro_f1", "balanced_accuracy", "mcc", "mel_recall", "mel_precision", "mel_f1")
    figure, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    colors = {
        "Real CE natural": "#3b6fb6",
        "Real CE weighted": "#2f855a",
        "Real Logit Adjustment": "#8a5a44",
        "Synthetic + DINO": "#c2413b",
    }
    for axis, metric in zip(axes.flat, metrics):
        for method, subset in frame.groupby("method", sort=False):
            subset = subset.sort_values("seed")
            axis.plot(
                subset["seed"],
                subset[metric],
                marker="o",
                linewidth=1.7,
                label=method,
                color=colors[method],
            )
        axis.set_title(metric.replace("_", " "))
        axis.set_xlabel("Seed")
        axis.grid(alpha=0.25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=4, frameon=False)
    figure.suptitle("Stage 8 validation metrics across matched seeds", fontsize=15)
    figure.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    root = Path(args.artifact_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    frame, predictions = load_runs(root)
    checks = validate_prediction_alignment(root, predictions)
    comparisons = paired_seed_comparisons(frame)
    bootstrap = pd.concat(
        [
            hierarchical_bootstrap(
                root,
                predictions,
                "stage8_synthetic_dino_384",
                baseline,
                args.bootstrap_replicates,
                args.seed + index,
            )
            for index, baseline in enumerate(
                ("stage8_real_ce_weighted_384", "stage8_real_ce_natural_384")
            )
        ],
        ignore_index=True,
    )
    threshold_frontier = mel_offset_frontier(predictions)

    frame.to_csv(out_dir / "stage8_seed_metrics.csv", index=False)
    checks.to_csv(out_dir / "prediction_alignment_checks.csv", index=False)
    comparisons.to_csv(out_dir / "paired_seed_comparisons.csv", index=False)
    bootstrap.to_csv(out_dir / "hierarchical_bootstrap_comparisons.csv", index=False)
    threshold_frontier.to_csv(out_dir / "mel_offset_frontier.csv", index=False)
    plot_seed_metrics(frame, out_dir / "stage8_seed_comparison.png")

    summary = {
        "artifact_root": str(root),
        "experiments": list(EXPERIMENTS),
        "seeds": [42, 43, 44, 45, 46],
        "validation_rows": 1280,
        "validation_groups": int(pd.read_csv(root / "val_real.csv")["group_id"].nunique()),
        "locked_test_evaluated": False,
        "prediction_alignment_passed": True,
        "bootstrap_replicates": args.bootstrap_replicates,
        "outputs": {
            "seed_metrics": "stage8_seed_metrics.csv",
            "paired_seed_comparisons": "paired_seed_comparisons.csv",
            "hierarchical_bootstrap": "hierarchical_bootstrap_comparisons.csv",
            "mel_offset_frontier": "mel_offset_frontier.csv",
            "figure": "stage8_seed_comparison.png",
        },
    }
    (out_dir / "analysis_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
