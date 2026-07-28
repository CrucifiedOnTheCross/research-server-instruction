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

from tools.analyze_stage9_results import CLASSES, metric_bundle


SEEDS = (42, 43, 44)
STRATA = ("strict_id", "aid_radial", "ood_far", "random_remaining")
ARMS = ("synthetic", "replay")
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
BOOTSTRAP_METRICS = (
    "macro_f1",
    "balanced_accuracy",
    "mcc",
    "worst_class_recall",
    "mel_precision",
    "mel_recall",
    "mel_f1",
)
REQUIRED_ARTIFACTS = (
    "summary.json",
    "val_metrics_best.json",
    "val_predictions_best.csv",
    "metrics.csv",
    "config.resolved.yaml",
    "environment.json",
    "sampling_plan.json",
    "model_initialization.json",
    "class_counts.json",
    "best.pt",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze completed Stage 10 geometry strata.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--data-root", default="/srv/research/projects/default/ham10000")
    parser.add_argument("--out-dir", default="outputs/reports/stage10_analysis")
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260728)
    return parser.parse_args()


def experiment_name(arm: str, stratum: str) -> str:
    return f"stage10_{arm}_{stratum}_384"


def exact_sign_flip_pvalue(differences: np.ndarray) -> float:
    observed = abs(float(differences.mean()))
    values = [
        abs(float(np.mean(differences * np.asarray(signs))))
        for signs in itertools.product((-1.0, 1.0), repeat=len(differences))
    ]
    return float(np.mean(np.asarray(values) >= observed - 1e-12))


def load_runs(
    project_root: Path,
) -> tuple[pd.DataFrame, dict[str, dict[str, dict[int, pd.DataFrame]]], pd.DataFrame]:
    outputs = project_root / "outputs"
    rows: list[dict[str, object]] = []
    checks: list[dict[str, object]] = []
    predictions: dict[str, dict[str, dict[int, pd.DataFrame]]] = {
        stratum: {arm: {} for arm in ARMS} for stratum in STRATA
    }
    probability_columns = [f"prob_{name}" for name in CLASSES]

    for stratum in STRATA:
        for arm in ARMS:
            experiment = experiment_name(arm, stratum)
            candidates = sorted((outputs / experiment).glob("*_*"))
            by_seed = {int(path.name.rsplit("_", 1)[-1]): path for path in candidates}
            if set(by_seed) != set(SEEDS):
                raise ValueError(f"{experiment}: expected seeds {SEEDS}, found {sorted(by_seed)}")
            for seed in SEEDS:
                run_dir = by_seed[seed]
                missing = [name for name in REQUIRED_ARTIFACTS if not (run_dir / name).is_file()]
                if missing:
                    raise ValueError(f"{run_dir}: missing {missing}")
                summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
                if summary.get("test_evaluated") is not False or summary.get("test"):
                    raise ValueError(f"{run_dir}: locked test was evaluated")
                metrics = json.loads(
                    (run_dir / "val_metrics_best.json").read_text(encoding="utf-8")
                )
                frame = pd.read_csv(run_dir / "val_predictions_best.csv")
                if len(frame) != 1280 or int(frame["is_synthetic"].astype(int).sum()) != 0:
                    raise ValueError(f"{run_dir}: invalid validation population")
                frame["image_id"] = frame["path"].map(lambda value: Path(value).stem)
                predictions[stratum][arm][seed] = frame

                target = pd.Categorical(frame["target"], categories=CLASSES).codes
                prediction = pd.Categorical(frame["prediction"], categories=CLASSES).codes
                probabilities = frame[probability_columns].to_numpy(float)
                recalculated = metric_bundle(target, prediction, probabilities)
                expected = {key: metrics[key] for key in METRICS if key in metrics}
                for key in ("precision", "recall", "f1", "auroc", "auprc"):
                    expected[f"mel_{key}"] = metrics["per_class"]["mel"][key]
                max_error = max(
                    abs(float(recalculated[key]) - float(value))
                    for key, value in expected.items()
                )
                if max_error > 5e-7:
                    raise ValueError(f"{run_dir}: metric recomputation error {max_error}")

                history = pd.read_csv(run_dir / "metrics.csv")
                best_row = history.loc[history["val/macro_f1"].idxmax()]
                if not np.isclose(
                    float(best_row["val/macro_f1"]),
                    float(summary["best_metric"]),
                    rtol=0,
                    atol=1e-10,
                ):
                    raise ValueError(f"{run_dir}: best checkpoint/history mismatch")
                rows.append(
                    {
                        "stratum": stratum,
                        "arm": arm,
                        "experiment": experiment,
                        "seed": seed,
                        "run": run_dir.name,
                        "best_epoch": int(summary["best_epoch"]),
                        "elapsed_seconds": float(summary["elapsed_seconds"]),
                        **recalculated,
                    }
                )
                checks.append(
                    {
                        "experiment": experiment,
                        "seed": seed,
                        "run": run_dir.name,
                        "missing_artifacts": ";".join(missing),
                        "test_evaluated": summary["test_evaluated"],
                        "validation_rows": len(frame),
                        "metric_recompute_max_abs_error": max_error,
                    }
                )
    return pd.DataFrame(rows), predictions, pd.DataFrame(checks)


def validate_alignment(
    split: pd.DataFrame,
    predictions: dict[str, dict[str, dict[int, pd.DataFrame]]],
) -> pd.DataFrame:
    if split["group_id"].isna().any() or split["image_id"].duplicated().any():
        raise ValueError("Validation split has invalid lesion groups or image IDs")
    expected = set(split["image_id"])
    reference: dict[str, str] | None = None
    rows = []
    for stratum in STRATA:
        for arm in ARMS:
            for seed in SEEDS:
                frame = predictions[stratum][arm][seed]
                target_map = dict(zip(frame["image_id"], frame["target"], strict=True))
                rows.append(
                    {
                        "stratum": stratum,
                        "arm": arm,
                        "seed": seed,
                        "image_set_matches": set(frame["image_id"]) == expected,
                        "targets_match": reference is None or target_map == reference,
                    }
                )
                if reference is None:
                    reference = target_map
    checks = pd.DataFrame(rows)
    if not checks[["image_set_matches", "targets_match"]].all().all():
        raise ValueError("Stage 10 prediction populations are not aligned")
    return checks


def summarize_runs(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (stratum, arm), subset in frame.groupby(["stratum", "arm"], sort=False):
        row: dict[str, object] = {"stratum": stratum, "arm": arm, "seeds": len(subset)}
        for metric in METRICS:
            row[f"{metric}_mean"] = subset[metric].mean()
            row[f"{metric}_sd"] = subset[metric].std(ddof=1)
        row["best_epoch_mean"] = subset["best_epoch"].mean()
        row["elapsed_seconds_mean"] = subset["elapsed_seconds"].mean()
        rows.append(row)
    return pd.DataFrame(rows)


def paired_seed_comparisons(frame: pd.DataFrame) -> pd.DataFrame:
    indexed = frame.set_index(["stratum", "arm", "seed"])
    rows = []
    for stratum in STRATA:
        for metric in METRICS:
            differences = np.asarray(
                [
                    indexed.loc[(stratum, "synthetic", seed), metric]
                    - indexed.loc[(stratum, "replay", seed), metric]
                    for seed in SEEDS
                ],
                dtype=float,
            )
            sem = stats.sem(differences)
            interval = stats.t.interval(
                0.95, len(differences) - 1, loc=differences.mean(), scale=sem
            )
            rows.append(
                {
                    "stratum": stratum,
                    "metric": metric,
                    "mean_difference_synthetic_minus_replay": differences.mean(),
                    "sd_difference": differences.std(ddof=1),
                    "ci95_low_seed_t": interval[0],
                    "ci95_high_seed_t": interval[1],
                    "synthetic_wins": int((differences > 0).sum()),
                    "exact_sign_flip_p_two_sided": exact_sign_flip_pvalue(differences),
                    "differences_by_seed": json.dumps(
                        dict(zip(map(str, SEEDS), differences.tolist(), strict=True))
                    ),
                }
            )
    return pd.DataFrame(rows)


def per_class_discrimination(
    predictions: dict[str, dict[str, dict[int, pd.DataFrame]]],
) -> pd.DataFrame:
    rows = []
    for stratum in STRATA:
        for arm in ARMS:
            for seed in SEEDS:
                frame = predictions[stratum][arm][seed]
                target = pd.Categorical(frame["target"], categories=CLASSES).codes
                for index, name in enumerate(CLASSES):
                    binary = (target == index).astype(int)
                    scores = frame[f"prob_{name}"].to_numpy(float)
                    rows.append(
                        {
                            "stratum": stratum,
                            "arm": arm,
                            "seed": seed,
                            "class": name,
                            "support": int(binary.sum()),
                            "auroc": roc_auc_score(binary, scores),
                            "auprc": average_precision_score(binary, scores),
                        }
                    )
    return pd.DataFrame(rows)


def hierarchical_bootstrap(
    split: pd.DataFrame,
    predictions: dict[str, dict[str, dict[int, pd.DataFrame]]],
    stratum: str,
    replicates: int,
    random_seed: int,
) -> pd.DataFrame:
    groups = split["group_id"].drop_duplicates().to_numpy()
    group_indices = {
        group: np.flatnonzero(split["group_id"].to_numpy() == group) for group in groups
    }
    prepared: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]] = {
        arm: {} for arm in ARMS
    }
    for arm in ARMS:
        for seed in SEEDS:
            aligned = predictions[stratum][arm][seed].set_index("image_id").loc[
                split["image_id"]
            ]
            prepared[arm][seed] = (
                pd.Categorical(aligned["target"], categories=CLASSES).codes,
                pd.Categorical(aligned["prediction"], categories=CLASSES).codes,
            )

    rng = np.random.default_rng(random_seed)
    sampled_values = {metric: [] for metric in BOOTSTRAP_METRICS}
    for _ in range(replicates):
        sampled_seeds = rng.choice(SEEDS, size=len(SEEDS), replace=True)
        sampled_groups = rng.choice(groups, size=len(groups), replace=True)
        indices = np.concatenate([group_indices[group] for group in sampled_groups])
        differences = {metric: [] for metric in BOOTSTRAP_METRICS}
        for seed in sampled_seeds:
            bundles = {}
            for arm in ARMS:
                target, prediction = prepared[arm][int(seed)]
                bundles[arm] = metric_bundle(target[indices], prediction[indices])
            for metric in BOOTSTRAP_METRICS:
                differences[metric].append(
                    bundles["synthetic"][metric] - bundles["replay"][metric]
                )
        for metric in BOOTSTRAP_METRICS:
            sampled_values[metric].append(float(np.mean(differences[metric])))

    rows = []
    for metric, values in sampled_values.items():
        array = np.asarray(values)
        rows.append(
            {
                "stratum": stratum,
                "metric": metric,
                "mean_difference": array.mean(),
                "ci95_low": np.quantile(array, 0.025),
                "ci95_high": np.quantile(array, 0.975),
                "probability_difference_gt_zero": (array > 0).mean(),
                "replicates": replicates,
            }
        )
    return pd.DataFrame(rows)


def load_calibration(project_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    fixed_rows = []
    for stratum in STRATA:
        for arm in ARMS:
            experiment = experiment_name(arm, stratum)
            by_seed = {
                int(path.name.rsplit("_", 1)[-1]): path
                for path in (project_root / "outputs" / experiment).glob("*_*")
            }
            for seed in SEEDS:
                path = by_seed[seed] / "calibration_diagnostic" / "calibration_diagnostic.json"
                report = json.loads(path.read_text(encoding="utf-8"))
                raw = report["evaluation_uncalibrated"]
                scaled = report["evaluation_temperature_scaled"]
                rows.append(
                    {
                        "stratum": stratum,
                        "arm": arm,
                        "seed": seed,
                        "temperature": report["temperature"],
                        "ece_before": raw["ece"],
                        "ece_after": scaled["ece"],
                        "nll_before": raw["nll"],
                        "nll_after": scaled["nll"],
                        "mel_auprc": scaled["mel"]["auprc"],
                        "mel_auroc": scaled["mel"]["auroc"],
                    }
                )
                for fixed in report["fixed_specificity"]:
                    fixed_rows.append(
                        {"stratum": stratum, "arm": arm, "seed": seed, **fixed}
                    )
    return pd.DataFrame(rows), pd.DataFrame(fixed_rows)


def plot_deltas(paired: pd.DataFrame, out_path: Path) -> None:
    metrics = ("macro_f1", "mcc", "mel_f1", "mel_recall")
    figure, axes = plt.subplots(1, 4, figsize=(14, 4.5), constrained_layout=True)
    for axis, metric in zip(axes, metrics, strict=True):
        subset = paired[paired["metric"] == metric].set_index("stratum").loc[list(STRATA)]
        values = subset["mean_difference_synthetic_minus_replay"].to_numpy()
        colors = ["#2f855a" if value >= 0 else "#c2413b" for value in values]
        axis.bar(range(len(STRATA)), values, color=colors)
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_title(metric.replace("_", " "))
        axis.set_xticks(range(len(STRATA)), STRATA, rotation=35, ha="right")
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("Stage 10 paired validation deltas: synthetic minus source replay")
    figure.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    data_root = Path(args.data_root).resolve()
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = project_root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    frame, predictions, checks = load_runs(project_root)
    split = pd.read_csv(data_root / "splits" / "stage8" / "val_real.csv")
    alignment = validate_alignment(split, predictions)
    method_summary = summarize_runs(frame)
    paired = paired_seed_comparisons(frame)
    per_class = per_class_discrimination(predictions)
    bootstrap = pd.concat(
        [
            hierarchical_bootstrap(
                split,
                predictions,
                stratum,
                args.bootstrap_replicates,
                args.seed + index,
            )
            for index, stratum in enumerate(STRATA)
        ],
        ignore_index=True,
    )
    calibration, fixed = load_calibration(project_root)

    frame.to_csv(out_dir / "seed_metrics.csv", index=False)
    method_summary.to_csv(out_dir / "method_summary.csv", index=False)
    paired.to_csv(out_dir / "paired_seed_comparisons.csv", index=False)
    bootstrap.to_csv(out_dir / "hierarchical_lesion_bootstrap.csv", index=False)
    per_class.to_csv(out_dir / "per_class_discrimination.csv", index=False)
    calibration.to_csv(out_dir / "calibration_summary.csv", index=False)
    fixed.to_csv(out_dir / "fixed_specificity_diagnostics.csv", index=False)
    checks.to_csv(out_dir / "artifact_integrity.csv", index=False)
    alignment.to_csv(out_dir / "prediction_alignment.csv", index=False)
    plot_deltas(paired, out_dir / "paired_validation_deltas.png")

    summary = {
        "status": "complete",
        "runs": len(frame),
        "strata": list(STRATA),
        "seeds": list(SEEDS),
        "locked_test_evaluated": False,
        "validation_rows": len(split),
        "validation_groups": int(split["group_id"].nunique()),
        "bootstrap_replicates": args.bootstrap_replicates,
        "metric_recompute_max_abs_error": float(
            checks["metric_recompute_max_abs_error"].max()
        ),
        "prediction_alignment_passed": True,
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
