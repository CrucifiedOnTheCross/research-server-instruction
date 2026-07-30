from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.image_transforms import CropDarkFieldOfView
from src.metrics import expected_calibration_error
from tools.analyze_stage16b_results import (
    class_names,
    latest_run_by_seed,
    metric_bundle,
)


SEED = 42
ARMS = {
    "natural_square_crop": "stage16_isic2019_real_ce_natural_384",
    "aspect_pad": "stage16p_isic2019_real_aspect_pad_natural_384",
    "dark_fov_pad": "stage16p_isic2019_real_dark_fov_pad_natural_384",
}
METRICS = (
    "macro_auprc",
    "macro_auroc",
    "macro_f1",
    "mcc",
    "balanced_accuracy",
    "ece",
    "worst_class_recall",
    "mel_auprc",
    "scc_auprc",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Stage 16P preprocessing screening.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--data-root", default="/srv/research/projects/default/isic2019")
    parser.add_argument(
        "--split",
        default="/srv/research/projects/default/isic2019/splits/val.csv",
    )
    parser.add_argument("--out-dir", default="outputs/reports/stage16p_analysis")
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260730)
    return parser.parse_args()


def load_predictions(
    project_root: Path,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, list[str]]:
    predictions = {}
    checks = []
    expected_classes = None
    for arm, experiment in ARMS.items():
        run_dir = latest_run_by_seed(project_root / "outputs" / experiment)[SEED]
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        if summary.get("test_evaluated") is not False or summary.get("test"):
            raise ValueError(f"{run_dir}: locked test was evaluated")
        frame = pd.read_csv(run_dir / "val_predictions_best.csv")
        frame["image_id"] = frame["path"].map(lambda value: Path(value).stem)
        classes = class_names(frame)
        if expected_classes is None:
            expected_classes = classes
        elif expected_classes != classes:
            raise ValueError(f"{run_dir}: class order changed")
        predictions[arm] = frame
        checks.append(
            {
                "arm": arm,
                "run": run_dir.name,
                "rows": len(frame),
                "test_evaluated": False,
                "best_epoch": int(summary["best_epoch"]),
                "elapsed_seconds": float(summary["elapsed_seconds"]),
            }
        )
    assert expected_classes is not None
    return predictions, pd.DataFrame(checks), expected_classes


def align(
    split: pd.DataFrame,
    predictions: dict[str, pd.DataFrame],
) -> None:
    image_ids = split["image_id"].astype(str).tolist()
    expected = set(image_ids)
    reference = None
    for arm, frame in predictions.items():
        if set(frame["image_id"]) != expected:
            raise ValueError(f"{arm}: validation IDs differ from the immutable split")
        aligned = frame.set_index("image_id").loc[image_ids].reset_index()
        targets = aligned["target"].tolist()
        if reference is not None and targets != reference:
            raise ValueError(f"{arm}: validation targets differ")
        reference = targets
        predictions[arm] = aligned


def screening_table(
    predictions: dict[str, pd.DataFrame],
    classes: list[str],
    checks: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    baseline = metric_bundle(predictions["natural_square_crop"], classes)
    for arm, frame in predictions.items():
        metrics = metric_bundle(frame, classes)
        elapsed = float(checks.loc[checks["arm"] == arm, "elapsed_seconds"].iloc[0])
        rows.append(
            {
                "arm": arm,
                **metrics,
                **{
                    f"delta_{metric}": metrics[metric] - baseline[metric]
                    for metric in METRICS
                },
                "elapsed_seconds": elapsed,
            }
        )
    return pd.DataFrame(rows)


def lesion_bootstrap(
    split: pd.DataFrame,
    predictions: dict[str, pd.DataFrame],
    classes: list[str],
    replicates: int,
    random_seed: int,
) -> pd.DataFrame:
    groups = split["group_id"].astype(str).drop_duplicates().to_numpy()
    values = split["group_id"].astype(str).to_numpy()
    indices = {group: np.flatnonzero(values == group) for group in groups}
    rng = np.random.default_rng(random_seed)
    samples = {
        (arm, metric): []
        for arm in ("aspect_pad", "dark_fov_pad")
        for metric in METRICS
    }
    for _ in range(replicates):
        selected = rng.choice(groups, size=len(groups), replace=True)
        sample_indices = np.concatenate([indices[group] for group in selected])
        baseline = metric_bundle(
            predictions["natural_square_crop"], classes, sample_indices
        )
        for arm in ("aspect_pad", "dark_fov_pad"):
            candidate = metric_bundle(predictions[arm], classes, sample_indices)
            for metric in METRICS:
                samples[(arm, metric)].append(candidate[metric] - baseline[metric])
    rows = []
    for (arm, metric), values in samples.items():
        array = np.asarray(values)
        rows.append(
            {
                "arm": arm,
                "metric": metric,
                "mean_difference": float(array.mean()),
                "ci95_low": float(np.quantile(array, 0.025)),
                "ci95_high": float(np.quantile(array, 0.975)),
                "probability_difference_gt_zero": float((array > 0).mean()),
                "replicates": replicates,
            }
        )
    return pd.DataFrame(rows)


def dark_fov_candidates(split: pd.DataFrame, data_root: Path) -> pd.DataFrame:
    transform = CropDarkFieldOfView()
    rows = []
    for row in split.itertuples(index=False):
        path = data_root / str(row.image_path)
        with Image.open(path) as image:
            original_size = image.size
            transformed_size = transform(image).size
        rows.append(
            {
                "image_id": str(row.image_id),
                "source": str(row.source),
                "label": str(row.label),
                "dark_fov_crop_applied": transformed_size != original_size,
                "original_width": original_size[0],
                "original_height": original_size[1],
                "cropped_width": transformed_size[0],
                "cropped_height": transformed_size[1],
            }
        )
    return pd.DataFrame(rows)


def subgroup_metrics(
    split: pd.DataFrame,
    predictions: dict[str, pd.DataFrame],
    classes: list[str],
    dark_candidates: pd.DataFrame,
) -> pd.DataFrame:
    masks = {
        f"source:{source}": split["source"].astype(str).to_numpy() == source
        for source in sorted(split["source"].astype(str).unique())
    }
    dark_mask = dark_candidates["dark_fov_crop_applied"].to_numpy(bool)
    masks["dark_fov:applied"] = dark_mask
    masks["dark_fov:not_applied"] = ~dark_mask

    rows = []
    for subgroup, mask in masks.items():
        indices = np.flatnonzero(mask)
        for arm, frame in predictions.items():
            subset = frame.iloc[indices]
            probabilities = subset[[f"prob_{name}" for name in classes]].to_numpy(float)
            target = pd.Categorical(subset["target"], categories=classes).codes
            prediction = pd.Categorical(subset["prediction"], categories=classes).codes
            present = np.unique(target)
            _, recall, _, _ = precision_recall_fscore_support(
                target, prediction, labels=present, zero_division=0
            )
            one_hot = np.eye(len(classes))[target]
            metrics = {
                "macro_auprc": float(
                    np.mean(
                        [
                            average_precision_score(one_hot[:, index], probabilities[:, index])
                            for index in present
                        ]
                    )
                ),
                "macro_auroc": float(
                    np.mean(
                        [
                            roc_auc_score(one_hot[:, index], probabilities[:, index])
                            for index in present
                        ]
                    )
                ),
                "macro_f1": float(
                    f1_score(target, prediction, labels=present, average="macro", zero_division=0)
                ),
                "mcc": float(matthews_corrcoef(target, prediction)),
                "balanced_accuracy": float(balanced_accuracy_score(target, prediction)),
                "ece": expected_calibration_error(probabilities, target, n_bins=15),
                "worst_class_recall": float(recall.min()),
            }
            for name in ("mel", "scc"):
                class_index = classes.index(name)
                binary = (target == class_index).astype(int)
                metrics[f"{name}_auprc"] = (
                    float(average_precision_score(binary, probabilities[:, class_index]))
                    if binary.any()
                    else float("nan")
                )
            rows.append(
                {
                    "subgroup": subgroup,
                    "arm": arm,
                    "rows": len(indices),
                    **{metric: metrics[metric] for metric in METRICS},
                }
            )
    return pd.DataFrame(rows)


def gate_decision(screening: pd.DataFrame) -> dict[str, object]:
    decisions = {}
    for arm in ("aspect_pad", "dark_fov_pad"):
        row = screening.set_index("arm").loc[arm]
        checks = {
            "macro_auprc_delta_at_least_0.005": row["delta_macro_auprc"] >= 0.005,
            "mcc_degradation_at_most_0.005": row["delta_mcc"] >= -0.005,
            "ece_degradation_at_most_0.02": row["delta_ece"] <= 0.02,
            "mel_auprc_degradation_at_most_0.01": row["delta_mel_auprc"] >= -0.01,
            "scc_auprc_degradation_at_most_0.01": row["delta_scc_auprc"] >= -0.01,
        }
        decisions[arm] = {
            "checks": {key: bool(value) for key, value in checks.items()},
            "passed": bool(all(checks.values())),
        }
    return {
        "arms": decisions,
        "selected_for_confirmation": None,
        "decision": "retain_natural_square_crop",
    }


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = project_root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    split = pd.read_csv(args.split)
    predictions, checks, classes = load_predictions(project_root)
    align(split, predictions)
    screening = screening_table(predictions, classes, checks)
    bootstrap = lesion_bootstrap(
        split,
        predictions,
        classes,
        args.bootstrap_replicates,
        args.seed,
    )
    candidates = dark_fov_candidates(split, data_root)
    subgroups = subgroup_metrics(split, predictions, classes, candidates)
    decision = gate_decision(screening)

    screening.to_csv(out_dir / "screening_metrics.csv", index=False)
    bootstrap.to_csv(out_dir / "paired_lesion_bootstrap.csv", index=False)
    candidates.to_csv(out_dir / "dark_fov_candidates.csv", index=False)
    subgroups.to_csv(out_dir / "subgroup_metrics.csv", index=False)
    checks.to_csv(out_dir / "artifact_integrity.csv", index=False)
    summary = {
        "status": "complete",
        "seed": SEED,
        "locked_test_evaluated": False,
        "validation_rows": len(split),
        "validation_groups": int(split["group_id"].nunique()),
        "bootstrap_replicates": args.bootstrap_replicates,
        "dark_fov_candidate_rows": int(candidates["dark_fov_crop_applied"].sum()),
        **decision,
    }
    (out_dir / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
