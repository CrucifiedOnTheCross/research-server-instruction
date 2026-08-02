#!/usr/bin/env python3
"""Analyze Stage 3B with split/model/lesion hierarchical resampling."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score, roc_curve

from longtail_medical.statistical_analysis import aggregate_by_lesion, fast_mcc_balanced_accuracy


CLASS_NAMES = ["nv", "mel", "bcc", "bkl", "ak", "scc", "vasc", "df"]
SPLIT_SEEDS = (101, 202, 303)
MODEL_SEEDS = (42, 43, 44)


def read_predictions(path: Path) -> dict:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    labels = np.asarray([int(row["label"]) for row in rows])
    probabilities = np.asarray([[float(row[f"prob_{name}"]) for name in CLASS_NAMES] for row in rows])
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
    image_ids = [row["image_id"] for row in rows]
    lesion_ids = [row["lesion_id"] for row in rows]
    lesion_labels, lesion_probabilities, ordered_ids = aggregate_by_lesion(labels, probabilities, lesion_ids, image_ids)
    return {"labels": lesion_labels, "probabilities": lesion_probabilities, "ids": ordered_ids}


def class_diagnostics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, dict[str, float]]:
    predictions = probabilities.argmax(axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, labels=range(len(CLASS_NAMES)), zero_division=0
    )
    one_hot = np.eye(len(CLASS_NAMES), dtype=np.int64)[labels]
    auprc = average_precision_score(one_hot, probabilities, average=None)
    auroc = roc_auc_score(one_hot, probabilities, average=None, multi_class="ovr")
    output = {}
    for index, name in enumerate(CLASS_NAMES):
        binary = (labels == index).astype(np.int64)
        fpr, tpr, _ = roc_curve(binary, probabilities[:, index])
        fixed = {}
        for specificity in (0.90, 0.95):
            candidates = tpr[fpr <= 1.0 - specificity + 1e-12]
            fixed[f"sensitivity_at_specificity_{int(specificity * 100)}"] = float(candidates.max()) if candidates.size else 0.0
        output[name] = {
            "precision": float(precision[index]), "recall": float(recall[index]),
            "f1": float(f1[index]), "auprc": float(auprc[index]),
            "auroc": float(auroc[index]), "support": int(support[index]), **fixed,
        }
    return output


def path_for(root: Path, variant: str, split_seed: int, model_seed: int, checkpoint: str) -> Path:
    return root / variant / f"split_{split_seed}" / f"model_seed_{model_seed}" / checkpoint / "predictions.csv"


def grouped_indices(labels: np.ndarray) -> dict[int, np.ndarray]:
    return {label: np.flatnonzero(labels == label) for label in range(len(CLASS_NAMES))}


def hierarchical_bootstrap(job: tuple[str, dict, int, int]) -> tuple[str, dict]:
    metric, pairs, repeats, random_seed = job
    rng = np.random.default_rng(random_seed)
    effects = np.empty(repeats, dtype=np.float64)
    split_values = np.asarray(SPLIT_SEEDS)
    model_values = np.asarray(MODEL_SEEDS)
    for repeat in range(repeats):
        deltas = []
        for split_seed in rng.choice(split_values, size=len(split_values), replace=True):
            for model_seed in rng.choice(model_values, size=len(model_values), replace=True):
                pair = pairs[(int(split_seed), int(model_seed))]
                sampled = []
                for indices in pair["groups"].values():
                    sampled.extend(rng.choice(indices, size=len(indices), replace=True).tolist())
                sampled = np.asarray(sampled)
                labels = pair["labels"][sampled]
                ldam = fast_mcc_balanced_accuracy(labels, pair["ldam"][sampled].argmax(axis=1), len(CLASS_NAMES))[metric]
                ce = fast_mcc_balanced_accuracy(labels, pair["ce"][sampled].argmax(axis=1), len(CLASS_NAMES))[metric]
                deltas.append(ldam - ce)
        effects[repeat] = float(np.mean(deltas))
    return metric, {
        "mean": float(effects.mean()), "ci95_low": float(np.quantile(effects, 0.025)),
        "ci95_high": float(np.quantile(effects, 0.975)),
        "probability_positive": float(np.mean(effects > 0)), "repeats": repeats,
    }


def mean_std(values) -> dict:
    values = np.asarray(values, dtype=np.float64)
    return {"mean": float(values.mean()), "std": float(values.std(ddof=1)), "values": values.tolist()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("outputs/stage3b_locked_test"))
    parser.add_argument("--output", type=Path, default=Path("outputs/stage3b_analysis/stage3b_analysis.json"))
    parser.add_argument("--bootstrap-repeats", type=int, default=10000)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    root = args.root if args.root.is_absolute() else project / args.root
    locked = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    if not locked.get("one_shot") or not locked.get("test_evaluated"):
        raise RuntimeError("Locked evaluation is incomplete")
    records = locked["records"]
    record_lookup = {
        (record["variant"] if record["variant"] != "raw" else record["arm"], record["checkpoint"], int(record["split_seed"]), int(record["model_seed"])): record
        for record in records
    }
    summary = defaultdict(lambda: defaultdict(list))
    for record in records:
        key = record["variant"] if record["variant"] != "raw" else record["arm"]
        checkpoint = record["checkpoint"]
        for level in ("image", "lesion"):
            for metric, value in record[level].items():
                if isinstance(value, (int, float)):
                    summary[(key, checkpoint, level)][metric].append(float(value))
    compact = {
        "/".join(key): {metric: mean_std(values) for metric, values in metrics.items()}
        for key, metrics in summary.items()
    }
    paired_comparisons = {}
    for name, treatment, control in (
        ("ldam_drw_minus_ce", "ldam_drw", "ce"),
        ("logit_adjustment_minus_ce", "logit_adjustment_tau1", "ce"),
        ("temperature_scaled_minus_ldam_raw", "temperature_scaled", "ldam_drw"),
    ):
        comparison = {"overall": {}, "by_split": {}}
        for metric in ("mcc", "balanced_accuracy", "macro_f1", "macro_auprc", "macro_auroc", "ece", "nll", "brier", "worst_class_recall"):
            values = []
            by_split = {}
            for split_seed in SPLIT_SEEDS:
                split_values = [
                    float(record_lookup[(treatment, "last", split_seed, model_seed)]["lesion"][metric])
                    - float(record_lookup[(control, "last", split_seed, model_seed)]["lesion"][metric])
                    for model_seed in MODEL_SEEDS
                ]
                by_split[str(split_seed)] = mean_std(split_values)
                values.extend(split_values)
            comparison["overall"][metric] = {
                **mean_std(values), "positive_pairs": int(np.sum(np.asarray(values) > 0)),
            }
            comparison["by_split"][metric] = by_split
        paired_comparisons[name] = comparison

    per_class = {}
    prediction_cache = {}
    diagnostics_cache = {}
    for variant in ("ce", "ldam_drw", "logit_adjustment_tau1", "temperature_scaled"):
        per_class[variant] = {}
        rows_by_class = defaultdict(list)
        for split_seed in SPLIT_SEEDS:
            for model_seed in MODEL_SEEDS:
                item = read_predictions(path_for(root, variant, split_seed, model_seed, "last"))
                prediction_cache[(variant, split_seed, model_seed)] = item
                diagnostics = class_diagnostics(item["labels"], item["probabilities"])
                diagnostics_cache[(variant, split_seed, model_seed)] = diagnostics
                for class_name, values in diagnostics.items():
                    rows_by_class[class_name].append(values)
        for class_name, rows in rows_by_class.items():
            per_class[variant][class_name] = {
                metric: mean_std([row[metric] for row in rows])
                for metric in rows[0] if metric != "support"
            }
            per_class[variant][class_name]["support_per_run"] = rows[0]["support"]

    per_class_ldam_delta = {}
    for class_name in CLASS_NAMES:
        per_class_ldam_delta[class_name] = {}
        for metric in ("precision", "recall", "f1", "auprc", "auroc", "sensitivity_at_specificity_90", "sensitivity_at_specificity_95"):
            values = []
            for split_seed in SPLIT_SEEDS:
                for model_seed in MODEL_SEEDS:
                    ldam = diagnostics_cache[("ldam_drw", split_seed, model_seed)][class_name][metric]
                    ce = diagnostics_cache[("ce", split_seed, model_seed)][class_name][metric]
                    values.append(ldam - ce)
            per_class_ldam_delta[class_name][metric] = {
                **mean_std(values), "positive_pairs": int(np.sum(np.asarray(values) > 0)),
            }
    pairs = {}
    split_deltas = defaultdict(list)
    model_deltas = defaultdict(list)
    for split_seed in SPLIT_SEEDS:
        for model_seed in MODEL_SEEDS:
            ce = read_predictions(path_for(root, "ce", split_seed, model_seed, "last"))
            ldam = read_predictions(path_for(root, "ldam_drw", split_seed, model_seed, "last"))
            if ce["ids"] != ldam["ids"] or not np.array_equal(ce["labels"], ldam["labels"]):
                raise RuntimeError("Paired lesion order mismatch")
            pairs[(split_seed, model_seed)] = {
                "labels": ce["labels"], "ce": ce["probabilities"], "ldam": ldam["probabilities"],
                "groups": grouped_indices(ce["labels"]),
            }
            ce_metrics = fast_mcc_balanced_accuracy(ce["labels"], ce["probabilities"].argmax(axis=1), len(CLASS_NAMES))
            ldam_metrics = fast_mcc_balanced_accuracy(ldam["labels"], ldam["probabilities"].argmax(axis=1), len(CLASS_NAMES))
            delta = ldam_metrics["mcc"] - ce_metrics["mcc"]
            split_deltas[str(split_seed)].append(delta)
            model_deltas[str(model_seed)].append(delta)
    jobs = [(metric, pairs, args.bootstrap_repeats, 20260802 + index) for index, metric in enumerate(("mcc", "balanced_accuracy"))]
    bootstrap = {}
    with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs))) as executor:
        for metric, result in executor.map(hierarchical_bootstrap, jobs):
            bootstrap[metric] = result
    result = {
        "protocol": "stage3b_split_model_lesion_hierarchical",
        "git_commit": locked["git_commit"], "primary_checkpoint": "last",
        "primary_comparison": "ldam_drw_minus_ce", "records": compact,
        "paired_comparisons_last_lesion": paired_comparisons,
        "per_class_last_lesion": per_class,
        "per_class_ldam_minus_ce_last_lesion": per_class_ldam_delta,
        "paired_delta_mcc_by_split": {key: mean_std(value) for key, value in split_deltas.items()},
        "paired_delta_mcc_by_model_seed": {key: mean_std(value) for key, value in model_deltas.items()},
        "hierarchical_bootstrap": bootstrap, "test_evaluated_once": True,
    }
    output = args.output if args.output.is_absolute() else project / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"status": "completed", "output": str(output), "bootstrap": bootstrap}, indent=2))


if __name__ == "__main__":
    main()
