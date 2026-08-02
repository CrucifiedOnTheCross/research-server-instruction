#!/usr/bin/env python3
"""Structured validation-only analysis for the Stage 3A loss screening."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)

from longtail_medical.statistical_analysis import (
    aggregate_by_lesion,
    fast_mcc_balanced_accuracy,
)


CLASS_NAMES = ["nv", "mel", "bcc", "bkl", "ak", "scc", "vasc", "df"]
TRAINED_METHODS = {
    "stage3a_ce_resnet50_lesion_disjoint": "CE",
    "stage3a_weighted_ce_resnet50_lesion_disjoint": "Weighted CE",
    "stage3a_focal_resnet50_lesion_disjoint": "Focal",
    "stage3a_cb_focal_resnet50_lesion_disjoint": "CB-Focal",
    "stage3a_balanced_softmax_resnet50_lesion_disjoint": "Balanced Softmax",
    "stage3a_ldam_drw_resnet50_lesion_disjoint": "LDAM-DRW",
}
DERIVED_METHOD = "Logit Adjustment"
SEEDS = (42, 43, 44)
METRIC_NAMES = (
    "mcc",
    "balanced_accuracy",
    "macro_f1",
    "macro_auprc",
    "macro_auroc",
    "worst_class_recall",
    "ece",
    "nll",
    "brier",
)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_predictions(path: Path) -> dict:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    probabilities = np.asarray(
        [[float(row[f"prob_{name}"]) for name in CLASS_NAMES] for row in rows],
        dtype=np.float64,
    )
    lesion_labels, lesion_probabilities, lesion_ids = aggregate_by_lesion(
        labels,
        probabilities,
        [row["lesion_id"] for row in rows],
        [row["image_id"] for row in rows],
    )
    return {
        "labels": lesion_labels,
        "probabilities": lesion_probabilities,
        "lesion_ids": lesion_ids,
    }


def class_diagnostics(labels: np.ndarray, probabilities: np.ndarray) -> dict:
    predictions = probabilities.argmax(axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels,
        predictions,
        labels=np.arange(len(CLASS_NAMES)),
        zero_division=0,
    )
    result = {}
    for index, name in enumerate(CLASS_NAMES):
        binary = (labels == index).astype(np.int64)
        fpr, tpr, _ = roc_curve(binary, probabilities[:, index])
        fixed = {}
        for specificity in (0.90, 0.95):
            valid = tpr[fpr <= 1.0 - specificity + 1e-12]
            fixed[f"sensitivity_at_specificity_{specificity:.2f}"] = (
                float(valid.max()) if len(valid) else 0.0
            )
        result[name] = {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
            "auroc": float(roc_auc_score(binary, probabilities[:, index])),
            "auprc": float(average_precision_score(binary, probabilities[:, index])),
            **fixed,
        }
    return result


def mean_std(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "values": array.tolist(),
    }


def hierarchical_paired_bootstrap(
    method_runs: dict[int, dict],
    ce_runs: dict[int, dict],
    repeats: int,
    random_seed: int,
) -> dict:
    rng = np.random.default_rng(random_seed)
    effects = {"mcc": np.empty(repeats), "balanced_accuracy": np.empty(repeats)}
    seeds = np.asarray(SEEDS)
    class_indices = {}
    for seed in SEEDS:
        method = method_runs[seed]
        control = ce_runs[seed]
        if method["lesion_ids"] != control["lesion_ids"]:
            raise ValueError(f"Lesion order mismatch for seed {seed}")
        if not np.array_equal(method["labels"], control["labels"]):
            raise ValueError(f"Label mismatch for seed {seed}")
        class_indices[seed] = {
            label: np.flatnonzero(method["labels"] == label)
            for label in range(len(CLASS_NAMES))
        }
    for repeat in range(repeats):
        seed_draws = rng.choice(seeds, size=len(seeds), replace=True)
        repeat_effects = {"mcc": [], "balanced_accuracy": []}
        for seed_value in seed_draws:
            seed = int(seed_value)
            sampled = np.concatenate([
                rng.choice(indices, size=len(indices), replace=True)
                for indices in class_indices[seed].values()
            ])
            labels = method_runs[seed]["labels"][sampled]
            method_predictions = method_runs[seed]["probabilities"][sampled].argmax(axis=1)
            ce_predictions = ce_runs[seed]["probabilities"][sampled].argmax(axis=1)
            method_metrics = fast_mcc_balanced_accuracy(labels, method_predictions, len(CLASS_NAMES))
            ce_metrics = fast_mcc_balanced_accuracy(labels, ce_predictions, len(CLASS_NAMES))
            for metric in repeat_effects:
                repeat_effects[metric].append(method_metrics[metric] - ce_metrics[metric])
        for metric in effects:
            effects[metric][repeat] = np.mean(repeat_effects[metric])
    return {
        metric: {
            "mean": float(values.mean()),
            "ci95_low": float(np.quantile(values, 0.025)),
            "ci95_high": float(np.quantile(values, 0.975)),
            "probability_positive": float(np.mean(values > 0)),
            "repeats": repeats,
        }
        for metric, values in effects.items()
    }


def bootstrap_job(args: tuple) -> tuple[str, dict]:
    method, method_runs, ce_runs, repeats, random_seed = args
    return method, hierarchical_paired_bootstrap(
        method_runs,
        ce_runs,
        repeats=repeats,
        random_seed=random_seed,
    )


def collect_runs(root: Path) -> tuple[dict, dict]:
    records: dict[str, dict[int, dict]] = defaultdict(dict)
    predictions: dict[str, dict[str, dict[int, dict]]] = {
        "last": defaultdict(dict),
        "best": defaultdict(dict),
    }
    for directory_name, display_name in TRAINED_METHODS.items():
        for run_dir in sorted((root / "stage3a_screening" / directory_name).iterdir()):
            if not run_dir.is_dir() or not (run_dir / "summary.json").exists():
                continue
            summary = read_json(run_dir / "summary.json")
            seed = int(summary["run_id"].rsplit("_", 1)[-1])
            if seed in records[display_name]:
                raise ValueError(f"Duplicate {display_name} seed {seed}")
            if summary["test_evaluated"] or summary["epochs_completed"] != 50:
                raise ValueError(f"Invalid completed run: {run_dir}")
            record = {"summary": summary, "run_dir": str(run_dir)}
            for checkpoint in ("last", "best"):
                record[checkpoint] = read_json(run_dir / f"val_metrics_{checkpoint}.json")
                predictions[checkpoint][display_name][seed] = load_predictions(
                    run_dir / f"val_predictions_{checkpoint}.csv"
                )
            with (run_dir / "metrics.csv").open(encoding="utf-8", newline="") as handle:
                epochs = list(csv.DictReader(handle))
            record["peak_vram_gib"] = max(float(row["gpu_max_memory_gib"]) for row in epochs)
            record["mean_epoch_seconds"] = float(np.mean([float(row["epoch_seconds"]) for row in epochs]))
            records[display_name][seed] = record

    derived_root = root / "stage3a_logit_adjustment_tau1"
    for seed in SEEDS:
        run_dir = derived_root / f"seed_{seed}"
        summary = read_json(run_dir / "summary.json")
        if summary["test_evaluated"]:
            raise ValueError(f"Derived run evaluated test: {run_dir}")
        record = {"summary": summary, "run_dir": str(run_dir)}
        for checkpoint in ("last", "best"):
            record[checkpoint] = read_json(run_dir / f"val_metrics_{checkpoint}.json")
            predictions[checkpoint][DERIVED_METHOD][seed] = load_predictions(
                run_dir / f"val_predictions_{checkpoint}.csv"
            )
        records[DERIVED_METHOD][seed] = record
    return records, predictions


def summarize(records: dict, predictions: dict, checkpoint: str) -> dict:
    output = {}
    for method, seed_records in records.items():
        method_summary = {"metrics": {}, "per_class": {}, "seeds": {}}
        for level in ("image", "lesion"):
            method_summary["metrics"][level] = {
                metric: mean_std([
                    float(seed_records[seed][checkpoint][level][metric]) for seed in SEEDS
                ])
                for metric in METRIC_NAMES
            }
        for class_name in CLASS_NAMES:
            diagnostics = []
            for seed in SEEDS:
                item = predictions[checkpoint][method][seed]
                diagnostics.append(class_diagnostics(item["labels"], item["probabilities"])[class_name])
            method_summary["per_class"][class_name] = {
                metric: mean_std([row[metric] for row in diagnostics])
                for metric in diagnostics[0]
                if metric != "support"
            }
            method_summary["per_class"][class_name]["support"] = diagnostics[0]["support"]
        for seed in SEEDS:
            method_summary["seeds"][str(seed)] = {
                "lesion_mcc": float(seed_records[seed][checkpoint]["lesion"]["mcc"]),
                "lesion_balanced_accuracy": float(
                    seed_records[seed][checkpoint]["lesion"]["balanced_accuracy"]
                ),
                "lesion_macro_auprc": float(
                    seed_records[seed][checkpoint]["lesion"]["macro_auprc"]
                ),
            }
        if method != DERIVED_METHOD:
            method_summary["compute"] = {
                "elapsed_seconds": mean_std([
                    float(seed_records[seed]["summary"]["elapsed_seconds"]) for seed in SEEDS
                ]),
                "peak_vram_gib": mean_std([
                    float(seed_records[seed]["peak_vram_gib"]) for seed in SEEDS
                ]),
                "best_epoch": mean_std([
                    float(seed_records[seed]["summary"]["best_epoch"]) for seed in SEEDS
                ]),
            }
        output[method] = method_summary
    ce = output["CE"]
    for method, method_summary in output.items():
        method_summary["paired_seed_delta_vs_ce"] = {}
        for metric in ("mcc", "balanced_accuracy", "macro_auprc", "ece", "nll", "brier"):
            deltas = np.asarray(method_summary["metrics"]["lesion"][metric]["values"]) - np.asarray(
                ce["metrics"]["lesion"][metric]["values"]
            )
            method_summary["paired_seed_delta_vs_ce"][metric] = {
                **mean_std(deltas.tolist()),
                "positive_seeds": int(np.sum(deltas > 0)),
            }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/stage3a_analysis/stage3a_analysis.json")
    )
    parser.add_argument("--bootstrap-repeats", type=int, default=10000)
    parser.add_argument("--workers", type=int, default=min(6, os.cpu_count() or 1))
    args = parser.parse_args()
    readiness = read_json(args.root / "stage3a_screening" / "readiness.json")
    if not readiness["ready"] or readiness["test_evaluated"]:
        raise RuntimeError("Stage 3A readiness gate is not valid")
    records, predictions = collect_runs(args.root)
    if set(records) != set(TRAINED_METHODS.values()) | {DERIVED_METHOD}:
        raise RuntimeError(f"Unexpected methods: {sorted(records)}")
    if any(set(seed_records) != set(SEEDS) for seed_records in records.values()):
        raise RuntimeError("Every method must contain exactly seeds 42, 43, and 44")

    result = {
        "protocol": "stage3a_lesion_disjoint_validation_only",
        "git_commit": readiness["git_commit"],
        "test_loaded": False,
        "test_evaluated": False,
        "trained_runs": len(readiness["accepted_training_runs"]),
        "derived_runs": len(readiness["accepted_derived_runs"]),
        "last_primary": summarize(records, predictions, "last"),
        "best_secondary": summarize(records, predictions, "best"),
        "hierarchical_bootstrap_last_vs_ce": {},
    }
    methods = sorted(set(records) - {"CE"})
    jobs = [
        (
            method,
            predictions["last"][method],
            predictions["last"]["CE"],
            args.bootstrap_repeats,
            20260802 + offset,
        )
        for offset, method in enumerate(methods)
    ]
    with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs))) as executor:
        futures = [executor.submit(bootstrap_job, job) for job in jobs]
        for future in as_completed(futures):
            method, bootstrap = future.result()
            result["hierarchical_bootstrap_last_vs_ce"][method] = bootstrap
    result["hierarchical_bootstrap_last_vs_ce"] = {
        method: result["hierarchical_bootstrap_last_vs_ce"][method]
        for method in methods
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": "completed",
        "output": str(args.output),
        "trained_runs": result["trained_runs"],
        "derived_runs": result["derived_runs"],
        "bootstrap_repeats": args.bootstrap_repeats,
        "workers": min(args.workers, len(jobs)),
        "test_evaluated": False,
    }, indent=2))


if __name__ == "__main__":
    main()
