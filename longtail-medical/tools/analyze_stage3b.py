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

from longtail_medical.statistical_analysis import aggregate_by_lesion, fast_mcc_balanced_accuracy


CLASS_NAMES = ["nv", "mel", "bcc", "bkl", "ak", "scc", "vasc", "df"]
SPLIT_SEEDS = (101, 202, 303)
MODEL_SEEDS = (42, 43, 44)


def read_predictions(path: Path) -> dict:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    labels = np.asarray([int(row["label"]) for row in rows])
    probabilities = np.asarray([[float(row[f"prob_{name}"]) for name in CLASS_NAMES] for row in rows])
    image_ids = [row["image_id"] for row in rows]
    lesion_ids = [row["lesion_id"] for row in rows]
    lesion_labels, lesion_probabilities, ordered_ids = aggregate_by_lesion(labels, probabilities, lesion_ids, image_ids)
    return {"labels": lesion_labels, "probabilities": lesion_probabilities, "ids": ordered_ids}


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
