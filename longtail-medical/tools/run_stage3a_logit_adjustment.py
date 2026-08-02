#!/usr/bin/env python3
"""Derive preregistered post-hoc logit adjustment from CE validation predictions."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import yaml

from longtail_medical.metrics import classification_metrics
from longtail_medical.statistical_analysis import aggregate_by_lesion


TAU = 1.0
SEEDS = (42, 43, 44)
SOURCE_ARM = "stage3a_ce_resnet50_lesion_disjoint"


def write_predictions(path: Path, labels, probabilities, image_ids, lesion_ids, class_names):
    with path.open("w", newline="", encoding="utf-8") as handle:
        fields = ["image_id", "lesion_id", "label", "prediction"] + [f"prob_{name}" for name in class_names]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for image_id, lesion_id, label, probs in zip(image_ids, lesion_ids, labels, probabilities):
            writer.writerow({
                "image_id": image_id,
                "lesion_id": lesion_id,
                "label": int(label),
                "prediction": int(np.argmax(probs)),
                **{f"prob_{name}": float(probs[index]) for index, name in enumerate(class_names)},
            })


def read_predictions(path: Path, class_names: list[str]):
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    labels = np.asarray([int(row["label"]) for row in rows])
    probabilities = np.asarray([
        [float(row[f"prob_{name}"]) for name in class_names] for row in rows
    ])
    return labels, probabilities, [row["image_id"] for row in rows], [row["lesion_id"] for row in rows]


def adjust(probabilities: np.ndarray, priors: np.ndarray, tau: float) -> np.ndarray:
    logits = np.log(np.clip(probabilities, 1e-12, 1.0)) - tau * np.log(priors)[None, :]
    logits -= logits.max(axis=1, keepdims=True)
    adjusted = np.exp(logits)
    return adjusted / adjusted.sum(axis=1, keepdims=True)


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    source_root = project / "outputs" / "stage3a_screening" / SOURCE_ARM
    output_root = project / "outputs" / "stage3a_logit_adjustment_tau1"
    output_root.mkdir(parents=True, exist_ok=True)
    for seed in SEEDS:
        candidates = sorted(source_root.glob(f"*_{seed}/summary.json"))
        valid = [
            path for path in candidates
            if json.loads(path.read_text(encoding="utf-8")).get("test_evaluated") is False
        ]
        if len(valid) != 1:
            raise RuntimeError(f"Expected one CE source run for seed {seed}, found {len(valid)}")
        source = valid[0].parent
        config = yaml.safe_load((source / "config.resolved.yaml").read_text(encoding="utf-8"))
        class_names = config["data"]["class_names"]
        counts = json.loads((source / "class_counts.json").read_text(encoding="utf-8"))
        priors = np.asarray([counts[name] for name in class_names], dtype=np.float64)
        priors /= priors.sum()
        destination = output_root / f"seed_{seed}"
        destination.mkdir(parents=True, exist_ok=True)
        checkpoint_results = {}
        for checkpoint in ("last", "best"):
            labels, probabilities, image_ids, lesion_ids = read_predictions(
                source / f"val_predictions_{checkpoint}.csv", class_names
            )
            adjusted = adjust(probabilities, priors, TAU)
            image_metrics = classification_metrics(labels, adjusted, class_names, int(config["evaluation"]["ece_bins"]))
            lesion_labels, lesion_probabilities, _ = aggregate_by_lesion(
                labels, adjusted, lesion_ids, image_ids
            )
            lesion_metrics = classification_metrics(
                lesion_labels, lesion_probabilities, class_names, int(config["evaluation"]["ece_bins"])
            )
            write_predictions(
                destination / f"val_predictions_{checkpoint}.csv",
                labels, adjusted, image_ids, lesion_ids, class_names,
            )
            payload = {"tau": TAU, "image": image_metrics, "lesion": lesion_metrics}
            (destination / f"val_metrics_{checkpoint}.json").write_text(
                json.dumps(payload, indent=2), encoding="utf-8"
            )
            checkpoint_results[checkpoint] = payload
        source_summary = json.loads((source / "summary.json").read_text(encoding="utf-8"))
        (destination / "summary.json").write_text(json.dumps({
            "status": "completed",
            "method": "posthoc_logit_adjustment",
            "tau": TAU,
            "seed": seed,
            "source_run": str(source),
            "source_run_signature": source_summary["run_signature"],
            "source_checkpoint_policy": "last_primary_best_secondary",
            "test_loaded": False,
            "test_evaluated": False,
            "selection_data": "validation_only",
        }, indent=2), encoding="utf-8")
    print(json.dumps({"completed_seeds": list(SEEDS), "tau": TAU, "test_evaluated": False}, indent=2))


if __name__ == "__main__":
    main()
