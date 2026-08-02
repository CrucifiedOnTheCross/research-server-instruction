#!/usr/bin/env python3
"""Finish Stage 2 statistics from persisted one-shot test predictions."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

from longtail_medical.provenance import code_commit, sha256_file
from longtail_medical.statistical_analysis import (
    lesion_stratified_metric_bootstrap,
    paired_lesion_stratified_bootstrap,
)


def load_predictions(path: Path, class_names: list[str]) -> dict:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    return {
        "labels": np.asarray([int(row["label"]) for row in rows]),
        "probabilities": np.asarray([
            [float(row[f"prob_{name}"]) for name in class_names] for row in rows
        ]),
        "image_ids": [row["image_id"] for row in rows],
        "lesion_ids": [row["lesion_id"] for row in rows],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, default=Path("configs/stage2_test_registry.yaml"))
    parser.add_argument("--output", type=Path, default=Path("outputs/stage2_locked_test"))
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    output = project / args.output
    marker_path = output / "TEST_EVALUATION_STARTED.json"
    if not marker_path.exists():
        raise RuntimeError("One-shot test marker is missing; analysis-only recovery is forbidden")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    registry_path = project / args.registry
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    if marker["registry_sha256"] != sha256_file(registry_path):
        raise RuntimeError("Registry differs from the one-shot inference registry")
    if len(marker.get("accepted_runs", [])) != 12:
        raise RuntimeError("One-shot marker does not contain exactly 12 accepted runs")

    results: dict[tuple[str, int, str], dict] = {}
    compact = []
    for arm in registry["arms"]:
        for seed in registry["seeds"]:
            for checkpoint in ("last", "best"):
                directory = output / arm / f"seed_{seed}" / checkpoint
                required = (
                    directory / "predictions.csv",
                    directory / "image_metrics.json",
                    directory / "lesion_metrics.json",
                )
                if not all(path.exists() for path in required):
                    raise RuntimeError(f"Incomplete persisted inference artifacts: {directory}")
                run = next(
                    Path(path) for path in marker["accepted_runs"]
                    if f"/{arm}/" in path and path.endswith(f"_{seed}")
                )
                config = yaml.safe_load((run / "config.resolved.yaml").read_text(encoding="utf-8"))
                result = load_predictions(required[0], config["data"]["class_names"])
                result["class_names"] = config["data"]["class_names"]
                result["image_metrics"] = json.loads(required[1].read_text(encoding="utf-8"))
                results[(arm, seed, checkpoint)] = result
                compact.append({
                    "arm": arm,
                    "seed": seed,
                    "checkpoint": f"{checkpoint}.pt",
                    **{
                        key: value for key, value in result["image_metrics"].items()
                        if isinstance(value, (int, float))
                    },
                })

    original = "stage2_monica_original_resnet50_ce"
    comparisons = {
        "exposure_matched_minus_original": "stage2_monica_decontaminated_exposure_matched_resnet50_ce",
        "unmatched_minus_original": "stage2_monica_decontaminated_unmatched_resnet50_ce",
    }
    comparison_report = {}
    for comparison_name, treatment_arm in comparisons.items():
        seed_reports = []
        bootstrap_by_metric = defaultdict(list)
        point_by_metric = defaultdict(list)
        for seed in registry["seeds"]:
            treatment = results[(treatment_arm, seed, "last")]
            control = results[(original, seed, "last")]
            if treatment["image_ids"] != control["image_ids"]:
                raise RuntimeError("Paired comparison image order mismatch")
            point = {
                metric: treatment["image_metrics"][metric] - control["image_metrics"][metric]
                for metric in ("mcc", "balanced_accuracy", "macro_f1", "macro_auroc", "macro_auprc", "nll", "brier", "ece")
            }
            bootstrap = paired_lesion_stratified_bootstrap(
                treatment["labels"], treatment["probabilities"], control["probabilities"],
                treatment["lesion_ids"], treatment["image_ids"], treatment["class_names"],
                repeats=int(registry["bootstrap"]["repeats"]), seed=20260802 + seed,
            )
            for metric, value in point.items():
                point_by_metric[metric].append(float(value))
            for metric in ("mcc", "balanced_accuracy"):
                bootstrap_by_metric[metric].append(np.asarray(bootstrap[metric].pop("samples")))
            seed_reports.append({"seed": seed, "point_effect": point, "bootstrap": bootstrap})
        aggregate = {}
        for metric, values in point_by_metric.items():
            aggregate[metric] = {
                "mean": float(np.mean(values)),
                "seed_sd": float(np.std(values, ddof=1)),
            }
            if metric in bootstrap_by_metric:
                mean_samples = np.vstack(bootstrap_by_metric[metric]).mean(axis=0)
                aggregate[metric].update({
                    "paired_lesion_bootstrap_ci95_low": float(np.quantile(mean_samples, 0.025)),
                    "paired_lesion_bootstrap_ci95_high": float(np.quantile(mean_samples, 0.975)),
                })
        comparison_report[comparison_name] = {"per_seed": seed_reports, "aggregate": aggregate}
    (output / "controlled_contamination_effects.json").write_text(
        json.dumps(comparison_report, indent=2), encoding="utf-8"
    )

    strict_arm = "stage2_lesion_disjoint_ir100_resnet50_ce"
    strict_bootstrap = []
    for seed in registry["seeds"]:
        result = results[(strict_arm, seed, "last")]
        strict_bootstrap.append({
            "seed": seed,
            "intervals": lesion_stratified_metric_bootstrap(
                result["labels"], result["probabilities"], result["lesion_ids"],
                result["image_ids"], result["class_names"],
                repeats=int(registry["bootstrap"]["repeats"]), seed=20260802 + seed,
            ),
        })
    (output / "lesion_disjoint_bootstrap_intervals.json").write_text(
        json.dumps(strict_bootstrap, indent=2), encoding="utf-8"
    )
    (output / "all_run_metrics.json").write_text(json.dumps(compact, indent=2), encoding="utf-8")
    analysis_commit = code_commit(project)
    (output / "evaluation_summary.json").write_text(json.dumps({
        "status": "completed",
        "test_evaluated": True,
        "one_shot_inference_git_commit": marker["git_commit"],
        "analysis_git_commit": analysis_commit,
        "analysis_recovered_from_persisted_predictions": True,
        "test_inference_repeated": False,
        "primary_checkpoint": "last.pt",
        "secondary_checkpoint": "best.pt",
        "bootstrap_repeats": int(registry["bootstrap"]["repeats"]),
    }, indent=2), encoding="utf-8")
    print(json.dumps({"status": "completed", "analysis_git_commit": analysis_commit}, indent=2))


if __name__ == "__main__":
    main()
