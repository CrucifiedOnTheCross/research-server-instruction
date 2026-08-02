#!/usr/bin/env python3
"""One-shot locked-test evaluation for the preregistered Stage 2 matrix."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from torchvision.models import resnet50

from longtail_medical.data import ManifestDataset, build_transforms
from longtail_medical.metrics import classification_metrics
from longtail_medical.provenance import code_commit, sha256_file
from longtail_medical.statistical_analysis import (
    aggregate_by_lesion,
    lesion_stratified_metric_bootstrap,
    paired_lesion_stratified_bootstrap,
)
from train import evaluate, make_loader, write_predictions


def load_run(run_dir: Path, checkpoint_name: str, expected_policy: str, test_manifest: Path):
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    checkpoint = torch.load(run_dir / checkpoint_name, map_location="cpu", weights_only=False)
    if checkpoint.get("run_signature") != summary["run_signature"]:
        raise RuntimeError(f"Checkpoint signature mismatch: {run_dir}/{checkpoint_name}")
    if checkpoint.get("checkpoint_policy") != expected_policy:
        raise RuntimeError(f"Checkpoint policy mismatch: {run_dir}/{checkpoint_name}")
    config = checkpoint["config"]
    dataset = ManifestDataset(test_manifest, build_transforms(config, train=False))
    loader = make_loader(
        dataset, int(config["training"]["physical_batch_size"]),
        int(config["training"]["num_workers"]), False,
        int(config["experiment"]["seed"]),
    )
    model = resnet50(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, int(config["data"]["num_classes"]))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    device = torch.device("cuda")
    model = model.to(device, memory_format=torch.channels_last)
    _, validation_payload, labels, probabilities, image_ids, lesion_ids = evaluate(
        model, loader, device, config["training"]["amp_dtype"],
        config["data"]["class_names"], int(config["evaluation"]["ece_bins"]),
    )
    lesion_labels, lesion_probabilities, aggregated_ids = aggregate_by_lesion(
        labels, probabilities, lesion_ids, image_ids
    )
    lesion_metrics = classification_metrics(
        lesion_labels, lesion_probabilities, config["data"]["class_names"],
        int(config["evaluation"]["ece_bins"]),
    )
    return {
        "config": config,
        "image_metrics": validation_payload["image"],
        "lesion_metrics": lesion_metrics,
        "labels": labels,
        "probabilities": probabilities,
        "image_ids": image_ids,
        "lesion_ids": lesion_ids,
        "aggregated_lesion_ids": aggregated_ids,
    }


def subset_metrics(result: dict, selected_ids: set[str]) -> dict:
    indices = np.asarray([image_id in selected_ids for image_id in result["image_ids"]])
    return classification_metrics(
        result["labels"][indices], result["probabilities"][indices],
        result["config"]["data"]["class_names"],
        int(result["config"]["evaluation"]["ece_bins"]),
    )


def scalar_metrics(result: dict) -> dict[str, float]:
    return {
        key: float(value) for key, value in result["image_metrics"].items()
        if isinstance(value, (int, float))
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, default=Path("configs/stage2_test_registry.yaml"))
    parser.add_argument("--readiness", type=Path, default=Path("outputs/stage2_confirmatory/readiness.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/stage2_locked_test"))
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    registry_path = project / args.registry
    readiness_path = project / args.readiness
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
    if not readiness.get("ready") or len(readiness.get("accepted_runs", [])) != 12:
        raise RuntimeError("Readiness gate is not open")
    if readiness["git_commit"] != code_commit(project):
        raise RuntimeError("Evaluation code commit differs from training commit")
    if readiness["registry_sha256"] != sha256_file(registry_path):
        raise RuntimeError("Registry changed after readiness check")
    output = project / args.output
    marker = output / "TEST_EVALUATION_STARTED.json"
    if marker.exists() or output.exists():
        raise RuntimeError("Locked test evaluation has already started; refusing a second invocation")
    output.mkdir(parents=True, exist_ok=False)
    marker.write_text(json.dumps({
        "git_commit": readiness["git_commit"],
        "registry_sha256": readiness["registry_sha256"],
        "accepted_runs": readiness["accepted_runs"],
        "one_shot": True,
    }, indent=2), encoding="utf-8")

    run_lookup: dict[tuple[str, int], Path] = {}
    for run in readiness["accepted_runs"]:
        run_path = Path(run)
        summary = json.loads((run_path / "summary.json").read_text(encoding="utf-8"))
        config = yaml.safe_load((run_path / "config.resolved.yaml").read_text(encoding="utf-8"))
        run_lookup[(config["experiment"]["name"], int(config["experiment"]["seed"]))] = run_path

    results: dict[tuple[str, int, str], dict] = {}
    compact = []
    policy_names = {"last.pt": "last", "best.pt": "best_validation"}
    for arm, expected in registry["arms"].items():
        test_manifest = project / expected["test_manifest"]
        if sha256_file(test_manifest) != expected["test_sha256"]:
            raise RuntimeError(f"Test hash mismatch immediately before inference: {arm}")
        for seed in registry["seeds"]:
            run_dir = run_lookup[(arm, seed)]
            for checkpoint_name, expected_policy in policy_names.items():
                result = load_run(run_dir, checkpoint_name, expected_policy, test_manifest)
                results[(arm, seed, checkpoint_name)] = result
                destination = output / arm / f"seed_{seed}" / checkpoint_name.removesuffix(".pt")
                destination.mkdir(parents=True, exist_ok=True)
                write_predictions(
                    destination / "predictions.csv", result["labels"], result["probabilities"],
                    result["image_ids"], result["lesion_ids"], result["config"]["data"]["class_names"],
                )
                (destination / "image_metrics.json").write_text(json.dumps(result["image_metrics"], indent=2), encoding="utf-8")
                (destination / "lesion_metrics.json").write_text(json.dumps(result["lesion_metrics"], indent=2), encoding="utf-8")
                compact.append({
                    "arm": arm, "seed": seed, "checkpoint": checkpoint_name,
                    **scalar_metrics(result),
                })

    leaked_ids = {row["image_id"] for row in ManifestDataset(project / "data_splits/stage2/monica_causal/test_leaked.csv").rows}
    clean_ids = {row["image_id"] for row in ManifestDataset(project / "data_splits/stage2/monica_causal/test_clean.csv").rows}
    descriptive = []
    original = "stage2_monica_original_resnet50_ce"
    for seed in registry["seeds"]:
        result = results[(original, seed, "last.pt")]
        descriptive.append({
            "seed": seed,
            "full": result["image_metrics"],
            "leaked": subset_metrics(result, leaked_ids),
            "clean": subset_metrics(result, clean_ids),
            "interpretation": "descriptive association; test lesion IDs informed protocol construction",
        })
    (output / "monica_leaked_clean_descriptive.json").write_text(json.dumps(descriptive, indent=2), encoding="utf-8")

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
            treatment = results[(treatment_arm, seed, "last.pt")]
            control = results[(original, seed, "last.pt")]
            if treatment["image_ids"] != control["image_ids"]:
                raise RuntimeError("Paired comparison image order mismatch")
            point = {
                metric: treatment["image_metrics"][metric] - control["image_metrics"][metric]
                for metric in ("mcc", "balanced_accuracy", "macro_f1", "macro_auroc", "macro_auprc", "nll", "brier", "ece")
            }
            bootstrap = paired_lesion_stratified_bootstrap(
                treatment["labels"], treatment["probabilities"], control["probabilities"],
                treatment["lesion_ids"], treatment["image_ids"], treatment["config"]["data"]["class_names"],
                repeats=int(registry["bootstrap"]["repeats"]), seed=20260802 + seed,
            )
            for metric, value in point.items():
                point_by_metric[metric].append(float(value))
            for metric in ("mcc", "balanced_accuracy"):
                bootstrap_by_metric[metric].append(np.asarray(bootstrap[metric].pop("samples")))
            seed_reports.append({"seed": seed, "point_effect": point, "bootstrap": bootstrap})
        aggregate = {}
        for metric, values in point_by_metric.items():
            aggregate[metric] = {"mean": float(np.mean(values)), "seed_sd": float(np.std(values, ddof=1))}
            if metric in bootstrap_by_metric:
                mean_samples = np.vstack(bootstrap_by_metric[metric]).mean(axis=0)
                aggregate[metric].update({
                    "paired_lesion_bootstrap_ci95_low": float(np.quantile(mean_samples, 0.025)),
                    "paired_lesion_bootstrap_ci95_high": float(np.quantile(mean_samples, 0.975)),
                })
        comparison_report[comparison_name] = {"per_seed": seed_reports, "aggregate": aggregate}
    (output / "controlled_contamination_effects.json").write_text(json.dumps(comparison_report, indent=2), encoding="utf-8")
    strict_arm = "stage2_lesion_disjoint_ir100_resnet50_ce"
    strict_bootstrap = []
    for seed in registry["seeds"]:
        result = results[(strict_arm, seed, "last.pt")]
        strict_bootstrap.append({
            "seed": seed,
            "intervals": lesion_stratified_metric_bootstrap(
                result["labels"], result["probabilities"], result["lesion_ids"],
                result["image_ids"], result["config"]["data"]["class_names"],
                repeats=int(registry["bootstrap"]["repeats"]), seed=20260802 + seed,
            ),
        })
    (output / "lesion_disjoint_bootstrap_intervals.json").write_text(
        json.dumps(strict_bootstrap, indent=2), encoding="utf-8"
    )
    (output / "all_run_metrics.json").write_text(json.dumps(compact, indent=2), encoding="utf-8")
    (output / "evaluation_summary.json").write_text(json.dumps({
        "status": "completed",
        "test_evaluated": True,
        "checkpoint_results_separated": True,
        "primary_checkpoint": "last.pt",
        "secondary_checkpoint": "best.pt",
        "bootstrap_repeats": registry["bootstrap"]["repeats"],
        "git_commit": readiness["git_commit"],
        "registry_sha256": readiness["registry_sha256"],
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
