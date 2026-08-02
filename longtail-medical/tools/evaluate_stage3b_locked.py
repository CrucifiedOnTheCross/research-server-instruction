#!/usr/bin/env python3
"""Perform the single preregistered Stage 3B locked-test evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import yaml
from torchvision.models import resnet50

from longtail_medical.calibration import apply_temperature
from longtail_medical.data import ManifestDataset, build_transforms
from longtail_medical.losses import NormedLinear, prediction_scale
from longtail_medical.metrics import classification_metrics
from longtail_medical.provenance import code_commit, sha256_file
from longtail_medical.statistical_analysis import aggregate_by_lesion
from tools.run_stage3a_logit_adjustment import adjust
from tools.run_stage3b_training_matrix import experiment_name
from tools.run_stage3b_validation_derivations import write_predictions
from train import evaluate, make_loader


def load_run(run: Path, checkpoint_name: str, test_manifest: Path) -> dict:
    checkpoint = torch.load(run / checkpoint_name, map_location="cpu", weights_only=False)
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    if checkpoint["run_signature"] != summary["run_signature"]:
        raise RuntimeError(f"Checkpoint signature mismatch: {run}")
    config = checkpoint["config"]
    loss = config["loss"]
    model = resnet50(weights=None)
    if loss["method"] == "ldam_drw":
        model.fc = NormedLinear(model.fc.in_features, int(config["data"]["num_classes"]))
    else:
        model.fc = torch.nn.Linear(model.fc.in_features, int(config["data"]["num_classes"]))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    device = torch.device("cuda")
    model = model.to(device, memory_format=torch.channels_last)
    dataset = ManifestDataset(test_manifest, build_transforms(config, train=False))
    loader = make_loader(dataset, int(config["training"]["physical_batch_size"]), int(config["training"]["num_workers"]), False, int(config["experiment"]["seed"]))
    _, payload, labels, probabilities, image_ids, lesion_ids = evaluate(
        model, loader, device, config["training"]["amp_dtype"], config["data"]["class_names"],
        int(config["evaluation"]["ece_bins"]), prediction_scale(loss),
    )
    del model
    torch.cuda.empty_cache()
    return {
        "config": config, "labels": labels, "probabilities": probabilities,
        "image_ids": image_ids, "lesion_ids": lesion_ids, "metrics": payload,
    }


def transformed_metrics(result: dict, probabilities: np.ndarray) -> dict:
    config = result["config"]
    image = classification_metrics(result["labels"], probabilities, config["data"]["class_names"], int(config["evaluation"]["ece_bins"]))
    lesion_labels, lesion_probabilities, _ = aggregate_by_lesion(
        result["labels"], probabilities, result["lesion_ids"], result["image_ids"]
    )
    lesion = classification_metrics(lesion_labels, lesion_probabilities, config["data"]["class_names"], int(config["evaluation"]["ece_bins"]))
    return {"image": image, "lesion": lesion}


def save_result(destination: Path, result: dict, probabilities: np.ndarray, metrics: dict, metadata: dict) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    write_predictions(destination / "predictions.csv", result["labels"], probabilities, result["image_ids"], result["lesion_ids"], result["config"]["data"]["class_names"])
    (destination / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (destination / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    readiness_path = project / "outputs/stage3b_confirmation/readiness.json"
    readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
    registry_path = project / "data_splits/stage3b/split_registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    if not readiness.get("ready") or len(readiness.get("accepted_runs", [])) != 18:
        raise RuntimeError("Stage 3B readiness gate is closed")
    if readiness["git_commit"] != code_commit(project):
        raise RuntimeError("Code commit changed after training")
    if readiness["split_registry_sha256"] != sha256_file(registry_path):
        raise RuntimeError("Split registry changed after readiness")
    output = project / "outputs/stage3b_locked_test"
    marker = output / "TEST_EVALUATION_STARTED.json"
    if output.exists() or marker.exists():
        raise RuntimeError("Stage 3B locked test has already started")
    output.mkdir(parents=True, exist_ok=False)
    marker.write_text(json.dumps({
        "one_shot": True, "git_commit": readiness["git_commit"],
        "split_registry_sha256": readiness["split_registry_sha256"],
        "accepted_runs": readiness["accepted_runs"],
    }, indent=2), encoding="utf-8")
    run_lookup = {}
    for value in readiness["accepted_runs"]:
        run = Path(value)
        config = yaml.safe_load((run / "config.resolved.yaml").read_text(encoding="utf-8"))
        run_lookup[(config["experiment"]["name"], int(config["experiment"]["seed"]))] = run
    compact = []
    for split_seed in registry["split_seeds"]:
        split = registry["splits"][str(split_seed)]
        test_manifest = project / split["test_manifest"]
        if sha256_file(test_manifest) != split["test_sha256"]:
            raise RuntimeError(f"Test manifest changed: split {split_seed}")
        for model_seed in (42, 43, 44):
            for arm in ("ce", "ldam_drw"):
                run = run_lookup[(experiment_name(arm, split_seed), model_seed)]
                counts = json.loads((run / "class_counts.json").read_text(encoding="utf-8"))
                for checkpoint in ("last", "best"):
                    result = load_run(run, f"{checkpoint}.pt", test_manifest)
                    base = output / arm / f"split_{split_seed}" / f"model_seed_{model_seed}" / checkpoint
                    metadata = {"arm": arm, "split_seed": split_seed, "model_seed": model_seed, "checkpoint": checkpoint, "source_run": str(run)}
                    save_result(base, result, result["probabilities"], result["metrics"], metadata)
                    compact.append({**metadata, "variant": "raw", "image": result["metrics"]["image"], "lesion": result["metrics"]["lesion"]})
                    if arm == "ce":
                        priors = np.asarray([counts[name] for name in result["config"]["data"]["class_names"]], dtype=np.float64)
                        priors /= priors.sum()
                        derived = adjust(result["probabilities"], priors, 1.0)
                        variant, parameters = "logit_adjustment_tau1", {"tau": 1.0}
                    else:
                        calibration = project / "outputs/stage3b_derivations/temperature_scaling" / f"split_{split_seed}" / f"model_seed_{model_seed}" / "summary.json"
                        fitted = json.loads(calibration.read_text(encoding="utf-8"))["parameters"][checkpoint]
                        derived = apply_temperature(result["probabilities"], fitted["temperature"])
                        variant, parameters = "temperature_scaled", fitted
                    derived_metrics = transformed_metrics(result, derived)
                    derived_base = output / variant / f"split_{split_seed}" / f"model_seed_{model_seed}" / checkpoint
                    save_result(derived_base, result, derived, derived_metrics, {**metadata, "variant": variant, "parameters": parameters})
                    compact.append({**metadata, "variant": variant, "image": derived_metrics["image"], "lesion": derived_metrics["lesion"]})
    summary = {
        "status": "completed", "one_shot": True, "test_evaluated": True,
        "git_commit": readiness["git_commit"], "records": compact,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"status": "completed", "records": len(compact), "test_evaluated": True}, indent=2))


if __name__ == "__main__":
    main()
