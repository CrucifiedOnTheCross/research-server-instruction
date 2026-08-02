#!/usr/bin/env python3
"""Create validation-only logit-adjustment and LDAM temperature results in parallel."""

from __future__ import annotations

import csv
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import yaml

from longtail_medical.calibration import apply_temperature, fit_temperature
from longtail_medical.metrics import classification_metrics
from longtail_medical.statistical_analysis import aggregate_by_lesion
from tools.build_stage3b_splits import SPLIT_SEEDS
from tools.run_stage3a_logit_adjustment import adjust
from tools.run_stage3b_training_matrix import MODEL_SEEDS, experiment_name


TAU = 1.0


def read_predictions(path: Path, class_names: list[str]):
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return (
        np.asarray([int(row["label"]) for row in rows]),
        np.asarray([[float(row[f"prob_{name}"]) for name in class_names] for row in rows]),
        [row["image_id"] for row in rows],
        [row["lesion_id"] for row in rows],
    )


def write_predictions(path: Path, labels, probabilities, image_ids, lesion_ids, class_names):
    with path.open("w", encoding="utf-8", newline="") as handle:
        fields = ["image_id", "lesion_id", "label", "prediction"] + [f"prob_{name}" for name in class_names]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for image_id, lesion_id, label, probs in zip(image_ids, lesion_ids, labels, probabilities):
            writer.writerow({
                "image_id": image_id, "lesion_id": lesion_id, "label": int(label),
                "prediction": int(np.argmax(probs)),
                **{f"prob_{name}": float(probs[index]) for index, name in enumerate(class_names)},
            })


def metrics(labels, probabilities, image_ids, lesion_ids, class_names, bins):
    image = classification_metrics(labels, probabilities, class_names, bins)
    lesion_labels, lesion_probabilities, _ = aggregate_by_lesion(labels, probabilities, lesion_ids, image_ids)
    lesion = classification_metrics(lesion_labels, lesion_probabilities, class_names, bins)
    return {"image": image, "lesion": lesion}


def source_run(project: Path, arm: str, split_seed: int, model_seed: int) -> Path:
    name = experiment_name(arm, split_seed)
    candidates = sorted((project / "outputs/stage3b_confirmation" / name).glob(f"*_{model_seed}/summary.json"))
    valid = [path.parent for path in candidates if json.loads(path.read_text(encoding="utf-8")).get("test_evaluated") is False]
    if len(valid) != 1:
        raise RuntimeError(f"Expected one source for {name}/{model_seed}, found {len(valid)}")
    return valid[0]


def derive(job: tuple[str, int, int, str]) -> dict:
    project = Path(job[0])
    split_seed, model_seed, kind = job[1:]
    arm = "ce" if kind == "logit_adjustment" else "ldam_drw"
    source = source_run(project, arm, split_seed, model_seed)
    config = yaml.safe_load((source / "config.resolved.yaml").read_text(encoding="utf-8"))
    class_names = config["data"]["class_names"]
    bins = int(config["evaluation"]["ece_bins"])
    destination = project / "outputs/stage3b_derivations" / kind / f"split_{split_seed}" / f"model_seed_{model_seed}"
    destination.mkdir(parents=True, exist_ok=True)
    source_summary = json.loads((source / "summary.json").read_text(encoding="utf-8"))
    counts = json.loads((source / "class_counts.json").read_text(encoding="utf-8"))
    priors = np.asarray([counts[name] for name in class_names], dtype=np.float64)
    priors /= priors.sum()
    fitted = {}
    for checkpoint in ("last", "best"):
        labels, probabilities, image_ids, lesion_ids = read_predictions(
            source / f"val_predictions_{checkpoint}.csv", class_names
        )
        if kind == "logit_adjustment":
            transformed = adjust(probabilities, priors, TAU)
            parameters = {"tau": TAU}
        else:
            parameters = fit_temperature(labels, probabilities)
            transformed = apply_temperature(probabilities, parameters["temperature"])
            if int(np.sum(transformed.argmax(axis=1) != probabilities.argmax(axis=1))) != 0:
                raise RuntimeError("Scalar temperature changed argmax predictions")
        payload = {"parameters": parameters, **metrics(
            labels, transformed, image_ids, lesion_ids, class_names, bins
        )}
        (destination / f"val_metrics_{checkpoint}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        write_predictions(destination / f"val_predictions_{checkpoint}.csv", labels, transformed, image_ids, lesion_ids, class_names)
        fitted[checkpoint] = parameters
    summary = {
        "status": "completed", "method": kind, "split_seed": split_seed,
        "model_seed": model_seed, "source_run": str(source),
        "source_run_signature": source_summary["run_signature"],
        "parameters": fitted, "selection_data": "validation_only",
        "test_loaded": False, "test_evaluated": False,
    }
    (destination / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def sync_mlflow(project: Path, summary: dict) -> None:
    try:
        import mlflow
        from mlflow.tracking import MlflowClient

        mlflow.set_tracking_uri("http://10.200.1.180:5000")
        experiment = mlflow.set_experiment("ISIC2019-LT Stage3B Split Confirmation")
        client = MlflowClient()
        signature = summary["source_run_signature"]
        kind = summary["method"]
        existing = client.search_runs(
            [experiment.experiment_id],
            filter_string=f"tags.source_run_signature = '{signature}' and tags.derivation = '{kind}'",
            max_results=2,
        )
        if existing:
            return
        destination = project / "outputs/stage3b_derivations" / kind / f"split_{summary['split_seed']}" / f"model_seed_{summary['model_seed']}"
        with mlflow.start_run(
            experiment_id=experiment.experiment_id,
            run_name=f"stage3b_{kind}/split_{summary['split_seed']}/model_seed_{summary['model_seed']}",
            tags={
                "project": "longtail-medical", "stage": "stage3b_derivation",
                "derivation": kind, "split_seed": str(summary["split_seed"]),
                "model_seed": str(summary["model_seed"]),
                "source_run_signature": signature, "selection_data": "validation_only",
                "test_evaluated": "false",
            },
        ):
            if kind == "logit_adjustment":
                mlflow.log_param("tau", 1.0)
            else:
                for checkpoint, values in summary["parameters"].items():
                    mlflow.log_param(f"temperature_{checkpoint}", values["temperature"])
                    mlflow.log_metric(f"{checkpoint}/nll_before", values["nll_before"])
                    mlflow.log_metric(f"{checkpoint}/nll_after", values["nll_after"])
            for artifact in destination.glob("*"):
                if artifact.is_file():
                    mlflow.log_artifact(str(artifact))
    except Exception as error:
        print(f"MLflow sync disabled for derivation {summary['source_run_signature']}: {error}")


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    jobs = [
        (str(project), split_seed, model_seed, kind)
        for split_seed in SPLIT_SEEDS for model_seed in MODEL_SEEDS
        for kind in ("logit_adjustment", "temperature_scaling")
    ]
    workers = min(9, os.cpu_count() or 1, len(jobs))
    with ProcessPoolExecutor(max_workers=workers) as executor:
        completed = list(executor.map(derive, jobs))
    for summary in completed:
        sync_mlflow(project, summary)
    print(json.dumps({"completed": len(completed), "workers": workers, "test_evaluated": False}, indent=2))


if __name__ == "__main__":
    main()
