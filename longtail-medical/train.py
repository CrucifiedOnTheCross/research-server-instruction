#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import platform
import random
import sys
import time
from collections import Counter
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models import ResNet50_Weights, resnet50

from longtail_medical.config import load_config, write_resolved_config
from longtail_medical.data import ManifestDataset, build_transforms
from longtail_medical.losses import LongTailObjective, NormedLinear, prediction_scale
from longtail_medical.metrics import classification_metrics
from longtail_medical.provenance import code_commit, resolved_config_sha256, run_signature, sha256_file
from longtail_medical.statistical_analysis import aggregate_by_lesion
from longtail_medical.tracking import build_mlflow_tags, namespace_epoch_metrics


def seed_everything(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.use_deterministic_algorithms(deterministic, warn_only=True)


def resolve_manifest(project_root: Path, configured: str) -> Path:
    path = Path(configured)
    return path if path.is_absolute() else project_root / path


def environment_payload(commit: str) -> dict:
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": __import__("torchvision").__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": gpu,
        "cuda_available": torch.cuda.is_available(),
        "git_commit": commit,
    }


def make_loader(dataset, batch_size: int, workers: int, shuffle: bool, seed: int):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        prefetch_factor=4 if workers > 0 else None,
        generator=generator,
        drop_last=False,
    )


def autocast_context(device: torch.device, dtype_name: str):
    if device.type != "cuda":
        return nullcontext()
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[dtype_name]
    return torch.autocast(device_type="cuda", dtype=dtype)


@torch.inference_mode()
def evaluate(
    model, loader, device, dtype_name: str, class_names: list[str], ece_bins: int,
    logit_scale: float = 1.0,
):
    model.eval()
    labels, probabilities, image_ids, lesion_ids = [], [], [], []
    loss_sum = 0.0
    for images, target, batch_ids, batch_lesions in loader:
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        if device.type == "cuda":
            images = images.contiguous(memory_format=torch.channels_last)
        with autocast_context(device, dtype_name):
            logits = model(images)
            predictive_logits = logits * logit_scale
            loss = F.cross_entropy(predictive_logits, target)
        loss_sum += float(loss) * len(target)
        labels.append(target.cpu().numpy())
        probabilities.append(torch.softmax(predictive_logits.float(), dim=1).cpu().numpy())
        image_ids.extend(batch_ids)
        lesion_ids.extend(batch_lesions)
    labels_array = np.concatenate(labels)
    probability_array = np.concatenate(probabilities)
    image_metrics = classification_metrics(labels_array, probability_array, class_names, ece_bins)
    image_metrics["loss"] = loss_sum / len(labels_array)
    lesion_labels, lesion_probabilities, _ = aggregate_by_lesion(
        labels_array, probability_array, lesion_ids, image_ids
    )
    lesion_metrics = classification_metrics(
        lesion_labels, lesion_probabilities, class_names, ece_bins
    )
    metrics = {
        **{key: value for key, value in image_metrics.items() if isinstance(value, (int, float))},
        **{f"image_{key}": value for key, value in image_metrics.items() if isinstance(value, (int, float))},
        **{f"lesion_{key}": value for key, value in lesion_metrics.items() if isinstance(value, (int, float))},
    }
    payload = {"image": image_metrics, "lesion": lesion_metrics}
    return metrics, payload, labels_array, probability_array, image_ids, lesion_ids


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


def maybe_start_mlflow(
    config: dict,
    run_name: str,
    *,
    git_commit: str,
    run_signature_value: str,
):
    try:
        import mlflow

        mlflow.set_tracking_uri(config["tracking"]["mlflow_tracking_uri"])
        mlflow.set_experiment(config["tracking"]["mlflow_experiment"])
        run = mlflow.start_run(
            run_name=run_name,
            tags=build_mlflow_tags(
                config,
                git_commit=git_commit,
                run_signature=run_signature_value,
            ),
        )
        mlflow.log_params({
            "seed": config["experiment"]["seed"],
            "model": config["model"]["name"],
            "epochs": config["training"]["epochs"],
            "physical_batch_size": config["training"]["physical_batch_size"],
            "accumulation_steps": config["training"]["accumulation_steps"],
            "test_evaluated": False,
            "loss_method": config.get("loss", {}).get("method", "cross_entropy"),
            "checkpoint_monitor": config["checkpoint"]["monitor"],
            "loss_gamma": config.get("loss", {}).get("gamma", 0.0),
            "loss_beta": config.get("loss", {}).get("beta", 0.0),
            "ldam_max_margin": config.get("loss", {}).get("max_margin", 0.0),
            "ldam_scale": config.get("loss", {}).get("scale", 1.0),
            "drw_start_epoch": config.get("loss", {}).get("drw_start_epoch", 0),
        })
        return mlflow, run
    except Exception as error:
        print(f"MLflow disabled: {error}", file=sys.stderr)
        return None, None


def train(config_path: Path) -> Path:
    project_root = Path(__file__).resolve().parent
    config = load_config(config_path)
    commit = code_commit(project_root)
    seed = int(config["experiment"]["seed"])
    seed_everything(seed, bool(config["training"]["deterministic"]))
    if not torch.cuda.is_available():
        raise RuntimeError("Stage 1 training requires a CUDA GPU")
    device = torch.device("cuda")
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + f"_{seed}"
    output = project_root / config["tracking"]["output_root"] / config["experiment"]["name"] / run_id
    output.mkdir(parents=True, exist_ok=False)
    write_resolved_config(output / "config.resolved.yaml", config)
    (output / "environment.json").write_text(json.dumps(environment_payload(commit), indent=2), encoding="utf-8")

    train_csv = resolve_manifest(project_root, config["data"]["train_csv"])
    val_csv = resolve_manifest(project_root, config["data"]["val_csv"])
    for manifest in (train_csv, val_csv):
        if not manifest.exists():
            raise FileNotFoundError(manifest)
    split_artifact = {
        "train": {"path": str(train_csv), "sha256": sha256_file(train_csv)},
        "validation": {"path": str(val_csv), "sha256": sha256_file(val_csv)},
        "test_loaded": False,
        "test_evaluated": False,
    }
    (output / "split_manifest.json").write_text(json.dumps(split_artifact, indent=2), encoding="utf-8")
    signature, signature_fields = run_signature(
        commit=commit,
        config_hash=resolved_config_sha256(config),
        train_hash=split_artifact["train"]["sha256"],
        validation_hash=split_artifact["validation"]["sha256"],
        protocol_version=config["data"]["protocol_version"],
        checkpoint_policy=config["checkpoint"]["primary_policy"],
        epochs=int(config["training"]["epochs"]),
    )
    (output / "run_signature.json").write_text(
        json.dumps({"run_signature": signature, **signature_fields}, indent=2),
        encoding="utf-8",
    )

    train_data = ManifestDataset(train_csv, build_transforms(config, train=True))
    val_data = ManifestDataset(val_csv, build_transforms(config, train=False))
    counts = Counter(int(row["label"]) for row in train_data.rows)
    class_counts = {name: counts.get(index, 0) for index, name in enumerate(config["data"]["class_names"])}
    (output / "class_counts.json").write_text(json.dumps(class_counts, indent=2), encoding="utf-8")
    missing = [row["image_path"] for row in train_data.rows + val_data.rows if not Path(row["image_path"]).exists()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} images; first: {missing[0]}")

    workers = int(config["training"]["num_workers"])
    batch = int(config["training"]["physical_batch_size"])
    train_loader = make_loader(train_data, batch, workers, True, seed)
    val_loader = make_loader(val_data, batch, workers, False, seed)
    weights = ResNet50_Weights.IMAGENET1K_V2 if config["model"]["pretrained"] else None
    model = resnet50(weights=weights)
    classifier_input = model.fc.in_features
    loss_config = config.get("loss", {"method": "cross_entropy"})
    if loss_config.get("method") == "ldam_drw":
        model.fc = NormedLinear(classifier_input, int(config["data"]["num_classes"]))
        classifier_type = "cosine_normed_linear"
    else:
        model.fc = torch.nn.Linear(classifier_input, int(config["data"]["num_classes"]))
        classifier_type = "linear"
    model = model.to(device)
    if config["training"]["channels_last"]:
        model = model.to(memory_format=torch.channels_last)
    initialization = {
        "architecture": "torchvision.resnet50",
        "weights": str(weights),
        "classifier_randomly_initialized": True,
        "classifier_type": classifier_type,
        "prediction_logit_scale": prediction_scale(loss_config),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }
    (output / "model_initialization.json").write_text(json.dumps(initialization, indent=2), encoding="utf-8")
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"])
    )
    ordered_counts = [counts.get(index, 0) for index in range(int(config["data"]["num_classes"]))]
    objective = LongTailObjective(loss_config, ordered_counts).to(device)
    (output / "loss_initialization.json").write_text(
        json.dumps(objective.metadata(), indent=2), encoding="utf-8"
    )
    accumulation = int(config["training"]["accumulation_steps"])
    dtype_name = config["training"]["amp_dtype"]
    scaler = torch.amp.GradScaler("cuda", enabled=dtype_name == "float16")
    monitor = config["checkpoint"]["monitor"]
    best_value = -float("inf")
    metrics_path = output / "metrics.csv"
    mlflow, mlflow_run = maybe_start_mlflow(
        config,
        f"{config['experiment']['name']}/{run_id}",
        git_commit=commit,
        run_signature_value=signature,
    )
    started = time.monotonic()
    try:
        for epoch in range(1, int(config["training"]["epochs"]) + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss_sum = 0.0
            epoch_started = time.monotonic()
            for step, (images, target, _, _) in enumerate(train_loader, start=1):
                images = images.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
                target = target.to(device, non_blocking=True)
                with autocast_context(device, dtype_name):
                    loss = objective(model(images), target, epoch)
                    scaled_loss = loss / accumulation
                scaler.scale(scaled_loss).backward()
                if step % accumulation == 0 or step == len(train_loader):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                loss_sum += float(loss) * len(target)
            val_metrics, val_payload, labels, probs, ids, lesions = evaluate(
                model, val_loader, device, dtype_name, config["data"]["class_names"],
                int(config["evaluation"]["ece_bins"]), prediction_scale(loss_config),
            )
            row = {
                "epoch": epoch,
                "train_loss": loss_sum / len(train_data),
                **{f"val_{key}": value for key, value in val_metrics.items() if isinstance(value, (int, float))},
                "epoch_seconds": time.monotonic() - epoch_started,
                "gpu_max_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
            }
            with metrics_path.open("a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                if handle.tell() == 0:
                    writer.writeheader()
                writer.writerow(row)
            if mlflow:
                mlflow.log_metrics(namespace_epoch_metrics(row), step=epoch)
            if float(val_metrics[monitor]) > best_value:
                best_value = float(val_metrics[monitor])
                torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "monitor": monitor, "monitor_value": best_value, "checkpoint_policy": "best_validation", "run_signature": signature, "config": config}, output / "best.pt")
                (output / "val_metrics_best.json").write_text(
                    json.dumps({"epoch": epoch, "monitor": monitor, "monitor_value": best_value, **val_payload}, indent=2),
                    encoding="utf-8",
                )
                write_predictions(output / "val_predictions_best.csv", labels, probs, ids, lesions, config["data"]["class_names"])
            if epoch == int(config["training"]["epochs"]):
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "checkpoint_policy": "last",
                    "config": config,
                    "run_signature": signature,
                }, output / "last.pt")
                (output / "val_metrics_last.json").write_text(
                    json.dumps({"epoch": epoch, "monitor": monitor, "monitor_value": float(val_metrics[monitor]), **val_payload}, indent=2),
                    encoding="utf-8",
                )
                write_predictions(
                    output / "val_predictions_last.csv", labels, probs, ids, lesions,
                    config["data"]["class_names"],
                )
        summary = {
            "status": "completed",
            "run_id": run_id,
            "best_epoch": json.loads((output / "val_metrics_best.json").read_text())["epoch"],
            "epochs_completed": int(config["training"]["epochs"]),
            "primary_checkpoint": "last.pt",
            "secondary_checkpoint": "best.pt",
            "checkpoint_policy": "last",
            "monitor": monitor,
            "best_monitor_value": best_value,
            "elapsed_seconds": time.monotonic() - started,
            "effective_batch_size": batch * accumulation,
            "test_evaluated": False,
            "run_signature": signature,
            "git_commit": commit,
            "protocol_version": config["data"]["protocol_version"],
            "loss_method": loss_config.get("method", "cross_entropy"),
            "classifier_type": classifier_type,
        }
        (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        if mlflow:
            for artifact in ("config.resolved.yaml", "environment.json", "split_manifest.json", "run_signature.json", "class_counts.json", "model_initialization.json", "loss_initialization.json", "metrics.csv", "val_metrics_best.json", "val_predictions_best.csv", "val_metrics_last.json", "val_predictions_last.csv", "summary.json"):
                mlflow.log_artifact(str(output / artifact))
            mlflow.set_tag("checkpoint_server_path", str(output / "best.pt"))
            mlflow.set_tag("primary_checkpoint_server_path", str(output / "last.pt"))
    finally:
        if mlflow and mlflow_run:
            mlflow.end_run()
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    print(train(parse_args().config))
