from __future__ import annotations

import argparse
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from timm.optim import create_optimizer_v2
from timm.utils import ModelEmaV3
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, LinearLR, SequentialLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .config import load_config, save_config
from .datasets import make_dataloaders
from .logging_utils import append_csv, append_jsonl, setup_logging, write_json
from .losses import build_loss
from .metrics import compute_metrics, softmax
from .models import configure_classifier_only, create_model, load_initial_checkpoint
from .reproducibility import collect_environment, set_seed
from .tracking import MlflowTracker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("overrides", nargs="*", help="Dotlist overrides, e.g. training.batch_size=128")
    return parser.parse_args()


def make_run_dir(config: dict[str, Any]) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    seed = config["runtime"]["seed"]
    output_dir = Path(config["experiment"]["output_dir"]) / config["experiment"]["name"] / f"{stamp}_{seed}"
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def resolve_device(config: dict[str, Any]) -> torch.device:
    requested = config["runtime"]["device"]
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "runtime.device=cuda was requested, but CUDA is unavailable. "
            "Refusing silent CPU fallback; recreate the GPU container and retry."
        )
    return torch.device(requested)


def autocast_dtype(config: dict[str, Any]) -> torch.dtype | None:
    amp = str(config["runtime"]["amp"]).lower()
    if amp == "bf16":
        return torch.bfloat16
    if amp == "fp16":
        return torch.float16
    return None


def make_optimizer(config: dict[str, Any], model: nn.Module) -> torch.optim.Optimizer:
    train = config["training"]
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("Model has no trainable parameters")
    layer_decay = float(train.get("layer_decay", 1.0))
    if not 0.0 < layer_decay <= 1.0:
        raise ValueError("training.layer_decay must be in (0, 1]")
    if layer_decay < 1.0:
        optimizer = create_optimizer_v2(
            model,
            opt=str(train["optimizer"]),
            lr=float(train["lr"]),
            weight_decay=float(train["weight_decay"]),
            momentum=0.9,
            layer_decay=layer_decay,
        )
        for group in optimizer.param_groups:
            group["lr"] = float(train["lr"]) * float(group.get("lr_scale", 1.0))
        return optimizer
    if train["optimizer"] == "adamw":
        return torch.optim.AdamW(parameters, lr=float(train["lr"]), weight_decay=float(train["weight_decay"]))
    if train["optimizer"] == "sgd":
        return torch.optim.SGD(
            parameters,
            lr=float(train["lr"]),
            momentum=0.9,
            weight_decay=float(train["weight_decay"]),
            nesterov=True,
        )
    raise ValueError(f"Unknown optimizer: {train['optimizer']}")


def optimizer_group_metadata(optimizer: torch.optim.Optimizer) -> list[dict[str, Any]]:
    base_lr = max(float(group["lr"]) for group in optimizer.param_groups)
    return [
        {
            "group_index": index,
            "parameter_tensors": len(group["params"]),
            "parameter_count": sum(parameter.numel() for parameter in group["params"]),
            "lr": float(group["lr"]),
            "lr_scale": float(group.get("lr_scale", float(group["lr"]) / base_lr if base_lr > 0 else 1.0)),
            "weight_decay": float(group.get("weight_decay", 0.0)),
        }
        for index, group in enumerate(optimizer.param_groups)
    ]


def make_scheduler(config: dict[str, Any], optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler.LRScheduler:
    train = config["training"]
    if train["scheduler"] != "cosine":
        raise ValueError(f"Unknown scheduler: {train['scheduler']}")
    epochs = int(train["epochs"])
    warmup = int(train["warmup_epochs"])
    if float(train.get("layer_decay", 1.0)) < 1.0:
        min_factor = float(train["min_lr"]) / float(train["lr"])

        def schedule_factor(epoch: int) -> float:
            if warmup > 0 and epoch < warmup:
                return 0.01 + 0.99 * epoch / warmup
            progress = (epoch - warmup) / max(1, epochs - warmup)
            progress = min(max(progress, 0.0), 1.0)
            return min_factor + 0.5 * (1.0 - min_factor) * (1.0 + math.cos(math.pi * progress))

        return LambdaLR(optimizer, lr_lambda=schedule_factor)
    cosine = CosineAnnealingLR(optimizer, T_max=max(1, epochs - warmup), eta_min=float(train["min_lr"]))
    if warmup <= 0:
        return cosine
    warm = LinearLR(optimizer, start_factor=0.01, total_iters=warmup)
    return SequentialLR(optimizer, schedulers=[warm, cosine], milestones=[warmup])


def class_counts_for_loss(bundle: Any) -> list[int]:
    counts = bundle.class_counts["train"]
    return [int(counts[bundle.idx_to_class[idx]]) for idx in range(len(bundle.idx_to_class))]


def ema_update_step(
    epoch: int,
    batch_step: int,
    batches_per_epoch: int,
    accumulation_steps: int,
) -> int:
    updates_per_epoch = math.ceil(batches_per_epoch / accumulation_steps)
    updates_in_epoch = math.ceil(batch_step / accumulation_steps)
    return (epoch - 1) * updates_per_epoch + updates_in_epoch


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    config: dict[str, Any],
    scaler: torch.amp.GradScaler | None,
    epoch: int,
    synthetic_weight_by_class: torch.Tensor | None = None,
    model_ema: ModelEmaV3 | None = None,
) -> dict[str, float]:
    if bool(config["training"].get("classifier_only", False)):
        model.eval()
        model_for_classifier = model._orig_mod if hasattr(model, "_orig_mod") else model
        model_for_classifier.get_classifier().train()
    else:
        model.train()
    if hasattr(loader.sampler, "set_epoch"):
        loader.sampler.set_epoch(epoch)
    dtype = autocast_dtype(config)
    total_loss = 0.0
    total_examples = 0
    accumulation = int(config["training"]["accumulation_steps"])
    optimizer.zero_grad(set_to_none=True)

    progress = tqdm(loader, desc=f"train {epoch}", leave=False)
    synthetic_weight = float(config["training"].get("synthetic_weight", 1.0))
    use_sample_weights = bool(config["training"].get("use_sample_weights", False))
    use_weighting = (
        synthetic_weight != 1.0
        or synthetic_weight_by_class is not None
        or use_sample_weights
    )
    for step, batch in enumerate(progress, start=1):
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        if bool(config["runtime"]["channels_last"]) and images.ndim == 4:
            images = images.to(memory_format=torch.channels_last)

        with torch.amp.autocast(device_type=device.type, dtype=dtype, enabled=dtype is not None and device.type == "cuda"):
            logits = model(images)
            loss_raw = criterion(logits, targets)
            if use_weighting:
                is_synthetic = batch["is_synthetic"].to(device, non_blocking=True).float()
                weights = batch["sample_weight"].to(device, non_blocking=True).float()
                if synthetic_weight_by_class is not None:
                    class_weights = synthetic_weight_by_class.to(device=device, dtype=is_synthetic.dtype)[targets]
                else:
                    class_weights = torch.full_like(is_synthetic, synthetic_weight)
                synthetic_weights = torch.where(
                    is_synthetic > 0,
                    class_weights,
                    torch.ones_like(is_synthetic),
                )
                weights = weights * synthetic_weights
                loss = (loss_raw * weights).sum() / weights.sum().clamp_min(1.0)
            else:
                loss = loss_raw.mean() if loss_raw.ndim > 0 else loss_raw
            loss = loss / accumulation

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if step % accumulation == 0 or step == len(loader):
            if float(config["training"]["max_grad_norm"]) > 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["training"]["max_grad_norm"]))
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            if model_ema is not None:
                model_ema.update(
                    model,
                    step=ema_update_step(epoch, step, len(loader), accumulation),
                )
            optimizer.zero_grad(set_to_none=True)

        examples = targets.numel()
        total_loss += float(loss.detach().cpu()) * accumulation * examples
        total_examples += examples
        progress.set_postfix(loss=total_loss / max(1, total_examples))

    return {"loss": total_loss / max(1, total_examples)}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    config: dict[str, Any],
    idx_to_class: dict[int, str],
) -> tuple[dict[str, Any], pd.DataFrame]:
    model.eval()
    dtype = autocast_dtype(config)
    logits_all: list[np.ndarray] = []
    targets_all: list[np.ndarray] = []
    paths: list[str] = []
    synthetic_flags: list[int] = []
    total_loss = 0.0
    total_examples = 0

    for batch in tqdm(loader, desc="eval", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        if bool(config["runtime"]["channels_last"]) and images.ndim == 4:
            images = images.to(memory_format=torch.channels_last)
        with torch.amp.autocast(device_type=device.type, dtype=dtype, enabled=dtype is not None and device.type == "cuda"):
            logits = model(images)
            loss = criterion(logits, targets)
        total_loss += float(loss.detach().cpu()) * targets.numel()
        total_examples += targets.numel()
        logits_all.append(logits.detach().float().cpu().numpy())
        targets_all.append(targets.detach().cpu().numpy())
        paths.extend(batch["path"])
        synthetic_flags.extend([int(x) for x in batch["is_synthetic"]])

    logits_np = np.concatenate(logits_all)
    targets_np = np.concatenate(targets_all)
    metrics = compute_metrics(logits_np, targets_np, idx_to_class, int(config["evaluation"]["ece_bins"]))
    metrics["loss"] = total_loss / max(1, total_examples)

    probs = softmax(logits_np)
    prediction = probs.argmax(axis=1)
    rows = {
        "path": paths,
        "target": [idx_to_class[int(x)] for x in targets_np],
        "prediction": [idx_to_class[int(x)] for x in prediction],
        "confidence": probs.max(axis=1),
        "is_synthetic": synthetic_flags,
    }
    for idx, name in idx_to_class.items():
        rows[f"prob_{name}"] = probs[:, idx]
        rows[f"logit_{name}"] = logits_np[:, idx]
    predictions = pd.DataFrame(rows)
    return metrics, predictions


def flatten_metrics(prefix: str, metrics: dict[str, Any]) -> dict[str, float]:
    flat = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            flat[f"{prefix}/{key}"] = float(value)
    return flat


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    best_metric: float,
    config: dict[str, Any],
    class_to_idx: dict[str, int],
    include_optimizer: bool,
    weights_source: str = "online",
) -> None:
    model_to_save = model._orig_mod if hasattr(model, "_orig_mod") else model
    checkpoint = {
        "epoch": epoch,
        "model": model_to_save.state_dict(),
        "best_metric": best_metric,
        "config": config,
        "class_to_idx": class_to_idx,
        "weights_source": weights_source,
    }
    if include_optimizer:
        checkpoint["optimizer"] = optimizer.state_dict()
        checkpoint["scheduler"] = scheduler.state_dict()
    torch.save(checkpoint, path)


def load_model_state(model: nn.Module, state_dict: dict[str, torch.Tensor]) -> None:
    target = model._orig_mod if hasattr(model, "_orig_mod") else model
    target.load_state_dict(state_dict)


def synthetic_weight_vector(config: dict[str, Any], class_to_idx: dict[str, int], device: torch.device) -> torch.Tensor | None:
    weights = config["training"].get("synthetic_weight_per_class")
    if not weights:
        return None
    default = float(config["training"].get("synthetic_weight", 1.0))
    values = torch.full((len(class_to_idx),), default, dtype=torch.float32, device=device)
    for label, value in weights.items():
        if label not in class_to_idx:
            raise ValueError(f"Unknown class in training.synthetic_weight_per_class: {label}")
        values[class_to_idx[label]] = float(value)
    return values


def main() -> None:
    args = parse_args()
    config = load_config(args.config, args.overrides)
    run_dir = make_run_dir(config)
    logger = setup_logging(run_dir)
    save_config(config, run_dir / "config.resolved.yaml")

    set_seed(int(config["runtime"]["seed"]), bool(config["runtime"]["deterministic"]))
    if config["runtime"]["matmul_precision"]:
        torch.set_float32_matmul_precision(str(config["runtime"]["matmul_precision"]))

    environment = collect_environment()
    write_json(run_dir / "environment.json", environment)
    logger.info("Run directory: %s", run_dir)
    logger.info("Environment: %s", environment)

    device = resolve_device(config)
    bundle = make_dataloaders(config)
    write_json(run_dir / "class_to_idx.json", bundle.class_to_idx)
    write_json(run_dir / "class_counts.json", bundle.class_counts)
    write_json(run_dir / "sampling_plan.json", bundle.sampling_plan)
    logger.info("Class counts: %s", bundle.class_counts)
    logger.info("Sampling plan: %s", bundle.sampling_plan)

    model = create_model(config, num_classes=len(bundle.class_to_idx))
    initialization = load_initial_checkpoint(model, config["model"].get("initial_checkpoint"))
    trainable_parameters: list[str] = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    if bool(config["training"].get("classifier_only", False)):
        trainable_parameters = configure_classifier_only(
            model,
            num_classes=len(bundle.class_to_idx),
            reinitialize_classifier=bool(
                config["training"].get("reinitialize_classifier", False)
            ),
        )
    initialization.update(
        {
            "classifier_only": bool(config["training"].get("classifier_only", False)),
            "reinitialize_classifier": bool(
                config["training"].get("reinitialize_classifier", False)
            ),
            "trainable_parameter_count": sum(
                parameter.numel() for parameter in model.parameters() if parameter.requires_grad
            ),
            "total_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        }
    )
    write_json(run_dir / "model_initialization.json", initialization)
    write_json(run_dir / "trainable_parameters.json", trainable_parameters)
    model = model.to(device)
    if bool(config["runtime"]["channels_last"]):
        model = model.to(memory_format=torch.channels_last)
    use_model_ema = bool(config["training"].get("model_ema", False))
    if use_model_ema and bool(config["runtime"]["compile"]):
        raise ValueError("training.model_ema=true is not supported with runtime.compile=true")
    if float(config["training"].get("layer_decay", 1.0)) < 1.0 and bool(config["runtime"]["compile"]):
        raise ValueError("training.layer_decay<1 is not supported with runtime.compile=true")
    model_ema = (
        ModelEmaV3(
            model,
            decay=float(config["training"].get("model_ema_decay", 0.9999)),
            use_warmup=bool(config["training"].get("model_ema_warmup", True)),
        )
        if use_model_ema
        else None
    )
    if bool(config["runtime"]["compile"]) and hasattr(torch, "compile"):
        model = torch.compile(model)

    synthetic_weight = float(config["training"].get("synthetic_weight", 1.0))
    synthetic_weight_by_class = synthetic_weight_vector(config, bundle.class_to_idx, device)
    if synthetic_weight_by_class is not None:
        write_json(
            run_dir / "synthetic_weight_by_class.json",
            {label: float(synthetic_weight_by_class[idx].detach().cpu()) for label, idx in bundle.class_to_idx.items()},
        )
        logger.info(
            "Synthetic weights by class: %s",
            {label: float(synthetic_weight_by_class[idx].detach().cpu()) for label, idx in bundle.class_to_idx.items()},
        )
    train_criterion = build_loss(
        config,
        class_counts_for_loss(bundle),
        device,
        reduction=(
            "none"
            if synthetic_weight != 1.0
            or synthetic_weight_by_class is not None
            or bool(config["training"].get("use_sample_weights", False))
            else "mean"
        ),
        label_smoothing=float(config["training"].get("label_smoothing", 0.0)),
    )
    eval_criterion = build_loss(config, class_counts_for_loss(bundle), device, reduction="mean")
    optimizer = make_optimizer(config, model)
    write_json(run_dir / "optimizer_groups.json", optimizer_group_metadata(optimizer))
    scheduler = make_scheduler(config, optimizer)
    dtype = autocast_dtype(config)
    scaler = torch.amp.GradScaler("cuda") if dtype == torch.float16 and device.type == "cuda" else None
    writer = SummaryWriter(log_dir=str(run_dir / "tensorboard"))
    tracker = MlflowTracker(config, run_dir, logger)
    tracker.start()

    monitor = str(config["training"]["monitor"])
    mode = str(config["training"]["monitor_mode"])
    best_metric = -math.inf if mode == "max" else math.inf
    best_epoch: int | None = None
    bad_epochs = 0
    start = time.time()

    for epoch in range(1, int(config["training"]["epochs"]) + 1):
        train_metrics = train_one_epoch(
            model,
            bundle.loaders["train"],
            train_criterion,
            optimizer,
            device,
            config,
            scaler,
            epoch,
            synthetic_weight_by_class,
            model_ema,
        )
        selection_model = model_ema.module if model_ema is not None else model
        val_metrics, val_predictions = evaluate(
            selection_model,
            bundle.loaders["val"],
            eval_criterion,
            device,
            config,
            bundle.idx_to_class,
        )
        scheduler.step()

        group_lrs = [float(group["lr"]) for group in optimizer.param_groups]
        metrics_row = {
            "epoch": epoch,
            "lr": max(group_lrs),
            "lr_min": min(group_lrs),
            "lr_max": max(group_lrs),
        }
        metrics_row.update(flatten_metrics("train", train_metrics))
        metrics_row.update(flatten_metrics("val", val_metrics))
        append_csv(run_dir / "metrics.csv", metrics_row)
        append_jsonl(run_dir / "metrics.jsonl", metrics_row)
        tracker.log_epoch(metrics_row)
        for key, value in metrics_row.items():
            if key != "epoch":
                writer.add_scalar(key, value, epoch)

        current = metrics_row[monitor]
        improved = current > best_metric if mode == "max" else current < best_metric
        if improved:
            best_metric = current
            best_epoch = epoch
            bad_epochs = 0
            save_checkpoint(
                run_dir / "best.pt",
                selection_model,
                optimizer,
                scheduler,
                epoch,
                best_metric,
                config,
                bundle.class_to_idx,
                bool(config["training"].get("checkpoint_include_optimizer", True)),
                "ema" if model_ema is not None else "online",
            )
            if bool(config["evaluation"]["save_predictions"]):
                val_predictions.to_csv(run_dir / "val_predictions_best.csv", index=False)
            write_json(run_dir / "val_metrics_best.json", val_metrics)
            if bool(config["evaluation"]["save_confusion_matrix"]):
                write_json(run_dir / "val_confusion_matrix_best.json", val_metrics["confusion_matrix"])
        else:
            bad_epochs += 1

        if bool(config["training"].get("save_last_checkpoint", True)):
            save_checkpoint(
                run_dir / "last.pt",
                selection_model,
                optimizer,
                scheduler,
                epoch,
                best_metric,
                config,
                bundle.class_to_idx,
                bool(config["training"].get("checkpoint_include_optimizer", True)),
                "ema" if model_ema is not None else "online",
            )
        logger.info("Epoch %03d | val %s=%.6f | best=%.6f | bad_epochs=%d", epoch, monitor, current, best_metric, bad_epochs)
        if bad_epochs >= int(config["training"]["early_stopping_patience"]):
            logger.info("Early stopping triggered.")
            break

    test_metrics: dict[str, Any] = {}
    if bool(config["evaluation"].get("run_test", True)):
        checkpoint = torch.load(run_dir / "best.pt", map_location=device)
        load_model_state(model, checkpoint["model"])
        test_metrics, test_predictions = evaluate(
            model,
            bundle.loaders["test"],
            eval_criterion,
            device,
            config,
            bundle.idx_to_class,
        )
        write_json(run_dir / "test_metrics.json", test_metrics)
        if bool(config["evaluation"]["save_predictions"]):
            test_predictions.to_csv(run_dir / "test_predictions.csv", index=False)
        if bool(config["evaluation"]["save_confusion_matrix"]):
            write_json(run_dir / "test_confusion_matrix.json", test_metrics["confusion_matrix"])
    else:
        logger.info("Locked test evaluation skipped by evaluation.run_test=false.")

    elapsed = time.time() - start
    write_json(
        run_dir / "summary.json",
        {
            "best_metric": best_metric,
            "best_epoch": best_epoch,
            "elapsed_seconds": elapsed,
            "test_evaluated": bool(config["evaluation"].get("run_test", True)),
            "weights_source": "ema" if model_ema is not None else "online",
            "test": test_metrics,
        },
    )
    writer.close()
    tracker.finish()
    logger.info("Finished in %.1f minutes. Test metrics: %s", elapsed / 60.0, test_metrics)


if __name__ == "__main__":
    main()
