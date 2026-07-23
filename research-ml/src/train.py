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
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .config import load_config, save_config
from .datasets import make_dataloaders
from .logging_utils import append_csv, append_jsonl, setup_logging, write_json
from .losses import build_loss
from .metrics import compute_metrics, softmax
from .models import create_model
from .reproducibility import collect_environment, set_seed


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
        return torch.device("cpu")
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
    if train["optimizer"] == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=float(train["lr"]), weight_decay=float(train["weight_decay"]))
    if train["optimizer"] == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=float(train["lr"]),
            momentum=0.9,
            weight_decay=float(train["weight_decay"]),
            nesterov=True,
        )
    raise ValueError(f"Unknown optimizer: {train['optimizer']}")


def make_scheduler(config: dict[str, Any], optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler.LRScheduler:
    train = config["training"]
    if train["scheduler"] != "cosine":
        raise ValueError(f"Unknown scheduler: {train['scheduler']}")
    epochs = int(train["epochs"])
    warmup = int(train["warmup_epochs"])
    cosine = CosineAnnealingLR(optimizer, T_max=max(1, epochs - warmup), eta_min=float(train["min_lr"]))
    if warmup <= 0:
        return cosine
    warm = LinearLR(optimizer, start_factor=0.01, total_iters=warmup)
    return SequentialLR(optimizer, schedulers=[warm, cosine], milestones=[warmup])


def class_counts_for_loss(bundle: Any) -> list[int]:
    counts = bundle.class_counts["train"]
    return [int(counts[bundle.idx_to_class[idx]]) for idx in range(len(bundle.idx_to_class))]


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
) -> dict[str, float]:
    model.train()
    dtype = autocast_dtype(config)
    total_loss = 0.0
    total_examples = 0
    accumulation = int(config["training"]["accumulation_steps"])
    optimizer.zero_grad(set_to_none=True)

    progress = tqdm(loader, desc=f"train {epoch}", leave=False)
    synthetic_weight = float(config["training"].get("synthetic_weight", 1.0))
    use_weighting = synthetic_weight != 1.0 or synthetic_weight_by_class is not None
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
                if synthetic_weight_by_class is not None:
                    class_weights = synthetic_weight_by_class.to(device=device, dtype=is_synthetic.dtype)[targets]
                else:
                    class_weights = torch.full_like(is_synthetic, synthetic_weight)
                weights = torch.where(is_synthetic > 0, class_weights, torch.ones_like(is_synthetic))
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
) -> None:
    model_to_save = model._orig_mod if hasattr(model, "_orig_mod") else model
    torch.save(
        {
            "epoch": epoch,
            "model": model_to_save.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_metric": best_metric,
            "config": config,
            "class_to_idx": class_to_idx,
        },
        path,
    )


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
    logger.info("Class counts: %s", bundle.class_counts)

    model = create_model(config, num_classes=len(bundle.class_to_idx)).to(device)
    if bool(config["runtime"]["channels_last"]):
        model = model.to(memory_format=torch.channels_last)
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
        reduction="none" if synthetic_weight != 1.0 or synthetic_weight_by_class is not None else "mean",
    )
    eval_criterion = build_loss(config, class_counts_for_loss(bundle), device, reduction="mean")
    optimizer = make_optimizer(config, model)
    scheduler = make_scheduler(config, optimizer)
    dtype = autocast_dtype(config)
    scaler = torch.amp.GradScaler("cuda") if dtype == torch.float16 and device.type == "cuda" else None
    writer = SummaryWriter(log_dir=str(run_dir / "tensorboard"))

    monitor = str(config["training"]["monitor"])
    mode = str(config["training"]["monitor_mode"])
    best_metric = -math.inf if mode == "max" else math.inf
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
        )
        val_metrics, val_predictions = evaluate(model, bundle.loaders["val"], eval_criterion, device, config, bundle.idx_to_class)
        scheduler.step()

        metrics_row = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"]}
        metrics_row.update(flatten_metrics("train", train_metrics))
        metrics_row.update(flatten_metrics("val", val_metrics))
        append_csv(run_dir / "metrics.csv", metrics_row)
        append_jsonl(run_dir / "metrics.jsonl", metrics_row)
        for key, value in metrics_row.items():
            if key != "epoch":
                writer.add_scalar(key, value, epoch)

        current = metrics_row[monitor]
        improved = current > best_metric if mode == "max" else current < best_metric
        if improved:
            best_metric = current
            bad_epochs = 0
            save_checkpoint(run_dir / "best.pt", model, optimizer, scheduler, epoch, best_metric, config, bundle.class_to_idx)
            if bool(config["evaluation"]["save_predictions"]):
                val_predictions.to_csv(run_dir / "val_predictions_best.csv", index=False)
            write_json(run_dir / "val_metrics_best.json", val_metrics)
            if bool(config["evaluation"]["save_confusion_matrix"]):
                write_json(run_dir / "val_confusion_matrix_best.json", val_metrics["confusion_matrix"])
        else:
            bad_epochs += 1

        save_checkpoint(run_dir / "last.pt", model, optimizer, scheduler, epoch, best_metric, config, bundle.class_to_idx)
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
            "elapsed_seconds": elapsed,
            "test_evaluated": bool(config["evaluation"].get("run_test", True)),
            "test": test_metrics,
        },
    )
    writer.close()
    logger.info("Finished in %.1f minutes. Test metrics: %s", elapsed / 60.0, test_metrics)


if __name__ == "__main__":
    main()
