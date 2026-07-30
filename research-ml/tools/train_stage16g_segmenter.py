from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models.segmentation import (
    DeepLabV3_ResNet50_Weights,
    deeplabv3_resnet50,
)
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as transform


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Stage 16G lesion segmenter.")
    parser.add_argument(
        "--config", default="configs/stage16g_segmentation_qualification.yaml"
    )
    parser.add_argument(
        "--output-root", default="outputs/stage16g_segmentation"
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def resize_pad(
    image: Image.Image, mask: Image.Image, size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    width, height = image.size
    scale = size / max(width, height)
    resized = (max(1, round(width * scale)), max(1, round(height * scale)))
    image = image.resize(resized, Image.Resampling.BILINEAR)
    mask = mask.resize(resized, Image.Resampling.NEAREST)
    left = (size - resized[0]) // 2
    top = (size - resized[1]) // 2
    image_canvas = Image.new("RGB", (size, size), (0, 0, 0))
    mask_canvas = Image.new("L", (size, size), 0)
    image_canvas.paste(image, (left, top))
    mask_canvas.paste(mask, (left, top))
    image_tensor = transform.to_tensor(image_canvas)
    image_tensor = transform.normalize(
        image_tensor,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    mask_tensor = (transform.pil_to_tensor(mask_canvas).float() / 255.0 >= 0.5).float()
    return image_tensor, mask_tensor


class SegmentationDataset(Dataset):
    def __init__(self, root: Path, rows: list[dict[str, str]], size: int, train: bool):
        self.root = root
        self.rows = rows
        self.size = size
        self.train = train

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.rows[index]
        with Image.open(self.root / row["image_path"]) as handle:
            image = handle.convert("RGB")
        with Image.open(self.root / row["mask_path"]) as handle:
            mask = handle.convert("L")
        if self.train and random.random() < 0.5:
            image = transform.hflip(image)
            mask = transform.hflip(mask)
        if self.train and random.random() < 0.5:
            angle = random.uniform(-20.0, 20.0)
            image = transform.rotate(
                image, angle, interpolation=InterpolationMode.BILINEAR
            )
            mask = transform.rotate(
                mask, angle, interpolation=InterpolationMode.NEAREST
            )
        return resize_pad(image, mask, self.size)


def dice_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    probabilities = logits.sigmoid()
    intersection = (probabilities * targets).sum(dim=(1, 2, 3))
    denominator = probabilities.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    return 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()


def binary_metrics(
    logits: torch.Tensor, targets: torch.Tensor, threshold: float
) -> tuple[float, float]:
    prediction = logits.sigmoid() >= threshold
    truth = targets >= 0.5
    intersection = (prediction & truth).sum(dim=(1, 2, 3)).float()
    pred_sum = prediction.sum(dim=(1, 2, 3)).float()
    truth_sum = truth.sum(dim=(1, 2, 3)).float()
    union = (prediction | truth).sum(dim=(1, 2, 3)).float()
    dice = ((2 * intersection + 1) / (pred_sum + truth_sum + 1)).mean()
    iou = ((intersection + 1) / (union + 1)).mean()
    return float(dice), float(iou)


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    threshold: float,
) -> tuple[float, float, float]:
    model.train(optimizer is not None)
    losses: list[float] = []
    dices: list[float] = []
    ious: list[float] = []
    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        with torch.set_grad_enabled(optimizer is not None):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(images)["out"]
                loss = functional.binary_cross_entropy_with_logits(logits, masks)
                loss = loss + dice_loss(logits, masks)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        dice, iou = binary_metrics(logits.detach(), masks, threshold)
        losses.append(float(loss.detach()))
        dices.append(dice)
        ious.append(iou)
    return float(np.mean(losses)), float(np.mean(dices)), float(np.mean(ious))


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    seed = int(config["experiment"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    if not torch.cuda.is_available():
        raise RuntimeError("Stage 16G segmenter requires CUDA")

    data_root = Path(config["data"]["root"])
    train_rows = read_csv(data_root / config["data"]["train_manifest"])
    val_rows = read_csv(data_root / config["data"]["validation_manifest"])
    size = int(config["data"]["image_size"])
    batch_size = int(config["training"]["batch_size"])
    workers = int(config["training"]["num_workers"])
    train_loader = DataLoader(
        SegmentationDataset(data_root, train_rows, size, train=True),
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        drop_last=True,
    )
    val_loader = DataLoader(
        SegmentationDataset(data_root, val_rows, size, train=False),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
    )
    model = deeplabv3_resnet50(
        weights=DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1,
        aux_loss=True,
    )
    model.classifier[-1] = torch.nn.Conv2d(256, 1, kernel_size=1)
    device = torch.device("cuda")
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    threshold = float(config["evaluation"]["threshold"])
    output = Path(args.output_root) / config["experiment"]["name"]
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    (output / "model_initialization.json").write_text(
        json.dumps(
            {
                "architecture": config["model"]["architecture"],
                "weights": config["model"]["pretrained_weights"],
                "seed": seed,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    history: list[dict[str, float | int]] = []
    best_dice = -1.0
    best_epoch = 0
    patience = 0
    started = time.time()
    for epoch in range(1, int(config["training"]["epochs"]) + 1):
        train_loss, train_dice, train_iou = run_epoch(
            model, train_loader, device, optimizer, threshold
        )
        with torch.no_grad():
            val_loss, val_dice, val_iou = run_epoch(
                model, val_loader, device, None, threshold
            )
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_dice": train_dice,
            "train_iou": train_iou,
            "val_loss": val_loss,
            "val_dice": val_dice,
            "val_iou": val_iou,
            "elapsed_seconds": time.time() - started,
        }
        history.append(row)
        with (output / "metrics.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(history)
        if val_dice > best_dice:
            best_dice = val_dice
            best_epoch = epoch
            patience = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "val_dice": val_dice,
                    "val_iou": val_iou,
                },
                output / "best.pt",
            )
            (output / "val_metrics_best.json").write_text(
                json.dumps(row, indent=2), encoding="utf-8"
            )
        else:
            patience += 1
        if patience >= int(config["training"]["early_stopping_patience"]):
            break

    summary = {
        "status": "complete",
        "test_evaluated": False,
        "best_epoch": best_epoch,
        "best_val_dice": best_dice,
        "train_images": len(train_rows),
        "qualification_images": len(val_rows),
        "elapsed_seconds": time.time() - started,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
