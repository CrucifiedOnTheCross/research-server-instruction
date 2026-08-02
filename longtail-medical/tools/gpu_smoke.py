#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models import ResNet50_Weights, resnet50

from longtail_medical.config import load_config
from longtail_medical.data import ManifestDataset, build_transforms


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    project = Path(__file__).resolve().parents[1]
    manifest = Path(config["data"]["train_csv"])
    if not manifest.is_absolute():
        manifest = project / manifest
    dataset = ManifestDataset(manifest, build_transforms(config, train=True))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)
    images, labels, _, _ = next(iter(loader))
    device = torch.device("cuda")
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    model.fc = torch.nn.Linear(model.fc.in_features, config["data"]["num_classes"])
    model = model.to(device, memory_format=torch.channels_last).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])
    images = images.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
    labels = labels.to(device, non_blocking=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(images)
        loss = F.cross_entropy(logits, labels)
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    result = {
        "status": "passed",
        "batch_size": args.batch_size,
        "loss": float(loss),
        "finite_loss": bool(torch.isfinite(loss)),
        "gpu": torch.cuda.get_device_name(0),
        "max_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
        "test_evaluated": False,
    }
    if not result["finite_loss"]:
        raise RuntimeError("Non-finite smoke loss")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
