#!/usr/bin/env python3
"""Run one finite forward/backward step for every Stage 3A objective."""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from torchvision.models import ResNet50_Weights, resnet50

from longtail_medical.data import ManifestDataset, build_transforms
from longtail_medical.losses import LongTailObjective, NormedLinear
from tools.run_stage3a_training_matrix import ARMS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output", type=Path, default=Path("outputs/stage3a_smoke.json"))
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    config = yaml.safe_load(
        (project / "configs" / "stage3a_lesion_disjoint_screening.yaml").read_text(encoding="utf-8")
    )
    dataset = ManifestDataset(project / config["data"]["train_csv"], build_transforms(config, train=True))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)
    images, labels, _, _ = next(iter(loader))
    counts = Counter(int(row["label"]) for row in dataset.rows)
    ordered_counts = [counts[index] for index in range(config["data"]["num_classes"])]
    device = torch.device("cuda")
    images = images.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
    labels = labels.to(device, non_blocking=True)
    results = []
    for name, overrides in ARMS:
        loss_config = copy.deepcopy(config["loss"])
        loss_config.update(overrides)
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        features = model.fc.in_features
        model.fc = (
            NormedLinear(features, config["data"]["num_classes"])
            if loss_config["method"] == "ldam_drw"
            else torch.nn.Linear(features, config["data"]["num_classes"])
        )
        model = model.to(device, memory_format=torch.channels_last).train()
        objective = LongTailObjective(loss_config, ordered_counts).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = objective(model(images), labels, epoch=41)
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        finite = bool(torch.isfinite(loss))
        if not finite:
            raise RuntimeError(f"Non-finite Stage 3A smoke loss: {name}")
        results.append({
            "arm": name,
            "method": loss_config["method"],
            "loss": float(loss.detach()),
            "finite": finite,
        })
        del optimizer, objective, model
        torch.cuda.empty_cache()
    payload = {
        "status": "passed",
        "batch_size": args.batch_size,
        "gpu": torch.cuda.get_device_name(0),
        "cuda_capability": list(torch.cuda.get_device_capability(0)),
        "methods": results,
        "test_loaded": False,
        "test_evaluated": False,
    }
    destination = project / args.output
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
