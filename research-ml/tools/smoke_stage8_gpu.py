from __future__ import annotations

import argparse
import json

import torch

from src.config import load_config
from src.models import create_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify Stage 8 model forward/backward memory on the target GPU.")
    parser.add_argument("--config", default="configs/ham10000_stage8_real_ce_weighted_384.yaml")
    parser.add_argument("--batch-size", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    batch_size = int(args.batch_size or config["training"]["batch_size"])
    image_size = int(config["data"]["image_size"])
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    model = create_model(config, num_classes=7).to(device, memory_format=torch.channels_last)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    images = torch.randn(
        batch_size,
        3,
        image_size,
        image_size,
        device=device,
        dtype=torch.float32,
    ).to(memory_format=torch.channels_last)
    targets = torch.randint(0, 7, (batch_size,), device=device)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        logits = model(images)
        loss = torch.nn.functional.cross_entropy(logits, targets)
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    report = {
        "model": config["model"]["name"],
        "batch_size": batch_size,
        "image_size": image_size,
        "loss": float(loss.detach().cpu()),
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
        "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 1024**3,
        "gpu": torch.cuda.get_device_name(device),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
