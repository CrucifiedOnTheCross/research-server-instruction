from __future__ import annotations

from typing import Any

import timm
import torch
from torch import nn


def create_model(config: dict[str, Any], num_classes: int) -> nn.Module:
    model_config = config["model"]
    return timm.create_model(
        model_config["name"],
        pretrained=bool(model_config["pretrained"]),
        num_classes=num_classes,
        drop_rate=float(model_config["drop_rate"]),
        drop_path_rate=float(model_config["drop_path_rate"]),
    )


def extract_features(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    if hasattr(model, "forward_features"):
        features = model.forward_features(images)
        if features.ndim == 4:
            features = features.mean(dim=(2, 3))
        elif features.ndim == 3:
            features = features[:, 0]
        return features
    if hasattr(model, "get_intermediate_layers"):
        features = model.get_intermediate_layers(images, n=1)[0]
        return features.mean(dim=1)
    raise RuntimeError("Model does not expose forward_features; choose a timm-compatible backbone.")

