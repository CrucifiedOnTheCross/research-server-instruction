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


def load_initial_checkpoint(model: nn.Module, checkpoint_path: str | None) -> dict[str, Any]:
    if not checkpoint_path:
        return {"checkpoint": None, "checkpoint_epoch": None, "checkpoint_best_metric": None}
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint)
    model.load_state_dict(state_dict)
    return {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_best_metric": checkpoint.get("best_metric"),
    }


def configure_classifier_only(
    model: nn.Module,
    num_classes: int,
    reinitialize_classifier: bool,
) -> list[str]:
    if reinitialize_classifier:
        if not hasattr(model, "reset_classifier"):
            raise RuntimeError("classifier reinitialization requires a timm model with reset_classifier")
        model.reset_classifier(num_classes)
    if not hasattr(model, "get_classifier"):
        raise RuntimeError("classifier-only training requires a timm model with get_classifier")
    for parameter in model.parameters():
        parameter.requires_grad = False
    classifier = model.get_classifier()
    if not isinstance(classifier, nn.Module):
        raise RuntimeError("model.get_classifier() did not return a torch module")
    for parameter in classifier.parameters():
        parameter.requires_grad = True
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("classifier-only configuration left no trainable parameters")
    return trainable


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
