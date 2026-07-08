from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None, reduction: str = "mean") -> None:
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.register_buffer("weight", weight if weight is not None else None)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, target, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        loss = (1.0 - pt) ** self.gamma * ce
        if self.reduction == "none":
            return loss
        if self.reduction == "sum":
            return loss.sum()
        return loss.mean()


class BalancedSoftmaxLoss(nn.Module):
    def __init__(self, class_counts: torch.Tensor, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction
        self.register_buffer("log_counts", torch.log(class_counts.float().clamp_min(1.0)))

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits + self.log_counts.to(logits.device), target, reduction=self.reduction)


class LogitAdjustedCELoss(nn.Module):
    def __init__(self, class_counts: torch.Tensor, tau: float, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction
        priors = class_counts.float().clamp_min(1.0)
        priors = priors / priors.sum()
        self.tau = tau
        self.register_buffer("log_priors", torch.log(priors))

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits + self.tau * self.log_priors.to(logits.device), target, reduction=self.reduction)


def class_weight_tensor(strategy: str, class_counts: torch.Tensor) -> torch.Tensor | None:
    if strategy in {"none", None}:
        return None
    counts = class_counts.float().clamp_min(1.0)
    if strategy == "inverse":
        weights = 1.0 / counts
    elif strategy == "effective_number":
        beta = 0.9999
        weights = (1.0 - beta) / (1.0 - torch.pow(torch.tensor(beta), counts))
    else:
        raise ValueError(f"Unknown class weight strategy: {strategy}")
    return weights / weights.mean()


def build_loss(config: dict[str, Any], class_counts: list[int], device: torch.device, reduction: str = "mean") -> nn.Module:
    imbalance = config["imbalance"]
    counts = torch.tensor(class_counts, dtype=torch.float32, device=device)
    if bool(imbalance.get("balanced_softmax", False)) or imbalance["loss"] == "balanced_softmax":
        return BalancedSoftmaxLoss(counts, reduction=reduction).to(device)
    if float(imbalance.get("logit_adjustment_tau", 0.0)) > 0 or imbalance["loss"] == "logit_adjustment":
        tau = float(imbalance.get("logit_adjustment_tau", 1.0))
        return LogitAdjustedCELoss(counts, tau=tau, reduction=reduction).to(device)
    weights = class_weight_tensor(imbalance.get("class_weights", "none"), counts)
    if weights is not None:
        weights = weights.to(device)
    if imbalance["loss"] == "focal":
        return FocalLoss(gamma=float(imbalance["focal_gamma"]), weight=weights, reduction=reduction).to(device)
    if imbalance["loss"] == "cross_entropy":
        return nn.CrossEntropyLoss(weight=weights, reduction=reduction).to(device)
    raise ValueError(f"Unknown loss: {imbalance['loss']}")
