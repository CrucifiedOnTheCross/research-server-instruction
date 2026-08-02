"""Preregistered long-tail objectives used by Stage 3A."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


SUPPORTED_METHODS = {
    "cross_entropy",
    "weighted_cross_entropy",
    "focal",
    "class_balanced_focal",
    "balanced_softmax",
    "ldam_drw",
}


def normalized_inverse_frequency(counts: torch.Tensor) -> torch.Tensor:
    weights = counts.sum() / counts
    return weights / weights.mean()


def effective_number_weights(counts: torch.Tensor, beta: float) -> torch.Tensor:
    effective = 1.0 - torch.pow(torch.full_like(counts, beta), counts)
    weights = (1.0 - beta) / effective
    return weights / weights.mean()


class LongTailObjective(nn.Module):
    def __init__(self, config: Mapping, class_counts: Sequence[int]):
        super().__init__()
        method = str(config.get("method", "cross_entropy"))
        if method not in SUPPORTED_METHODS:
            raise ValueError(f"Unsupported long-tail objective: {method}")
        counts = torch.as_tensor(class_counts, dtype=torch.float32)
        if counts.ndim != 1 or torch.any(counts <= 0):
            raise ValueError("Every class must have a positive training count")
        self.method = method
        self.gamma = float(config.get("gamma", 2.0))
        self.beta = float(config.get("beta", 0.9999))
        self.max_margin = float(config.get("max_margin", 0.5))
        self.scale = float(config.get("scale", 30.0))
        self.drw_start_epoch = int(config.get("drw_start_epoch", 40))
        self.register_buffer("class_counts", counts)
        self.register_buffer("inverse_weights", normalized_inverse_frequency(counts))
        self.register_buffer("effective_weights", effective_number_weights(counts, self.beta))
        margins = torch.pow(counts, -0.25)
        margins = margins * (self.max_margin / margins.max())
        self.register_buffer("margins", margins)

    def forward(self, logits: torch.Tensor, target: torch.Tensor, epoch: int) -> torch.Tensor:
        if self.method == "cross_entropy":
            return F.cross_entropy(logits, target)
        if self.method == "weighted_cross_entropy":
            return F.cross_entropy(logits, target, weight=self.inverse_weights)
        if self.method in {"focal", "class_balanced_focal"}:
            ce = F.cross_entropy(logits, target, reduction="none")
            focal = torch.pow(1.0 - torch.exp(-ce), self.gamma) * ce
            if self.method == "class_balanced_focal":
                focal = focal * self.effective_weights[target]
            return focal.mean()
        if self.method == "balanced_softmax":
            return F.cross_entropy(logits + self.class_counts.log().unsqueeze(0), target)
        if self.method == "ldam_drw":
            adjusted = logits.clone()
            rows = torch.arange(len(target), device=target.device)
            adjusted[rows, target] -= self.margins[target]
            weights = self.effective_weights if epoch > self.drw_start_epoch else None
            return F.cross_entropy(self.scale * adjusted, target, weight=weights)
        raise AssertionError(self.method)

    def metadata(self) -> dict:
        return {
            "method": self.method,
            "class_counts": self.class_counts.tolist(),
            "inverse_frequency_weights_mean_one": self.inverse_weights.tolist(),
            "effective_number_beta": self.beta,
            "effective_number_weights_mean_one": self.effective_weights.tolist(),
            "focal_gamma": self.gamma,
            "ldam_max_margin": self.max_margin,
            "ldam_margins": self.margins.tolist(),
            "ldam_scale": self.scale,
            "drw_start_epoch_exclusive": self.drw_start_epoch,
        }


class NormedLinear(nn.Module):
    """Cosine classifier used by the canonical LDAM implementation."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.uniform_(self.weight, -1.0, 1.0)
        with torch.no_grad():
            self.weight.renorm_(2, 0, 1e-5).mul_(1e5)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return F.linear(F.normalize(features, dim=1), F.normalize(self.weight, dim=1))


def prediction_scale(loss_config: Mapping) -> float:
    return float(loss_config.get("scale", 30.0)) if loss_config.get("method") == "ldam_drw" else 1.0
