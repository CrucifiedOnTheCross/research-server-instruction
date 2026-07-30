from __future__ import annotations

import hashlib
from collections import defaultdict

import numpy as np
from scipy import ndimage


def mask_characteristics(
    probability: np.ndarray, threshold: float
) -> dict[str, float | int]:
    mask = probability >= threshold
    labels, components = ndimage.label(mask)
    component_sizes = np.bincount(labels.ravel())[1:]
    foreground = int(mask.sum())
    largest = int(component_sizes.max()) if component_sizes.size else 0
    border = np.concatenate([mask[0], mask[-1], mask[:, 0], mask[:, -1]])
    confidence = float(probability[mask].mean()) if foreground else 0.0
    return {
        "area_fraction": foreground / mask.size,
        "foreground_confidence": confidence,
        "component_count": int(components),
        "largest_component_fraction": largest / max(foreground, 1),
        "border_foreground_fraction": float(border.mean()),
    }


def overlap_metrics(prediction: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    prediction = prediction.astype(bool)
    truth = truth.astype(bool)
    intersection = int(np.logical_and(prediction, truth).sum())
    union = int(np.logical_or(prediction, truth).sum())
    return {
        "dice": (2 * intersection + 1) / (prediction.sum() + truth.sum() + 1),
        "iou": (intersection + 1) / (union + 1),
        "gt_area_fraction": float(truth.mean()),
    }


def deterministic_group_sample(
    rows: list[dict[str, str]], classes: list[str], maximum: int
) -> list[dict[str, str]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if str(row["label"]) in classes:
            grouped[str(row["label"])].append(row)
    selected: list[dict[str, str]] = []
    for label in classes:
        unique: dict[str, dict[str, str]] = {}
        for row in grouped[label]:
            group = str(row.get("group_id") or row["image_id"])
            unique.setdefault(group, row)
        ordered = sorted(
            unique.values(),
            key=lambda row: hashlib.sha256(
                f"{label}:{row.get('group_id') or row['image_id']}".encode()
            ).hexdigest(),
        )
        selected.extend(ordered[:maximum])
    return selected
