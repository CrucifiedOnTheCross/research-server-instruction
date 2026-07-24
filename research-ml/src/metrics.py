from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)


def expected_calibration_error(probs: np.ndarray, y_true: np.ndarray, n_bins: int) -> float:
    confidence = probs.max(axis=1)
    prediction = probs.argmax(axis=1)
    correct = prediction == y_true
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for start, end in zip(bins[:-1], bins[1:]):
        mask = (confidence > start) & (confidence <= end)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(ece)


def brier_multiclass(probs: np.ndarray, y_true: np.ndarray, num_classes: int) -> float:
    if num_classes == 2:
        return float(brier_score_loss(y_true, probs[:, 1]))
    one_hot = np.eye(num_classes)[y_true]
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


def compute_metrics(
    logits: np.ndarray,
    y_true: np.ndarray,
    idx_to_class: dict[int, str],
    ece_bins: int,
) -> dict[str, Any]:
    probs = softmax(logits)
    y_pred = probs.argmax(axis=1)
    labels = list(range(len(idx_to_class)))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "nll": float(log_loss(y_true, probs, labels=labels)),
        "brier": brier_multiclass(probs, y_true, len(labels)),
        "ece": expected_calibration_error(probs, y_true, ece_bins),
        "worst_class_recall": float(np.min(recall)) if len(recall) else 0.0,
    }
    try:
        if len(labels) == 2:
            metrics["auroc"] = float(roc_auc_score(y_true, probs[:, 1]))
            metrics["auprc"] = float(average_precision_score(y_true, probs[:, 1]))
        else:
            metrics["auroc_ovr_macro"] = float(roc_auc_score(y_true, probs, multi_class="ovr", average="macro"))
            one_hot = np.eye(len(labels))[y_true]
            metrics["auprc_ovr_macro"] = float(average_precision_score(one_hot, probs, average="macro"))
    except ValueError:
        pass

    per_class = {}
    for idx, name in idx_to_class.items():
        class_metrics = {
            "precision": float(precision[idx]),
            "recall": float(recall[idx]),
            "f1": float(f1[idx]),
            "support": int(support[idx]),
        }
        binary_target = (y_true == idx).astype(int)
        if np.unique(binary_target).size == 2:
            class_metrics["auroc"] = float(roc_auc_score(binary_target, probs[:, idx]))
            class_metrics["auprc"] = float(
                average_precision_score(binary_target, probs[:, idx])
            )
        per_class[name] = class_metrics
    metrics["per_class"] = per_class
    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    return metrics


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)
