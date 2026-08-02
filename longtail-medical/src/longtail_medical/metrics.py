from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    roc_auc_score,
)


def expected_calibration_error(
    labels: np.ndarray, probabilities: np.ndarray, bins: int = 15
) -> float:
    confidence = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    correct = predictions == labels
    edges = np.linspace(0.0, 1.0, bins + 1)
    result = 0.0
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        included = (confidence > lower) & (confidence <= upper)
        if index == 0:
            included |= confidence == 0.0
        if included.any():
            result += included.mean() * abs(
                correct[included].mean() - confidence[included].mean()
            )
    return float(result)


def multiclass_brier(labels: np.ndarray, probabilities: np.ndarray) -> float:
    one_hot = np.eye(probabilities.shape[1], dtype=np.float64)[labels]
    return float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))


def classification_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    class_names: list[str],
    ece_bins: int = 15,
) -> dict[str, object]:
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.shape != (len(labels), len(class_names)):
        raise ValueError("Probability matrix shape does not match labels/classes")
    if labels.size == 0 or labels.min() < 0 or labels.max() >= len(class_names):
        raise ValueError("Labels are empty or outside the configured class range")
    if not np.isfinite(probabilities).all():
        raise ValueError("Probabilities contain non-finite values")
    row_sums = probabilities.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-5):
        raise ValueError("Probability rows must sum to one")

    predictions = probabilities.argmax(axis=1)
    one_hot = np.eye(len(class_names), dtype=np.int64)[labels]
    per_class_recall = []
    matrix = confusion_matrix(labels, predictions, labels=range(len(class_names)))
    for index in range(len(class_names)):
        denominator = matrix[index].sum()
        per_class_recall.append(
            float(matrix[index, index] / denominator) if denominator else float("nan")
        )
    per_class_auroc = roc_auc_score(
        one_hot, probabilities, average=None, multi_class="ovr"
    )
    per_class_auprc = average_precision_score(
        one_hot, probabilities, average=None
    )
    return {
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
        "mcc": float(matthews_corrcoef(labels, predictions)),
        "macro_auroc": float(np.mean(per_class_auroc)),
        "macro_auprc": float(np.mean(per_class_auprc)),
        "nll": float(log_loss(labels, probabilities, labels=range(len(class_names)))),
        "brier": multiclass_brier(labels, probabilities),
        "ece": expected_calibration_error(labels, probabilities, bins=ece_bins),
        "worst_class_recall": float(np.nanmin(per_class_recall)),
        "per_class": {
            name: {
                "recall": per_class_recall[index],
                "auroc": float(per_class_auroc[index]),
                "auprc": float(per_class_auprc[index]),
            }
            for index, name in enumerate(class_names)
        },
        "confusion_matrix": matrix.tolist(),
    }
