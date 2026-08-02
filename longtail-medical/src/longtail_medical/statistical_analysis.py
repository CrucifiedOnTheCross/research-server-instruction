from __future__ import annotations

from collections import defaultdict

import numpy as np

from longtail_medical.metrics import classification_metrics


def fast_mcc_balanced_accuracy(labels, predictions, num_classes: int) -> dict[str, float]:
    """Compute bootstrap target metrics without unrelated ranking metrics."""
    labels = np.asarray(labels, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    matrix = np.bincount(
        labels * num_classes + predictions,
        minlength=num_classes * num_classes,
    ).reshape(num_classes, num_classes)
    true_sum = matrix.sum(axis=1, dtype=np.float64)
    pred_sum = matrix.sum(axis=0, dtype=np.float64)
    correct = float(np.trace(matrix))
    total = float(matrix.sum())
    recalls = np.divide(
        np.diag(matrix), true_sum,
        out=np.full(num_classes, np.nan, dtype=np.float64),
        where=true_sum != 0,
    )
    numerator = correct * total - float(np.dot(true_sum, pred_sum))
    denominator = np.sqrt(
        (total * total - float(np.dot(pred_sum, pred_sum)))
        * (total * total - float(np.dot(true_sum, true_sum)))
    )
    return {
        "mcc": float(numerator / denominator) if denominator else 0.0,
        "balanced_accuracy": float(np.nanmean(recalls)),
    }


def aggregate_by_lesion(labels, probabilities, lesion_ids, image_ids):
    groups: dict[str, list[int]] = defaultdict(list)
    for index, (lesion_id, image_id) in enumerate(zip(lesion_ids, image_ids)):
        groups[lesion_id or f"image:{image_id}"].append(index)
    lesion_labels, lesion_probabilities, ordered_ids = [], [], []
    for lesion_id in sorted(groups):
        indices = groups[lesion_id]
        unique_labels = set(np.asarray(labels)[indices].tolist())
        if len(unique_labels) != 1:
            raise ValueError(f"Conflicting labels for lesion {lesion_id}")
        lesion_labels.append(unique_labels.pop())
        lesion_probabilities.append(np.asarray(probabilities)[indices].mean(axis=0))
        ordered_ids.append(lesion_id)
    return np.asarray(lesion_labels), np.asarray(lesion_probabilities), ordered_ids


def paired_lesion_stratified_bootstrap(
    labels, probabilities_a, probabilities_b, lesion_ids, image_ids, class_names,
    repeats: int = 10000, seed: int = 20260802,
):
    labels = np.asarray(labels)
    groups: dict[int, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for index, (label, lesion_id, image_id) in enumerate(zip(labels, lesion_ids, image_ids)):
        groups[int(label)][lesion_id or f"image:{image_id}"].append(index)
    rng = np.random.default_rng(seed)
    effects = {"mcc": [], "balanced_accuracy": []}
    for _ in range(repeats):
        sampled_indices: list[int] = []
        for label in range(len(class_names)):
            lesion_groups = list(groups[label].values())
            draws = rng.integers(0, len(lesion_groups), size=len(lesion_groups))
            for draw in draws:
                sampled_indices.extend(lesion_groups[int(draw)])
        sampled = np.asarray(sampled_indices)
        metrics_a = fast_mcc_balanced_accuracy(
            labels[sampled], np.asarray(probabilities_a)[sampled].argmax(axis=1), len(class_names)
        )
        metrics_b = fast_mcc_balanced_accuracy(
            labels[sampled], np.asarray(probabilities_b)[sampled].argmax(axis=1), len(class_names)
        )
        for metric in effects:
            effects[metric].append(metrics_a[metric] - metrics_b[metric])
    return {
        metric: {
            "mean": float(np.mean(values)),
            "ci95_low": float(np.quantile(values, 0.025)),
            "ci95_high": float(np.quantile(values, 0.975)),
            "samples": values,
        }
        for metric, values in effects.items()
    }


def lesion_stratified_metric_bootstrap(
    labels, probabilities, lesion_ids, image_ids, class_names,
    repeats: int = 10000, seed: int = 20260802,
):
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities)
    groups: dict[int, list[list[int]]] = defaultdict(list)
    temporary: dict[tuple[int, str], list[int]] = defaultdict(list)
    for index, (label, lesion_id, image_id) in enumerate(zip(labels, lesion_ids, image_ids)):
        temporary[(int(label), lesion_id or f"image:{image_id}")].append(index)
    for (label, _), indices in temporary.items():
        groups[label].append(indices)
    rng = np.random.default_rng(seed)
    distributions = {
        "image_level": {"mcc": [], "balanced_accuracy": []},
        "lesion_level": {"mcc": [], "balanced_accuracy": []},
    }
    for _ in range(repeats):
        image_indices = []
        lesion_labels = []
        lesion_probabilities = []
        for label in range(len(class_names)):
            class_groups = groups[label]
            draws = rng.integers(0, len(class_groups), size=len(class_groups))
            for draw in draws:
                indices = class_groups[int(draw)]
                image_indices.extend(indices)
                lesion_labels.append(label)
                lesion_probabilities.append(probabilities[indices].mean(axis=0))
        image_indices_array = np.asarray(image_indices)
        image_metrics = fast_mcc_balanced_accuracy(
            labels[image_indices_array], probabilities[image_indices_array].argmax(axis=1), len(class_names)
        )
        lesion_probabilities_array = np.asarray(lesion_probabilities)
        lesion_metrics = fast_mcc_balanced_accuracy(
            np.asarray(lesion_labels), lesion_probabilities_array.argmax(axis=1), len(class_names)
        )
        for metric in ("mcc", "balanced_accuracy"):
            distributions["image_level"][metric].append(image_metrics[metric])
            distributions["lesion_level"][metric].append(lesion_metrics[metric])
    return {
        level: {
            metric: {
                "mean": float(np.mean(values)),
                "ci95_low": float(np.quantile(values, 0.025)),
                "ci95_high": float(np.quantile(values, 0.975)),
            }
            for metric, values in metrics.items()
        }
        for level, metrics in distributions.items()
    }
