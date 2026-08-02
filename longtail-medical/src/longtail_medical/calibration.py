from __future__ import annotations

import numpy as np


def apply_temperature(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    """Apply scalar temperature scaling to probabilities via equivalent log logits."""
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be finite and positive")
    logits = np.log(np.clip(np.asarray(probabilities, dtype=np.float64), 1e-12, 1.0))
    logits = logits / float(temperature)
    logits -= logits.max(axis=1, keepdims=True)
    calibrated = np.exp(logits)
    return calibrated / calibrated.sum(axis=1, keepdims=True)


def temperature_nll(labels: np.ndarray, probabilities: np.ndarray, temperature: float) -> float:
    calibrated = apply_temperature(probabilities, temperature)
    return float(-np.log(np.clip(calibrated[np.arange(len(labels)), labels], 1e-12, 1.0)).mean())


def fit_temperature(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    log_temperature_bounds: tuple[float, float] = (-4.0, 4.0),
    iterations: int = 96,
) -> dict[str, float]:
    """Fit one temperature by deterministic golden-section minimization of validation NLL."""
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[0] != labels.size:
        raise ValueError("Labels and probabilities have incompatible shapes")
    lower, upper = map(float, log_temperature_bounds)
    ratio = (np.sqrt(5.0) - 1.0) / 2.0
    left = upper - ratio * (upper - lower)
    right = lower + ratio * (upper - lower)

    def objective(log_temperature: float) -> float:
        return temperature_nll(labels, probabilities, float(np.exp(log_temperature)))

    left_value, right_value = objective(left), objective(right)
    for _ in range(iterations):
        if left_value <= right_value:
            upper, right, right_value = right, left, left_value
            left = upper - ratio * (upper - lower)
            left_value = objective(left)
        else:
            lower, left, left_value = left, right, right_value
            right = lower + ratio * (upper - lower)
            right_value = objective(right)
    temperature = float(np.exp((lower + upper) / 2.0))
    return {
        "temperature": temperature,
        "nll_before": temperature_nll(labels, probabilities, 1.0),
        "nll_after": temperature_nll(labels, probabilities, temperature),
        "iterations": iterations,
    }
