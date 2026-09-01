"""Convergence helpers."""

from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit


def rounds_to_accuracy(accuracies: list[float], target: float = 0.8) -> int | None:
    for idx, value in enumerate(accuracies, start=1):
        if value >= target:
            return idx
    return None


def fit_inverse_sqrt(values: list[float]) -> dict[str, float]:
    y = np.asarray(values, dtype=float)
    x = np.arange(1, len(y) + 1, dtype=float)

    def fn(t, a, b):
        return a / np.sqrt(t) + b

    params, _ = curve_fit(fn, x, y, maxfev=10000)
    pred = fn(x, *params)
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {"a": float(params[0]), "b": float(params[1]), "r2": 1 - ss_res / ss_tot if ss_tot else 1.0, "rmse": float(np.sqrt(np.mean((y - pred) ** 2)))}

