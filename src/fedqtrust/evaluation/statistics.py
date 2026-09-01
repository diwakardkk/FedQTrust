"""Statistical summaries."""

from __future__ import annotations

import numpy as np
from scipy.stats import wilcoxon


def summarize(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    return {"mean": float(np.mean(arr)), "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0, "median": float(np.median(arr))}


def paired_wilcoxon(a: list[float], b: list[float]) -> float:
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    return float(wilcoxon(a, b, zero_method="zsplit").pvalue)


def bonferroni_threshold(alpha: float = 0.05, comparisons: int = 42) -> float:
    return alpha / comparisons

