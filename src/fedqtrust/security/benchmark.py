"""Timing helpers for crypto and protocol components."""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable


def benchmark(fn: Callable[[], object], trials: int) -> dict[str, float]:
    values = []
    for _ in range(trials):
        start = time.perf_counter()
        fn()
        values.append(time.perf_counter() - start)
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }

