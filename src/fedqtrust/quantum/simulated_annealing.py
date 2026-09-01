"""Reproducible simulated annealing for the manuscript QUBO."""

from __future__ import annotations

import time

import numpy as np

from .exact_solver import QuboSolution
from .qubo import QuboInstance, project_cardinality, qubo_objective


def solve_sa(instance: QuboInstance, seed: int = 42, steps: int = 1000) -> QuboSolution:
    rng = np.random.default_rng(seed)
    start = time.perf_counter()
    n = len(instance.trust)
    x = np.zeros(n, dtype=int)
    chosen = rng.choice(n, size=min(instance.k, n), replace=False)
    x[chosen] = 1
    best = x.copy()
    best_value = qubo_objective(best, instance.trust, instance.distances, instance.lambda_, instance.mu, instance.k)
    current_value = best_value
    for step in range(steps):
        temp = max(1e-4, 1.0 - step / max(steps - 1, 1))
        candidate = x.copy()
        idx = int(rng.integers(0, n))
        candidate[idx] = 1 - candidate[idx]
        candidate = project_cardinality(candidate, instance.trust, instance.k)
        value = qubo_objective(candidate, instance.trust, instance.distances, instance.lambda_, instance.mu, instance.k)
        if value < current_value or rng.random() < np.exp((current_value - value) / temp):
            x = candidate
            current_value = value
        if value < best_value:
            best = candidate.copy()
            best_value = value
    return QuboSolution(best, best_value, time.perf_counter() - start, "SA", int(best.sum()) == instance.k)

