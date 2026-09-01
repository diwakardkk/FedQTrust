"""Exact QUBO enumeration for small instances."""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass

import numpy as np

from .qubo import QuboInstance, qubo_objective


@dataclass
class QuboSolution:
    bitstring: np.ndarray
    objective: float
    runtime_s: float
    solver_used: str
    feasible_cardinality: bool


def solve_exact(instance: QuboInstance) -> QuboSolution:
    if len(instance.trust) > 20:
        raise ValueError("exact enumeration is intended for small QUBOs")
    start = time.perf_counter()
    best_x: np.ndarray | None = None
    best_value = float("inf")
    for bits in itertools.product([0, 1], repeat=len(instance.trust)):
        x = np.array(bits, dtype=int)
        value = qubo_objective(x, instance.trust, instance.distances, instance.lambda_, instance.mu, instance.k)
        if value < best_value:
            best_value = value
            best_x = x
    assert best_x is not None
    return QuboSolution(best_x, best_value, time.perf_counter() - start, "EXACT", int(best_x.sum()) == instance.k)

