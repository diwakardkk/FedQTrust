"""Server-side selection and aggregation helpers."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from fedqtrust.quantum.exact_solver import solve_exact
from fedqtrust.quantum.qubo import QuboInstance


@dataclass
class ServerState:
    round_index: int = 0
    last_update_cache: dict[int, np.ndarray] = field(default_factory=dict)


def select_with_exact_qubo(trust: dict[int, float], updates: dict[int, np.ndarray], k: int, lambda_: float = 1.0, mu: float = 2.0, theta: float = 0.4) -> list[int]:
    eligible = [cid for cid, value in trust.items() if value >= theta]
    threshold = theta
    while len(eligible) < k and threshold > 0:
        threshold = max(0.0, threshold - 0.1)
        eligible = [cid for cid, value in trust.items() if value >= threshold]
    if len(eligible) <= k:
        return eligible
    vectors = [updates.get(cid, np.zeros(1)) for cid in eligible]
    from fedqtrust.quantum.qubo import pairwise_cosine_distances

    instance = QuboInstance(np.array([trust[cid] for cid in eligible]), pairwise_cosine_distances(vectors), lambda_, mu, k, eligible)
    solution = solve_exact(instance)
    return [cid for cid, bit in zip(eligible, solution.bitstring) if bit == 1]

