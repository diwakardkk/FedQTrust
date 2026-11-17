"""QUBO objective and conversion helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class QuboInstance:
    trust: np.ndarray
    distances: np.ndarray
    lambda_: float
    mu: float
    k: int
    client_ids: list[int]


def qubo_objective(x: np.ndarray, trust: np.ndarray, distances: np.ndarray, lambda_: float, mu: float, k: int) -> float:
    x = np.asarray(x, dtype=float)
    trust_term = -float(np.sum(trust * x))
    pair_term = 0.0
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            pair_term += float(distances[i, j] * x[i] * x[j])
    cardinality = float((np.sum(x) - k) ** 2)
    return trust_term + lambda_ * pair_term + mu * cardinality


def qubo_matrix(trust: np.ndarray, distances: np.ndarray, lambda_: float, mu: float, k: int) -> np.ndarray:
    n = len(trust)
    q = np.zeros((n, n), dtype=float)
    for i in range(n):
        q[i, i] = -trust[i] + mu * (1 - 2 * k)
    for i in range(n):
        for j in range(i + 1, n):
            q[i, j] = lambda_ * distances[i, j] + 2 * mu
    return q


def matrix_objective_upper(x: np.ndarray, q: np.ndarray, constant: float) -> float:
    x = np.asarray(x, dtype=float)
    total = constant
    for i in range(len(x)):
        total += q[i, i] * x[i]
        for j in range(i + 1, len(x)):
            total += q[i, j] * x[i] * x[j]
    return float(total)


def project_cardinality(x: np.ndarray, trust: np.ndarray, k: int) -> np.ndarray:
    x = np.asarray(x, dtype=int).copy()
    while int(x.sum()) > k:
        selected = np.where(x == 1)[0]
        remove = selected[np.argmin(trust[selected])]
        x[remove] = 0
    while int(x.sum()) < k and int(x.sum()) < len(x):
        remaining = np.where(x == 0)[0]
        add = remaining[np.argmax(trust[remaining])]
        x[add] = 1
    return x


def pairwise_cosine_distances(updates: list[np.ndarray]) -> np.ndarray:
    n = len(updates)
    distances = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            denom = float(np.linalg.norm(updates[i]) * np.linalg.norm(updates[j]))
            cos = 0.0 if denom == 0.0 else float(np.dot(updates[i], updates[j]) / denom)
            distances[i, j] = distances[j, i] = 1.0 - cos
    return distances

