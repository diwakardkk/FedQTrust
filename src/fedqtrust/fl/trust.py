"""Trust score equations from the manuscript."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field

import numpy as np


def safe_cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


@dataclass
class TrustManager:
    num_clients: int
    alpha: float = 0.4
    beta: float = 0.3
    gamma: float = 0.3
    tau_min: float = 0.1
    reputation_decay: float = 0.9
    consistency_window: int = 5
    accuracy: dict[int, float] = field(default_factory=dict)
    reputation: dict[int, float] = field(default_factory=dict)
    trust: dict[int, float] = field(default_factory=dict)
    consistency_history: dict[int, deque[float]] = field(default_factory=lambda: defaultdict(deque))

    def __post_init__(self) -> None:
        for cid in range(self.num_clients):
            self.accuracy[cid] = 0.5
            self.reputation[cid] = 0.5
            self.trust[cid] = self.compute_trust(cid, 1.0)

    def compute_trust(self, client_id: int, consistency: float) -> float:
        raw = (
            self.alpha * self.accuracy.get(client_id, 0.5)
            + self.beta * consistency
            + self.gamma * self.reputation.get(client_id, 0.5)
        )
        return max(self.tau_min, float(raw))

    def update_consistency(self, client_id: int, update: np.ndarray, global_reference: np.ndarray) -> float:
        value = safe_cosine(update, global_reference)
        history = self.consistency_history[client_id]
        history.append(value)
        while len(history) > self.consistency_window:
            history.popleft()
        return float(np.mean(history)) if history else 1.0

    def update_round(self, client_id: int, validation_accuracy: float, consistency: float) -> float:
        previous_tau = self.trust.get(client_id, 0.65)
        self.accuracy[client_id] = float(validation_accuracy)
        self.reputation[client_id] = self.reputation_decay * self.reputation[client_id] + (1 - self.reputation_decay) * previous_tau
        self.trust[client_id] = self.compute_trust(client_id, consistency)
        return self.trust[client_id]

