"""Client selection diagnostics."""

from __future__ import annotations

import numpy as np


def l2_anomaly_scores(updates: list[np.ndarray], reference: np.ndarray) -> np.ndarray:
    return np.array([np.linalg.norm(update - reference) for update in updates], dtype=float)


def qubo_anomaly_scores(trust: np.ndarray, distances: np.ndarray) -> np.ndarray:
    mean_divergence = distances.mean(axis=1) if distances.size else np.zeros_like(trust)
    return (1.0 - trust) + mean_divergence

