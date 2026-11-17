"""FLTrust scoring helper."""

from __future__ import annotations

import numpy as np


def fltrust_scores(client_gradients: list[np.ndarray], server_gradient: np.ndarray) -> np.ndarray:
    scores = []
    for grad in client_gradients:
        denom = float(np.linalg.norm(grad) * np.linalg.norm(server_gradient))
        cos = 0.0 if denom == 0.0 else float(np.dot(grad, server_gradient) / denom)
        scores.append(max(0.0, cos))
    return np.array(scores, dtype=float)

