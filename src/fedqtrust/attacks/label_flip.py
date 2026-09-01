"""Label flipping attack."""

from __future__ import annotations

import numpy as np


def flip_labels(labels: np.ndarray, num_classes: int, ratio: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels).astype(int).copy()
    n = int(round(len(labels) * ratio))
    if n == 0:
        return labels
    chosen = rng.choice(len(labels), size=n, replace=False)
    for idx in chosen:
        choices = [c for c in range(num_classes) if c != int(labels[idx])]
        labels[idx] = int(rng.choice(choices))
    return labels

