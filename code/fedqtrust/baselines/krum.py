"""Krum aggregation selection from scratch."""

from __future__ import annotations

import numpy as np


def krum_select(vectors: list[np.ndarray], f_requested: int) -> tuple[int, int]:
    n = len(vectors)
    f_effective = min(f_requested, max(0, (n - 3) // 2))
    if n <= 2 * f_effective + 2:
        f_effective = max(0, (n - 3) // 2)
    neighbor_count = n - f_effective - 2
    if neighbor_count <= 0:
        return 0, f_effective
    scores = []
    for i, vec in enumerate(vectors):
        distances = sorted(float(np.sum((vec - other) ** 2)) for j, other in enumerate(vectors) if j != i)
        scores.append(sum(distances[:neighbor_count]))
    return int(np.argmin(scores)), f_effective

