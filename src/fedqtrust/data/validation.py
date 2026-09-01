"""Data validation helpers."""

from __future__ import annotations

import numpy as np


def validate_labels(labels: np.ndarray, num_classes: int) -> bool:
    labels = np.asarray(labels).astype(int)
    return bool(labels.size == 0 or (labels.min() >= 0 and labels.max() < num_classes))

