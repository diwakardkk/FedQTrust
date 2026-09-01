"""Little-is-enough model poisoning."""

from __future__ import annotations

import numpy as np
import torch

from fedqtrust.fl.state import StateDict


def lie_vector(honest_vectors: list[np.ndarray], z_max: float = 1.0) -> np.ndarray:
    matrix = np.stack(honest_vectors, axis=0)
    return matrix.mean(axis=0) - z_max * matrix.std(axis=0)


def vector_to_delta(vector: np.ndarray, template: StateDict) -> StateDict:
    out: StateDict = {}
    offset = 0
    tensor = torch.from_numpy(vector.astype("float32"))
    for key, value in sorted(template.items()):
        size = value.numel()
        out[key] = tensor[offset : offset + size].reshape(value.shape).to(dtype=value.dtype)
        offset += size
    return out

