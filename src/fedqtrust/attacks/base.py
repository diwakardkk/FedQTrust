"""Attack interfaces and utilities."""

from __future__ import annotations

import numpy as np
import torch

from fedqtrust.fl.state import StateDict


def clone_delta(delta: StateDict) -> StateDict:
    return {k: v.detach().clone() for k, v in delta.items()}


def flatten_delta(delta: StateDict) -> torch.Tensor:
    return torch.cat([v.detach().reshape(-1).float().cpu() for _, v in sorted(delta.items())])


class MaliciousClient:
    def transform_delta(self, delta: StateDict, rng: np.random.Generator) -> StateDict:
        return clone_delta(delta)

