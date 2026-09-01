"""Free-rider attack."""

from __future__ import annotations

import torch

from fedqtrust.fl.state import StateDict


def free_ride(delta: StateDict) -> StateDict:
    return {k: torch.zeros_like(v) for k, v in delta.items()}

