"""Gaussian model-poisoning attack."""

from __future__ import annotations

import torch

from fedqtrust.fl.state import StateDict


def gaussian_noise(delta: StateDict, sigma: float, seed: int) -> StateDict:
    out: StateDict = {}
    for offset, (key, value) in enumerate(delta.items()):
        gen = torch.Generator(device=value.device).manual_seed(seed + offset)
        noise = torch.randn(value.shape, generator=gen, dtype=value.dtype, device=value.device) * sigma
        out[key] = value.detach().clone() + noise
    return out
