"""Gaussian model-poisoning attack."""

from __future__ import annotations

import torch

from fedqtrust.fl.state import StateDict


def gaussian_noise(delta: StateDict, sigma: float, seed: int) -> StateDict:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    return {k: v.detach().clone() + torch.randn(v.shape, generator=gen, dtype=v.dtype) * sigma for k, v in delta.items()}

