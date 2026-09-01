"""Patch extraction helpers."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def image_patches(x: torch.Tensor, patch_size: int = 2, stride: int = 2) -> torch.Tensor:
    return F.unfold(x, kernel_size=patch_size, stride=stride).transpose(1, 2)

