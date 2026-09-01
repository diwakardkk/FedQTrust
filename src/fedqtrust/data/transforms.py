"""Deterministic image preprocessing."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


class ToGrayTensor:
    """Convert PIL/NumPy images to 1x28x28 float tensors in [0, 1]."""

    def __call__(self, image: Any) -> torch.Tensor:
        arr = np.asarray(image)
        if arr.ndim == 3:
            if arr.shape[-1] == 1:
                arr = arr[..., 0]
            else:
                arr = np.dot(arr[..., :3], np.array([0.299, 0.587, 0.114]))
        arr = arr.astype("float32")
        if arr.max(initial=0.0) > 1.0:
            arr = arr / 255.0
        tensor = torch.from_numpy(arr).unsqueeze(0)
        return tensor.contiguous()

