"""State serialization helpers."""

from __future__ import annotations

import hashlib
import io

import torch


def serialize_state(state: dict[str, torch.Tensor]) -> bytes:
    buffer = io.BytesIO()
    torch.save({k: v.detach().cpu() for k, v in state.items()}, buffer)
    return buffer.getvalue()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

