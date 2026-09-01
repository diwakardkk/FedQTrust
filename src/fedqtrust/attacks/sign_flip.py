"""Sign-flipping attack."""

from __future__ import annotations

from fedqtrust.fl.state import StateDict


def sign_flip(delta: StateDict) -> StateDict:
    return {k: -v.detach().clone() for k, v in delta.items()}

