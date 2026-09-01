"""Blockchain backend protocol."""

from __future__ import annotations

from typing import Protocol


class BlockchainBackend(Protocol):
    def ping(self) -> bool: ...

    def set_trust(self, client_id: int, value: float) -> None: ...

    def get_trust(self, client_id: int) -> float | None: ...

    def log_selection(self, record: dict) -> None: ...

