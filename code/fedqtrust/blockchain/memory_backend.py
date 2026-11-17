"""In-memory blockchain-compatible backend for smoke and tests."""

from __future__ import annotations


class InMemoryBlockchain:
    def __init__(self):
        self.trust: dict[int, float] = {}
        self.selection_log: list[dict] = []

    def ping(self) -> bool:
        return True

    def set_trust(self, client_id: int, value: float) -> None:
        self.trust[int(client_id)] = float(value)

    def get_trust(self, client_id: int) -> float | None:
        return self.trust.get(int(client_id))

    def log_selection(self, record: dict) -> None:
        self.selection_log.append(dict(record))

