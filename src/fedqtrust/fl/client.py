"""Federated client primitives."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Client:
    client_id: int
    dataset: str
    train_count: int
    validation_count: int
    malicious: bool = False

