"""Client metadata for malicious participants."""

from __future__ import annotations

from dataclasses import dataclass

from .client import Client


@dataclass
class MaliciousFLClient(Client):
    attack: str = "label_flip"
    malicious: bool = True

