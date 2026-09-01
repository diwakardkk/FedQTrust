"""JSON ledger backend for development runs."""

from __future__ import annotations

import json
from pathlib import Path


class JsonBlockchain:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text(json.dumps({"trust": {}, "selection_log": []}, indent=2), encoding="utf-8")

    def _read(self) -> dict:
        return json.loads(self.path.read_text(encoding="utf-8"))

    def _write(self, data: dict) -> None:
        self.path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def ping(self) -> bool:
        return self.path.exists()

    def set_trust(self, client_id: int, value: float) -> None:
        data = self._read()
        data["trust"][str(client_id)] = float(value)
        self._write(data)

    def get_trust(self, client_id: int) -> float | None:
        value = self._read()["trust"].get(str(client_id))
        return None if value is None else float(value)

    def log_selection(self, record: dict) -> None:
        data = self._read()
        data["selection_log"].append(record)
        self._write(data)

