"""Hyperledger Fabric CLI adapter."""

from __future__ import annotations

import shutil
import subprocess
import time
from dataclasses import dataclass


@dataclass
class FabricResult:
    stdout: str
    stderr: str
    returncode: int
    latency_s: float


class HyperledgerFabricBlockchain:
    def __init__(self, chaincode: str = "fedqtrust"):
        self.chaincode = chaincode
        if shutil.which("peer") is None:
            raise RuntimeError("Hyperledger Fabric peer CLI is not available")

    def _peer(self, args: list[str]) -> FabricResult:
        start = time.perf_counter()
        proc = subprocess.run(["peer", *args], capture_output=True, text=True, check=False)
        return FabricResult(proc.stdout, proc.stderr, proc.returncode, time.perf_counter() - start)

    def ping(self) -> bool:
        result = self._peer(["chaincode", "query", "-n", self.chaincode, "-c", '{"Args":["Ping"]}'])
        return result.returncode == 0

    def set_trust(self, client_id: int, value: float) -> None:
        result = self._peer(["chaincode", "invoke", "-n", self.chaincode, "-c", f'{{"Args":["SetTrust","{client_id}","{value}"]}}'])
        if result.returncode != 0:
            raise RuntimeError(result.stderr)

    def get_trust(self, client_id: int) -> float | None:
        result = self._peer(["chaincode", "query", "-n", self.chaincode, "-c", f'{{"Args":["GetTrust","{client_id}"]}}'])
        if result.returncode != 0:
            return None
        try:
            return float(result.stdout.strip())
        except ValueError:
            return None

    def log_selection(self, record: dict) -> None:
        import json

        payload = json.dumps(record, separators=(",", ":"))
        result = self._peer(["chaincode", "invoke", "-n", self.chaincode, "-c", f'{{"Args":["LogSelection",{json.dumps(payload)}]}}'])
        if result.returncode != 0:
            raise RuntimeError(result.stderr)

