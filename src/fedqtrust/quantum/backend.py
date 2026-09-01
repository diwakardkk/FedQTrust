"""Quantum backend detection."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass


@dataclass
class QuantumBackendInfo:
    quantum_backend: str
    quantum_device: str
    simulation_method: str
    precision: str
    shots: int
    gpu_available: bool
    fallback_reason: str | None


def detect_quantum_backend(shots: int = 1024) -> QuantumBackendInfo:
    if importlib.util.find_spec("qiskit_aer") is None:
        return QuantumBackendInfo("unavailable", "none", "none", "double", shots, False, "qiskit_aer is not installed")
    return QuantumBackendInfo(
        "qiskit_aer",
        "CPU",
        "automatic",
        "double",
        shots,
        False,
        "safe smoke probe avoids importing AerSimulator; run preflight on the GPU server for device='GPU' verification",
    )
