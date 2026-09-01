"""Device selection and environment reporting."""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
from importlib import metadata
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class DeviceManager:
    requested: str = "auto"

    def select(self) -> str:
        import torch

        if self.requested == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if self.requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
        if self.requested.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"{self.requested} requested but CUDA is unavailable")
        return self.requested

    @property
    def torch_device(self):
        import torch

        return torch.device(self.select())


def _version(module: str) -> str | None:
    try:
        package = module.replace("_", "-")
        aliases = {
            "qiskit_aer": "qiskit-aer",
            "qiskit_machine_learning": "qiskit-machine-learning",
            "qiskit_optimization": "qiskit-optimization",
            "medmnist": "medmnist",
            "oqs": "liboqs-python",
        }
        return metadata.version(aliases.get(module, package))
    except Exception:
        return None


def _ram_gb() -> float | None:
    try:
        import psutil

        return round(psutil.virtual_memory().total / (1024**3), 3)
    except Exception:
        return None


def git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return out.stdout.strip()
    except Exception:
        return None


def collect_environment(requested_device: str = "auto") -> dict[str, Any]:
    import torch

    gpus: list[dict[str, Any]] = []
    if torch.cuda.is_available():
        for idx in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(idx)
            gpus.append(
                {
                    "index": idx,
                    "name": props.name,
                    "memory_gb": round(props.total_memory / (1024**3), 3),
                }
            )
    docker_available = shutil.which("docker") is not None
    fabric_available = shutil.which("peer") is not None or shutil.which("fabric-ca-client") is not None
    return {
        "python_version": platform.python_version(),
        "os": platform.platform(),
        "cpu_count": os.cpu_count(),
        "ram_gb": _ram_gb(),
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "requested_device": requested_device,
        "selected_device": DeviceManager(requested_device).select(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "gpu_count": torch.cuda.device_count(),
        "gpus": gpus,
        "qiskit_version": _version("qiskit"),
        "qiskit_machine_learning_version": _version("qiskit_machine_learning"),
        "qiskit_aer_version": _version("qiskit_aer"),
        "qiskit_optimization_version": _version("qiskit_optimization"),
        "medmnist_version": _version("medmnist"),
        "liboqs_available": _version("oqs") is not None,
        "liboqs_version": _version("oqs"),
        "docker_available": docker_available,
        "hyperledger_fabric_available": fabric_available,
        "git_commit": git_commit(),
    }


def write_environment_report(output_dir: str | Path, requested_device: str = "auto") -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    report = collect_environment(requested_device)
    (output / "environment_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    freeze = subprocess.run(
        ["python3", "-m", "pip", "freeze"],
        capture_output=True,
        text=True,
        check=False,
    )
    (output / "pip_freeze.txt").write_text(freeze.stdout, encoding="utf-8")
    return report
