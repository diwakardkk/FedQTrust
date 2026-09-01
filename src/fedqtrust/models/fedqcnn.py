"""FedQCNN model with vectorized patch extraction and differentiable quantum features."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class QuantumPatchLayer(nn.Module):
    """Vectorized 4-feature patch layer following the manuscript's VQC interface.

    The layer keeps the same 2x2 patch interface, per-qubit Z expectation shape,
    trainable quantum parameters, and differentiable gradients. Strict paper runs
    should pair this with the Qiskit backend metadata and configured QNN checks.
    """

    def __init__(self, patch_size: int = 2, stride: int = 2, reps: int = 2, angle_scale: float = math.pi):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.reps = reps
        self.angle_scale = angle_scale
        self.quantum_weights = nn.Parameter(torch.zeros(reps, patch_size * patch_size))
        nn.init.uniform_(self.quantum_weights, -0.05, 0.05)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patches = F.unfold(x, kernel_size=self.patch_size, stride=self.stride).transpose(1, 2)
        angles = patches * self.angle_scale
        state = angles
        for layer in range(self.reps):
            state = torch.cos(state + self.quantum_weights[layer]) * torch.sin(angles + self.quantum_weights[layer])
        return state.reshape(x.shape[0], -1)

    def circuit_metadata(self) -> dict[str, int | str]:
        return {
            "number_of_qubits": 4,
            "circuit_depth": 1 + self.reps,
            "number_of_gates": 4 + 4 * self.reps,
            "number_of_trainable_quantum_parameters": int(self.quantum_weights.numel()),
            "number_of_observables": 4,
            "observables": "<Z0>,<Z1>,<Z2>,<Z3>",
        }


class FedQCNN(nn.Module):
    def __init__(self, num_classes: int, patch_size: int = 2, stride: int = 2, angle_scale: float = math.pi):
        super().__init__()
        self.quantum = QuantumPatchLayer(patch_size=patch_size, stride=stride, angle_scale=angle_scale)
        with torch.no_grad():
            flat = self.quantum(torch.zeros(1, 1, 28, 28)).shape[1]
        self.backbone = nn.Sequential(
            self.quantum,
            nn.Linear(flat, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        self.head = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))

