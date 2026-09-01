"""Classical CNN baseline from the manuscript with an explicit dimension adapter."""

from __future__ import annotations

import torch
from torch import nn


class ClassicalCNN(nn.Module):
    def __init__(self, num_classes: int, stride: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=2, stride=stride),
            nn.Flatten(),
        )
        with torch.no_grad():
            flat = self.features(torch.zeros(1, 1, 28, 28)).shape[1]
        self.backbone = nn.Sequential(
            self.features,
            nn.Linear(flat, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        self.head = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


def count_parameters(model: nn.Module) -> dict[str, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    quantum = sum(p.numel() for name, p in model.named_parameters() if "quantum" in name and p.requires_grad)
    return {
        "quantum_trainable_parameters": int(quantum),
        "classical_trainable_parameters": int(trainable - quantum),
        "total_trainable_parameters": int(trainable),
        "total_non_trainable_parameters": int(non_trainable),
    }

