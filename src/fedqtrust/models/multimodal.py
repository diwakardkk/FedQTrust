"""Multi-modal shared-backbone model with dataset-specific heads."""

from __future__ import annotations

import torch
from torch import nn

from fedqtrust.config import DATASETS
from .fedqcnn import QuantumPatchLayer


class MultiModalFedQCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.quantum = QuantumPatchLayer()
        with torch.no_grad():
            flat = self.quantum(torch.zeros(1, 1, 28, 28)).shape[1]
        self.shared_backbone = nn.Sequential(
            self.quantum,
            nn.Linear(flat, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        self.heads = nn.ModuleDict({key: nn.Linear(256, spec["classes"]) for key, spec in DATASETS.items()})

    def forward(self, x: torch.Tensor, dataset: str) -> torch.Tensor:
        return self.heads[dataset](self.shared_backbone(x))

