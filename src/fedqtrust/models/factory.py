"""Model factory."""

from __future__ import annotations

from .classical_cnn import ClassicalCNN
from .fedqcnn import FedQCNN
from .multimodal import MultiModalFedQCNN


def build_model(name: str, num_classes: int = 2):
    if name == "classical_cnn":
        return ClassicalCNN(num_classes)
    if name == "fedqcnn":
        return FedQCNN(num_classes)
    if name == "multimodal_fedqcnn":
        return MultiModalFedQCNN()
    raise ValueError(f"unknown model {name}")

