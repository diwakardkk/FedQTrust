import torch

from fedqtrust.models.classical_cnn import ClassicalCNN, count_parameters
from fedqtrust.models.fedqcnn import FedQCNN
from fedqtrust.models.multimodal import MultiModalFedQCNN


def test_model_shapes_and_quantum_gradient():
    x = torch.rand(2, 1, 28, 28)
    cnn = ClassicalCNN(2)
    assert cnn(x).shape == (2, 2)
    q = FedQCNN(2)
    y = q(x[:1]).sum()
    y.backward()
    assert q.quantum.quantum_weights.grad is not None
    counts = count_parameters(q)
    assert counts["quantum_trainable_parameters"] > 0


def test_multimodal_heads():
    model = MultiModalFedQCNN()
    x = torch.rand(1, 1, 28, 28)
    assert model(x, "pathmnist").shape == (1, 9)
    assert model(x, "breastmnist").shape == (1, 2)

