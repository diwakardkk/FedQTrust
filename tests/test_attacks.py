import numpy as np
import torch

from fedqtrust.attacks.combined import combined_attack_sets
from fedqtrust.attacks.free_rider import free_ride
from fedqtrust.attacks.gaussian import gaussian_noise
from fedqtrust.attacks.label_flip import flip_labels
from fedqtrust.attacks.lie import lie_vector, vector_to_delta
from fedqtrust.attacks.sign_flip import sign_flip


def test_attack_transformations():
    labels = np.array([0, 1, 0, 1])
    flipped = flip_labels(labels, 2, 1.0, 42)
    assert np.all(flipped != labels)
    delta = {"w": torch.tensor([1.0, -2.0])}
    assert torch.equal(sign_flip(delta)["w"], torch.tensor([-1.0, 2.0]))
    assert torch.equal(free_ride(delta)["w"], torch.zeros(2))
    assert gaussian_noise(delta, 0.1, 42)["w"].shape == delta["w"].shape
    assert lie_vector([np.array([1.0, 2.0]), np.array([3.0, 4.0])], 1.0).shape == (2,)
    rebuilt = vector_to_delta(np.array([1.0, 2.0]), {"w": torch.zeros(2, dtype=torch.float64)})
    assert rebuilt["w"].dtype == torch.float64
    assert rebuilt["w"].device == delta["w"].device
    a, b = combined_attack_sets(40)
    assert a.isdisjoint(b)
