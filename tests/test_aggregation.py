import torch

from fedqtrust.fl.aggregation import fedavg, trust_weighted_delta


def test_trust_weighted_aggregation_manual():
    d1 = {"w": torch.tensor([1.0, 3.0])}
    d2 = {"w": torch.tensor([3.0, 5.0])}
    got = trust_weighted_delta([d1, d2], [1, 3], [0.5, 1.0])["w"]
    expected = (d1["w"] * 0.5 + d2["w"] * 3.0) / 3.5
    assert torch.allclose(got, expected)
    assert torch.allclose(fedavg([d1, d2], [1, 1])["w"], torch.tensor([2.0, 4.0]))

