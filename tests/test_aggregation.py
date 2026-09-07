import torch

from fedqtrust.fl.aggregation import fedavg, trust_weighted_delta
from fedqtrust.fl.state import add_delta, subtract_state


def test_trust_weighted_aggregation_manual():
    d1 = {"w": torch.tensor([1.0, 3.0])}
    d2 = {"w": torch.tensor([3.0, 5.0])}
    got = trust_weighted_delta([d1, d2], [1, 3], [0.5, 1.0])["w"]
    expected = (d1["w"] * 0.5 + d2["w"] * 3.0) / 3.5
    assert torch.allclose(got, expected)
    assert torch.allclose(fedavg([d1, d2], [1, 1])["w"], torch.tensor([2.0, 4.0]))


def test_state_math_aligns_delta_dtype_to_target():
    global_state = {"w": torch.tensor([1.0, 2.0], dtype=torch.float64)}
    delta = {"w": torch.tensor([0.5, -1.0], dtype=torch.float32)}
    updated = add_delta(global_state, delta)
    assert updated["w"].dtype == torch.float64
    assert torch.allclose(updated["w"], torch.tensor([1.5, 1.0], dtype=torch.float64))

    local = {"w": torch.tensor([2.0, 4.0], dtype=torch.float32)}
    diff = subtract_state(local, global_state)
    assert diff["w"].dtype == torch.float32
    assert torch.allclose(diff["w"], torch.tensor([1.0, 2.0], dtype=torch.float32))
