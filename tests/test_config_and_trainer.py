from fedqtrust.config import load_config
from fedqtrust.fl.trainer import tiny_fl_run


def test_config_validation_and_tiny_fl():
    cfg = load_config(None, profile="smoke")
    assert cfg.training.clients_per_round == 5
    rows = tiny_fl_run("cpu")
    assert len(rows) == 2
    assert rows[0]["round"] == 1

