import numpy as np

from fedqtrust.fl.trust import TrustManager, safe_cosine


def test_initial_trust_exact():
    manager = TrustManager(2)
    assert manager.trust[0] == 0.65


def test_reputation_and_consistency_update():
    manager = TrustManager(1)
    consistency = manager.update_consistency(0, np.array([1.0, 0.0]), np.array([1.0, 0.0]))
    trust = manager.update_round(0, 0.75, consistency)
    assert trust > 0.65
    assert safe_cosine(np.zeros(2), np.ones(2)) == 0.0

