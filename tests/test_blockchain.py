from fedqtrust.blockchain.memory_backend import InMemoryBlockchain


def test_inmemory_blockchain():
    backend = InMemoryBlockchain()
    assert backend.ping()
    backend.set_trust(3, 0.4)
    backend.log_selection({"round": 1, "selected": [3]})
    assert backend.get_trust(3) == 0.4
    assert backend.selection_log[0]["selected"] == [3]

