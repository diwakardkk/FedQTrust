import itertools

import numpy as np

from fedqtrust.quantum.exact_solver import solve_exact
from fedqtrust.quantum.qubo import QuboInstance, matrix_objective_upper, project_cardinality, qubo_matrix, qubo_objective
from fedqtrust.quantum.simulated_annealing import solve_sa


def test_qubo_matrix_matches_direct_objective():
    rng = np.random.default_rng(42)
    trust = rng.random(4)
    d = rng.random((4, 4))
    distances = (d + d.T) / 2
    np.fill_diagonal(distances, 0)
    q = qubo_matrix(trust, distances, 1.3, 2.1, 2)
    constant = 2.1 * 2 * 2
    for bits in itertools.product([0, 1], repeat=4):
        x = np.array(bits)
        assert np.isclose(qubo_objective(x, trust, distances, 1.3, 2.1, 2), matrix_objective_upper(x, q, constant))


def test_exact_sa_and_projection_cardinality():
    trust = np.array([0.9, 0.8, 0.1])
    distances = np.zeros((3, 3))
    instance = QuboInstance(trust, distances, 1.0, 2.0, 2, [0, 1, 2])
    exact = solve_exact(instance)
    sa = solve_sa(instance, seed=1, steps=20)
    assert exact.feasible_cardinality
    assert sa.feasible_cardinality
    assert project_cardinality(np.array([1, 1, 1]), trust, 2).sum() == 2

