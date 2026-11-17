"""Actual sampler-based QAOA; missing dependencies never become exact results."""
from __future__ import annotations
import time
import numpy as np
from .exact_solver import QuboSolution
from .qubo import QuboInstance, qubo_matrix, qubo_objective

class QAOATimeout(RuntimeError):
    """QAOA exceeded its time budget; the caller records an SA fallback (manuscript Sec. 3.4)."""


def solve_qaoa(
    instance: QuboInstance,
    strict=False,
    *,
    reps=3,
    shots=1024,
    maxiter=100,
    seed=42,
    sampler=None,
    timeout_s=None
):
    """Run QAOA and return the QAOA answer, not a classical best-of-samples search.

    The returned bitstring is the most probable outcome of the optimized QAOA
    circuit. MinimumEigenOptimizer's best-sampled bitstring and the distribution's
    expected objective <C> are attached as ``best_sampled_objective`` and
    ``expected_objective`` so E2 can report both honestly.
    """
    from qiskit.primitives import StatevectorSampler
    from qiskit_optimization import QuadraticProgram
    from qiskit_optimization.algorithms import MinimumEigenOptimizer
    from qiskit_optimization.minimum_eigensolvers import QAOA
    from qiskit_optimization.optimizers import COBYLA
    n = len(instance.trust)
    pass_manager = None
    if sampler is None and n > 12:
        from qiskit_aer.primitives import SamplerV2
        sampler = SamplerV2(
            default_shots=shots,
            seed=seed,
            # Shot-based statevector simulation (Sec. 3.4, |E| > 12). MPS is far slower for the
            # dense all-to-all QUBO couplings (>120 s at 15 qubits vs ~4 s).
            options={'backend_options': {'method': 'statevector'}}
        )
    if sampler is not None:
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
        pass_manager = generate_preset_pass_manager(
            optimization_level=1,
            basis_gates=getattr(
                sampler,
                'fedqtrust_basis_gates',
                ['rx', 'ry', 'rz', 'h', 'x', 'cx', 'measure']
            ),
            seed_transpiler=seed
        )
    start = time.perf_counter()
    q = qubo_matrix(instance.trust, instance.distances, instance.lambda_, instance.mu, instance.k)
    problem = QuadraticProgram()
    for i in range(n):
        problem.binary_var(f'x{i}')
    problem.minimize(
        constant=instance.mu * instance.k ** 2,
        linear={i: float(q[i, i]) for i in range(n)},
        quadratic={(i, j): float(q[i, j]) for i in range(n) for j in range(i + 1, n)}
    )
    rng = np.random.default_rng(seed)

    def watchdog(*_):
        if timeout_s is not None and time.perf_counter() - start > timeout_s:
            raise QAOATimeout(f'QAOA exceeded {timeout_s}s on {n} qubits')
    solver = QAOA(
        sampler=sampler or StatevectorSampler(default_shots=shots, seed=seed),
        optimizer=COBYLA(maxiter=maxiter),
        reps=reps,
        initial_point=rng.uniform(0, 2 * np.pi, 2 * reps),
        pass_manager=pass_manager,
        callback=watchdog
    )
    result = MinimumEigenOptimizer(solver).solve(problem)
    samples = list(result.samples or [])
    if samples:
        top = max(samples, key=lambda s: s.probability)
        x = np.rint(top.x).astype(int)
        expected = float(sum(s.probability * s.fval for s in samples) / max(sum(s.probability for s in samples), 1e-12))
    else:
        x = np.rint(result.x).astype(int)
        expected = float('nan')
    value = qubo_objective(x, instance.trust, instance.distances, instance.lambda_, instance.mu, instance.k)
    best_x = np.rint(result.x).astype(int)
    solution = QuboSolution(x, value, time.perf_counter() - start, 'QAOA', int(x.sum()) == instance.k)
    solution.expected_objective = expected
    solution.best_sampled_objective = qubo_objective(
        best_x, instance.trust, instance.distances, instance.lambda_, instance.mu, instance.k)
    return solution
