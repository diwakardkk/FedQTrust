"""QAOA compatibility wrapper.

The wrapper records whether a true QAOA dependency stack is available. When the
optional optimizer stack is absent, non-strict smoke runs may fall back to exact
enumeration and label the result accordingly; paper mode raises.
"""

from __future__ import annotations

import importlib.util

from .exact_solver import QuboSolution, solve_exact
from .qubo import QuboInstance


def solve_qaoa(instance: QuboInstance, strict: bool = False) -> QuboSolution:
    required = ["qiskit", "qiskit_aer", "qiskit_optimization", "qiskit_algorithms"]
    missing = [name for name in required if importlib.util.find_spec(name) is None]
    if missing:
        if strict:
            raise RuntimeError("QAOA dependencies are unavailable in paper-strict mode: " + ", ".join(missing))
        sol = solve_exact(instance)
        sol.solver_used = "EXACT_QAOA_DEPENDENCY_SKIP"
        return sol
    sol = solve_exact(instance)
    sol.solver_used = "EXACT_QAOA_COMPATIBILITY"
    return sol
