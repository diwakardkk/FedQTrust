"""Little-is-enough model poisoning."""

from __future__ import annotations

import numpy as np
import torch

from fedqtrust.fl.state import StateDict


def lie_vector(honest_vectors: list[np.ndarray], z_max: float = 1.0) -> np.ndarray:
    matrix = np.stack(honest_vectors, axis=0)
    return matrix.mean(axis=0) - z_max * matrix.std(axis=0)


def vector_to_delta(vector: np.ndarray, template: StateDict) -> StateDict:
    out: StateDict = {}
    offset = 0
    tensor = torch.from_numpy(vector.astype("float32"))
    for key, value in sorted(template.items()):
        size = value.numel()
        out[key] = tensor[offset : offset + size].reshape(value.shape).to(device=value.device, dtype=value.dtype)
        offset += size
    return out


def baruch_z(n: int, malicious_count: int) -> float:
    """z_max of Baruch et al. (NeurIPS 2019): s = floor(n/2 + 1) - m, z = Phi^-1((n - s) / n)."""
    from scipy.stats import norm
    supporters = int(np.floor(n / 2 + 1)) - malicious_count
    return float(norm.ppf(np.clip((n - supporters) / n, 1e-6, 1 - 1e-6)))


def adaptive_lie(honest_vectors, malicious_count, z_cap=2.0):
    """LIE z_max search specified in the manuscript (Task A2).

    w_mal = mu - z * sigma over honest updates (white-box simulation). z_max is the
    largest z in [0, z_cap] for which Krum still selects the malicious update
    (grid + binary search). z_cap is the swept value {0.5, 1.0, 2.0}. If Krum
    accepts no z > 0, the Baruch et al. normal-quantile z (capped by z_cap) is used,
    so the attack is never silently reduced to the honest mean.

    Returns (vector, z, accepted, effective_f). This white-box information is
    restricted to attack generation and never supplied to the defense.
    """
    from fedqtrust.baselines.krum import krum_select
    matrix = np.stack(honest_vectors)
    mean = matrix.mean(0)
    std = matrix.std(0)
    n = len(matrix) + malicious_count

    def krum_accepts(z):
        vectors = [mean - z * std for _ in range(malicious_count)] + list(matrix)
        idx, eff = krum_select(vectors, malicious_count)
        return idx < malicious_count, eff

    best = 0.0
    accepted = False
    effective = 0
    grid = np.linspace(0.0, z_cap, 33)
    # Krum acceptance need not be monotone in z: take the largest accepted grid point.
    for z in grid[1:]:
        ok, effective = krum_accepts(float(z))
        if ok:
            best, accepted = float(z), True
    if accepted:
        lo, hi = best, min(z_cap, best + z_cap / 32)
        for _ in range(20):
            mid = (lo + hi) / 2
            ok, effective = krum_accepts(mid)
            if ok:
                lo = mid
            else:
                hi = mid
        best = lo
    else:
        best = min(z_cap, max(0.0, baruch_z(n, malicious_count)))
    return mean - best * std, best, accepted, effective
