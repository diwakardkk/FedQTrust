"""Federated aggregation rules."""

from __future__ import annotations

import torch

from .state import StateDict


def fedavg(deltas: list[StateDict], sample_counts: list[int]) -> StateDict:
    return trust_weighted_delta(deltas, sample_counts, [1.0] * len(deltas))


def trust_weighted_delta(deltas: list[StateDict], sample_counts: list[int], trust_scores: list[float]) -> StateDict:
    if not deltas:
        raise ValueError("at least one delta is required")
    denom = sum(float(n) * float(t) for n, t in zip(sample_counts, trust_scores))
    if denom <= 0:
        raise ValueError("aggregation denominator is zero")
    keys = deltas[0].keys()
    out: StateDict = {}
    for key in keys:
        acc = torch.zeros_like(deltas[0][key], dtype=deltas[0][key].dtype)
        for delta, n, tau in zip(deltas, sample_counts, trust_scores):
            update = delta[key].to(device=acc.device, dtype=acc.dtype)
            acc = acc + update * (float(n) * float(tau) / denom)
        out[key] = acc
    return out
