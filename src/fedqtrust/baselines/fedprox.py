"""FedProx loss helper."""

from __future__ import annotations

import torch


def fedprox_penalty(model, global_state: dict[str, torch.Tensor], mu_p: float = 0.01) -> torch.Tensor:
    penalty = torch.zeros((), device=next(model.parameters()).device)
    for name, param in model.named_parameters():
        if name in global_state:
            penalty = penalty + torch.sum((param - global_state[name].to(param.device)) ** 2)
    return 0.5 * mu_p * penalty

