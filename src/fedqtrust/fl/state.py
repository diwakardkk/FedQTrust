"""State-dict utilities."""

from __future__ import annotations

import torch


StateDict = dict[str, torch.Tensor]


def clone_state(state: StateDict) -> StateDict:
    return {k: v.detach().clone() for k, v in state.items()}


def subtract_state(local: StateDict, global_state: StateDict) -> StateDict:
    return {k: local[k].detach() - global_state[k].detach() for k in global_state}


def add_delta(global_state: StateDict, delta: StateDict) -> StateDict:
    return {k: global_state[k].detach() + delta[k].detach() for k in global_state}


def flatten_state(state: StateDict) -> torch.Tensor:
    return torch.cat([v.detach().reshape(-1).float().cpu() for _, v in sorted(state.items())])

