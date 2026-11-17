"""Seed control across Python, NumPy, PyTorch, and Qiskit-compatible code."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np


@dataclass
class SeedState:
    seed: int
    deterministic: bool


def seed_everything(seed: int, deterministic: bool = True) -> SeedState:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.use_deterministic_algorithms(False)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    except Exception:
        ...
    try:
        from qiskit_algorithms.utils import algorithm_globals

        algorithm_globals.random_seed = seed
    except Exception:
        ...
    return SeedState(seed=seed, deterministic=deterministic)

