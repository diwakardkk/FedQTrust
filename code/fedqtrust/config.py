"""Configuration objects and validation for FedQTrust."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


SEEDS = [42, 123, 456, 789, 999]
DATASETS = {
    "pathmnist": {"display": "PathMNIST", "classes": 9, "clients": 8},
    "octmnist": {"display": "OCTMNIST", "classes": 4, "clients": 8},
    "pneumoniamnist": {"display": "PneumoniaMNIST", "classes": 2, "clients": 8},
    "retinamnist": {"display": "RetinaMNIST", "classes": 5, "clients": 8},
    "breastmnist": {"display": "BreastMNIST", "classes": 2, "clients": 8},
}


@dataclass
class TrustConfig:
    alpha: float = 0.4
    beta: float = 0.3
    gamma: float = 0.3
    tau_min: float = 0.1
    theta: float = 0.4
    reputation_decay: float = 0.9
    consistency_window: int = 5


@dataclass
class QuboConfig:
    lambda_: float = 1.0
    mu: float = 2.0
    qaoa_reps: int = 3
    qaoa_shots: int = 1024
    qaoa_maxiter: int = 100
    qaoa_timeout: float = 60.0


@dataclass
class TrainingConfig:
    rounds: int = 100
    local_epochs: int = 3
    batch_size: int = 32
    learning_rate: float = 0.01
    weight_decay: float = 1e-4
    clients_total: int = 40
    clients_per_round: int = 5
    amp: bool = False
    torch_compile: bool = False
    num_workers: int = 0
    prefetch_factor: int | None = None
    checkpoint_every: int = 5


@dataclass
class DataConfig:
    root: str = "data/raw"
    partitions: str = "data/partitions"
    partition_alpha: float = 0.5
    partition_seed: int = 42
    validation_fraction: float = 0.2
    min_samples_per_class: int = 10
    max_partition_retries: int = 20
    image_size: int = 28
    channels: int = 1
    angle_scale: float = 3.141592653589793


@dataclass
class ExperimentConfig:
    profile: str = "smoke"
    device: str = "auto"
    output_dir: str = "output"
    seeds: list[int] = field(default_factory=lambda: SEEDS.copy())
    datasets: list[str] = field(default_factory=lambda: list(DATASETS))
    trust: TrustConfig = field(default_factory=TrustConfig)
    qubo: QuboConfig = field(default_factory=QuboConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    attack_start_round: int = 10

    def validate(self) -> None:
        if abs((self.trust.alpha + self.trust.beta + self.trust.gamma) - 1.0) > 1e-8:
            raise ValueError("trust alpha+beta+gamma must equal 1")
        if not 0.0 <= self.trust.theta <= 1.0:
            raise ValueError("trust theta must be in [0, 1]")
        if self.training.clients_per_round > self.training.clients_total:
            raise ValueError("clients_per_round cannot exceed clients_total")
        if self.qubo.mu <= 0 or self.qubo.lambda_ <= 0:
            raise ValueError("QUBO lambda and mu must be positive")
        if self.qubo.qaoa_reps < 1:
            raise ValueError("QAOA reps must be >= 1")
        if self.training.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.profile not in {"smoke", "phase1", "phase2", "paper"}:
            raise ValueError("profile must be one of smoke, phase1, phase2, paper")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["qubo"]["lambda"] = data["qubo"].pop("lambda_")
        return data


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | None = None, **overrides: Any) -> ExperimentConfig:
    raw: dict[str, Any] = {}
    if path:
        with Path(path).open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
    raw = _merge(raw, overrides)
    trust = TrustConfig(**raw.get("trust", {}))
    qubo_raw = dict(raw.get("qubo", {}))
    if "lambda" in qubo_raw:
        qubo_raw["lambda_"] = qubo_raw.pop("lambda")
    qubo = QuboConfig(**qubo_raw)
    training = TrainingConfig(**raw.get("training", {}))
    data = DataConfig(**raw.get("data", {}))
    top = {k: v for k, v in raw.items() if k not in {"trust", "qubo", "training", "data"}}
    cfg = ExperimentConfig(trust=trust, qubo=qubo, training=training, data=data, **top)
    cfg.validate()
    return cfg

