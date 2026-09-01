"""Tiny FL trainer used by smoke tests and as the base orchestration path."""

from __future__ import annotations

import time

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from fedqtrust.fl.aggregation import fedavg
from fedqtrust.fl.state import add_delta, clone_state, subtract_state
from fedqtrust.models.classical_cnn import ClassicalCNN


def train_one_client(model: nn.Module, loader: DataLoader, global_state: dict[str, torch.Tensor], lr: float, device: str) -> tuple[dict[str, torch.Tensor], float]:
    local = type(model)(num_classes=model.head.out_features).to(device)
    local.load_state_dict(global_state)
    opt = torch.optim.Adam(local.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    batches = 0
    local.train()
    for x, y in loader:
        x = x.to(device)
        y = y.to(device).long()
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(local(x), y)
        loss.backward()
        opt.step()
        total_loss += float(loss.detach().cpu())
        batches += 1
    return subtract_state(local.state_dict(), global_state), total_loss / max(batches, 1)


def tiny_fl_run(device: str = "cpu") -> list[dict[str, float]]:
    gen = torch.Generator().manual_seed(42)
    x = torch.rand(24, 1, 28, 28, generator=gen)
    y = torch.randint(0, 2, (24,), generator=gen)
    loaders = [
        DataLoader(TensorDataset(x[:12], y[:12]), batch_size=4, shuffle=False),
        DataLoader(TensorDataset(x[12:], y[12:]), batch_size=4, shuffle=False),
    ]
    global_model = ClassicalCNN(2).to(device)
    rows: list[dict[str, float]] = []
    for rnd in range(1, 3):
        start = time.perf_counter()
        global_state = clone_state(global_model.state_dict())
        deltas = []
        losses = []
        for loader in loaders:
            delta, loss = train_one_client(global_model, loader, global_state, 0.01 / (rnd**0.5), device)
            deltas.append(delta)
            losses.append(loss)
        avg_delta = fedavg(deltas, [12, 12])
        global_model.load_state_dict(add_delta(global_state, avg_delta))
        rows.append({"round": rnd, "loss": sum(losses) / len(losses), "round_time": time.perf_counter() - start})
    return rows

