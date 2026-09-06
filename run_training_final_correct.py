#!/usr/bin/env python3
"""Known-good FedQTrust training runner with saved results.

This is a standalone compatibility entrypoint based on the previously validated
100-round script. It trains a simple CNN on the five MedMNIST datasets and saves
round metrics, final metrics, environment metadata, plots, and a zip bundle.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import medmnist
import numpy as np
import pandas as pd
import torch
from medmnist import BreastMNIST, INFO, OCTMNIST, PathMNIST, PneumoniaMNIST, RetinaMNIST
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms


DATASETS = {
    "pathmnist": PathMNIST,
    "octmnist": OCTMNIST,
    "pneumoniamnist": PneumoniaMNIST,
    "retinamnist": RetinaMNIST,
    "breastmnist": BreastMNIST,
}


class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.25)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.reshape(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


class ScalarLabelDataset(Dataset):
    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image, label = self.dataset[idx]
        return image, torch.tensor(_scalar_label(label), dtype=torch.long)


def _scalar_label(label: object) -> int:
    if isinstance(label, torch.Tensor):
        label = label.detach().cpu().numpy()
    if isinstance(label, np.ndarray):
        flat = label.reshape(-1)
        if flat.size > 1:
            return int(np.argmax(flat))
        return int(flat[0])
    return int(label)


def _num_classes(dataset_name: str) -> int:
    return len(INFO[dataset_name]["label"])


def _select_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested, but torch.cuda.is_available() is false")
    return requested


def _environment(device: str) -> dict[str, object]:
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_torch_version": torch.__version__,
        "medmnist_version": getattr(medmnist, "__version__", "unknown"),
        "cuda_available": torch.cuda.is_available(),
        "selected_device": device,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the known-good FedQTrust training workflow.")
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--clients", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--data-root", default="data/raw")
    parser.add_argument("--output-dir", default="output/final_correct")
    parser.add_argument("--datasets", default="pathmnist,octmnist,pneumoniamnist,retinamnist,breastmnist")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--download-data", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def _make_run_dir(output_dir: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_dir) / f"run_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _plot_losses(round_df: pd.DataFrame, run_dir: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(run_dir / "matplotlib_cache"))
    os.environ.setdefault("XDG_CACHE_HOME", str(run_dir / "cache"))
    (run_dir / "matplotlib_cache").mkdir(parents=True, exist_ok=True)
    (run_dir / "cache").mkdir(parents=True, exist_ok=True)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if round_df.empty:
        return
    plt.figure(figsize=(10, 6))
    for dataset, group in round_df.groupby("dataset"):
        plt.plot(group["round"], group["avg_round_loss"], label=dataset)
    plt.xlabel("Round")
    plt.ylabel("Average training loss")
    plt.title("FedQTrust known-good training loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(run_dir / "training_loss.png", dpi=200)
    plt.close()


def main() -> int:
    args = _parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = _select_device(args.device)
    run_dir = _make_run_dir(args.output_dir)
    print("=" * 60, flush=True)
    print("FEDQTRUST - FINAL CORRECT TRAINING", flush=True)
    print("=" * 60, flush=True)
    print(f"Device: {device}", flush=True)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"Output: {run_dir}", flush=True)
    print("=" * 60, flush=True)

    (run_dir / "environment.json").write_text(json.dumps(_environment(device), indent=2), encoding="utf-8")
    (run_dir / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    requested_datasets = [item.strip().lower() for item in args.datasets.split(",") if item.strip()]
    round_rows: list[dict[str, object]] = []
    final_rows: list[dict[str, object]] = []

    for dataset_name in requested_datasets:
        if dataset_name not in DATASETS:
            final_rows.append({"dataset": dataset_name, "status": "FAILED", "error": "unknown dataset"})
            continue

        print(f"\n{'=' * 50}", flush=True)
        print(f"TRAINING ON: {dataset_name.upper()}", flush=True)
        print(f"{'=' * 50}", flush=True)
        start_time = time.perf_counter()

        try:
            num_classes = _num_classes(dataset_name)
            raw_data = DATASETS[dataset_name](
                split="train",
                root=args.data_root,
                download=args.download_data,
                transform=transform,
            )
            train_data = ScalarLabelDataset(raw_data)
            indices = list(range(len(train_data)))
            if args.max_samples is not None:
                indices = indices[: max(1, min(args.max_samples, len(indices)))]
            train_subset = Subset(train_data, indices)

            print(f"Classes: {num_classes}", flush=True)
            print(f"Samples: {len(train_subset)}", flush=True)

            model = SimpleCNN(num_classes).to(device)
            global_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            losses: list[float] = []

            for round_idx in range(1, args.rounds + 1):
                client_losses: list[float] = []
                samples_per_client = max(1, len(train_subset) // args.clients)

                for client_idx in range(args.clients):
                    start_idx = client_idx * samples_per_client
                    end_idx = len(train_subset) if client_idx == args.clients - 1 else min(
                        (client_idx + 1) * samples_per_client,
                        len(train_subset),
                    )
                    if start_idx >= len(train_subset):
                        continue

                    subset = Subset(train_subset, range(start_idx, end_idx))
                    loader = DataLoader(subset, batch_size=args.batch_size, shuffle=True)
                    local_model = SimpleCNN(num_classes).to(device)
                    local_model.load_state_dict(global_state)
                    optimizer = torch.optim.Adam(local_model.parameters(), lr=args.learning_rate)
                    loss_fn = nn.CrossEntropyLoss()
                    local_model.train()

                    total_loss = 0.0
                    num_batches = 0
                    for batch_x, batch_y in loader:
                        batch_x = batch_x.to(device)
                        batch_y = batch_y.to(device).long()
                        optimizer.zero_grad(set_to_none=True)
                        output = local_model(batch_x)
                        loss = loss_fn(output, batch_y)
                        loss.backward()
                        optimizer.step()
                        total_loss += float(loss.item())
                        num_batches += 1

                    avg_loss = total_loss / max(num_batches, 1)
                    client_losses.append(avg_loss)
                    global_state = {k: v.detach().clone() for k, v in local_model.state_dict().items()}

                avg_round_loss = float(sum(client_losses) / max(len(client_losses), 1))
                losses.append(avg_round_loss)
                row: dict[str, object] = {
                    "dataset": dataset_name,
                    "round": round_idx,
                    "avg_round_loss": avg_round_loss,
                }
                for client_idx, client_loss in enumerate(client_losses):
                    row[f"client_{client_idx}_loss"] = client_loss
                round_rows.append(row)
                print(f"Round {round_idx}/{args.rounds} - Avg loss: {avg_round_loss:.4f}", flush=True)

            elapsed = time.perf_counter() - start_time
            torch.save(global_state, run_dir / f"{dataset_name}_simplecnn_state.pt")
            final_rows.append(
                {
                    "dataset": dataset_name,
                    "status": "COMPLETED",
                    "num_classes": num_classes,
                    "samples": len(train_subset),
                    "rounds": args.rounds,
                    "clients": args.clients,
                    "final_loss": losses[-1] if losses else None,
                    "best_loss": min(losses) if losses else None,
                    "time_sec": elapsed,
                    "device": device,
                }
            )
            print(f"{dataset_name.upper()} COMPLETE - final loss {losses[-1]:.4f}", flush=True)
        except Exception as exc:
            final_rows.append(
                {
                    "dataset": dataset_name,
                    "status": "FAILED",
                    "error": str(exc),
                    "time_sec": time.perf_counter() - start_time,
                    "device": device,
                }
            )
            print(f"ERROR on {dataset_name}: {exc}", flush=True)

    round_df = pd.DataFrame(round_rows)
    final_df = pd.DataFrame(final_rows)
    round_df.to_csv(run_dir / "round_metrics.csv", index=False)
    final_df.to_csv(run_dir / "final_metrics.csv", index=False)
    _plot_losses(round_df, run_dir)

    completed = int((final_df["status"] == "COMPLETED").sum()) if not final_df.empty else 0
    summary = {
        "run_dir": str(run_dir),
        "completed_datasets": completed,
        "total_datasets": len(requested_datasets),
        "rounds": args.rounds,
        "device": device,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (run_dir / "DONE").write_text("FedQTrust final-correct training complete.\n", encoding="utf-8")
    archive = shutil.make_archive(str(run_dir), "zip", run_dir)
    print("=" * 60, flush=True)
    print("ALL EXPERIMENTS COMPLETE", flush=True)
    print(f"Results: {run_dir}", flush=True)
    print(f"Zip: {archive}", flush=True)
    print("=" * 60, flush=True)
    return 0 if completed == len(requested_datasets) else 1


if __name__ == "__main__":
    raise SystemExit(main())
