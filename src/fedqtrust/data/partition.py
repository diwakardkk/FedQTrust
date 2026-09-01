"""Dirichlet client partitioning."""

from __future__ import annotations

import csv
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class ClientSplit:
    client_id: int
    dataset: str
    train_indices: list[int]
    validation_indices: list[int]
    class_counts: dict[int, int]


def dirichlet_partition(
    labels: np.ndarray,
    num_clients: int,
    alpha: float,
    seed: int,
    validation_fraction: float = 0.2,
    min_samples_per_class: int = 10,
    max_retries: int = 20,
) -> tuple[list[ClientSplit], list[str]]:
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels).astype(int)
    classes = sorted(int(c) for c in np.unique(labels))
    notes: list[str] = []
    best: list[list[int]] | None = None
    for attempt in range(max_retries):
        client_indices = [[] for _ in range(num_clients)]
        for cls in classes:
            cls_idx = np.where(labels == cls)[0]
            rng.shuffle(cls_idx)
            proportions = rng.dirichlet(np.full(num_clients, alpha))
            cuts = (np.cumsum(proportions)[:-1] * len(cls_idx)).astype(int)
            chunks = np.split(cls_idx, cuts)
            for cid, chunk in enumerate(chunks):
                client_indices[cid].extend(int(i) for i in chunk)
        counts = [np.bincount(labels[idxs], minlength=max(classes) + 1) if idxs else np.zeros(max(classes) + 1) for idxs in client_indices]
        best = client_indices
        if all(all(count[cls] >= min_samples_per_class for cls in classes) for count in counts):
            break
        if attempt == max_retries - 1:
            notes.append("Dirichlet sparse classes retained with global head dimensionality unchanged.")
    assert best is not None
    splits: list[ClientSplit] = []
    for cid, idxs in enumerate(best):
        idxs_arr = np.array(idxs, dtype=int)
        rng.shuffle(idxs_arr)
        val_n = int(round(len(idxs_arr) * validation_fraction))
        validation = idxs_arr[:val_n].tolist()
        train = idxs_arr[val_n:].tolist()
        counts = {int(cls): int(np.sum(labels[np.array(train, dtype=int)] == cls)) if train else 0 for cls in classes}
        splits.append(ClientSplit(cid, "", train, validation, counts))
    return splits, notes


def save_partitions(dataset: str, splits: list[ClientSplit], output_dir: str | Path, seed: int) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in splits:
        split.dataset = dataset
    path = output_dir / f"{dataset}_clients_seed{seed}.pkl"
    with path.open("wb") as handle:
        pickle.dump(splits, handle)
    return path


def write_data_stats(splits_by_dataset: dict[str, list[ClientSplit]], output_csv: str | Path) -> None:
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    classes = sorted({cls for splits in splits_by_dataset.values() for split in splits for cls in split.class_counts})
    fields = ["dataset", "client_id", "train_count", "validation_count"] + [f"class_{c}_count" for c in classes]
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for dataset, splits in splits_by_dataset.items():
            for split in splits:
                row = {
                    "dataset": dataset,
                    "client_id": split.client_id,
                    "train_count": len(split.train_indices),
                    "validation_count": len(split.validation_indices),
                }
                row.update({f"class_{c}_count": split.class_counts.get(c, 0) for c in classes})
                writer.writerow(row)

