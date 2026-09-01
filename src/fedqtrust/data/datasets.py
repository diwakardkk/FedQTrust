"""MedMNIST dataset loading and validation."""

from __future__ import annotations

import csv
import hashlib
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .transforms import ToGrayTensor


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    medmnist_flag: str
    display: str
    classes: int


DATASET_SPECS = [
    DatasetSpec("pathmnist", "pathmnist", "PathMNIST", 9),
    DatasetSpec("octmnist", "octmnist", "OCTMNIST", 4),
    DatasetSpec("pneumoniamnist", "pneumoniamnist", "PneumoniaMNIST", 2),
    DatasetSpec("retinamnist", "retinamnist", "RetinaMNIST", 5),
    DatasetSpec("breastmnist", "breastmnist", "BreastMNIST", 2),
]


def _medmnist_class(flag: str):
    import medmnist

    info = medmnist.INFO[flag]
    return getattr(medmnist, info["python_class"])


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _direct_download_if_needed(flag: str, root: Path) -> None:
    import medmnist

    info = medmnist.INFO[flag]
    path = root / f"{flag}.npz"
    expected = info.get("MD5")
    if path.exists() and (not expected or _md5(path) == expected):
        return
    url = info["url"]
    print(f"[DATA] Direct download fallback for {flag}", flush=True)
    proc = subprocess.run(["curl", "-L", "--fail", "-o", str(path), url], capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or f"curl failed for {url}")
    if expected and _md5(path) != expected:
        path.unlink(missing_ok=True)
        raise RuntimeError(f"MD5 mismatch for {path.name}")


def load_medmnist_dataset(flag: str, split: str, root: str | Path, download: bool):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    if download:
        _direct_download_if_needed(flag, root)
    dataset_cls = _medmnist_class(flag)
    return dataset_cls(
        split=split,
        transform=ToGrayTensor(),
        root=str(root),
        download=False,
    )


def label_array(dataset: Any) -> np.ndarray:
    labels = getattr(dataset, "labels", None)
    if labels is None:
        labels = getattr(dataset, "imgs", None)
    arr = np.asarray(labels)
    if arr.ndim > 1:
        arr = arr[:, 0]
    return arr.astype(int)


def ensure_all_datasets(root: str | Path, download: bool, metadata_dir: str | Path) -> list[dict[str, Any]]:
    import medmnist

    root = Path(root)
    metadata_dir = Path(metadata_dir)
    root.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for spec in DATASET_SPECS:
        print(f"[DATA] Checking {spec.display}", flush=True)
        splits = {}
        for split in ("train", "val", "test"):
            print(f"[DATA] {spec.display} {split}: {'download/check' if download else 'check'}", flush=True)
            ds = load_medmnist_dataset(spec.medmnist_flag, split, root, download)
            labels = label_array(ds)
            splits[split] = len(ds)
            if labels.size and (labels.min() < 0 or labels.max() >= spec.classes):
                raise ValueError(f"{spec.display} {split} labels outside [0,{spec.classes})")
        rows.append(
            {
                "dataset": spec.key,
                "display": spec.display,
                "medmnist_version": getattr(medmnist, "__version__", "unknown"),
                "actual_train_samples": splits["train"],
                "actual_val_samples": splits["val"],
                "actual_test_samples": splits["test"],
                "number_of_classes": spec.classes,
                "channel_count": 1,
                "image_resolution": "28x28",
                "download_status": "checked",
            }
        )
        print(f"[DATA] Verified {spec.display}", flush=True)
    out = metadata_dir / "dataset_manifest.csv"
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows
