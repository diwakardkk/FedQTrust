"""Dataset download command helpers."""

from __future__ import annotations

from fedqtrust.data.datasets import ensure_all_datasets


def download_all(root: str = "data/raw") -> list[dict]:
    return ensure_all_datasets(root, True, "data/metadata")

