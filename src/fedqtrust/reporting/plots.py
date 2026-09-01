"""Publication-quality plotting helpers."""

from __future__ import annotations

import os
from pathlib import Path

Path("/private/tmp/fedqtrust-matplotlib").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/fedqtrust-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def save_smoke_plot(output_dir: str | Path) -> tuple[Path, Path, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    data = pd.DataFrame({"round": [1, 2], "accuracy": [0.5, 0.55], "method": ["smoke", "smoke"]})
    csv_path = output / "smoke_accuracy.csv"
    data.to_csv(csv_path, index=False)
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    ax.plot(data["round"], data["accuracy"], marker="o", linewidth=2, label="smoke")
    ax.set_xlabel("Round")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0)
    pdf = output / "smoke_accuracy.pdf"
    png = output / "smoke_accuracy.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return csv_path, pdf, png
