"""Table generation helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def save_table(df: pd.DataFrame, output_dir: str | Path, stem: str) -> dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "csv": output / f"{stem}.csv",
        "tex": output / f"{stem}.tex",
        "md": output / f"{stem}.md",
    }
    df.to_csv(paths["csv"], index=False)
    paths["tex"].write_text(df.to_latex(index=False), encoding="utf-8")
    paths["md"].write_text(df.to_markdown(index=False), encoding="utf-8")
    return paths

