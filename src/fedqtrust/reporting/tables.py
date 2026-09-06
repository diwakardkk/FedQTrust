"""Table generation helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _fallback_markdown(df: pd.DataFrame) -> str:
    headers = [str(column) for column in df.columns]
    rows = [[str(value) for value in row] for row in df.to_numpy()]
    widths = [
        max([len(header), *(len(row[index]) for row in rows)] or [len(header)])
        for index, header in enumerate(headers)
    ]

    def fmt(values: list[str]) -> str:
        return "| " + " | ".join(value.ljust(widths[index]) for index, value in enumerate(values)) + " |"

    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    return "\n".join([fmt(headers), separator, *(fmt(row) for row in rows)])


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
    try:
        markdown = df.to_markdown(index=False)
    except ImportError:
        markdown = _fallback_markdown(df)
    paths["md"].write_text(markdown, encoding="utf-8")
    return paths
