"""LaTeX table helpers."""

from __future__ import annotations

import pandas as pd


def dataframe_to_latex(df: pd.DataFrame) -> str:
    return df.to_latex(index=False)

