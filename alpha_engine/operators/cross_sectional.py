from __future__ import annotations

import pandas as pd


def cs_rank_by_date(
    df: pd.DataFrame,
    value_col: str,
    *,
    date_col: str = "date",
    output_col: str | None = None,
    pct: bool = True,
) -> pd.Series:
    ranked = df.groupby(date_col, sort=False)[value_col].rank(method="average", pct=pct)
    if output_col:
        ranked.name = output_col
    return ranked
