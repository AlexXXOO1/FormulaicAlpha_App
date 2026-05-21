"""
WorldQuant 101 Formulaic Alphas - Alpha#003.

Paper formula:
    Alpha#003 =
        -1 * correlation(rank(open), rank(volume), 10)

Required paper-level input fields:
    symbol
    date
    open
    volume

Factor meaning:
    Alpha#003 measures the negative 10-day rolling correlation between:
    1. cross-sectional rank of open price
    2. cross-sectional rank of volume

    The paper multiplies by -1, so the final alpha is contrarian to
    recent positive co-movement between high open-price rank and high volume rank.

Usage in this project:
    The factor is computed from T0 and historical data only.
    It is valid for later T+1 / T+2 / T+3 forward-return validation.
"""

from __future__ import annotations

import pandas as pd

from alpha_engine.operators.cross_sectional import cs_rank_by_date
from alpha_engine.operators.time_series import rolling_corr_by_symbol


ALPHA_NAME = "alpha_003"


def compute_alpha_003(
    market_df: pd.DataFrame,
    *,
    symbol_col: str = "symbol",
    date_col: str = "date",
    open_col: str = "open",
    volume_col: str = "volume",
    output_col: str = ALPHA_NAME,
) -> pd.DataFrame:
    required = {symbol_col, date_col, open_col, volume_col}
    missing = sorted(required - set(market_df.columns))
    if missing:
        raise ValueError(f"Missing required columns for alpha_003: {missing}")

    df = market_df[[symbol_col, date_col, open_col, volume_col]].copy()

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce").dt.normalize()
    df[open_col] = pd.to_numeric(df[open_col], errors="coerce")
    df[volume_col] = pd.to_numeric(df[volume_col], errors="coerce")

    df = df.dropna(subset=[symbol_col, date_col])
    df = df.sort_values([symbol_col, date_col], kind="mergesort").reset_index(drop=True)

    df["_rank_open"] = cs_rank_by_date(df, open_col, date_col=date_col)
    df["_rank_volume"] = cs_rank_by_date(df, volume_col, date_col=date_col)

    raw_alpha = -1.0 * rolling_corr_by_symbol(
        df,
        "_rank_open",
        "_rank_volume",
        10,
        symbol_col=symbol_col,
    )

    df[output_col] = raw_alpha.clip(lower=-1.0, upper=1.0)

    return df[[symbol_col, date_col, output_col]].copy()
