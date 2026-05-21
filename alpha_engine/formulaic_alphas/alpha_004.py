"""
WorldQuant 101 Formulaic Alphas - Alpha#004.

Paper formula:
    Alpha#004 =
        -1 * Ts_Rank(rank(low), 9)

Required paper-level input fields:
    symbol
    date
    low

Factor meaning:
    Alpha#004 first ranks low price cross-sectionally by date, then calculates
    the time-series rank of that cross-sectional low rank over the latest
    9 trading days for each symbol.

    The final value is multiplied by -1, so higher recent time-series rank
    of low-price rank becomes a more negative alpha value.

Usage in this project:
    The factor is computed from T0 and historical data only.
    It is valid for later T+1 / T+2 / T+3 forward-return validation.
"""

from __future__ import annotations

import pandas as pd

from alpha_engine.operators.cross_sectional import cs_rank_by_date
from alpha_engine.operators.time_series import ts_rank_by_symbol


ALPHA_NAME = "alpha_004"


def compute_alpha_004(
    market_df: pd.DataFrame,
    *,
    symbol_col: str = "symbol",
    date_col: str = "date",
    low_col: str = "low",
    output_col: str = ALPHA_NAME,
) -> pd.DataFrame:
    required = {symbol_col, date_col, low_col}
    missing = sorted(required - set(market_df.columns))
    if missing:
        raise ValueError(f"Missing required columns for alpha_004: {missing}")

    df = market_df[[symbol_col, date_col, low_col]].copy()

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce").dt.normalize()
    df[low_col] = pd.to_numeric(df[low_col], errors="coerce")

    df = df.dropna(subset=[symbol_col, date_col, low_col])
    df = df.sort_values([symbol_col, date_col], kind="mergesort").reset_index(drop=True)

    df["_rank_low"] = cs_rank_by_date(df, low_col, date_col=date_col)
    df[output_col] = -1.0 * ts_rank_by_symbol(
        df,
        "_rank_low",
        9,
        symbol_col=symbol_col,
    )

    df[output_col] = df[output_col].clip(lower=-1.0, upper=0.0)

    return df[[symbol_col, date_col, output_col]].copy()
