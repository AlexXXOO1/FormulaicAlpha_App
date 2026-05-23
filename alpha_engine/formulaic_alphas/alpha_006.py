"""
WorldQuant 101 Formulaic Alphas - Alpha#006.

Paper formula:
    Alpha#006 =
        -1 * correlation(open, volume, 10)

Required paper-level input fields:
    symbol
    date
    open
    volume

Factor meaning:
    Alpha#006 measures the negative 10-day rolling correlation between
    open price and volume.

    If open and volume move together strongly over the recent 10-day window,
    the raw correlation is positive and the alpha becomes negative.

    If open and volume move in opposite directions, the raw correlation is
    negative and the alpha becomes positive.

Usage in this project:
    The factor is computed from T0 and historical data only.
    It is valid for later T+1 / T+2 / T+3 forward-return validation.
"""

from __future__ import annotations

import pandas as pd

from alpha_engine.operators.time_series import rolling_corr_by_symbol


ALPHA_NAME = "alpha_006"


def compute_alpha_006(
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
        raise ValueError(f"Missing required columns for alpha_006: {missing}")

    df = market_df[[symbol_col, date_col, open_col, volume_col]].copy()

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce").dt.normalize()
    df[open_col] = pd.to_numeric(df[open_col], errors="coerce")
    df[volume_col] = pd.to_numeric(df[volume_col], errors="coerce")

    df = df.dropna(subset=[symbol_col, date_col])
    df = df.sort_values([symbol_col, date_col], kind="mergesort").reset_index(drop=True)

    raw_alpha = -1.0 * rolling_corr_by_symbol(
        df,
        open_col,
        volume_col,
        10,
        symbol_col=symbol_col,
    )

    df[output_col] = raw_alpha.clip(lower=-1.0, upper=1.0)

    return df[[symbol_col, date_col, output_col]].copy()
