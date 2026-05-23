"""
WorldQuant 101 Formulaic Alphas - Alpha#012.

Paper formula:
Alpha#012 = (sign(delta(volume, 1)) * (-1 * delta(close, 1)))

Required paper-level input fields:
symbol
date
close
volume

Factor meaning:
Alpha#012 combines one-day volume direction with one-day price reversal.
If volume increases, the factor is negative one-day close delta.
If volume decreases, the factor is positive one-day close delta.
If volume is unchanged, the factor is zero.

Usage in this project:
The factor is computed from T0 and historical data only.
It is valid for later T+1 / T+2 / T+3 / T+4 forward-return validation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from alpha_engine.operators.time_series import ts_delta_by_symbol

ALPHA_NAME = "alpha_012"


def compute_alpha_012(
    market_df: pd.DataFrame,
    *,
    symbol_col: str = "symbol",
    date_col: str = "date",
    close_col: str = "close",
    volume_col: str = "volume",
    output_col: str = ALPHA_NAME,
) -> pd.DataFrame:
    required = {symbol_col, date_col, close_col, volume_col}
    missing = sorted(required - set(market_df.columns))
    if missing:
        raise ValueError(f"Missing required columns for alpha_012: {missing}")

    df = market_df[[symbol_col, date_col, close_col, volume_col]].copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce").dt.normalize()

    for col in [close_col, volume_col]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=[symbol_col, date_col])
    df = df.sort_values([symbol_col, date_col], kind="mergesort").reset_index(drop=True)

    df["_delta_volume_1"] = ts_delta_by_symbol(
        df,
        volume_col,
        1,
        symbol_col=symbol_col,
    )
    df["_delta_close_1"] = ts_delta_by_symbol(
        df,
        close_col,
        1,
        symbol_col=symbol_col,
    )

    df[output_col] = np.sign(df["_delta_volume_1"]) * (-1.0 * df["_delta_close_1"])

    return df[[symbol_col, date_col, output_col]].copy()
