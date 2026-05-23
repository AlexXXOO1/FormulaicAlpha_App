"""
WorldQuant 101 Formulaic Alphas - Alpha#009.

Paper formula:
    Alpha#009 =
        ((0 < ts_min(delta(close, 1), 5))
            ? delta(close, 1)
            : ((ts_max(delta(close, 1), 5) < 0)
                ? delta(close, 1)
                : (-1 * delta(close, 1))))

Required paper-level input fields:
    symbol
    date
    close

Factor meaning:
    Alpha#009 keeps the 1-day close change when the recent 5-day
    close-change sequence is consistently positive or consistently negative.

    When the recent 5-day close-change window is mixed, it flips the latest
    1-day close change.

Usage in this project:
    The factor is computed from T0 and historical data only.
    It is valid for later T+1 / T+2 / T+3 forward-return validation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from alpha_engine.operators.time_series import (
    rolling_max_by_symbol,
    rolling_min_by_symbol,
    ts_delta_by_symbol,
)


ALPHA_NAME = "alpha_009"


def compute_alpha_009(
    market_df: pd.DataFrame,
    *,
    symbol_col: str = "symbol",
    date_col: str = "date",
    close_col: str = "close",
    output_col: str = ALPHA_NAME,
) -> pd.DataFrame:
    required = {symbol_col, date_col, close_col}
    missing = sorted(required - set(market_df.columns))
    if missing:
        raise ValueError(f"Missing required columns for alpha_009: {missing}")

    df = market_df[[symbol_col, date_col, close_col]].copy()

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce").dt.normalize()
    df[close_col] = pd.to_numeric(df[close_col], errors="coerce")

    df = df.dropna(subset=[symbol_col, date_col])
    df = df.sort_values([symbol_col, date_col], kind="mergesort").reset_index(drop=True)

    df["_delta_close_1"] = ts_delta_by_symbol(
        df,
        close_col,
        1,
        symbol_col=symbol_col,
    )

    df["_min_delta_close_1_5"] = rolling_min_by_symbol(
        df,
        "_delta_close_1",
        5,
        symbol_col=symbol_col,
    )

    df["_max_delta_close_1_5"] = rolling_max_by_symbol(
        df,
        "_delta_close_1",
        5,
        symbol_col=symbol_col,
    )

    positive_trend = 0.0 < df["_min_delta_close_1_5"]
    negative_trend = df["_max_delta_close_1_5"] < 0.0

    df[output_col] = np.nan
    df.loc[positive_trend, output_col] = df.loc[positive_trend, "_delta_close_1"]
    df.loc[negative_trend, output_col] = df.loc[negative_trend, "_delta_close_1"]

    mixed = ~(positive_trend | negative_trend)
    df.loc[mixed, output_col] = -1.0 * df.loc[mixed, "_delta_close_1"]

    return df[[symbol_col, date_col, output_col]].copy()
