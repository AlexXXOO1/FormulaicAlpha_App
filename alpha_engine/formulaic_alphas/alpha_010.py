"""
WorldQuant 101 Formulaic Alphas - Alpha#010.

Paper formula:
    Alpha#010 =
        rank(
            ((0 < ts_min(delta(close, 1), 4))
                ? delta(close, 1)
                : ((ts_max(delta(close, 1), 4) < 0)
                    ? delta(close, 1)
                    : (-1 * delta(close, 1))))
        )

Required paper-level input fields:
    symbol
    date
    close

Factor meaning:
    Alpha#010 is the cross-sectional rank of an Alpha#009-like branch rule
    using a 4-day rolling window instead of 5.

    It keeps the latest 1-day close change when recent 1-day close changes
    are consistently positive or consistently negative. When the recent
    4-day close-change window is mixed, it flips the latest close change.
    The branch output is then ranked cross-sectionally by date.

Usage in this project:
    The factor is computed from T0 and historical data only.
    It is valid for later T+1 / T+2 / T+3 forward-return validation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from alpha_engine.operators.cross_sectional import cs_rank_by_date
from alpha_engine.operators.time_series import (
    rolling_max_by_symbol,
    rolling_min_by_symbol,
    ts_delta_by_symbol,
)


ALPHA_NAME = "alpha_010"


def compute_alpha_010(
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
        raise ValueError(f"Missing required columns for alpha_010: {missing}")

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

    df["_min_delta_close_1_4"] = rolling_min_by_symbol(
        df,
        "_delta_close_1",
        4,
        symbol_col=symbol_col,
    )

    df["_max_delta_close_1_4"] = rolling_max_by_symbol(
        df,
        "_delta_close_1",
        4,
        symbol_col=symbol_col,
    )

    positive_trend = 0.0 < df["_min_delta_close_1_4"]
    negative_trend = df["_max_delta_close_1_4"] < 0.0

    df["_raw"] = np.nan
    df.loc[positive_trend, "_raw"] = df.loc[positive_trend, "_delta_close_1"]
    df.loc[negative_trend, "_raw"] = df.loc[negative_trend, "_delta_close_1"]

    mixed = ~(positive_trend | negative_trend)
    df.loc[mixed, "_raw"] = -1.0 * df.loc[mixed, "_delta_close_1"]

    df[output_col] = cs_rank_by_date(
        df,
        "_raw",
        date_col=date_col,
    )

    df[output_col] = df[output_col].clip(lower=0.0, upper=1.0)

    return df[[symbol_col, date_col, output_col]].copy()
