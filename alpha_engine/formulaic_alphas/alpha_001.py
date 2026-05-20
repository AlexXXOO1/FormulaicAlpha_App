"""
WorldQuant 101 Formulaic Alphas - Alpha#001.

Paper formula:
    Alpha#001 =
        rank(
            ts_argmax(
                signedpower(
                    where(returns < 0, stddev(returns, 20), close),
                    2
                ),
                5
            )
        ) - 0.5

Required paper-level input fields:
    symbol
    date
    close
    returns

Factor meaning:
    Alpha#001 combines short-term return direction, recent return volatility,
    and the timing of the maximum squared signal within a 5-day window.

    If returns < 0, it uses 20-day return volatility.
    Otherwise, it uses close price.

    The final value is cross-sectionally ranked by date and shifted by -0.5,
    so the theoretical output range is approximately (-0.5, 0.5].

Usage in this project:
    The factor is computed from T0 and historical data only.
    It is valid for later T+1 / T+2 / T+3 forward-return validation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from alpha_engine.operators.cross_sectional import cs_rank_by_date
from alpha_engine.operators.math_ops import signed_power
from alpha_engine.operators.time_series import rolling_std_by_symbol, ts_argmax_by_symbol


ALPHA_NAME = "alpha_001"


def compute_alpha_001(
    market_df: pd.DataFrame,
    *,
    symbol_col: str = "symbol",
    date_col: str = "date",
    close_col: str = "close",
    returns_col: str = "returns",
    output_col: str = ALPHA_NAME,
) -> pd.DataFrame:
    required = {symbol_col, date_col, close_col, returns_col}
    missing = sorted(required - set(market_df.columns))
    if missing:
        raise ValueError(f"Missing required columns for alpha_001: {missing}")

    df = market_df[[symbol_col, date_col, close_col, returns_col]].copy()

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce").dt.normalize()
    df[close_col] = pd.to_numeric(df[close_col], errors="coerce")
    df[returns_col] = pd.to_numeric(df[returns_col], errors="coerce")

    df = df.dropna(subset=[symbol_col, date_col, close_col])
    df = df.sort_values([symbol_col, date_col], kind="mergesort").reset_index(drop=True)

    df["_returns"] = df[returns_col]
    df["_stddev_returns_20"] = rolling_std_by_symbol(df, "_returns", 20, symbol_col=symbol_col)

    base = np.where(
        df["_returns"].to_numpy(dtype="float64") < 0,
        df["_stddev_returns_20"].to_numpy(dtype="float64"),
        df[close_col].to_numpy(dtype="float64"),
    )

    df["_signed_power"] = signed_power(base, 2.0)
    df["_ts_argmax_5"] = ts_argmax_by_symbol(df, "_signed_power", 5, symbol_col=symbol_col)
    df[output_col] = cs_rank_by_date(df, "_ts_argmax_5", date_col=date_col) - 0.5

    return df[[symbol_col, date_col, output_col]].copy()
