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
    output_col: str = ALPHA_NAME,
) -> pd.DataFrame:
    """
    Alpha#001:
        rank(ts_argmax(signedpower(where(returns < 0, stddev(returns, 20), close), 2), 5)) - 0.5

    Required columns:
        symbol, date, close

    Output:
        symbol, date, alpha_001
    """
    required = {symbol_col, date_col, close_col}
    missing = sorted(required - set(market_df.columns))
    if missing:
        raise ValueError(f"Missing required columns for alpha_001: {missing}")

    df = market_df[[symbol_col, date_col, close_col]].copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce").dt.normalize()
    df[close_col] = pd.to_numeric(df[close_col], errors="coerce")
    df = df.dropna(subset=[symbol_col, date_col, close_col])
    df = df.sort_values([symbol_col, date_col], kind="mergesort").reset_index(drop=True)

    df["_returns"] = df.groupby(symbol_col, sort=False)[close_col].pct_change()
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
