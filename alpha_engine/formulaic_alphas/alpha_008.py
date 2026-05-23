"""
WorldQuant 101 Formulaic Alphas - Alpha#008.

Paper formula:
    Alpha#008 =
        -1 * rank(
            (
                (sum(open, 5) * sum(returns, 5))
                - delay((sum(open, 5) * sum(returns, 5)), 10)
            )
        )

Required paper-level input fields:
    symbol
    date
    open
    returns

Factor meaning:
    Alpha#008 compares today's 5-day open-price sum multiplied by the
    5-day return sum against the same product delayed by 10 trading days.

    The difference is ranked cross-sectionally by date, then multiplied by -1.
    A stronger recent open/return product acceleration receives a more negative
    alpha value.

Usage in this project:
    The factor is computed from T0 and historical data only.
    It is valid for later T+1 / T+2 / T+3 forward-return validation.
"""

from __future__ import annotations

import pandas as pd

from alpha_engine.operators.cross_sectional import cs_rank_by_date
from alpha_engine.operators.time_series import (
    rolling_sum_by_symbol,
    ts_delay_by_symbol,
)


ALPHA_NAME = "alpha_008"


def compute_alpha_008(
    market_df: pd.DataFrame,
    *,
    symbol_col: str = "symbol",
    date_col: str = "date",
    open_col: str = "open",
    returns_col: str = "returns",
    output_col: str = ALPHA_NAME,
) -> pd.DataFrame:
    required = {symbol_col, date_col, open_col, returns_col}
    missing = sorted(required - set(market_df.columns))
    if missing:
        raise ValueError(f"Missing required columns for alpha_008: {missing}")

    df = market_df[[symbol_col, date_col, open_col, returns_col]].copy()

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce").dt.normalize()
    df[open_col] = pd.to_numeric(df[open_col], errors="coerce")
    df[returns_col] = pd.to_numeric(df[returns_col], errors="coerce")

    df = df.dropna(subset=[symbol_col, date_col])
    df = df.sort_values([symbol_col, date_col], kind="mergesort").reset_index(drop=True)

    df["_sum_open_5"] = rolling_sum_by_symbol(
        df,
        open_col,
        5,
        symbol_col=symbol_col,
    )

    df["_sum_returns_5"] = rolling_sum_by_symbol(
        df,
        returns_col,
        5,
        symbol_col=symbol_col,
    )

    df["_term"] = df["_sum_open_5"] * df["_sum_returns_5"]

    df["_delay_term_10"] = ts_delay_by_symbol(
        df,
        "_term",
        10,
        symbol_col=symbol_col,
    )

    df["_raw"] = df["_term"] - df["_delay_term_10"]

    df[output_col] = -1.0 * cs_rank_by_date(
        df,
        "_raw",
        date_col=date_col,
    )

    df[output_col] = df[output_col].clip(lower=-1.0, upper=0.0)

    return df[[symbol_col, date_col, output_col]].copy()
