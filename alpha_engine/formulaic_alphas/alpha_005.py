"""
WorldQuant 101 Formulaic Alphas - Alpha#005.

Paper formula:
    Alpha#005 =
        rank((open - (sum(vwap, 10) / 10)))
        * (-1 * abs(rank((close - vwap))))

Required paper-level input fields:
    symbol
    date
    open
    close
    vwap

Factor meaning:
    Alpha#005 combines:
    1. Today's open relative to the 10-day average VWAP.
    2. Today's close relative to VWAP.

    The first term ranks open price deviation from recent VWAP.
    The second term is the negative absolute cross-sectional rank of close minus VWAP.

Usage in this project:
    The factor is computed from T0 and historical data only.
    It is valid for later T+1 / T+2 / T+3 forward-return validation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from alpha_engine.operators.cross_sectional import cs_rank_by_date


ALPHA_NAME = "alpha_005"


def compute_alpha_005(
    market_df: pd.DataFrame,
    *,
    symbol_col: str = "symbol",
    date_col: str = "date",
    open_col: str = "open",
    close_col: str = "close",
    vwap_col: str = "vwap",
    output_col: str = ALPHA_NAME,
) -> pd.DataFrame:
    required = {symbol_col, date_col, open_col, close_col, vwap_col}
    missing = sorted(required - set(market_df.columns))
    if missing:
        raise ValueError(f"Missing required columns for alpha_005: {missing}")

    df = market_df[[symbol_col, date_col, open_col, close_col, vwap_col]].copy()

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce").dt.normalize()
    for col in [open_col, close_col, vwap_col]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=[symbol_col, date_col, open_col, close_col, vwap_col])
    df = df.sort_values([symbol_col, date_col], kind="mergesort").reset_index(drop=True)

    df["_vwap_mean_10"] = (
        df.groupby(symbol_col, sort=False)[vwap_col]
        .rolling(10, min_periods=10)
        .mean()
        .reset_index(level=0, drop=True)
    )

    df["_open_minus_vwap_mean_10"] = df[open_col] - df["_vwap_mean_10"]
    df["_close_minus_vwap"] = df[close_col] - df[vwap_col]

    df["_rank_open_minus_vwap_mean_10"] = cs_rank_by_date(
        df,
        "_open_minus_vwap_mean_10",
        date_col=date_col,
    )
    df["_rank_close_minus_vwap"] = cs_rank_by_date(
        df,
        "_close_minus_vwap",
        date_col=date_col,
    )

    df[output_col] = (
        df["_rank_open_minus_vwap_mean_10"]
        * (-1.0 * np.abs(df["_rank_close_minus_vwap"]))
    )

    return df[[symbol_col, date_col, output_col]].copy()
