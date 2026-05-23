"""
WorldQuant 101 Formulaic Alphas - Alpha#011.

Paper formula:
Alpha#011 =
((rank(ts_max((vwap - close), 3)) + rank(ts_min((vwap - close), 3))) * rank(delta(volume, 3)))

Required paper-level input fields:
symbol
date
close
vwap
volume

Factor meaning:
Alpha#011 combines the recent 3-day range of vwap-minus-close with the 3-day change in volume.
The first two terms rank the 3-day maximum and minimum of vwap - close cross-sectionally by date.
The final term ranks the 3-day volume delta cross-sectionally by date.

Usage in this project:
The factor is computed from T0 and historical data only.
It is valid for later T+1 / T+2 / T+3 forward-return validation.
"""

from __future__ import annotations

import pandas as pd

from alpha_engine.operators.cross_sectional import cs_rank_by_date
from alpha_engine.operators.time_series import (
    rolling_max_by_symbol,
    rolling_min_by_symbol,
    ts_delta_by_symbol,
)

ALPHA_NAME = "alpha_011"


def compute_alpha_011(
    market_df: pd.DataFrame,
    *,
    symbol_col: str = "symbol",
    date_col: str = "date",
    close_col: str = "close",
    vwap_col: str = "vwap",
    volume_col: str = "volume",
    output_col: str = ALPHA_NAME,
) -> pd.DataFrame:
    required = {symbol_col, date_col, close_col, vwap_col, volume_col}
    missing = sorted(required - set(market_df.columns))
    if missing:
        raise ValueError(f"Missing required columns for alpha_011: {missing}")

    df = market_df[[symbol_col, date_col, close_col, vwap_col, volume_col]].copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce").dt.normalize()

    for col in [close_col, vwap_col, volume_col]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=[symbol_col, date_col])
    df = df.sort_values([symbol_col, date_col], kind="mergesort").reset_index(drop=True)

    df["_vwap_minus_close"] = df[vwap_col] - df[close_col]

    df["_max_vwap_minus_close_3"] = rolling_max_by_symbol(
        df,
        "_vwap_minus_close",
        3,
        symbol_col=symbol_col,
    )
    df["_min_vwap_minus_close_3"] = rolling_min_by_symbol(
        df,
        "_vwap_minus_close",
        3,
        symbol_col=symbol_col,
    )
    df["_delta_volume_3"] = ts_delta_by_symbol(
        df,
        volume_col,
        3,
        symbol_col=symbol_col,
    )

    df["_rank_max_vwap_minus_close_3"] = cs_rank_by_date(
        df,
        "_max_vwap_minus_close_3",
        date_col=date_col,
    )
    df["_rank_min_vwap_minus_close_3"] = cs_rank_by_date(
        df,
        "_min_vwap_minus_close_3",
        date_col=date_col,
    )
    df["_rank_delta_volume_3"] = cs_rank_by_date(
        df,
        "_delta_volume_3",
        date_col=date_col,
    )

    df[output_col] = (
        df["_rank_max_vwap_minus_close_3"] + df["_rank_min_vwap_minus_close_3"]
    ) * df["_rank_delta_volume_3"]

    return df[[symbol_col, date_col, output_col]].copy()
