"""
WorldQuant 101 Formulaic Alphas - Alpha#007.

Paper formula:
    Alpha#007 =
        ((adv20 < volume)
            ? ((-1 * ts_rank(abs(delta(close, 7)), 60)) * sign(delta(close, 7)))
            : (-1 * 1))

Required paper-level input fields:
    symbol
    date
    close
    volume

Factor meaning:
    Alpha#007 activates only when today's volume is above the recent
    20-day average volume.

    When active, it ranks the absolute 7-day close change over the latest
    60 trading days, then applies the sign of the 7-day close change.

    The final multiplier -1 makes strong recent upward movement negative,
    and strong recent downward movement positive.

Usage in this project:
    The factor is computed from T0 and historical data only.
    It is valid for later T+1 / T+2 / T+3 forward-return validation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from alpha_engine.operators.time_series import (
    rolling_mean_by_symbol,
    ts_delta_by_symbol,
    ts_rank_by_symbol,
)


ALPHA_NAME = "alpha_007"


def compute_alpha_007(
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
        raise ValueError(f"Missing required columns for alpha_007: {missing}")

    df = market_df[[symbol_col, date_col, close_col, volume_col]].copy()

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce").dt.normalize()
    df[close_col] = pd.to_numeric(df[close_col], errors="coerce")
    df[volume_col] = pd.to_numeric(df[volume_col], errors="coerce")

    df = df.dropna(subset=[symbol_col, date_col])
    df = df.sort_values([symbol_col, date_col], kind="mergesort").reset_index(drop=True)

    df["_adv20"] = rolling_mean_by_symbol(
        df,
        volume_col,
        20,
        symbol_col=symbol_col,
    )

    df["_delta_close_7"] = ts_delta_by_symbol(
        df,
        close_col,
        7,
        symbol_col=symbol_col,
    )

    df["_abs_delta_close_7"] = df["_delta_close_7"].abs()

    df["_ts_rank_abs_delta_close_7_60"] = ts_rank_by_symbol(
        df,
        "_abs_delta_close_7",
        60,
        symbol_col=symbol_col,
    )

    active = df["_adv20"] < df[volume_col]

    active_value = (
        -1.0
        * df["_ts_rank_abs_delta_close_7_60"]
        * np.sign(df["_delta_close_7"])
    )

    df[output_col] = -1.0
    df.loc[active, output_col] = active_value.loc[active]
    df[output_col] = df[output_col].clip(lower=-1.0, upper=1.0)

    return df[[symbol_col, date_col, output_col]].copy()
