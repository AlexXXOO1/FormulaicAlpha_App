"""
WorldQuant 101 Formulaic Alphas - Alpha#002.

Paper formula:
    Alpha#002 =
        -1 * correlation(
            rank(delta(log(volume), 2)),
            rank((close - open) / open),
            6
        )

Required paper-level input fields:
    symbol
    date
    open
    close
    volume

Factor meaning:
    Alpha#002 measures the negative 6-day rolling correlation between:
    1. cross-sectional rank of 2-day change in log volume
    2. cross-sectional rank of same-day intraday return

    A high positive raw correlation means volume acceleration and intraday strength
    moved together recently. The paper multiplies by -1, so the final alpha is
    contrarian to that relationship.

Usage in this project:
    The factor is computed from T0 and historical data only.
    It is valid for later T+1 / T+2 / T+3 forward-return validation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from alpha_engine.operators.cross_sectional import cs_rank_by_date
from alpha_engine.operators.time_series import rolling_corr_by_symbol, ts_delta_by_symbol


ALPHA_NAME = "alpha_002"


def compute_alpha_002(
    market_df: pd.DataFrame,
    *,
    symbol_col: str = "symbol",
    date_col: str = "date",
    open_col: str = "open",
    close_col: str = "close",
    volume_col: str = "volume",
    output_col: str = ALPHA_NAME,
) -> pd.DataFrame:
    required = {symbol_col, date_col, open_col, close_col, volume_col}
    missing = sorted(required - set(market_df.columns))
    if missing:
        raise ValueError(f"Missing required columns for alpha_002: {missing}")

    df = market_df[[symbol_col, date_col, open_col, close_col, volume_col]].copy()

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce").dt.normalize()
    df[open_col] = pd.to_numeric(df[open_col], errors="coerce")
    df[close_col] = pd.to_numeric(df[close_col], errors="coerce")
    df[volume_col] = pd.to_numeric(df[volume_col], errors="coerce")

    df = df.dropna(subset=[symbol_col, date_col])
    df = df.sort_values([symbol_col, date_col], kind="mergesort").reset_index(drop=True)

    df["_log_volume"] = np.nan
    positive_volume = df[volume_col] > 0
    df.loc[positive_volume, "_log_volume"] = np.log(
        df.loc[positive_volume, volume_col].to_numpy(dtype="float64")
    )

    df["_delta_log_volume_2"] = ts_delta_by_symbol(
        df,
        "_log_volume",
        2,
        symbol_col=symbol_col,
    )

    df["_intraday_return"] = np.nan
    valid_open = df[open_col].notna() & df[close_col].notna() & df[open_col].ne(0)
    df.loc[valid_open, "_intraday_return"] = (
        df.loc[valid_open, close_col].to_numpy(dtype="float64")
        - df.loc[valid_open, open_col].to_numpy(dtype="float64")
    ) / df.loc[valid_open, open_col].to_numpy(dtype="float64")

    df["_rank_delta_log_volume_2"] = cs_rank_by_date(
        df,
        "_delta_log_volume_2",
        date_col=date_col,
    )
    df["_rank_intraday_return"] = cs_rank_by_date(
        df,
        "_intraday_return",
        date_col=date_col,
    )

    raw_alpha = -1.0 * rolling_corr_by_symbol(
        df,
        "_rank_delta_log_volume_2",
        "_rank_intraday_return",
        6,
        symbol_col=symbol_col,
    )

    # Rolling correlation can produce tiny floating-point boundary drift
    # such as 1.0000000000000002. The mathematical range is [-1, 1].
    df[output_col] = raw_alpha.clip(lower=-1.0, upper=1.0)

    return df[[symbol_col, date_col, output_col]].copy()
