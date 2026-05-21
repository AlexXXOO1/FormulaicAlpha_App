from __future__ import annotations

import numpy as np
import pandas as pd


def rolling_std_by_symbol(
    df: pd.DataFrame,
    value_col: str,
    window: int,
    *,
    symbol_col: str = "symbol",
    min_periods: int | None = None,
) -> pd.Series:
    min_periods = window if min_periods is None else min_periods
    return (
        df.groupby(symbol_col, sort=False)[value_col]
        .rolling(window, min_periods=min_periods)
        .std()
        .reset_index(level=0, drop=True)
    )


def ts_argmax_current_1_to_window(values: np.ndarray) -> float:
    if values.size == 0 or np.all(np.isnan(values)):
        return np.nan

    pos_from_left = int(np.nanargmax(values))

    # 1 = max happened today/current row
    # window = max happened at oldest row in rolling window
    return float(len(values) - pos_from_left)


def ts_argmax_by_symbol(
    df: pd.DataFrame,
    value_col: str,
    window: int,
    *,
    symbol_col: str = "symbol",
    min_periods: int | None = None,
) -> pd.Series:
    min_periods = window if min_periods is None else min_periods
    return (
        df.groupby(symbol_col, sort=False)[value_col]
        .rolling(window, min_periods=min_periods)
        .apply(ts_argmax_current_1_to_window, raw=True)
        .reset_index(level=0, drop=True)
    )



def ts_delta_by_symbol(
    df: pd.DataFrame,
    value_col: str,
    periods: int,
    *,
    symbol_col: str = "symbol",
) -> pd.Series:
    return df.groupby(symbol_col, sort=False)[value_col].diff(periods)


def rolling_corr_by_symbol(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    window: int,
    *,
    symbol_col: str = "symbol",
    min_periods: int | None = None,
) -> pd.Series:
    min_periods = window if min_periods is None else min_periods
    out = pd.Series(np.nan, index=df.index, dtype="float64")

    for _, sub in df.groupby(symbol_col, sort=False):
        corr = sub[x_col].rolling(window, min_periods=min_periods).corr(sub[y_col])
        out.loc[sub.index] = corr.to_numpy(dtype="float64")

    return out


def _ts_rank_pct_numpy(
    values: np.ndarray,
    window: int,
    min_periods: int,
) -> np.ndarray:
    arr = np.asarray(values, dtype="float64")
    out = np.full(arr.shape[0], np.nan, dtype="float64")

    for i in range(arr.shape[0]):
        current = arr[i]
        if np.isnan(current):
            continue

        start = max(0, i - window + 1)
        win = arr[start:i + 1]
        win = win[~np.isnan(win)]

        if win.size < min_periods:
            continue

        less = np.sum(win < current)
        equal = np.sum(win == current)

        # pandas rank(method="average", pct=True) equivalent for current value
        avg_rank = less + (equal + 1.0) / 2.0
        out[i] = avg_rank / win.size

    return out


def ts_rank_by_symbol(
    df: pd.DataFrame,
    value_col: str,
    window: int,
    *,
    symbol_col: str = "symbol",
    min_periods: int | None = None,
) -> pd.Series:
    min_periods = window if min_periods is None else min_periods
    out = pd.Series(np.nan, index=df.index, dtype="float64")

    for _, sub in df.groupby(symbol_col, sort=False):
        values = sub[value_col].to_numpy(dtype="float64")
        ranked = _ts_rank_pct_numpy(values, window, min_periods)
        out.loc[sub.index] = ranked

    return out

