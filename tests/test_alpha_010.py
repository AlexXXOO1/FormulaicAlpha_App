from __future__ import annotations

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from alpha_engine.formulaic_alphas.alpha_010 import compute_alpha_010
from alpha_engine.formulaic_alphas.registry import get_formulaic_alpha


def test_alpha_010_matches_manual_formula():
    dates = pd.date_range("2024-01-01", periods=10, freq="D")

    close_map = {
        "AAA": [10.0, 11.0, 12.0, 13.0, 14.0, 13.0, 15.0, 14.0, 16.0, 17.0],
        "BBB": [20.0, 19.0, 18.0, 17.0, 16.0, 17.0, 15.0, 16.0, 14.0, 13.0],
        "CCC": [30.0, 30.5, 29.5, 30.2, 29.8, 31.0, 30.7, 31.5, 30.9, 32.0],
    }

    rows = []
    for symbol, closes in close_map.items():
        for date, close in zip(dates, closes):
            rows.append({"symbol": symbol, "date": date, "close": close})

    df = pd.DataFrame(rows)
    out = compute_alpha_010(df)

    manual = df.copy()
    manual["date"] = pd.to_datetime(manual["date"]).dt.normalize()
    manual = manual.sort_values(["symbol", "date"], kind="mergesort").reset_index(drop=True)

    manual["_delta_close_1"] = manual.groupby("symbol", sort=False)["close"].diff(1)

    manual["_min_delta_close_1_4"] = (
        manual.groupby("symbol", sort=False)["_delta_close_1"]
        .rolling(4, min_periods=4)
        .min()
        .reset_index(level=0, drop=True)
    )

    manual["_max_delta_close_1_4"] = (
        manual.groupby("symbol", sort=False)["_delta_close_1"]
        .rolling(4, min_periods=4)
        .max()
        .reset_index(level=0, drop=True)
    )

    positive_trend = 0.0 < manual["_min_delta_close_1_4"]
    negative_trend = manual["_max_delta_close_1_4"] < 0.0

    manual["_raw"] = np.nan
    manual.loc[positive_trend, "_raw"] = manual.loc[positive_trend, "_delta_close_1"]
    manual.loc[negative_trend, "_raw"] = manual.loc[negative_trend, "_delta_close_1"]

    mixed = ~(positive_trend | negative_trend)
    manual.loc[mixed, "_raw"] = -1.0 * manual.loc[mixed, "_delta_close_1"]

    manual["expected"] = manual.groupby("date")["_raw"].rank(pct=True)

    pdt.assert_series_equal(
        out["alpha_010"],
        manual["expected"],
        check_names=False,
        check_dtype=False,
    )

    valid = out["alpha_010"].dropna()
    assert len(valid) > 0
    assert valid.between(0.0, 1.0).all()


def test_alpha_010_rejects_missing_columns():
    df = pd.DataFrame(
        {
            "symbol": ["AAA"],
            "date": ["2024-01-01"],
        }
    )

    with pytest.raises(ValueError, match="Missing required columns for alpha_010"):
        compute_alpha_010(df)


def test_alpha_010_is_registered():
    assert get_formulaic_alpha("alpha_010") is compute_alpha_010
