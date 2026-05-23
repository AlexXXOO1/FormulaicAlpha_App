from __future__ import annotations

import pandas as pd
import pandas.testing as pdt
import pytest

from alpha_engine.formulaic_alphas.alpha_011 import compute_alpha_011
from alpha_engine.formulaic_alphas.registry import get_formulaic_alpha


def test_alpha_011_matches_manual_formula():
    dates = pd.date_range("2024-01-01", periods=8, freq="D")

    data = {
        "AAA": {
            "close": [10.0, 10.2, 10.1, 10.5, 10.4, 10.8, 10.7, 11.0],
            "vwap":  [10.1, 10.1, 10.3, 10.4, 10.6, 10.7, 10.9, 10.8],
            "volume": [100, 110, 130, 160, 150, 190, 210, 230],
        },
        "BBB": {
            "close": [20.0, 19.8, 19.9, 19.7, 20.1, 20.0, 20.3, 20.2],
            "vwap":  [19.9, 20.0, 19.8, 19.9, 20.0, 20.2, 20.1, 20.4],
            "volume": [200, 195, 210, 205, 250, 240, 260, 255],
        },
        "CCC": {
            "close": [30.0, 30.5, 30.2, 30.7, 30.6, 31.0, 30.9, 31.2],
            "vwap":  [30.2, 30.3, 30.4, 30.6, 30.8, 30.9, 31.1, 31.0],
            "volume": [300, 330, 320, 360, 350, 390, 410, 430],
        },
        "DDD": {
            "close": [40.0, 39.7, 39.9, 40.2, 40.0, 40.4, 40.3, 40.6],
            "vwap":  [39.8, 39.9, 40.0, 40.1, 40.2, 40.3, 40.5, 40.4],
            "volume": [400, 380, 420, 440, 430, 470, 460, 500],
        },
    }

    rows = []
    for symbol, values in data.items():
        for idx, date in enumerate(dates):
            rows.append(
                {
                    "symbol": symbol,
                    "date": date,
                    "close": values["close"][idx],
                    "vwap": values["vwap"][idx],
                    "volume": values["volume"][idx],
                }
            )

    df = pd.DataFrame(rows)
    out = compute_alpha_011(df)

    manual = df.copy()
    manual["date"] = pd.to_datetime(manual["date"]).dt.normalize()
    manual = manual.sort_values(["symbol", "date"], kind="mergesort").reset_index(drop=True)

    manual["_vwap_minus_close"] = manual["vwap"] - manual["close"]
    manual["_max_vwap_minus_close_3"] = (
        manual.groupby("symbol", sort=False)["_vwap_minus_close"]
        .rolling(3, min_periods=3)
        .max()
        .reset_index(level=0, drop=True)
    )
    manual["_min_vwap_minus_close_3"] = (
        manual.groupby("symbol", sort=False)["_vwap_minus_close"]
        .rolling(3, min_periods=3)
        .min()
        .reset_index(level=0, drop=True)
    )
    manual["_delta_volume_3"] = manual.groupby("symbol", sort=False)["volume"].diff(3)

    manual["_rank_max_vwap_minus_close_3"] = manual.groupby("date", sort=False)[
        "_max_vwap_minus_close_3"
    ].rank(method="average", pct=True)
    manual["_rank_min_vwap_minus_close_3"] = manual.groupby("date", sort=False)[
        "_min_vwap_minus_close_3"
    ].rank(method="average", pct=True)
    manual["_rank_delta_volume_3"] = manual.groupby("date", sort=False)[
        "_delta_volume_3"
    ].rank(method="average", pct=True)

    manual["expected"] = (
        manual["_rank_max_vwap_minus_close_3"] + manual["_rank_min_vwap_minus_close_3"]
    ) * manual["_rank_delta_volume_3"]

    assert list(out.columns) == ["symbol", "date", "alpha_011"]
    assert len(out) == len(manual)

    pdt.assert_series_equal(
        out["alpha_011"],
        manual["expected"],
        check_names=False,
        check_dtype=False,
    )

    valid = out["alpha_011"].dropna()
    assert len(valid) > 0
    assert valid.between(0.0, 2.0).all()


def test_alpha_011_rejects_missing_columns():
    df = pd.DataFrame(
        {
            "symbol": ["AAA"],
            "date": ["2024-01-01"],
            "close": [10.0],
            "vwap": [10.1],
        }
    )

    with pytest.raises(ValueError, match="Missing required columns for alpha_011"):
        compute_alpha_011(df)


def test_alpha_011_is_registered():
    assert get_formulaic_alpha("alpha_011") is compute_alpha_011
