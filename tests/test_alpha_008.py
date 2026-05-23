from __future__ import annotations

import pandas as pd
import pandas.testing as pdt
import pytest

from alpha_engine.formulaic_alphas.alpha_008 import compute_alpha_008
from alpha_engine.formulaic_alphas.registry import get_formulaic_alpha


def test_alpha_008_matches_manual_formula():
    dates = pd.date_range("2024-01-01", periods=25, freq="D")

    rows = []
    for symbol_idx, symbol in enumerate(["AAA", "BBB", "CCC"], start=1):
        for i, date in enumerate(dates):
            rows.append(
                {
                    "symbol": symbol,
                    "date": date,
                    "open": float(10 * symbol_idx + i),
                    "returns": float((symbol_idx * 0.001) + (i * 0.0002)),
                }
            )

    df = pd.DataFrame(rows)
    out = compute_alpha_008(df)

    manual = df.copy()
    manual["date"] = pd.to_datetime(manual["date"]).dt.normalize()
    manual = manual.sort_values(["symbol", "date"], kind="mergesort").reset_index(drop=True)

    manual["_sum_open_5"] = (
        manual.groupby("symbol", sort=False)["open"]
        .rolling(5, min_periods=5)
        .sum()
        .reset_index(level=0, drop=True)
    )
    manual["_sum_returns_5"] = (
        manual.groupby("symbol", sort=False)["returns"]
        .rolling(5, min_periods=5)
        .sum()
        .reset_index(level=0, drop=True)
    )
    manual["_term"] = manual["_sum_open_5"] * manual["_sum_returns_5"]
    manual["_delay_term_10"] = manual.groupby("symbol", sort=False)["_term"].shift(10)
    manual["_raw"] = manual["_term"] - manual["_delay_term_10"]
    manual["expected"] = -1.0 * manual.groupby("date")["_raw"].rank(pct=True)

    pdt.assert_series_equal(
        out["alpha_008"],
        manual["expected"],
        check_names=False,
        check_dtype=False,
    )

    valid = out["alpha_008"].dropna()
    assert len(valid) > 0
    assert valid.between(-1.0, 0.0).all()


def test_alpha_008_rejects_missing_columns():
    df = pd.DataFrame(
        {
            "symbol": ["AAA"],
            "date": ["2024-01-01"],
            "open": [10.0],
        }
    )

    with pytest.raises(ValueError, match="Missing required columns for alpha_008"):
        compute_alpha_008(df)


def test_alpha_008_is_registered():
    assert get_formulaic_alpha("alpha_008") is compute_alpha_008
