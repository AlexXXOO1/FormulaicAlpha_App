from __future__ import annotations

import pandas as pd
import pytest

from alpha_engine.formulaic_alphas.alpha_009 import compute_alpha_009
from alpha_engine.formulaic_alphas.registry import get_formulaic_alpha


def test_alpha_009_basic_branch_logic():
    dates = pd.date_range("2024-01-01", periods=8, freq="D")

    rows = []

    aaa_close = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 14.0, 16.0]
    bbb_close = [20.0, 19.0, 18.0, 17.0, 16.0, 15.0, 16.0, 14.0]

    for date, close in zip(dates, aaa_close):
        rows.append({"symbol": "AAA", "date": date, "close": close})

    for date, close in zip(dates, bbb_close):
        rows.append({"symbol": "BBB", "date": date, "close": close})

    out = compute_alpha_009(pd.DataFrame(rows))
    wide = out.pivot(index="date", columns="symbol", values="alpha_009")

    # First valid 5-delta window.
    # AAA has five positive deltas, so alpha keeps delta: +1.
    assert wide.loc[dates[5], "AAA"] == pytest.approx(1.0)

    # Mixed recent window at dates[6], latest delta is -1, so alpha flips to +1.
    assert wide.loc[dates[6], "AAA"] == pytest.approx(1.0)

    # BBB has five negative deltas, so alpha keeps delta: -1.
    assert wide.loc[dates[5], "BBB"] == pytest.approx(-1.0)

    # Mixed recent window at dates[6], latest delta is +1, so alpha flips to -1.
    assert wide.loc[dates[6], "BBB"] == pytest.approx(-1.0)


def test_alpha_009_rejects_missing_columns():
    df = pd.DataFrame(
        {
            "symbol": ["AAA"],
            "date": ["2024-01-01"],
        }
    )

    with pytest.raises(ValueError, match="Missing required columns for alpha_009"):
        compute_alpha_009(df)


def test_alpha_009_is_registered():
    assert get_formulaic_alpha("alpha_009") is compute_alpha_009
