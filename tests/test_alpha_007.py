from __future__ import annotations

import pandas as pd
import pytest

from alpha_engine.formulaic_alphas.alpha_007 import compute_alpha_007
from alpha_engine.formulaic_alphas.registry import get_formulaic_alpha


def test_alpha_007_basic_active_logic():
    dates = pd.date_range("2024-01-01", periods=75, freq="D")

    rows = []
    for i, date in enumerate(dates):
        rows.append(
            {
                "symbol": "AAA",
                "date": date,
                "close": float(1000 + i * i),
                "volume": float(1000 + i),
            }
        )
        rows.append(
            {
                "symbol": "BBB",
                "date": date,
                "close": float(10000 - i * i),
                "volume": float(1000 + i),
            }
        )

    df = pd.DataFrame(rows)
    out = compute_alpha_007(df)
    wide = out.pivot(index="date", columns="symbol", values="alpha_007")

    assert list(out.columns) == ["symbol", "date", "alpha_007"]
    assert len(out) == len(df)

    # adv20 is unavailable before 20 observations, so inactive branch returns -1.
    assert wide.loc[dates[18], "AAA"] == pytest.approx(-1.0)
    assert wide.loc[dates[18], "BBB"] == pytest.approx(-1.0)

    # By index 66, delta(close, 7) has 60 valid observations.
    # AAA has positive and increasing 7-day close change -> -1.
    # BBB has negative and increasingly large absolute 7-day close change -> +1.
    assert wide.loc[dates[66], "AAA"] == pytest.approx(-1.0)
    assert wide.loc[dates[66], "BBB"] == pytest.approx(1.0)

    valid = out["alpha_007"].dropna()
    assert valid.between(-1.0, 1.0).all()


def test_alpha_007_rejects_missing_columns():
    df = pd.DataFrame(
        {
            "symbol": ["AAA"],
            "date": ["2024-01-01"],
            "close": [10.0],
        }
    )

    with pytest.raises(ValueError, match="Missing required columns for alpha_007"):
        compute_alpha_007(df)


def test_alpha_007_is_registered():
    assert get_formulaic_alpha("alpha_007") is compute_alpha_007
