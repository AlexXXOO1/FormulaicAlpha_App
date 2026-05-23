import pandas as pd
import pytest

from alpha_engine.formulaic_alphas.alpha_006 import compute_alpha_006
from alpha_engine.formulaic_alphas.registry import get_formulaic_alpha


def test_alpha_006_computes_negative_rolling_open_volume_correlation():
    dates = pd.date_range("2024-01-01", periods=12, freq="D")

    rows = []
    for i, date in enumerate(dates, start=1):
        rows.append(
            {
                "symbol": "AAA",
                "date": date,
                "open": float(i),
                "volume": float(100 + i),
            }
        )
        rows.append(
            {
                "symbol": "BBB",
                "date": date,
                "open": float(i),
                "volume": float(100 - i),
            }
        )

    market_df = pd.DataFrame(rows)
    result = compute_alpha_006(market_df)

    wide = result.pivot(index="date", columns="symbol", values="alpha_006")

    assert pd.isna(wide.loc[dates[8], "AAA"])
    assert pd.isna(wide.loc[dates[8], "BBB"])

    assert wide.loc[dates[9], "AAA"] == pytest.approx(-1.0)
    assert wide.loc[dates[9], "BBB"] == pytest.approx(1.0)

    assert wide.loc[dates[11], "AAA"] == pytest.approx(-1.0)
    assert wide.loc[dates[11], "BBB"] == pytest.approx(1.0)


def test_alpha_006_rejects_missing_columns():
    market_df = pd.DataFrame(
        {
            "symbol": ["AAA"],
            "date": ["2024-01-01"],
            "open": [10.0],
        }
    )

    with pytest.raises(ValueError, match="Missing required columns for alpha_006"):
        compute_alpha_006(market_df)


def test_alpha_006_is_registered():
    assert get_formulaic_alpha("alpha_006") is compute_alpha_006
