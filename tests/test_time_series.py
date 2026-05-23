from __future__ import annotations

import pandas as pd

from alpha_engine.operators.time_series import rolling_mean_by_symbol




def test_rolling_mean_by_symbol():
    df = pd.DataFrame(
        {
            "symbol": ["AAA", "AAA", "AAA", "BBB", "BBB", "BBB"],
            "date": pd.date_range("2024-01-01", periods=6, freq="D"),
            "value": [1.0, 2.0, 3.0, 10.0, 20.0, 30.0],
        }
    )

    out = rolling_mean_by_symbol(df, "value", 2)

    assert pd.isna(out.iloc[0])
    assert out.iloc[1] == 1.5
    assert out.iloc[2] == 2.5
    assert pd.isna(out.iloc[3])
    assert out.iloc[4] == 15.0
    assert out.iloc[5] == 25.0
