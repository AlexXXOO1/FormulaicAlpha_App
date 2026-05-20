from __future__ import annotations

import pandas as pd

from alpha_engine.formulaic_alphas.alpha_001 import compute_alpha_001


def test_alpha_001_basic_output():
    rows = []

    for symbol, offset in [("AAA", 0.0), ("BBB", 0.2), ("CCC", 0.4)]:
        for i in range(40):
            rows.append({
                "symbol": symbol,
                "date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=i),
                "close": 10.0 + offset + i * 0.1,
            })

    df = pd.DataFrame(rows)
    out = compute_alpha_001(df)

    assert list(out.columns) == ["symbol", "date", "alpha_001"]
    assert len(out) == len(df)
    assert out["alpha_001"].notna().sum() > 0
    assert out["alpha_001"].dropna().between(-0.5, 0.5).all()
