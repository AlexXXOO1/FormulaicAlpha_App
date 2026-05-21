from __future__ import annotations

import pandas as pd

from alpha_engine.formulaic_alphas.alpha_004 import compute_alpha_004


def test_alpha_004_basic_output():
    rows = []

    for symbol_idx, symbol in enumerate(["AAA", "BBB", "CCC", "DDD", "EEE"]):
        for i in range(40):
            low_price = 10.0 + symbol_idx * 0.3 + ((i * (symbol_idx + 2)) % 13) * 0.05

            rows.append({
                "symbol": symbol,
                "date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=i),
                "low": low_price,
            })

    df = pd.DataFrame(rows)
    df = df.sort_values(["symbol", "date"], kind="mergesort").reset_index(drop=True)

    out = compute_alpha_004(df)

    assert list(out.columns) == ["symbol", "date", "alpha_004"]
    assert len(out) == len(df)

    valid = out["alpha_004"].dropna()
    assert len(valid) > 0
    assert valid.between(-1.0, 0.0).all()
