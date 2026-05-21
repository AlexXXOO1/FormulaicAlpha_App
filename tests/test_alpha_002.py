from __future__ import annotations

import pandas as pd

from alpha_engine.formulaic_alphas.alpha_002 import compute_alpha_002


def test_alpha_002_basic_output():
    rows = []

    for symbol_idx, symbol in enumerate(["AAA", "BBB", "CCC", "DDD"]):
        for i in range(30):
            open_price = 10.0 + symbol_idx * 0.5 + i * 0.03
            intraday_ret = ((((i + 2) * (symbol_idx + 1)) % 11) - 5) / 100.0
            close_price = open_price * (1.0 + intraday_ret)
            volume = (
                1000.0
                + (symbol_idx + 1) * 100.0
                + ((i + 1) * (i + symbol_idx + 2) * (symbol_idx + 2) * 3.0)
                + ((i % 4) * (5 - symbol_idx) * 11.0)
            )

            rows.append({
                "symbol": symbol,
                "date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=i),
                "open": open_price,
                "close": close_price,
                "volume": volume,
            })

    df = pd.DataFrame(rows)
    df = df.sort_values(["symbol", "date"], kind="mergesort").reset_index(drop=True)

    out = compute_alpha_002(df)

    assert list(out.columns) == ["symbol", "date", "alpha_002"]
    assert len(out) == len(df)

    valid = out["alpha_002"].dropna()
    assert len(valid) > 0
    assert valid.between(-1.0, 1.0).all()
