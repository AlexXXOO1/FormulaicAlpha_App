from __future__ import annotations

import pandas as pd

from alpha_engine.formulaic_alphas.alpha_003 import compute_alpha_003


def test_alpha_003_basic_output():
    rows = []

    for symbol_idx, symbol in enumerate(["AAA", "BBB", "CCC", "DDD", "EEE"]):
        for i in range(40):
            open_price = 10.0 + ((i * (symbol_idx + 2)) % 17) * 0.1 + symbol_idx * 0.03
            volume = 1000.0 + ((i + 3) * (symbol_idx + 5) * 37) % 700 + symbol_idx * 11

            rows.append({
                "symbol": symbol,
                "date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=i),
                "open": open_price,
                "volume": volume,
            })

    df = pd.DataFrame(rows)
    df = df.sort_values(["symbol", "date"], kind="mergesort").reset_index(drop=True)

    out = compute_alpha_003(df)

    assert list(out.columns) == ["symbol", "date", "alpha_003"]
    assert len(out) == len(df)

    valid = out["alpha_003"].dropna()
    assert len(valid) > 0
    assert valid.between(-1.0, 1.0).all()
