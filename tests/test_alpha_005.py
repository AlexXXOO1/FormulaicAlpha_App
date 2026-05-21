from __future__ import annotations

import pandas as pd

from alpha_engine.formulaic_alphas.alpha_005 import compute_alpha_005


def test_alpha_005_basic_output():
    rows = []

    for symbol_idx, symbol in enumerate(["AAA", "BBB", "CCC", "DDD", "EEE"]):
        for i in range(30):
            open_price = 10.0 + symbol_idx * 0.7 + i * 0.05
            vwap = open_price + ((i + symbol_idx) % 5 - 2) * 0.03
            close = vwap + ((i * (symbol_idx + 2)) % 7 - 3) * 0.04

            rows.append({
                "symbol": symbol,
                "date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=i),
                "open": open_price,
                "close": close,
                "vwap": vwap,
            })

    df = pd.DataFrame(rows)
    df = df.sort_values(["symbol", "date"], kind="mergesort").reset_index(drop=True)

    out = compute_alpha_005(df)

    assert list(out.columns) == ["symbol", "date", "alpha_005"]
    assert len(out) == len(df)

    valid = out["alpha_005"].dropna()
    assert len(valid) > 0
    assert valid.between(-1.0, 0.0).all()


def test_alpha_005_missing_required_columns():
    df = pd.DataFrame({
        "symbol": ["AAA"],
        "date": [pd.Timestamp("2024-01-01")],
        "open": [10.0],
        "close": [10.1],
    })

    try:
        compute_alpha_005(df)
    except ValueError as exc:
        assert "vwap" in str(exc)
    else:
        raise AssertionError("Expected ValueError for missing vwap")
