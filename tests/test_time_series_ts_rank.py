from __future__ import annotations

import pandas as pd

from alpha_engine.operators.time_series import ts_rank_by_symbol


def test_ts_rank_by_symbol_current_value_rank():
    rows = []
    for i in range(10):
        rows.append({
            "symbol": "AAA",
            "date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=i),
            "value": float(i + 1),
        })
        rows.append({
            "symbol": "BBB",
            "date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=i),
            "value": float(10 - i),
        })

    df = pd.DataFrame(rows).sort_values(["symbol", "date"], kind="mergesort").reset_index(drop=True)

    out = ts_rank_by_symbol(df, "value", 3)

    aaa = out[df["symbol"].eq("AAA")].reset_index(drop=True)
    bbb = out[df["symbol"].eq("BBB")].reset_index(drop=True)

    assert aaa.iloc[0:2].isna().all()
    assert bbb.iloc[0:2].isna().all()

    assert aaa.iloc[2:].eq(1.0).all()
    assert bbb.iloc[2:].eq(1.0 / 3.0).all()
