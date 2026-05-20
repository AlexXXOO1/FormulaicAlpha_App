from __future__ import annotations

from pathlib import Path

import pandas as pd

from alpha_engine.pipeline.factor_builder import build_formulaic_alpha


def make_market(symbol: str, offset: float) -> pd.DataFrame:
    rows = []

    for i in range(40):
        close = 10.0 + offset + i * 0.05 + ((-1) ** i) * 0.03

        rows.append({
            "symbol": symbol,
            "date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=i),
            "adjusted_open": close - 0.02,
            "adjusted_high": close + 0.05,
            "adjusted_low": close - 0.06,
            "adjusted_close": close,
            "volume": 1000000 + i,
            "adjusted_vwap": close - 0.01,
        })

    df = pd.DataFrame(rows)
    df["returns"] = df.groupby("symbol", sort=False)["adjusted_close"].pct_change()

    return df


def test_build_formulaic_alpha_writes_feature_parquet(tmp_path: Path):
    market_dir = tmp_path / "market"
    output_dir = tmp_path / "features"
    market_dir.mkdir(parents=True, exist_ok=True)

    for symbol, offset in [("AAA", 0.0), ("BBB", 0.3), ("CCC", 0.6)]:
        make_market(symbol, offset).to_parquet(market_dir / f"{symbol}.parquet", index=False)

    report = build_formulaic_alpha(
        alpha_name="alpha_001",
        input_dir=market_dir,
        output_dir=output_dir,
    )

    assert len(report) == 3

    out = pd.read_parquet(output_dir / "AAA.parquet")

    assert list(out.columns) == ["symbol", "date", "alpha_001"]
    assert len(out) == 40
    assert out["alpha_001"].notna().sum() > 0
