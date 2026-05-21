from __future__ import annotations

from pathlib import Path

import pandas as pd

from research.factor_analysis.analyze_single_factor import run_single_factor_analysis


def make_market(symbol: str, offset: float) -> pd.DataFrame:
    rows = []

    for i in range(30):
        close = 10.0 + offset + i * 0.1

        rows.append({
            "symbol": symbol,
            "date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=i),
            "adjusted_close": close,
        })

    return pd.DataFrame(rows)


def make_factor(symbol: str, offset: float) -> pd.DataFrame:
    rows = []

    for i in range(30):
        rows.append({
            "symbol": symbol,
            "date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=i),
            "alpha_001": offset + (i % 5) * 0.1,
        })

    return pd.DataFrame(rows)


def test_run_single_factor_analysis_with_market_and_factor_dirs(tmp_path: Path):
    market_dir = tmp_path / "market"
    factor_dir = tmp_path / "factor"
    output_dir = tmp_path / "output"

    market_dir.mkdir(parents=True, exist_ok=True)
    factor_dir.mkdir(parents=True, exist_ok=True)

    for symbol, offset in [("AAA", 0.0), ("BBB", 1.0), ("CCC", 2.0)]:
        make_market(symbol, offset).to_parquet(market_dir / f"{symbol}.parquet", index=False)
        make_factor(symbol, offset).to_parquet(factor_dir / f"{symbol}.parquet", index=False)

    result = run_single_factor_analysis(
        market_path=market_dir,
        factor_path=factor_dir,
        factor_col="alpha_001",
        output_dir=output_dir,
        horizons=[1, 2, 5],
        bucket_count=3,
    )

    assert result["market_rows"] == 90
    assert result["factor_rows"] == 90
    assert result["merged_rows"] == 90
    assert result["factor_non_null"] == 90
    assert result["bucket_summary_rows"] > 0
    assert result["yearly_bucket_summary_rows"] > 0
    assert result["daily_ic_rows"] > 0

    bucket = pd.read_csv(output_dir / "bucket_summary.csv")
    yearly_bucket = pd.read_csv(output_dir / "yearly_bucket_summary.csv")
    daily_ic = pd.read_csv(output_dir / "daily_ic.csv")
    detail = pd.read_parquet(output_dir / "factor_member_detail.parquet")

    assert set(bucket["target"]) == {
        "fwd_return_pct_T1",
        "fwd_return_pct_T2",
        "fwd_return_pct_T5",
    }
    assert "year" in yearly_bucket.columns
    assert "spearman_ic" in daily_ic.columns
    assert "alpha_001_bucket" in detail.columns
