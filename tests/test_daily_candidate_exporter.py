from __future__ import annotations

from pathlib import Path

import pandas as pd

from research.portfolio_analysis.daily_candidate_exporter import export_daily_candidates


def _write_symbol_factor_file(
    factor_dir: Path,
    symbol: str,
    *,
    test_alpha_001: float,
    test_alpha_002: float,
    test_alpha_005: float,
) -> None:
    train_dates = pd.date_range("2021-01-01", periods=100, freq="D")
    test_date = pd.Timestamp("2026-05-20")

    values = list(range(100))

    df = pd.DataFrame(
        {
            "symbol": [symbol] * 101,
            "date": list(train_dates) + [test_date],
            "alpha_001": values + [test_alpha_001],
            "alpha_002": values + [test_alpha_002],
            "alpha_005": values + [test_alpha_005],
        }
    )

    df.to_parquet(factor_dir / f"{symbol}.parquet", index=False)


def test_export_daily_candidates_creates_wide_precision_and_summary(tmp_path: Path) -> None:
    factor_dir = tmp_path / "factor"
    output_dir = tmp_path / "candidate_output"
    regime_path = tmp_path / "custom_market_regime.csv"

    factor_dir.mkdir()

    _write_symbol_factor_file(
        factor_dir,
        "SZ#000001",
        test_alpha_001=35,
        test_alpha_002=35,
        test_alpha_005=45,
    )

    _write_symbol_factor_file(
        factor_dir,
        "SZ#000002",
        test_alpha_001=35,
        test_alpha_002=85,
        test_alpha_005=35,
    )

    pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-05-20")],
            "market_regime": ["range_bound"],
        }
    ).to_csv(regime_path, index=False)

    result = export_daily_candidates(
        factor_dir=factor_dir,
        regime_path=regime_path,
        output_dir=output_dir,
        target_date="2026-05-20",
    )

    assert result.wide_count == 2
    assert result.precision_count == 1
    assert result.wide_path.exists()
    assert result.precision_path.exists()
    assert result.summary_path.exists()

    wide = pd.read_csv(result.wide_path)
    precision = pd.read_csv(result.precision_path)
    summary = pd.read_csv(result.summary_path)

    assert set(wide["symbol"]) == {"SZ#000001", "SZ#000002"}
    assert precision["symbol"].tolist() == ["SZ#000001"]

    summary_counts = dict(zip(summary["pool_name"], summary["candidate_count"]))
    assert summary_counts["wide_pool"] == 2
    assert summary_counts["precision_pool"] == 1
