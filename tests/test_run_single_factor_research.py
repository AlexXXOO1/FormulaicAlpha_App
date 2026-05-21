from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from research.factor_analysis.run_single_factor_research import run_single_factor_research


def make_symbol_data(symbol: str, dates: pd.DatetimeIndex, offset: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    n = len(dates)
    base = 10.0 + offset + np.arange(n) * 0.01

    market = pd.DataFrame({
        "symbol": symbol,
        "date": dates,
        "adjusted_open": base + 0.01,
        "adjusted_close": base + 0.03,
    })

    factor = pd.DataFrame({
        "symbol": symbol,
        "date": dates,
        "alpha_001": np.sin(np.arange(n) / 7.0 + offset),
    })

    return market, factor


def test_run_single_factor_research_writes_step_outputs(tmp_path: Path):
    market_dir = tmp_path / "market"
    factor_dir = tmp_path / "factor"
    output_dir = tmp_path / "output"
    regime_path = tmp_path / "custom_market_regime.csv"

    market_dir.mkdir()
    factor_dir.mkdir()

    dates = pd.DatetimeIndex(
        list(pd.date_range("2021-01-01", periods=80, freq="D"))
        + list(pd.date_range("2025-01-01", periods=80, freq="D"))
    )

    for symbol, offset in [("AAA", 0.0), ("BBB", 1.0), ("CCC", 2.0), ("DDD", 3.0)]:
        market, factor = make_symbol_data(symbol, dates, offset)
        market.to_parquet(market_dir / f"{symbol}.parquet", index=False)
        factor.to_parquet(factor_dir / f"{symbol}.parquet", index=False)

    regimes = pd.DataFrame({
        "date": dates,
        "market_regime": ["risk_off", "range_bound", "strong_repair", "strong_trend"] * (len(dates) // 4),
        "risk_score": [3, 2, 1, 0] * (len(dates) // 4),
        "trend_score": [0, 1, 2, 3] * (len(dates) // 4),
        "repair_score": [0, 1, 3, 1] * (len(dates) // 4),
    })
    regimes.to_csv(regime_path, index=False)

    run_single_factor_research(
        market_dir=market_dir,
        factor_dir=factor_dir,
        factor_col="alpha_001",
        output_dir=output_dir,
        custom_market_regime=regime_path,
        bucket_count=10,
        research_horizons=[1, 2],
        trade_horizons=[2, 3],
        train_start=2021,
        train_end=2024,
        test_start=2025,
        test_end=2026,
    )

    expected = [
        "step1_alpha_001_input_validation.csv",
        "step1_alpha_001_symbol_alignment.csv",
        "step2_alpha_001_factor_distribution.csv",
        "step2_alpha_001_daily_coverage.csv",
        "step3_alpha_001_research_bucket_summary.csv",
        "step3_alpha_001_research_daily_ic.csv",
        "step3_alpha_001_research_yearly_bucket_summary.csv",
        "step3_alpha_001_research_factor_diagnosis.csv",
        "step4_alpha_001_trade_bucket_summary.csv",
        "step4_alpha_001_trade_daily_ic.csv",
        "step4_alpha_001_trade_yearly_bucket_summary.csv",
        "step4_alpha_001_trade_member_detail.parquet",
        "step5_alpha_001_bucket_pattern_summary.csv",
        "step5_alpha_001_bucket_group_summary.csv",
        "step5_alpha_001_bucket_group_win_summary.csv",
        "step6_alpha_001_train_bucket_edges.csv",
        "step6_alpha_001_train_test_bucket_check.csv",
        "step7_alpha_001_daily_candidate_count.csv",
        "step7_alpha_001_train_defined_daily_candidate_count.csv",
        "step8_alpha_001_custom_regime_group_summary.csv",
        "step8_alpha_001_custom_regime_yearly_stability.csv",
        "step8_alpha_001_custom_regime_middle_yearly.csv",
        "step9_alpha_001_train_defined_bucket_regime_check.csv",
        "step10_alpha_001_factor_conclusion.csv",
        "step10_alpha_001_factor_conclusion.md",
    ]

    for name in expected:
        assert (output_dir / name).exists(), name



def test_assign_train_bucket_handles_duplicate_edges():
    import numpy as np
    import pandas as pd

    from research.factor_analysis.run_single_factor_research import assign_train_bucket

    series = pd.Series([-1.0, -0.9, -0.8, -0.7, -0.6])
    edges = np.array([-np.inf, -1.0, -1.0, -0.9, -0.8, -0.7, -0.6, np.inf])

    out = assign_train_bucket(series, edges, bucket_count=7)

    assert len(out) == len(series)
    assert out.notna().all()
    assert out.min() >= 1
    assert out.max() <= 6
