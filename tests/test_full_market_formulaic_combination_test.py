
from __future__ import annotations

from pathlib import Path

import pandas as pd

from research.candidate_pool.full_market_formulaic_combination_test import (
    build_full_market_formulaic_combination_test,
)


def _write_market_files(market_dir: Path) -> None:
    market_dir.mkdir(parents=True, exist_ok=True)

    dates = pd.date_range("2021-01-01", periods=8, freq="D")

    pd.DataFrame(
        {
            "symbol": ["AAA"] * len(dates),
            "date": dates,
            "adjusted_open": [10.0, 10.2, 10.4, 10.8, 11.0, 11.2, 11.4, 11.6],
            "adjusted_high": [10.1, 10.3, 10.5, 10.9, 11.1, 11.3, 11.5, 11.7],
            "adjusted_low": [9.9, 10.1, 10.3, 10.7, 10.9, 11.1, 11.3, 11.5],
            "adjusted_close": [10.1, 10.3, 10.7, 11.0, 11.2, 11.5, 11.7, 12.0],
            "adjusted_vwap": [10.05, 10.25, 10.55, 10.9, 11.1, 11.35, 11.55, 11.8],
            "volume": [1000] * len(dates),
            "amount": [1000000.0] * len(dates),
            "returns": [0.01] * len(dates),
        }
    ).to_parquet(market_dir / "AAA.parquet", index=False)

    pd.DataFrame(
        {
            "symbol": ["BBB"] * len(dates),
            "date": dates,
            "adjusted_open": [20.0] * len(dates),
            "adjusted_high": [20.1] * len(dates),
            "adjusted_low": [19.9] * len(dates),
            "adjusted_close": [20.0] * len(dates),
            "adjusted_vwap": [20.0] * len(dates),
            "volume": [1000] * len(dates),
            "amount": [1000000.0] * len(dates),
            "returns": [0.0] * len(dates),
        }
    ).to_parquet(market_dir / "BBB.parquet", index=False)


def _write_factor_files(factor_dir: Path) -> None:
    factor_dir.mkdir(parents=True, exist_ok=True)
    dates = pd.date_range("2021-01-01", periods=8, freq="D")

    pd.DataFrame(
        {
            "symbol": ["AAA"] * len(dates),
            "date": dates,
            "alpha_005": [0.35] * len(dates),
            "alpha_002": [0.35] * len(dates),
        }
    ).to_parquet(factor_dir / "AAA.parquet", index=False)

    pd.DataFrame(
        {
            "symbol": ["BBB"] * len(dates),
            "date": dates,
            "alpha_005": [0.15] * len(dates),
            "alpha_002": [0.35] * len(dates),
        }
    ).to_parquet(factor_dir / "BBB.parquet", index=False)


def _write_research_files(root: Path) -> Path:
    research_root = root / "research_output" / "factor_analysis"
    research_root.mkdir(parents=True, exist_ok=True)

    for factor in ["alpha_005", "alpha_002"]:
        factor_dir = research_root / f"{factor}_research"
        factor_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "bucket": [1, 2, 3, 4, 5],
                "left_edge": [0.0, 0.1, 0.2, 0.3, 0.4],
                "right_edge": [0.1, 0.2, 0.3, 0.4, 0.5],
            }
        ).to_csv(factor_dir / f"step6_{factor}_train_bucket_edges.csv", index=False)

    pd.DataFrame(
        {
            "factor_name": ["alpha_005", "alpha_002"],
            "factor_type_manual": ["regime_aware_filter", "regime_aware_filter_low_confidence"],
            "bucket_rule": ["train_defined_bucket in [4]", "train_defined_bucket in [4]"],
            "valid_regime": ["range_bound,strong_repair,strong_trend", "range_bound,risk_off,strong_repair"],
            "weak_regime": ["risk_off", "strong_trend"],
            "is_filter_factor": [True, True],
            "manual_override": [True, True],
            "ml_baseline_role": ["eligible_filter_candidate", "eligible_low_confidence_filter"],
        }
    ).to_csv(research_root / "manual_factor_conclusion_summary.csv", index=False)

    return research_root


def test_build_full_market_formulaic_combination_test_outputs_summaries(tmp_path: Path) -> None:
    market_dir = tmp_path / "market"
    factor_dir = tmp_path / "factor"
    _write_market_files(market_dir)
    _write_factor_files(factor_dir)
    research_root = _write_research_files(tmp_path)

    regime_path = tmp_path / "custom_market_regime.csv"
    dates = pd.date_range("2021-01-01", periods=8, freq="D")
    pd.DataFrame(
        {
            "date": dates,
            "market_regime": ["range_bound"] * len(dates),
        }
    ).to_csv(regime_path, index=False)

    output_dir = tmp_path / "combination_output"

    result = build_full_market_formulaic_combination_test(
        market_dir=market_dir,
        factor_dir=factor_dir,
        research_root=research_root,
        manual_summary_path=research_root / "manual_factor_conclusion_summary.csv",
        regime_path=regime_path,
        output_dir=output_dir,
        factor_names=["alpha_005", "alpha_002"],
        horizons=(3, 4),
        start_date="2021-01-01",
        end_date="2021-01-08",
    )

    assert not result["daily"].empty
    assert not result["train_test_summary"].empty
    assert not result["train_test_regime_summary"].empty
    assert not result["yearly_summary"].empty

    assert (output_dir / "full_market_formulaic_combination_test_daily.parquet").exists()
    assert (output_dir / "full_market_formulaic_combination_test_train_test_summary.csv").exists()
    assert (output_dir / "full_market_formulaic_combination_test_train_test_regime_summary.csv").exists()
    assert (output_dir / "full_market_formulaic_combination_test_yearly_summary.csv").exists()

    names = set(result["train_test_summary"]["combined_rule_name"])
    assert "alpha_005_only" in names
    assert "alpha_005_and_alpha_002" in names

    assert set(result["train_test_summary"]["target"]) == {"fwd_return_pct_T3", "fwd_return_pct_T4"}
    assert not result["recency_period_summary"].empty
    assert "recent_120d" in set(result["recency_period_summary"]["period"])
    assert not result["combination_conclusion_summary"].empty
    assert "auto_conclusion" in result["combination_conclusion_summary"].columns
    assert "is_live_eligible" in result["combination_conclusion_summary"].columns
    assert "short_window_clear_negative" in result["combination_conclusion_summary"].columns
    assert not result["market_recent_period_summary"].empty
    assert "market_recent_20d" in set(result["market_recent_period_summary"]["period"])
