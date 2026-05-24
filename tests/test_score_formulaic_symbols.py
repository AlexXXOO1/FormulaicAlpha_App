
from __future__ import annotations

from pathlib import Path

import pandas as pd

from research.candidate_pool.formulaic_symbol_score import (
    normalize_symbol,
    score_formulaic_symbols,
)


def _write_market_and_factor_files(market_dir: Path, factor_dir: Path) -> None:
    market_dir.mkdir(parents=True, exist_ok=True)
    factor_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        {
            "symbol": ["SH#600051"],
            "date": ["2026-01-02"],
            "adjusted_open": [10.0],
            "adjusted_high": [10.5],
            "adjusted_low": [9.8],
            "adjusted_close": [10.2],
            "adjusted_vwap": [10.1],
            "volume": [1000],
            "amount": [1000000.0],
            "returns": [0.02],
        }
    ).to_parquet(market_dir / "SH#600051.parquet", index=False)

    pd.DataFrame(
        {
            "symbol": ["SZ#000001"],
            "date": ["2026-01-02"],
            "adjusted_open": [20.0],
            "adjusted_high": [20.5],
            "adjusted_low": [19.8],
            "adjusted_close": [20.2],
            "adjusted_vwap": [20.1],
            "volume": [1000],
            "amount": [1000000.0],
            "returns": [0.01],
        }
    ).to_parquet(market_dir / "SZ#000001.parquet", index=False)

    pd.DataFrame(
        {
            "symbol": ["SH#600051"],
            "date": ["2026-01-02"],
            "alpha_002": [0.35],
            "alpha_006": [0.35],
        }
    ).to_parquet(factor_dir / "SH#600051.parquet", index=False)

    pd.DataFrame(
        {
            "symbol": ["SZ#000001"],
            "date": ["2026-01-02"],
            "alpha_002": [0.15],
            "alpha_006": [0.35],
        }
    ).to_parquet(factor_dir / "SZ#000001.parquet", index=False)


def _write_research_files(root: Path) -> Path:
    research_root = root / "research_output" / "factor_analysis"
    research_root.mkdir(parents=True, exist_ok=True)

    for factor in ["alpha_002", "alpha_006"]:
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
            "factor_name": ["alpha_002", "alpha_006"],
            "factor_type_manual": [
                "regime_aware_filter_low_confidence",
                "regime_aware_filter_low_confidence",
            ],
            "bucket_rule": [
                "train_defined_bucket in [4]",
                "train_defined_bucket in [4,5]",
            ],
            "valid_regime": [
                "range_bound,risk_off,strong_repair",
                "range_bound,risk_off,strong_repair",
            ],
            "weak_regime": ["strong_trend", "strong_trend"],
            "is_filter_factor": [True, True],
            "manual_override": [True, True],
            "ml_baseline_role": [
                "eligible_low_confidence_filter",
                "eligible_low_confidence_filter",
            ],
        }
    ).to_csv(research_root / "manual_factor_conclusion_summary.csv", index=False)

    return research_root


def test_normalize_symbol() -> None:
    assert normalize_symbol("600051") == "SH#600051"
    assert normalize_symbol("000001") == "SZ#000001"
    assert normalize_symbol("SH#600051") == "SH#600051"


def test_score_formulaic_symbols_scores_pass_and_partial(tmp_path: Path) -> None:
    market_dir = tmp_path / "market"
    factor_dir = tmp_path / "factor"
    _write_market_and_factor_files(market_dir, factor_dir)
    research_root = _write_research_files(tmp_path)

    regime_path = tmp_path / "custom_market_regime.csv"
    pd.DataFrame(
        {
            "date": ["2026-01-02"],
            "market_regime": ["strong_repair"],
        }
    ).to_csv(regime_path, index=False)

    output_path = tmp_path / "score.csv"

    out = score_formulaic_symbols(
        symbols=["600051", "000001"],
        target_date="2026-01-02",
        market_dir=market_dir,
        factor_dir=factor_dir,
        research_root=research_root,
        manual_summary_path=research_root / "manual_factor_conclusion_summary.csv",
        regime_path=regime_path,
        output_path=output_path,
        factor_names=["alpha_002", "alpha_006"],
    )

    assert output_path.exists()
    assert len(out) == 2

    passed = out[out["symbol"].eq("SH#600051")].iloc[0]
    assert passed["combo_pass"] is True or bool(passed["combo_pass"]) is True
    assert passed["score"] == 100.0
    assert passed["status"] == "PAPER_WATCH_PASS"
    assert passed["allow_new_entry"] is False or bool(passed["allow_new_entry"]) is False

    partial = out[out["symbol"].eq("SZ#000001")].iloc[0]
    assert partial["combo_pass"] is False or bool(partial["combo_pass"]) is False
    assert partial["score"] == 60.0
    assert partial["status"] == "PARTIAL_MATCH"


def test_score_formulaic_symbols_rejects_by_regime(tmp_path: Path) -> None:
    market_dir = tmp_path / "market"
    factor_dir = tmp_path / "factor"
    _write_market_and_factor_files(market_dir, factor_dir)
    research_root = _write_research_files(tmp_path)

    regime_path = tmp_path / "custom_market_regime.csv"
    pd.DataFrame(
        {
            "date": ["2026-01-02"],
            "market_regime": ["strong_trend"],
        }
    ).to_csv(regime_path, index=False)

    out = score_formulaic_symbols(
        symbols=["600051"],
        target_date="2026-01-02",
        market_dir=market_dir,
        factor_dir=factor_dir,
        research_root=research_root,
        manual_summary_path=research_root / "manual_factor_conclusion_summary.csv",
        regime_path=regime_path,
        output_path=None,
        factor_names=["alpha_002", "alpha_006"],
    )

    row = out.iloc[0]
    assert row["combo_pass"] is True or bool(row["combo_pass"]) is True
    assert row["status"] == "FORCE_REJECT_BY_REGIME"
    assert "regime_not_allowed" in row["reject_reason"]
