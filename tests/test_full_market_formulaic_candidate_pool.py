
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from research.candidate_pool.full_market_formulaic_candidate_pool import (
    assign_train_defined_bucket,
    build_full_market_formulaic_candidate_pool,
    load_manual_factor_rules,
)


def _write_market_and_factor_files(market_dir: Path, factor_dir: Path) -> None:
    market_dir.mkdir(parents=True, exist_ok=True)
    factor_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        {
            "symbol": ["AAA"],
            "date": ["2026-01-02"],
            "adjusted_open": [10.0],
            "adjusted_high": [10.5],
            "adjusted_low": [9.8],
            "adjusted_close": [10.2],
            "adjusted_vwap": [10.15],
            "volume": [1000],
            "amount": [1000000.0],
            "returns": [0.02],
        }
    ).to_parquet(market_dir / "AAA.parquet", index=False)

    pd.DataFrame(
        {
            "symbol": ["BBB"],
            "date": ["2026-01-02"],
            "adjusted_open": [20.0],
            "adjusted_high": [20.4],
            "adjusted_low": [19.9],
            "adjusted_close": [20.1],
            "adjusted_vwap": [20.05],
            "volume": [1000],
            "amount": [1000000.0],
            "returns": [0.01],
        }
    ).to_parquet(market_dir / "BBB.parquet", index=False)

    pd.DataFrame(
        {
            "symbol": ["AAA"],
            "date": ["2026-01-02"],
            "alpha_005": [0.35],
            "alpha_002": [0.36],
        }
    ).to_parquet(factor_dir / "AAA.parquet", index=False)

    pd.DataFrame(
        {
            "symbol": ["BBB"],
            "date": ["2026-01-02"],
            "alpha_005": [0.15],
            "alpha_002": [0.36],
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

    summary_path = research_root / "manual_factor_conclusion_summary.csv"
    pd.DataFrame(
        {
            "factor_name": ["alpha_005", "alpha_002", "alpha_012"],
            "factor_type_manual": [
                "regime_aware_filter",
                "regime_aware_filter_low_confidence",
                "reject",
            ],
            "bucket_rule": [
                "train_defined_bucket in [4]",
                "train_defined_bucket in [4]",
                "none",
            ],
            "valid_regime": [
                "range_bound,strong_repair,strong_trend",
                "range_bound,risk_off,strong_repair",
                "none",
            ],
            "weak_regime": ["risk_off", "strong_trend", "none"],
            "is_filter_factor": [True, True, False],
            "manual_override": [True, True, True],
            "ml_baseline_role": [
                "eligible_filter_candidate",
                "eligible_low_confidence_filter",
                "exclude_rejected",
            ],
        }
    ).to_csv(summary_path, index=False)

    return research_root


def test_assign_train_defined_bucket_uses_train_edges() -> None:
    edges = pd.DataFrame(
        {
            "bucket": [1, 2, 3],
            "left_edge": [float("-inf"), 0.1, 0.2],
            "right_edge": [0.1, 0.2, float("inf")],
        }
    )

    values = pd.Series([0.05, 0.15, 0.25, None])
    out = assign_train_defined_bucket(values, edges)

    assert out.tolist()[:3] == [1.0, 2.0, 3.0]
    assert pd.isna(out.iloc[3])


def test_build_full_market_formulaic_candidate_pool_outputs_combinations(tmp_path: Path) -> None:
    market_dir = tmp_path / "market"
    factor_dir = tmp_path / "factor"
    _write_market_and_factor_files(market_dir, factor_dir)

    research_root = _write_research_files(tmp_path)

    regime_path = tmp_path / "custom_market_regime.csv"
    pd.DataFrame(
        {
            "date": ["2026-01-02"],
            "market_regime": ["range_bound"],
        }
    ).to_csv(regime_path, index=False)

    output_path = tmp_path / "full_market_formulaic_candidate_pool.parquet"

    out = build_full_market_formulaic_candidate_pool(
        market_dir=market_dir,
        factor_dir=factor_dir,
        research_root=research_root,
        manual_summary_path=research_root / "manual_factor_conclusion_summary.csv",
        regime_path=regime_path,
        target_date="2026-01-02",
        output_path=output_path,
        factor_names=["alpha_005", "alpha_002"],
        combinations=None,
    )

    assert output_path.exists()
    assert output_path.with_suffix(".csv").exists()
    assert Path(str(output_path.with_suffix("")) + "_daily_summary.csv").exists()
    assert Path(str(output_path.with_suffix("")) + "_rule_summary.csv").exists()
    assert Path(str(output_path.with_suffix("")) + "_reject_reason_summary.csv").exists()

    assert set(out["symbol"]) == {"AAA"}
    assert set(out["combined_rule_name"]) == {"alpha_005_only", "alpha_005_and_alpha_002"}
    assert out["market_regime"].eq("range_bound").all()
    assert out["is_tradable_base"].all()
    assert out["alpha_005_train_defined_bucket"].eq(4.0).all()
    assert out["alpha_002_train_defined_bucket"].eq(4.0).all()


def test_load_manual_factor_rules_rejects_inactive_regime(tmp_path: Path) -> None:
    research_root = _write_research_files(tmp_path)

    with pytest.raises(ValueError, match="No active factor rules"):
        load_manual_factor_rules(
            research_root / "manual_factor_conclusion_summary.csv",
            target_regime="risk_off",
            factor_names=["alpha_005"],
        )
