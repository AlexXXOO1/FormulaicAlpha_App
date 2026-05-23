from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml_engine.dataset import (
    DEFAULT_ALPHA_COLS,
    MODEL_HOLDOUT_ALPHA_COLS,
    REJECTED_ALPHA_COLS,
    add_trade_return_label,
    build_ml_dataset_from_symbol_dirs,
    summarize_ml_dataset,
)


def make_symbol_frames(symbol: str, dates: pd.DatetimeIndex, offset: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    n = len(dates)
    base = 10.0 + offset + np.arange(n, dtype=float)

    market = pd.DataFrame(
        {
            "symbol": symbol,
            "date": dates,
            "adjusted_open": base + 0.10,
            "adjusted_close": base + 0.50,
        }
    )

    feature = pd.DataFrame(
        {
            "symbol": symbol,
            "date": dates,
            "alpha_001": np.linspace(-1.0, 1.0, n) + offset,
            "alpha_002": np.linspace(1.0, -1.0, n) - offset,
        }
    )
    return market, feature


def test_build_ml_dataset_creates_t1_open_to_t3_label_and_splits(tmp_path: Path):
    market_dir = tmp_path / "market"
    factor_dir = tmp_path / "factor"
    market_dir.mkdir()
    factor_dir.mkdir()

    dates = pd.DatetimeIndex(
        list(pd.date_range("2021-01-01", periods=8, freq="D"))
        + list(pd.date_range("2025-01-01", periods=8, freq="D"))
    )

    expected_market = None
    for symbol, offset in [("AAA", 0.0), ("BBB", 5.0)]:
        market, feature = make_symbol_frames(symbol, dates, offset)
        market.to_parquet(market_dir / f"{symbol}.parquet", index=False)
        feature.to_parquet(factor_dir / f"{symbol}.parquet", index=False)
        if symbol == "AAA":
            expected_market = market

    regime = pd.DataFrame(
        {
            "date": dates,
            "market_regime": ["range_bound", "risk_off"] * (len(dates) // 2),
            "risk_score": [1, 2] * (len(dates) // 2),
        }
    )
    regime_path = tmp_path / "custom_market_regime.csv"
    regime.to_csv(regime_path, index=False)

    out = build_ml_dataset_from_symbol_dirs(
        market_dir=market_dir,
        factor_dir=factor_dir,
        factor_cols=["alpha_001", "alpha_002"],
        horizon=3,
        custom_market_regime=regime_path,
        train_start=2021,
        train_end=2024,
        test_start=2025,
        test_end=2026,
    )

    assert {"symbol", "date", "split", "alpha_001", "alpha_002", "label_return_pct", "label_up", "target_horizon"}.issubset(out.columns)
    assert {"market_regime", "risk_score"}.issubset(out.columns)
    assert set(out["split"]) == {"train", "test"}
    assert set(out["target_horizon"]) == {3}

    first = out[(out["symbol"] == "AAA") & (out["date"] == pd.Timestamp("2021-01-01"))].iloc[0]
    expected = (
        expected_market.loc[3, "adjusted_close"] / expected_market.loc[1, "adjusted_open"] - 1.0
    ) * 100.0
    assert first["label_return_pct"] == pytest.approx(expected)
    assert first["label_up"] == (expected > 0)


def test_add_trade_return_label_rejects_t1_horizon():
    market = pd.DataFrame(
        {
            "symbol": ["AAA", "AAA", "AAA"],
            "date": pd.date_range("2021-01-01", periods=3, freq="D"),
            "adjusted_open": [10.0, 11.0, 12.0],
            "adjusted_close": [10.5, 11.5, 12.5],
        }
    )

    with pytest.raises(ValueError, match="horizon >= 2"):
        add_trade_return_label(market, horizon=1)


def test_build_ml_dataset_rejects_missing_factor_column(tmp_path: Path):
    market_dir = tmp_path / "market"
    factor_dir = tmp_path / "factor"
    market_dir.mkdir()
    factor_dir.mkdir()

    dates = pd.date_range("2021-01-01", periods=5, freq="D")
    market, feature = make_symbol_frames("AAA", dates, 0.0)
    market.to_parquet(market_dir / "AAA.parquet", index=False)
    feature.drop(columns=["alpha_002"]).to_parquet(factor_dir / "AAA.parquet", index=False)

    with pytest.raises(ValueError, match="Feature data missing columns"):
        build_ml_dataset_from_symbol_dirs(
            market_dir=market_dir,
            factor_dir=factor_dir,
            factor_cols=["alpha_001", "alpha_002"],
            horizon=3,
        )


def test_summarize_ml_dataset_counts_complete_feature_rows(tmp_path: Path):
    market_dir = tmp_path / "market"
    factor_dir = tmp_path / "factor"
    market_dir.mkdir()
    factor_dir.mkdir()

    dates = pd.date_range("2021-01-01", periods=6, freq="D")
    market, feature = make_symbol_frames("AAA", dates, 0.0)
    feature.loc[0, "alpha_002"] = np.nan
    market.to_parquet(market_dir / "AAA.parquet", index=False)
    feature.to_parquet(factor_dir / "AAA.parquet", index=False)

    out = build_ml_dataset_from_symbol_dirs(
        market_dir=market_dir,
        factor_dir=factor_dir,
        factor_cols=["alpha_001", "alpha_002"],
        horizon=3,
    )
    summary = summarize_ml_dataset(out, factor_cols=["alpha_001", "alpha_002"])

    train_row = summary[summary["split"] == "train"].iloc[0]
    assert train_row["rows"] == len(out)
    assert train_row["feature_complete_rows"] == len(out) - 1

def test_default_ml_feature_set_excludes_rejected_and_holdout_alphas():
    assert "alpha_003" in MODEL_HOLDOUT_ALPHA_COLS
    assert "alpha_004" in MODEL_HOLDOUT_ALPHA_COLS
    assert "alpha_007" in REJECTED_ALPHA_COLS
    assert "alpha_009" in REJECTED_ALPHA_COLS

    assert "alpha_003" not in DEFAULT_ALPHA_COLS
    assert "alpha_004" not in DEFAULT_ALPHA_COLS
    assert "alpha_007" not in DEFAULT_ALPHA_COLS
    assert "alpha_009" not in DEFAULT_ALPHA_COLS

    assert DEFAULT_ALPHA_COLS == (
        "alpha_001",
        "alpha_002",
        "alpha_005",
        "alpha_006",
        "alpha_008",
        "alpha_010",
    )

