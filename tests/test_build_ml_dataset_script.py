from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


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


def test_build_ml_dataset_script_writes_dataset_and_summary(tmp_path: Path):
    market_dir = tmp_path / "market"
    factor_dir = tmp_path / "factor"
    output_dir = tmp_path / "output"
    market_dir.mkdir()
    factor_dir.mkdir()

    dates = pd.DatetimeIndex(
        list(pd.date_range("2021-01-01", periods=8, freq="D"))
        + list(pd.date_range("2025-01-01", periods=8, freq="D"))
    )

    for symbol, offset in [("AAA", 0.0), ("BBB", 5.0)]:
        market, feature = make_symbol_frames(symbol, dates, offset)
        market.to_parquet(market_dir / f"{symbol}.parquet", index=False)
        feature.to_parquet(factor_dir / f"{symbol}.parquet", index=False)

    cmd = [
        sys.executable,
        "scripts/build_ml_dataset.py",
        "--market-dir",
        str(market_dir),
        "--factor-dir",
        str(factor_dir),
        "--output-dir",
        str(output_dir),
        "--output-name",
        "unit_ml_dataset",
        "--factor-cols",
        "alpha_001,alpha_002",
        "--horizon",
        "3",
        "--no-regime",
    ]

    result = subprocess.run(cmd, check=True, capture_output=True, text=True)

    dataset_path = output_dir / "unit_ml_dataset.parquet"
    summary_path = output_dir / "unit_ml_dataset_summary.csv"

    assert dataset_path.exists()
    assert summary_path.exists()
    assert "saved dataset:" in result.stdout
    assert "saved summary:" in result.stdout

    dataset = pd.read_parquet(dataset_path)
    summary = pd.read_csv(summary_path)

    assert {"symbol", "date", "split", "alpha_001", "alpha_002", "label_return_pct", "label_up", "target_horizon"}.issubset(dataset.columns)
    assert set(dataset["split"]) == {"train", "test"}
    assert set(dataset["target_horizon"]) == {3}

    assert {"split", "rows", "dates", "symbols", "label_non_null", "feature_complete_rows"}.issubset(summary.columns)
    assert summary["rows"].sum() == len(dataset)


def test_build_ml_dataset_script_rejects_duplicate_factor_cols(tmp_path: Path):
    cmd = [
        sys.executable,
        "scripts/build_ml_dataset.py",
        "--market-dir",
        str(tmp_path / "market"),
        "--factor-dir",
        str(tmp_path / "factor"),
        "--output-dir",
        str(tmp_path / "output"),
        "--factor-cols",
        "alpha_001,alpha_001",
        "--no-regime",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    assert result.returncode != 0
    assert "Duplicated factor columns" in result.stderr
