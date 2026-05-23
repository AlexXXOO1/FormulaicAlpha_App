from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml_engine.profile import (
    build_ml_dataset_profile,
    factor_distribution_drift,
    factor_missing_summary,
    infer_regime_col,
    label_by_regime,
    write_ml_dataset_profile,
)


def make_profile_df() -> pd.DataFrame:
    dates_train = pd.date_range("2021-01-01", periods=6, freq="D")
    dates_test = pd.date_range("2025-01-01", periods=6, freq="D")

    rows = []
    for split, dates, shift in [("train", dates_train, 0.0), ("test", dates_test, 1.0)]:
        for i, date in enumerate(dates):
            for symbol_idx, symbol in enumerate(["AAA", "BBB", "CCC"]):
                rows.append(
                    {
                        "symbol": symbol,
                        "date": date,
                        "split": split,
                        "alpha_001": i + symbol_idx + shift,
                        "alpha_002": (i - symbol_idx) * 0.5 + shift,
                        "alpha_005": np.nan if (split == "train" and i == 0 and symbol == "AAA") else i * 0.1,
                        "label_return_pct": float(i - 2 + symbol_idx),
                        "market_regime": "range_bound" if i % 2 == 0 else "risk_off",
                    }
                )
    return pd.DataFrame(rows)


def test_factor_missing_summary_reports_missing_rate():
    df = make_profile_df()
    out = factor_missing_summary(df, ["alpha_001", "alpha_002", "alpha_005"])

    hit = out[(out["split"] == "train") & (out["factor"] == "alpha_005")].iloc[0]
    assert hit["missing_count"] == 1
    assert hit["rows"] == 18
    assert hit["missing_rate"] == pytest.approx(1 / 18)


def test_factor_distribution_drift_contains_psi_and_mean_diff():
    df = make_profile_df()
    out = factor_distribution_drift(df, ["alpha_001", "alpha_002"])

    assert {"factor", "mean_diff", "psi_train_to_test", "train_mean", "test_mean"}.issubset(out.columns)
    assert out["psi_train_to_test"].notna().all()


def test_label_by_regime_uses_inferred_market_regime():
    df = make_profile_df()
    assert infer_regime_col(df) == "market_regime"

    out = label_by_regime(df, ["alpha_001", "alpha_002", "alpha_005"])
    assert {"split", "regime_col", "regime", "label_mean_pct", "label_win_rate"}.issubset(out.columns)
    assert set(out["regime_col"]) == {"market_regime"}


def test_build_profile_returns_expected_frames():
    df = make_profile_df()
    out = build_ml_dataset_profile(df, factor_cols=["alpha_001", "alpha_002", "alpha_005"])

    assert {
        "factor_missing",
        "factor_distribution_drift",
        "factor_correlation_train",
        "factor_correlation_all",
        "label_by_year",
        "label_by_regime",
        "daily_sample_stability",
        "missing_pattern",
    }.issubset(out.keys())

    assert not out["daily_sample_stability"].empty
    assert not out["label_by_year"].empty


def test_write_ml_dataset_profile_creates_csv_files(tmp_path: Path):
    df = make_profile_df()
    paths = write_ml_dataset_profile(
        df,
        output_dir=tmp_path,
        output_name="unit_profile",
        factor_cols=["alpha_001", "alpha_002", "alpha_005"],
    )

    assert "factor_missing" in paths
    assert "factor_distribution_drift" in paths
    assert all(path.exists() for path in paths.values())


def test_profile_ml_dataset_script_writes_outputs(tmp_path: Path):
    df = make_profile_df()
    dataset_path = tmp_path / "ml_dataset.parquet"
    output_dir = tmp_path / "profile"
    df.to_parquet(dataset_path, index=False)

    cmd = [
        sys.executable,
        "scripts/profile_ml_dataset.py",
        "--dataset-path",
        str(dataset_path),
        "--output-dir",
        str(output_dir),
        "--output-name",
        "unit_cli_profile",
        "--factor-cols",
        "alpha_001,alpha_002,alpha_005",
    ]

    result = subprocess.run(cmd, check=True, capture_output=True, text=True)

    assert "saved profile files:" in result.stdout
    assert (output_dir / "unit_cli_profile_factor_missing.csv").exists()
    assert (output_dir / "unit_cli_profile_factor_distribution_drift.csv").exists()
    assert (output_dir / "unit_cli_profile_label_by_year.csv").exists()
