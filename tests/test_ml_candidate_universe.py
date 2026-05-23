from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ml_engine.candidate_universe import (
    build_candidate_universe_dataset,
    daily_candidate_count,
    parse_bucket_numbers,
    parse_feature_cols,
    summarize_candidate_universe_dataset,
)


def make_dataset() -> pd.DataFrame:
    rows = []

    for split, start, n_dates in [("train", "2021-01-01", 12), ("test", "2025-01-01", 6)]:
        dates = pd.date_range(start, periods=n_dates, freq="D")
        for date_idx, date in enumerate(dates):
            for symbol_idx in range(10):
                symbol = f"S{symbol_idx:03d}"
                alpha_005 = symbol_idx / 9
                rows.append(
                    {
                        "symbol": symbol,
                        "date": date,
                        "split": split,
                        "alpha_001": symbol_idx * 0.1,
                        "alpha_002": date_idx * 0.1,
                        "alpha_005": alpha_005,
                        "alpha_006": symbol_idx * -0.1,
                        "alpha_008": alpha_005 - 0.5,
                        "alpha_010": 0.5 - alpha_005,
                        "label_return_pct": float(symbol_idx - 5 + date_idx * 0.1),
                        "label_up": symbol_idx > 5,
                        "target_horizon": 3,
                        "market_regime": "range_bound",
                    }
                )

    return pd.DataFrame(rows)


def test_parse_helpers():
    assert parse_bucket_numbers("7,4,4,5") == [4, 5, 7]
    assert parse_feature_cols("alpha_001,alpha_002") == ["alpha_001", "alpha_002"]


def test_build_candidate_universe_dataset_uses_train_defined_bucket():
    df = make_dataset()

    candidate, edges = build_candidate_universe_dataset(
        df,
        bucket_factor="alpha_005",
        candidate_buckets=[4, 5, 6, 7],
        feature_cols=["alpha_001", "alpha_002", "alpha_006", "alpha_008", "alpha_010"],
    )

    assert not candidate.empty
    assert "alpha_005_train_bucket" in candidate.columns
    assert "candidate_bucket_factor" in candidate.columns
    assert set(candidate["alpha_005_train_bucket"].dropna().astype(int)).issubset({4, 5, 6, 7})
    assert set(candidate["candidate_bucket_factor"]) == {"alpha_005"}

    assert "alpha_005" in candidate.columns
    assert {"alpha_001", "alpha_002", "alpha_006", "alpha_008", "alpha_010"}.issubset(candidate.columns)

    assert not edges.empty
    assert edges["selected"].sum() == 4


def test_candidate_summary_and_daily_count():
    df = make_dataset()
    candidate, _ = build_candidate_universe_dataset(
        df,
        feature_cols=["alpha_001", "alpha_002", "alpha_006", "alpha_008", "alpha_010"],
    )

    summary = summarize_candidate_universe_dataset(
        candidate,
        feature_cols=["alpha_001", "alpha_002", "alpha_006", "alpha_008", "alpha_010"],
    )
    daily = daily_candidate_count(candidate)

    assert set(summary["split"]) == {"train", "test"}
    assert not daily.empty
    assert daily["symbols"].gt(0).all()


def test_build_ml_candidate_dataset_script_writes_outputs(tmp_path: Path):
    dataset_path = tmp_path / "ml_dataset.parquet"
    output_dir = tmp_path / "candidate"
    make_dataset().to_parquet(dataset_path, index=False)

    cmd = [
        sys.executable,
        "scripts/build_ml_candidate_dataset.py",
        "--dataset-path",
        str(dataset_path),
        "--output-dir",
        str(output_dir),
        "--output-name",
        "unit_candidate",
        "--bucket-factor",
        "alpha_005",
        "--candidate-buckets",
        "4,5,6,7",
        "--feature-cols",
        "alpha_001,alpha_002,alpha_006,alpha_008,alpha_010",
    ]

    result = subprocess.run(cmd, check=True, capture_output=True, text=True)

    assert "Candidate summary:" in result.stdout
    assert (output_dir / "unit_candidate.parquet").exists()
    assert (output_dir / "unit_candidate_summary.csv").exists()
    assert (output_dir / "unit_candidate_bucket_edges.csv").exists()
    assert (output_dir / "unit_candidate_daily_count.csv").exists()

    out = pd.read_parquet(output_dir / "unit_candidate.parquet")
    assert "alpha_005_train_bucket" in out.columns
    assert "candidate_bucket_rule" in out.columns
