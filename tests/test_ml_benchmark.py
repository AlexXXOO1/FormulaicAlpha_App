from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

from ml_engine.benchmark import (
    build_benchmark_comparison,
    build_market_daily_benchmark,
    build_middle_bucket_benchmark,
    build_random_topk_benchmark,
    load_ml_topk_daily,
    summarize_benchmark_daily,
)


def make_dataset() -> pd.DataFrame:
    rows = []

    for split, start, n_dates in [("train", "2021-01-01", 20), ("test", "2025-01-01", 8)]:
        dates = pd.date_range(start, periods=n_dates, freq="D")
        for date_idx, date in enumerate(dates):
            for symbol_idx, symbol in enumerate(["AAA", "BBB", "CCC", "DDD", "EEE"]):
                alpha_001 = symbol_idx / 4
                alpha_005 = date_idx / max(n_dates - 1, 1)
                label = symbol_idx - 2 + date_idx * 0.1
                rows.append(
                    {
                        "symbol": symbol,
                        "date": date,
                        "split": split,
                        "alpha_001": alpha_001,
                        "alpha_005": alpha_005,
                        "label_return_pct": float(label),
                        "market_regime": "range_bound" if date_idx % 2 == 0 else "risk_off",
                    }
                )

    return pd.DataFrame(rows)


def make_topk_daily(path: Path) -> Path:
    rows = []
    for top_k in [1, 3]:
        for date in pd.date_range("2025-01-01", periods=8, freq="D"):
            rows.append(
                {
                    "split": "test",
                    "date": date,
                    "top_k": top_k,
                    "selected_count": top_k,
                    "return_mean_pct": 1.0 / top_k,
                    "return_median_pct": 1.0 / top_k,
                    "symbol_win_rate": 0.6,
                    "regime_col": "market_regime",
                    "regime": "range_bound",
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return path


def test_market_and_random_benchmarks():
    df = make_dataset()

    market = build_market_daily_benchmark(df)
    random = build_random_topk_benchmark(df, top_k=[1, 3], random_state=1)

    assert set(market["benchmark_name"]) == {"market_all"}
    assert set(random["benchmark_name"]) == {"random_top_1", "random_top_3"}
    assert market["selected_count"].min() == 5
    assert random[random["benchmark_name"] == "random_top_1"]["selected_count"].eq(1).all()


def test_middle_bucket_benchmark_creates_single_and_combined_rules():
    df = make_dataset()

    out = build_middle_bucket_benchmark(
        df,
        factor_cols=["alpha_001", "alpha_005"],
        bucket_numbers=[4, 5, 6, 7],
    )

    assert "alpha_001_bucket_4_7" in set(out["benchmark_name"])
    assert "alpha_005_bucket_4_7" in set(out["benchmark_name"])
    assert "alpha_001_alpha_005_bucket_4_7_AND" in set(out["benchmark_name"])
    assert out["selected_count"].gt(0).all()


def test_load_ml_topk_daily_and_summary(tmp_path: Path):
    path = make_topk_daily(tmp_path / "topk_daily.csv")
    out = load_ml_topk_daily(path)

    assert set(out["benchmark_name"]) == {"ml_top_1", "ml_top_3"}
    assert set(out["benchmark_type"]) == {"ml_topk"}

    summary = summarize_benchmark_daily(out)
    assert {"benchmark_name", "mean_daily_return_pct", "daily_win_rate"}.issubset(summary.columns)


def test_build_benchmark_comparison_returns_all_frames(tmp_path: Path):
    df = make_dataset()
    ml = load_ml_topk_daily(make_topk_daily(tmp_path / "topk_daily.csv"))

    out = build_benchmark_comparison(
        df,
        ml_topk_daily=ml,
        top_k=[1, 3],
        bucket_factors=["alpha_001", "alpha_005"],
    )

    assert {"benchmark_daily", "benchmark_summary", "benchmark_by_year", "benchmark_by_regime"}.issubset(out.keys())
    assert not out["benchmark_summary"].empty
    assert "market_all" in set(out["benchmark_daily"]["benchmark_name"])
    assert "ml_top_1" in set(out["benchmark_daily"]["benchmark_name"])


def test_evaluate_ml_benchmark_script_writes_outputs(tmp_path: Path):
    dataset_path = tmp_path / "ml_dataset.parquet"
    topk_path = tmp_path / "topk_daily.csv"
    output_dir = tmp_path / "benchmark"

    make_dataset().to_parquet(dataset_path, index=False)
    make_topk_daily(topk_path)

    cmd = [
        sys.executable,
        "scripts/evaluate_ml_benchmark.py",
        "--dataset-path",
        str(dataset_path),
        "--topk-daily-path",
        str(topk_path),
        "--output-dir",
        str(output_dir),
        "--output-name",
        "unit_benchmark",
        "--top-k",
        "1,3",
        "--bucket-factors",
        "alpha_001,alpha_005",
    ]

    result = subprocess.run(cmd, check=True, capture_output=True, text=True)

    assert "Benchmark summary:" in result.stdout
    assert (output_dir / "unit_benchmark_benchmark_daily.csv").exists()
    assert (output_dir / "unit_benchmark_benchmark_summary.csv").exists()
    assert (output_dir / "unit_benchmark_benchmark_by_year.csv").exists()
    assert (output_dir / "unit_benchmark_benchmark_by_regime.csv").exists()
