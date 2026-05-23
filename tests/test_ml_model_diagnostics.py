from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

from ml_engine.model_diagnostics import (
    build_prediction_bucket_detail,
    build_prediction_diagnostics,
    diagnose_prediction_gate,
    summarize_prediction_buckets,
)


def make_predictions(*, monotonic: bool) -> pd.DataFrame:
    rows = []
    for date in pd.date_range("2025-01-01", periods=5, freq="D"):
        for idx in range(200):
            prediction = idx / 199
            if monotonic:
                label = prediction * 2.0
            else:
                label = (1.0 - prediction) * 2.0

            rows.append(
                {
                    "symbol": f"S{idx:04d}",
                    "date": date,
                    "split": "test",
                    "prediction": prediction,
                    "label_return_pct": label,
                }
            )
    return pd.DataFrame(rows)


def make_topk_summary(*, monotonic: bool) -> pd.DataFrame:
    if monotonic:
        values = {10: 1.0, 20: 0.8, 50: 0.6}
    else:
        values = {10: 0.4, 20: 0.6, 50: 0.5}

    return pd.DataFrame(
        [
            {"top_k": k, "mean_daily_return_pct": v}
            for k, v in values.items()
        ]
    )


def make_benchmark_summary() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "benchmark_name": "market_all",
                "benchmark_type": "market",
                "mean_daily_return_pct": 0.5,
            }
        ]
    )


def test_prediction_bucket_summary_detects_monotonic_success():
    pred = make_predictions(monotonic=True)

    detail = build_prediction_bucket_detail(pred, bucket_count=10, min_daily_rows=100)
    summary = summarize_prediction_buckets(detail)
    gate = diagnose_prediction_gate(
        summary,
        topk_summary=make_topk_summary(monotonic=True),
        benchmark_summary=make_benchmark_summary(),
    )

    assert not detail.empty
    assert len(summary) == 10
    assert gate.iloc[0]["model_status"] == "passed"
    assert gate.iloc[0]["top_bucket_beats_bottom"]


def test_prediction_gate_fails_when_high_prediction_bucket_underperforms():
    pred = make_predictions(monotonic=False)

    diagnostics = build_prediction_diagnostics(
        pred,
        topk_summary=make_topk_summary(monotonic=False),
        benchmark_summary=make_benchmark_summary(),
        bucket_count=10,
        min_daily_rows=100,
    )

    gate = diagnostics["prediction_bucket_gate"].iloc[0]

    assert gate["model_status"] == "failed"
    assert gate["reason"] == "no_monotonic_prediction_edge"
    assert gate["top_bucket_return_pct"] < gate["bottom_bucket_return_pct"]


def test_diagnose_ml_predictions_script_writes_outputs(tmp_path: Path):
    pred_path = tmp_path / "predictions.parquet"
    topk_path = tmp_path / "topk_summary.csv"
    benchmark_path = tmp_path / "benchmark_summary.csv"
    output_dir = tmp_path / "diagnostics"

    make_predictions(monotonic=False).to_parquet(pred_path, index=False)
    make_topk_summary(monotonic=False).to_csv(topk_path, index=False)
    make_benchmark_summary().to_csv(benchmark_path, index=False)

    cmd = [
        sys.executable,
        "scripts/diagnose_ml_predictions.py",
        "--predictions-path",
        str(pred_path),
        "--topk-summary-path",
        str(topk_path),
        "--benchmark-summary-path",
        str(benchmark_path),
        "--output-dir",
        str(output_dir),
        "--output-name",
        "unit_diag",
    ]

    result = subprocess.run(cmd, check=True, capture_output=True, text=True)

    assert "Prediction gate:" in result.stdout
    assert (output_dir / "unit_diag_prediction_bucket_detail.csv").exists()
    assert (output_dir / "unit_diag_prediction_bucket_summary.csv").exists()
    assert (output_dir / "unit_diag_prediction_bucket_gate.csv").exists()

    gate = pd.read_csv(output_dir / "unit_diag_prediction_bucket_gate.csv")
    assert gate.iloc[0]["model_status"] == "failed"
