from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ml_engine.model import (
    build_prediction_frame,
    load_model_bundle,
    save_model_bundle,
    train_model_from_dataset,
)


def make_ml_df() -> pd.DataFrame:
    rows = []
    for split, start, n_dates in [("train", "2021-01-01", 20), ("test", "2025-01-01", 8)]:
        dates = pd.date_range(start, periods=n_dates, freq="D")
        for date_idx, date in enumerate(dates):
            for symbol_idx, symbol in enumerate(["AAA", "BBB", "CCC", "DDD", "EEE"]):
                a1 = date_idx * 0.1 + symbol_idx * 0.2
                a2 = symbol_idx * -0.1
                label = 2.0 * a1 - 0.5 * a2
                rows.append(
                    {
                        "symbol": symbol,
                        "date": date,
                        "split": split,
                        "alpha_001": a1,
                        "alpha_002": a2,
                        "label_return_pct": label,
                        "label_up": label > 0,
                        "target_horizon": 3,
                        "market_regime": "range_bound" if date_idx % 2 == 0 else "risk_off",
                    }
                )
    return pd.DataFrame(rows)


def test_train_model_from_dataset_and_predict():
    df = make_ml_df()
    model, metadata = train_model_from_dataset(
        df,
        factor_cols=["alpha_001", "alpha_002"],
        max_iter=10,
        random_state=7,
    )

    pred = build_prediction_frame(
        df,
        model,
        factor_cols=["alpha_001", "alpha_002"],
        predict_split="test",
    )

    assert metadata["model_type"] == "HistGradientBoostingRegressor"
    assert metadata["train_rows"] == 100
    assert len(pred) == 40
    assert pred["prediction"].notna().all()
    assert {"symbol", "date", "split", "alpha_001", "alpha_002", "label_return_pct", "prediction"}.issubset(pred.columns)


def test_save_and_load_model_bundle(tmp_path: Path):
    df = make_ml_df()
    model, metadata = train_model_from_dataset(
        df,
        factor_cols=["alpha_001", "alpha_002"],
        max_iter=5,
    )

    path = save_model_bundle(model=model, metadata=metadata, path=tmp_path / "model.pkl")
    bundle = load_model_bundle(path)

    assert path.exists()
    assert "model" in bundle
    assert bundle["metadata"]["factor_cols"] == ["alpha_001", "alpha_002"]


def test_train_ml_stock_selector_script_writes_outputs(tmp_path: Path):
    df = make_ml_df()
    dataset_path = tmp_path / "ml_dataset.parquet"
    output_dir = tmp_path / "model"
    df.to_parquet(dataset_path, index=False)

    cmd = [
        sys.executable,
        "scripts/train_ml_stock_selector.py",
        "--dataset-path",
        str(dataset_path),
        "--output-dir",
        str(output_dir),
        "--output-name",
        "unit_model",
        "--factor-cols",
        "alpha_001,alpha_002",
        "--top-k",
        "1,3",
        "--max-iter",
        "5",
    ]

    result = subprocess.run(cmd, check=True, capture_output=True, text=True)

    assert "[progress]" in result.stdout
    assert "Top-k summary:" in result.stdout
    assert (output_dir / "unit_model_model.pkl").exists()
    assert (output_dir / "unit_model_metadata.json").exists()
    assert (output_dir / "unit_model_predictions.parquet").exists()
    assert (output_dir / "unit_model_topk_daily.csv").exists()
    assert (output_dir / "unit_model_topk_summary.csv").exists()
    assert (output_dir / "unit_model_topk_by_year.csv").exists()
    assert (output_dir / "unit_model_topk_by_regime.csv").exists()
