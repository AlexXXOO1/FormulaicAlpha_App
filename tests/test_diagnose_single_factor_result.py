from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from research.factor_analysis.diagnose_single_factor_result import diagnose_single_factor_result


def test_diagnose_single_factor_result_prefers_middle(tmp_path: Path):
    bucket = pd.DataFrame({
        "target": ["fwd_return_pct_T3"] * 10,
        "alpha_001_bucket": [float(i) for i in range(1, 11)],
        "sample_count": [100] * 10,
        "mean_return_pct": [-0.2, 0.0, 0.2, 0.8, 1.0, 0.9, 0.3, 0.1, 0.0, -0.1],
        "median_return_pct": [-0.1] * 10,
        "up_ratio": [0.5] * 10,
        "min_factor": [0.0] * 10,
        "max_factor": [1.0] * 10,
    })

    ic = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=3),
        "target": ["fwd_return_pct_T3"] * 3,
        "spearman_ic": [0.1, 0.2, 0.3],
        "pearson_ic": [0.05, 0.15, 0.25],
        "sample_count": [3000, 3001, 3002],
    })

    bucket_path = tmp_path / "bucket_summary.csv"
    ic_path = tmp_path / "daily_ic.csv"
    output_path = tmp_path / "factor_diagnosis.csv"

    bucket.to_csv(bucket_path, index=False)
    ic.to_csv(ic_path, index=False)

    out = diagnose_single_factor_result(
        bucket_summary_path=bucket_path,
        daily_ic_path=ic_path,
        output_path=output_path,
    )

    assert output_path.exists()
    assert len(out) == 1
    assert out.loc[0, "target"] == "fwd_return_pct_T3"
    assert out.loc[0, "best_bucket"] == 5.0
    assert out.loc[0, "conclusion"] == "prefer_middle"
    assert out.loc[0, "spearman_ic_mean"] == pytest.approx(0.2)
