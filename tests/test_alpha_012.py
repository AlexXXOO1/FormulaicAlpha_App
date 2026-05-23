from __future__ import annotations

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from alpha_engine.formulaic_alphas.alpha_012 import compute_alpha_012
from alpha_engine.formulaic_alphas.registry import get_formulaic_alpha


def test_alpha_012_matches_manual_formula():
    df = pd.DataFrame(
        {
            "symbol": ["AAA", "AAA", "AAA", "AAA", "BBB", "BBB", "BBB", "BBB"],
            "date": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-01-02",
                    "2024-01-03",
                    "2024-01-04",
                    "2024-01-01",
                    "2024-01-02",
                    "2024-01-03",
                    "2024-01-04",
                ]
            ),
            "close": [10.0, 11.0, 10.5, 10.8, 20.0, 19.5, 19.7, 19.1],
            "volume": [100, 120, 110, 110, 200, 180, 190, 170],
        }
    )

    out = compute_alpha_012(df)

    manual = df.copy()
    manual = manual.sort_values(["symbol", "date"], kind="mergesort").reset_index(drop=True)
    delta_volume = manual.groupby("symbol", sort=False)["volume"].diff(1)
    delta_close = manual.groupby("symbol", sort=False)["close"].diff(1)
    expected = np.sign(delta_volume) * (-1.0 * delta_close)

    assert list(out.columns) == ["symbol", "date", "alpha_012"]
    pdt.assert_series_equal(
        out["alpha_012"],
        expected,
        check_names=False,
        check_dtype=False,
    )


def test_alpha_012_zero_volume_delta_outputs_zero():
    df = pd.DataFrame(
        {
            "symbol": ["AAA", "AAA"],
            "date": ["2024-01-01", "2024-01-02"],
            "close": [10.0, 11.0],
            "volume": [100, 100],
        }
    )

    out = compute_alpha_012(df)
    assert out["alpha_012"].iloc[0] != out["alpha_012"].iloc[0]
    assert out["alpha_012"].iloc[1] == 0.0


def test_alpha_012_rejects_missing_columns():
    df = pd.DataFrame(
        {
            "symbol": ["AAA"],
            "date": ["2024-01-01"],
            "close": [10.0],
        }
    )

    with pytest.raises(ValueError, match="Missing required columns for alpha_012"):
        compute_alpha_012(df)


def test_alpha_012_is_registered():
    assert get_formulaic_alpha("alpha_012") is compute_alpha_012
