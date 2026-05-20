from __future__ import annotations

from pathlib import Path

import pytest
import numpy as np
import pandas as pd

from data_engine.market_data.loader import (
    NORMALIZED_MARKER,
    import_tdx_txt_to_daily_parquet,
    parse_tdx_txt_file,
)


def write_tdx_file(path: Path, rows: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join([
            "?????? ?? ???",
            "??,??,??,??,??,???,???",
            *rows,
            "#????:???",
            "",
        ]),
        encoding="gb18030",
    )


def test_loader_normalizes_txt_and_parses_english_utf8(tmp_path: Path):
    path = tmp_path / "SZ#000001.txt"

    write_tdx_file(
        path,
        [
            "02/08/2021,15.56,16.12,15.43,15.93,1500000,27000000.00",
            "03/08/2021,15.91,15.98,15.70,15.81,1600000,28800000.00",
        ],
    )

    adjusted_dir = tmp_path / "adjusted"
    unadjusted_dir = tmp_path / "unadjusted"
    output_dir = tmp_path / "out"

    adjusted_file = adjusted_dir / "SZ#000001.txt"
    unadjusted_file = unadjusted_dir / "SZ#000001.txt"

    write_tdx_file(
        adjusted_file,
        [
            "02/08/2021,15.56,16.12,15.43,15.93,1500000,27000000.00",
            "03/08/2021,15.91,15.98,15.70,15.81,1600000,28800000.00",
        ],
    )

    write_tdx_file(
        unadjusted_file,
        [
            "02/08/2021,17.64,18.20,17.51,18.01,1500000,27000000.00",
            "03/08/2021,17.99,18.06,17.78,17.89,1600000,28800000.00",
        ],
    )

    import_tdx_txt_to_daily_parquet(
        adjusted_input=adjusted_file,
        unadjusted_input=unadjusted_file,
        output_dir=output_dir,
        normalize_raw=True,
        incremental=True,
    )

    normalized_text = adjusted_file.read_text(encoding="utf-8")
    assert NORMALIZED_MARKER in normalized_text
    assert "date,open,high,low,close,volume,amount" in normalized_text
    assert "??" not in normalized_text

    df = pd.read_parquet(output_dir / "SZ#000001.parquet")
    assert len(df) == 2

    expected_unadjusted_vwap = df["amount"] / df["volume"]
    expected_adjustment_factor = df["adjusted_close"] / df["unadjusted_close"]
    expected_adjusted_vwap = expected_unadjusted_vwap * expected_adjustment_factor

    assert np.allclose(df["unadjusted_vwap"], expected_unadjusted_vwap)
    assert np.allclose(df["adjustment_factor"], expected_adjustment_factor)
    assert np.allclose(df["adjusted_vwap"], expected_adjusted_vwap)


def test_loader_incremental_update_overwrites_overlap_and_appends_new_date(tmp_path: Path):
    adjusted_file = tmp_path / "adjusted" / "SZ#000001.txt"
    unadjusted_file = tmp_path / "unadjusted" / "SZ#000001.txt"
    output_dir = tmp_path / "out"

    write_tdx_file(
        adjusted_file,
        [
            "02/08/2021,15.56,16.12,15.43,15.93,1500000,27000000.00",
            "03/08/2021,15.91,15.98,15.70,15.81,1600000,28800000.00",
        ],
    )
    write_tdx_file(
        unadjusted_file,
        [
            "02/08/2021,17.64,18.20,17.51,18.01,1500000,27000000.00",
            "03/08/2021,17.99,18.06,17.78,17.89,1600000,28800000.00",
        ],
    )

    import_tdx_txt_to_daily_parquet(
        adjusted_input=adjusted_file,
        unadjusted_input=unadjusted_file,
        output_dir=output_dir,
        normalize_raw=True,
        incremental=True,
    )

    write_tdx_file(
        adjusted_file,
        [
            "03/08/2021,15.91,16.05,15.70,15.99,1600000,28800000.00",
            "04/08/2021,15.96,16.05,15.60,15.73,1700000,30600000.00",
        ],
    )
    write_tdx_file(
        unadjusted_file,
        [
            "03/08/2021,17.99,18.10,17.78,18.07,1600000,28800000.00",
            "04/08/2021,18.04,18.13,17.68,17.81,1700000,30600000.00",
        ],
    )

    import_tdx_txt_to_daily_parquet(
        adjusted_input=adjusted_file,
        unadjusted_input=unadjusted_file,
        output_dir=output_dir,
        normalize_raw=True,
        incremental=True,
    )

    df = pd.read_parquet(output_dir / "SZ#000001.parquet")

    assert len(df) == 3
    assert df[["symbol", "date"]].duplicated().sum() == 0

    row = df.loc[df["date"].eq(pd.Timestamp("2021-08-03"))].iloc[0]
    assert row["adjusted_close"] == 15.99
    assert row["unadjusted_close"] == 18.07

    assert df["returns"].isna().sum() == 1


def test_loader_rejects_invalid_ohlc_rows(tmp_path: Path):
    path = tmp_path / "SZ#000001.txt"

    write_tdx_file(
        path,
        [
            # Invalid: close > high
            "03/08/2021,15.91,15.98,15.70,15.99,1600000,28800000.00",
        ],
    )

    with pytest.raises(ValueError, match="Invalid OHLCV rows"):
        parse_tdx_txt_file(path)

