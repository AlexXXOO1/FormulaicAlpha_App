from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

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
    adjusted_file = tmp_path / "adjusted" / "SZ#000001.txt"
    unadjusted_file = tmp_path / "unadjusted" / "SZ#000001.txt"
    output_dir = tmp_path / "out"

    write_tdx_file(
        adjusted_file,
        [
            "02/08/2021,15.56,16.12,15.43,15.93,1500000,27000000.00",
            # Latest day is anchored to unadjusted price basis.
            "03/08/2021,17.99,18.06,17.78,17.89,1600000,28800000.00",
        ],
    )

    write_tdx_file(
        unadjusted_file,
        [
            "02/08/2021,17.64,18.20,17.51,18.01,1500000,27000000.00",
            "03/08/2021,17.99,18.06,17.78,17.89,1600000,28800000.00",
        ],
    )

    report = import_tdx_txt_to_daily_parquet(
        adjusted_input=adjusted_file,
        unadjusted_input=unadjusted_file,
        output_dir=output_dir,
        normalize_raw=True,
        incremental=True,
    )

    assert report.loc[0, "status"] == "ok"

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
            "03/08/2021,17.99,18.06,17.78,17.89,1600000,28800000.00",
        ],
    )
    write_tdx_file(
        unadjusted_file,
        [
            "02/08/2021,17.64,18.20,17.51,18.01,1500000,27000000.00",
            "03/08/2021,17.99,18.06,17.78,17.89,1600000,28800000.00",
        ],
    )

    report_1 = import_tdx_txt_to_daily_parquet(
        adjusted_input=adjusted_file,
        unadjusted_input=unadjusted_file,
        output_dir=output_dir,
        normalize_raw=True,
        incremental=True,
    )

    assert report_1.loc[0, "status"] == "ok"

    write_tdx_file(
        adjusted_file,
        [
            "03/08/2021,17.99,18.10,17.78,18.07,1600000,28800000.00",
            "04/08/2021,18.04,18.13,17.68,17.81,1700000,30600000.00",
        ],
    )
    write_tdx_file(
        unadjusted_file,
        [
            "03/08/2021,17.99,18.10,17.78,18.07,1600000,28800000.00",
            "04/08/2021,18.04,18.13,17.68,17.81,1700000,30600000.00",
        ],
    )

    report_2 = import_tdx_txt_to_daily_parquet(
        adjusted_input=adjusted_file,
        unadjusted_input=unadjusted_file,
        output_dir=output_dir,
        normalize_raw=True,
        incremental=True,
    )

    assert report_2.loc[0, "status"] == "ok"

    df = pd.read_parquet(output_dir / "SZ#000001.parquet")

    assert len(df) == 3
    assert df[["symbol", "date"]].duplicated().sum() == 0

    row = df.loc[df["date"].eq(pd.Timestamp("2021-08-03"))].iloc[0]
    assert row["adjusted_close"] == 18.07
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


def test_loader_rejects_swapped_adjusted_and_unadjusted_inputs(tmp_path: Path):
    adjusted_file = tmp_path / "adjusted" / "SZ#000001.txt"
    unadjusted_file = tmp_path / "unadjusted" / "SZ#000001.txt"
    output_dir = tmp_path / "out"

    # Wrong on purpose: adjusted folder receives unadjusted-like prices.
    write_tdx_file(
        adjusted_file,
        [
            "02/08/2021,17.64,18.20,17.51,18.01,1500000,27000000.00",
            "03/08/2021,17.99,18.06,17.78,17.89,1600000,28800000.00",
        ],
    )

    # Wrong on purpose: unadjusted folder receives adjusted-like prices.
    write_tdx_file(
        unadjusted_file,
        [
            "02/08/2021,15.56,16.12,15.43,15.93,1500000,27000000.00",
            "03/08/2021,15.91,15.98,15.70,15.81,1600000,28800000.00",
        ],
    )

    report = import_tdx_txt_to_daily_parquet(
        adjusted_input=adjusted_file,
        unadjusted_input=unadjusted_file,
        output_dir=output_dir,
        normalize_raw=True,
        incremental=True,
    )

    assert report.loc[0, "status"] == "failed"
    assert "swapped" in report.loc[0, "error"] or "adjustment_factor" in report.loc[0, "error"]


def test_loader_rejects_normalized_file_with_wrong_price_basis(tmp_path: Path):
    adjusted_file = tmp_path / "adjusted" / "SZ#000001.txt"
    unadjusted_file = tmp_path / "unadjusted" / "SZ#000001.txt"
    output_dir = tmp_path / "out"

    adjusted_file.parent.mkdir(parents=True, exist_ok=True)
    adjusted_file.write_text(
        "\n".join([
            NORMALIZED_MARKER,
            "# symbol=SZ#000001",
            "# source_format=tdx_daily_txt",
            "# price_basis=unadjusted_ohlcv",
            "# original_encoding=gb18030",
            "date,open,high,low,close,volume,amount",
            "02/08/2021,15.56,16.12,15.43,15.93,1500000,27000000.00",
            "",
        ]),
        encoding="utf-8",
    )

    write_tdx_file(
        unadjusted_file,
        [
            "02/08/2021,17.64,18.20,17.51,18.01,1500000,27000000.00",
        ],
    )

    report = import_tdx_txt_to_daily_parquet(
        adjusted_input=adjusted_file,
        unadjusted_input=unadjusted_file,
        output_dir=output_dir,
        normalize_raw=True,
        incremental=True,
    )

    assert report.loc[0, "status"] == "failed"
    assert "price_basis mismatch" in report.loc[0, "error"]
