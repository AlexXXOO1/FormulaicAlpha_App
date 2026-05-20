from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = [
    "symbol",
    "date",
    "adjusted_open",
    "adjusted_high",
    "adjusted_low",
    "adjusted_close",
    "unadjusted_open",
    "unadjusted_high",
    "unadjusted_low",
    "unadjusted_close",
    "volume",
    "amount",
    "unadjusted_vwap",
    "adjustment_factor",
    "adjusted_vwap",
    "returns",
    "source_adjusted_ohlc_file",
    "source_adjusted_ohlc_encoding",
    "source_unadjusted_ohlcv_file",
    "source_unadjusted_ohlcv_encoding",
]


def fail(message: str) -> None:
    raise RuntimeError(message)


def validate_ohlc(df: pd.DataFrame, prefix: str) -> None:
    open_col = f"{prefix}_open"
    high_col = f"{prefix}_high"
    low_col = f"{prefix}_low"
    close_col = f"{prefix}_close"

    bad = (
        (df[open_col] <= 0)
        | (df[high_col] <= 0)
        | (df[low_col] <= 0)
        | (df[close_col] <= 0)
        | (df[high_col] < df[[open_col, low_col, close_col]].max(axis=1))
        | (df[low_col] > df[[open_col, high_col, close_col]].min(axis=1))
    )

    if bad.any():
        sample = df.loc[
            bad,
            ["symbol", "date", open_col, high_col, low_col, close_col],
        ].head(20)
        fail(f"Invalid {prefix} OHLC rows:\n{sample.to_string(index=False)}")


def validate_market_parquet(path: Path) -> None:
    df = pd.read_parquet(path)

    print("file:", path)
    print("rows:", len(df))

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        fail(f"Missing required columns: {missing}")

    if df.empty:
        fail("DataFrame is empty.")

    df = df[REQUIRED_COLUMNS].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()

    if df["date"].isna().any():
        sample = df.loc[df["date"].isna()].head(20)
        fail(f"Invalid date rows:\n{sample.to_string(index=False)}")

    if df[["symbol", "date"]].duplicated().any():
        dup = df[df[["symbol", "date"]].duplicated(keep=False)].head(20)
        fail(f"Duplicate symbol/date rows:\n{dup.to_string(index=False)}")

    df = df.sort_values(["symbol", "date"], kind="mergesort").reset_index(drop=True)

    for symbol, part in df.groupby("symbol", sort=False):
        if not part["date"].is_monotonic_increasing:
            fail(f"Date is not sorted for symbol: {symbol}")

    numeric_cols = [
        "adjusted_open",
        "adjusted_high",
        "adjusted_low",
        "adjusted_close",
        "unadjusted_open",
        "unadjusted_high",
        "unadjusted_low",
        "unadjusted_close",
        "volume",
        "amount",
        "unadjusted_vwap",
        "adjustment_factor",
        "adjusted_vwap",
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

        if df[col].isna().any():
            sample = df.loc[df[col].isna(), ["symbol", "date", col]].head(20)
            fail(f"NaN found in {col}:\n{sample.to_string(index=False)}")

    if (df["volume"] <= 0).any():
        sample = df.loc[df["volume"] <= 0, ["symbol", "date", "volume"]].head(20)
        fail(f"Non-positive volume rows:\n{sample.to_string(index=False)}")

    if (df["amount"] < 0).any():
        sample = df.loc[df["amount"] < 0, ["symbol", "date", "amount"]].head(20)
        fail(f"Negative amount rows:\n{sample.to_string(index=False)}")

    validate_ohlc(df, "adjusted")
    validate_ohlc(df, "unadjusted")

    expected_unadjusted_vwap = df["amount"] / df["volume"]
    expected_adjustment_factor = df["adjusted_close"] / df["unadjusted_close"]
    expected_adjusted_vwap = expected_unadjusted_vwap * expected_adjustment_factor

    if not np.allclose(
        df["unadjusted_vwap"],
        expected_unadjusted_vwap,
        rtol=1e-10,
        atol=1e-10,
        equal_nan=True,
    ):
        fail("unadjusted_vwap mismatch: expected amount / volume")

    if not np.allclose(
        df["adjustment_factor"],
        expected_adjustment_factor,
        rtol=1e-10,
        atol=1e-10,
        equal_nan=True,
    ):
        fail("adjustment_factor mismatch: expected adjusted_close / unadjusted_close")

    if not np.allclose(
        df["adjusted_vwap"],
        expected_adjusted_vwap,
        rtol=1e-10,
        atol=1e-10,
        equal_nan=True,
    ):
        fail("adjusted_vwap mismatch: expected unadjusted_vwap * adjustment_factor")

    expected_returns = df.groupby("symbol", sort=False)["adjusted_close"].pct_change()

    if not np.allclose(
        df["returns"],
        expected_returns,
        rtol=1e-10,
        atol=1e-10,
        equal_nan=True,
    ):
        fail("returns mismatch: expected pct_change of adjusted_close")

    print("symbol_count:", df["symbol"].nunique())
    print("date_min:", df["date"].min())
    print("date_max:", df["date"].max())
    print("validation: PASS")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True)
    args = parser.parse_args()

    validate_market_parquet(Path(args.path))


if __name__ == "__main__":
    main()
