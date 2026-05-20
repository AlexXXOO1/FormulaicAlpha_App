from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from alpha_engine.formulaic_alphas.alpha_001 import compute_alpha_001


def read_market_data(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()

    if suffix == ".parquet":
        return pd.read_parquet(path)

    if suffix == ".csv":
        return pd.read_csv(path)

    raise ValueError(f"Unsupported input file type: {suffix}. Use .csv or .parquet")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Market data file with symbol/date/close")
    parser.add_argument("--output", required=True, help="Output parquet path")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    df = read_market_data(input_path)
    out = compute_alpha_001(df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_path, index=False)

    print(f"saved: {output_path}")
    print(f"rows: {len(out)}")
    print(f"non_null_alpha_001: {out['alpha_001'].notna().sum()}")


if __name__ == "__main__":
    main()
