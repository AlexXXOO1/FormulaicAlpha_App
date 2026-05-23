from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from ml_engine.dataset import (  # noqa: E402
    DEFAULT_ALPHA_COLS,
    build_ml_dataset_from_symbol_dirs,
    summarize_ml_dataset,
)


DEFAULT_MARKET_DIR = Path(r"C:\Users\zyf37\Desktop\FormulaicAlpha_Data\market_data\daily_bars_by_symbol")
DEFAULT_FACTOR_DIR = Path(r"C:\Users\zyf37\Desktop\FormulaicAlpha_Data\feature_data\formulaic_alphas_by_symbol")
DEFAULT_OUTPUT_DIR = Path(r"C:\Users\zyf37\Desktop\FormulaicAlpha_Data\research_output\ml_dataset")
DEFAULT_CUSTOM_MARKET_REGIME = Path(r"C:\Users\zyf37\Desktop\FormulaicAlpha_Data\market_regime\custom_market_regime.csv")


def parse_factor_cols(raw: str | None) -> list[str]:
    if raw is None or not raw.strip():
        return list(DEFAULT_ALPHA_COLS)

    cols = [item.strip() for item in raw.split(",") if item.strip()]
    if not cols:
        raise ValueError("No valid factor columns parsed from --factor-cols.")

    duplicated = sorted({col for col in cols if cols.count(col) > 1})
    if duplicated:
        raise ValueError(f"Duplicated factor columns in --factor-cols: {duplicated}")

    return cols


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build ML stock selector dataset from market parquet and formulaic alpha feature parquet."
    )
    parser.add_argument("--market-dir", type=Path, default=DEFAULT_MARKET_DIR)
    parser.add_argument("--factor-dir", type=Path, default=DEFAULT_FACTOR_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-name", type=str, default="ml_dataset_v0")

    parser.add_argument(
        "--factor-cols",
        type=str,
        default=",".join(DEFAULT_ALPHA_COLS),
        help="Comma-separated alpha columns. Default uses stable ML feature set.",
    )
    parser.add_argument("--horizon", type=int, default=3)

    parser.add_argument("--train-start", type=int, default=2021)
    parser.add_argument("--train-end", type=int, default=2024)
    parser.add_argument("--test-start", type=int, default=2025)
    parser.add_argument("--test-end", type=int, default=2026)

    parser.add_argument(
        "--custom-market-regime",
        type=Path,
        default=DEFAULT_CUSTOM_MARKET_REGIME,
        help="CSV/JSON market regime file. Ignored if path does not exist or --no-regime is set.",
    )
    parser.add_argument("--no-regime", action="store_true")
    parser.add_argument("--keep-missing-label", action="store_true")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    factor_cols = parse_factor_cols(args.factor_cols)

    custom_market_regime = None
    if not args.no_regime and args.custom_market_regime is not None and args.custom_market_regime.exists():
        custom_market_regime = args.custom_market_regime

    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset = build_ml_dataset_from_symbol_dirs(
        market_dir=args.market_dir,
        factor_dir=args.factor_dir,
        factor_cols=factor_cols,
        horizon=args.horizon,
        custom_market_regime=custom_market_regime,
        train_start=args.train_start,
        train_end=args.train_end,
        test_start=args.test_start,
        test_end=args.test_end,
        drop_missing_label=not args.keep_missing_label,
    )

    summary = summarize_ml_dataset(dataset, factor_cols=factor_cols)

    dataset_path = args.output_dir / f"{args.output_name}.parquet"
    summary_path = args.output_dir / f"{args.output_name}_summary.csv"

    dataset.to_parquet(dataset_path, index=False)
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print(f"saved dataset: {dataset_path}")
    print(f"saved summary: {summary_path}")
    print(f"rows: {len(dataset)}")
    print(f"factor_cols: {','.join(factor_cols)}")
    print(summary.to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
