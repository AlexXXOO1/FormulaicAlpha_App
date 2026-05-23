from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import pandas as pd

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from ml_engine.benchmark import (  # noqa: E402
    DEFAULT_BUCKET_FACTORS,
    DEFAULT_MIDDLE_BUCKETS,
    build_benchmark_comparison,
    load_ml_topk_daily,
    write_benchmark_comparison,
)
from ml_engine.evaluator import parse_top_k  # noqa: E402


DEFAULT_DATASET_PATH = Path(r"C:\Users\zyf37\Desktop\FormulaicAlpha_Data\research_output\ml_dataset\ml_dataset_v0.parquet")
DEFAULT_MODEL_DIR = Path(r"C:\Users\zyf37\Desktop\FormulaicAlpha_Data\research_output\ml_dataset\model")
DEFAULT_TOPK_DAILY_PATH = DEFAULT_MODEL_DIR / "ml_model_v0_topk_daily.csv"
DEFAULT_OUTPUT_DIR = DEFAULT_MODEL_DIR / "benchmark"


def parse_csv_cols(raw: str | None, default: Sequence[str]) -> list[str]:
    if raw is None or not raw.strip():
        return list(default)

    cols = [item.strip() for item in raw.split(",") if item.strip()]
    if not cols:
        raise ValueError("No valid values parsed.")

    duplicated = sorted({col for col in cols if cols.count(col) > 1})
    if duplicated:
        raise ValueError(f"Duplicated values: {duplicated}")

    return cols


def parse_int_list(raw: str | None, default: Sequence[int]) -> list[int]:
    if raw is None or not raw.strip():
        return [int(x) for x in default]
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("No valid integer values parsed.")
    return sorted(set(values))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare ML top-k result against market, random, and manual middle-bucket benchmarks.")
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--topk-daily-path", type=Path, default=DEFAULT_TOPK_DAILY_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-name", type=str, default="ml_model_v0")

    parser.add_argument("--top-k", type=str, default="10,20,50")
    parser.add_argument("--bucket-factors", type=str, default=",".join(DEFAULT_BUCKET_FACTORS))
    parser.add_argument("--bucket-numbers", type=str, default=",".join(str(x) for x in DEFAULT_MIDDLE_BUCKETS))

    parser.add_argument("--label-col", type=str, default="label_return_pct")
    parser.add_argument("--train-split", type=str, default="train")
    parser.add_argument("--test-split", type=str, default="test")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--regime-col", type=str, default=None)
    parser.add_argument("--no-ml-topk", action="store_true")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    top_k = parse_top_k(args.top_k)
    bucket_factors = parse_csv_cols(args.bucket_factors, DEFAULT_BUCKET_FACTORS)
    bucket_numbers = parse_int_list(args.bucket_numbers, DEFAULT_MIDDLE_BUCKETS)

    print(f"loading dataset: {args.dataset_path}", flush=True)
    dataset = pd.read_parquet(args.dataset_path)
    dataset["date"] = pd.to_datetime(dataset["date"], errors="coerce").dt.normalize()
    print(f"dataset rows: {len(dataset)}", flush=True)

    ml_topk_daily = None
    if not args.no_ml_topk:
        print(f"loading ml top-k daily: {args.topk_daily_path}", flush=True)
        ml_topk_daily = load_ml_topk_daily(args.topk_daily_path)

    comparison = build_benchmark_comparison(
        dataset,
        ml_topk_daily=ml_topk_daily,
        top_k=top_k,
        bucket_factors=bucket_factors,
        bucket_numbers=bucket_numbers,
        label_col=args.label_col,
        train_split=args.train_split,
        test_split=args.test_split,
        random_state=args.random_state,
        regime_col=args.regime_col,
    )

    paths = write_benchmark_comparison(
        comparison,
        output_dir=args.output_dir,
        output_name=args.output_name,
    )

    print("saved benchmark files:")
    for name, path in paths.items():
        print(f"{name}: {path}")

    print("\nBenchmark summary:")
    print(comparison["benchmark_summary"].to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
