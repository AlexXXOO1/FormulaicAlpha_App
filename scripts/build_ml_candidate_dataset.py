from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import pandas as pd

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from ml_engine.candidate_universe import (  # noqa: E402
    DEFAULT_BUCKET_COUNT,
    DEFAULT_CANDIDATE_BUCKET_FACTOR,
    DEFAULT_CANDIDATE_BUCKETS,
    DEFAULT_CANDIDATE_FEATURE_COLS,
    build_candidate_universe_dataset,
    daily_candidate_count,
    parse_bucket_numbers,
    parse_feature_cols,
    summarize_candidate_universe_dataset,
    write_candidate_universe_outputs,
)


DEFAULT_DATASET_PATH = Path(r"C:\Users\zyf37\Desktop\FormulaicAlpha_Data\research_output\ml_dataset\ml_dataset_v0.parquet")
DEFAULT_OUTPUT_DIR = Path(r"C:\Users\zyf37\Desktop\FormulaicAlpha_Data\research_output\ml_dataset\candidate")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build ML candidate-universe dataset using train-defined factor buckets."
    )
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-name", type=str, default="ml_candidate_alpha005_bucket_4_7")

    parser.add_argument("--bucket-factor", type=str, default=DEFAULT_CANDIDATE_BUCKET_FACTOR)
    parser.add_argument(
        "--candidate-buckets",
        type=str,
        default=",".join(str(x) for x in DEFAULT_CANDIDATE_BUCKETS),
    )
    parser.add_argument("--bucket-count", type=int, default=DEFAULT_BUCKET_COUNT)

    parser.add_argument(
        "--feature-cols",
        type=str,
        default=",".join(DEFAULT_CANDIDATE_FEATURE_COLS),
        help="Model features inside candidate universe. Default excludes alpha_005 because it is used as the universe filter.",
    )
    parser.add_argument("--label-col", type=str, default="label_return_pct")
    parser.add_argument("--train-split", type=str, default="train")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    candidate_buckets = parse_bucket_numbers(args.candidate_buckets)
    feature_cols = parse_feature_cols(args.feature_cols)

    print(f"loading dataset: {args.dataset_path}", flush=True)
    df = pd.read_parquet(args.dataset_path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    print(f"input rows: {len(df)}", flush=True)

    candidate_df, bucket_edges = build_candidate_universe_dataset(
        df,
        bucket_factor=args.bucket_factor,
        candidate_buckets=candidate_buckets,
        bucket_count=args.bucket_count,
        feature_cols=feature_cols,
        label_col=args.label_col,
        train_split=args.train_split,
    )

    paths = write_candidate_universe_outputs(
        candidate_df=candidate_df,
        bucket_edges=bucket_edges,
        output_dir=args.output_dir,
        output_name=args.output_name,
        feature_cols=feature_cols,
    )

    summary = summarize_candidate_universe_dataset(
        candidate_df,
        feature_cols=feature_cols,
        label_col=args.label_col,
    )
    daily_count = daily_candidate_count(candidate_df)

    print(f"candidate rows: {len(candidate_df)}")
    print(f"bucket_factor: {args.bucket_factor}")
    print(f"candidate_buckets: {','.join(str(x) for x in candidate_buckets)}")
    print(f"feature_cols: {','.join(feature_cols)}")
    print("saved candidate files:")
    for name, path in paths.items():
        print(f"{name}: {path}")

    print("\nCandidate summary:")
    print(summary.to_string(index=False))

    print("\nDaily candidate count summary:")
    print(daily_count.groupby("split")["symbols"].describe().to_string())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
