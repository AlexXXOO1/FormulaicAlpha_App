from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from ml_engine.model_diagnostics import (  # noqa: E402
    build_prediction_diagnostics,
    load_optional_csv,
    load_predictions,
    write_prediction_diagnostics,
)


DEFAULT_PREDICTIONS_PATH = Path(r"C:\Users\zyf37\Desktop\FormulaicAlpha_Data\research_output\ml_dataset\candidate_model\ml_candidate_model_v1_predictions.parquet")
DEFAULT_TOPK_SUMMARY_PATH = Path(r"C:\Users\zyf37\Desktop\FormulaicAlpha_Data\research_output\ml_dataset\candidate_model\ml_candidate_model_v1_topk_summary.csv")
DEFAULT_BENCHMARK_SUMMARY_PATH = Path(r"C:\Users\zyf37\Desktop\FormulaicAlpha_Data\research_output\ml_dataset\candidate_model\benchmark\ml_candidate_model_v1_benchmark_summary.csv")
DEFAULT_OUTPUT_DIR = Path(r"C:\Users\zyf37\Desktop\FormulaicAlpha_Data\research_output\ml_dataset\candidate_model\diagnostics")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diagnose whether ML prediction scores have monotonic forward-return edge.")
    parser.add_argument("--predictions-path", type=Path, default=DEFAULT_PREDICTIONS_PATH)
    parser.add_argument("--topk-summary-path", type=Path, default=DEFAULT_TOPK_SUMMARY_PATH)
    parser.add_argument("--benchmark-summary-path", type=Path, default=DEFAULT_BENCHMARK_SUMMARY_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-name", type=str, default="ml_candidate_model_v1")

    parser.add_argument("--prediction-col", type=str, default="prediction")
    parser.add_argument("--label-col", type=str, default="label_return_pct")
    parser.add_argument("--bucket-count", type=int, default=10)
    parser.add_argument("--min-daily-rows", type=int, default=100)
    parser.add_argument("--candidate-benchmark-name", type=str, default="market_all")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    print(f"loading predictions: {args.predictions_path}", flush=True)
    predictions = load_predictions(args.predictions_path)
    print(f"prediction rows: {len(predictions)}", flush=True)

    topk_summary = load_optional_csv(args.topk_summary_path)
    benchmark_summary = load_optional_csv(args.benchmark_summary_path)

    diagnostics = build_prediction_diagnostics(
        predictions,
        topk_summary=topk_summary,
        benchmark_summary=benchmark_summary,
        prediction_col=args.prediction_col,
        label_col=args.label_col,
        bucket_count=args.bucket_count,
        min_daily_rows=args.min_daily_rows,
        candidate_benchmark_name=args.candidate_benchmark_name,
    )

    paths = write_prediction_diagnostics(
        diagnostics,
        output_dir=args.output_dir,
        output_name=args.output_name,
    )

    print("saved diagnostics files:")
    for name, path in paths.items():
        print(f"{name}: {path}")

    print("\nPrediction bucket summary:")
    print(diagnostics["prediction_bucket_summary"].to_string(index=False))

    print("\nPrediction gate:")
    print(diagnostics["prediction_bucket_gate"].to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
