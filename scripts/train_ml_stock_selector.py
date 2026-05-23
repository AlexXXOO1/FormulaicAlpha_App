from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import pandas as pd

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from ml_engine.dataset import DEFAULT_ALPHA_COLS  # noqa: E402
from ml_engine.evaluator import evaluate_topk_predictions, parse_top_k, write_topk_evaluation  # noqa: E402
from ml_engine.model import build_prediction_frame, save_model_bundle, train_model_from_dataset  # noqa: E402


DEFAULT_DATASET_PATH = Path(r"C:\Users\zyf37\Desktop\FormulaicAlpha_Data\research_output\ml_dataset\ml_dataset_v0.parquet")
DEFAULT_OUTPUT_DIR = Path(r"C:\Users\zyf37\Desktop\FormulaicAlpha_Data\research_output\ml_dataset\model")


class StepProgress:
    def __init__(self, total: int, *, width: int = 30) -> None:
        if total <= 0:
            raise ValueError("total must be positive")
        self.total = total
        self.width = width
        self.current = 0

    def advance(self, message: str) -> None:
        self.current = min(self.current + 1, self.total)
        ratio = self.current / self.total
        filled = int(self.width * ratio)
        bar = "#" * filled + "-" * (self.width - filled)
        print(
            f"[progress] [{bar}] {self.current}/{self.total} {ratio * 100:6.2f}% {message}",
            flush=True,
        )


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
    parser = argparse.ArgumentParser(description="Train ML stock selector baseline and evaluate daily top-k selections.")
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-name", type=str, default="ml_model_v0")

    parser.add_argument("--factor-cols", type=str, default=",".join(DEFAULT_ALPHA_COLS))
    parser.add_argument("--label-col", type=str, default="label_return_pct")
    parser.add_argument("--train-split", type=str, default="train")
    parser.add_argument("--predict-split", type=str, default="test")
    parser.add_argument("--top-k", type=str, default="10,20,50")
    parser.add_argument("--regime-col", type=str, default=None)

    parser.add_argument("--max-iter", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-leaf-nodes", type=int, default=31)
    parser.add_argument("--l2-regularization", type=float, default=0.0)
    parser.add_argument("--random-state", type=int, default=42)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    factor_cols = parse_factor_cols(args.factor_cols)
    top_k = parse_top_k(args.top_k)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    progress = StepProgress(total=7)

    df = pd.read_parquet(args.dataset_path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    progress.advance(f"loaded dataset rows={len(df)}")

    model, metadata = train_model_from_dataset(
        df,
        factor_cols=factor_cols,
        label_col=args.label_col,
        train_split=args.train_split,
        max_iter=args.max_iter,
        learning_rate=args.learning_rate,
        max_leaf_nodes=args.max_leaf_nodes,
        l2_regularization=args.l2_regularization,
        random_state=args.random_state,
    )
    progress.advance(f"trained model train_rows={metadata['train_rows']}")

    predictions = build_prediction_frame(
        df,
        model,
        factor_cols=factor_cols,
        prediction_col="prediction",
        label_col=args.label_col,
        predict_split=args.predict_split,
    )
    progress.advance(f"built predictions rows={len(predictions)}")

    evaluation = evaluate_topk_predictions(
        predictions,
        top_k=top_k,
        prediction_col="prediction",
        label_col=args.label_col,
        split=args.predict_split,
        regime_col=args.regime_col,
    )
    progress.advance("evaluated top-k portfolios")

    model_path = args.output_dir / f"{args.output_name}_model.pkl"
    metadata_path = args.output_dir / f"{args.output_name}_metadata.json"
    predictions_path = args.output_dir / f"{args.output_name}_predictions.parquet"

    save_model_bundle(model=model, metadata=metadata, path=model_path)
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    predictions.to_parquet(predictions_path, index=False)
    progress.advance("saved model metadata and predictions")

    eval_paths = write_topk_evaluation(evaluation, output_dir=args.output_dir, output_name=args.output_name)
    progress.advance("saved evaluation csv files")
    progress.advance("done")

    print(f"dataset: {args.dataset_path}")
    print(f"rows: {len(df)}")
    print(f"factor_cols: {','.join(factor_cols)}")
    print(f"train_rows: {metadata['train_rows']}")
    print(f"prediction_rows: {len(predictions)}")
    print(f"saved model: {model_path}")
    print(f"saved metadata: {metadata_path}")
    print(f"saved predictions: {predictions_path}")
    print("saved evaluation files:")
    for name, path in eval_paths.items():
        print(f"{name}: {path}")

    print("\nTop-k summary:")
    print(evaluation["topk_summary"].to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
