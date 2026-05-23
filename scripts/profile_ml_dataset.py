from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from ml_engine.dataset import DEFAULT_ALPHA_COLS  # noqa: E402
from ml_engine.profile import load_ml_dataset, write_ml_dataset_profile  # noqa: E402


DEFAULT_DATASET_PATH = Path(r"C:\Users\zyf37\Desktop\FormulaicAlpha_Data\research_output\ml_dataset\ml_dataset_v0.parquet")
DEFAULT_OUTPUT_DIR = Path(r"C:\Users\zyf37\Desktop\FormulaicAlpha_Data\research_output\ml_dataset\profile")


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
    parser = argparse.ArgumentParser(description="Profile ML dataset before model training.")
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-name", type=str, default="ml_dataset_v0_profile")
    parser.add_argument(
        "--factor-cols",
        type=str,
        default=",".join(DEFAULT_ALPHA_COLS),
        help="Comma-separated feature columns. Default uses ML baseline factors.",
    )
    parser.add_argument("--regime-col", type=str, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    factor_cols = parse_factor_cols(args.factor_cols)
    df = load_ml_dataset(args.dataset_path)

    paths = write_ml_dataset_profile(
        df,
        output_dir=args.output_dir,
        output_name=args.output_name,
        factor_cols=factor_cols,
        regime_col=args.regime_col,
    )

    print(f"dataset: {args.dataset_path}")
    print(f"rows: {len(df)}")
    print(f"factor_cols: {','.join(factor_cols)}")
    print("saved profile files:")
    for name, path in paths.items():
        print(f"{name}: {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
