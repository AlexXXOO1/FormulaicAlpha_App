from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from research.factor_analysis.factor_ml_applicability import (  # noqa: E402
    build_ml_factor_applicability_table,
    write_ml_factor_applicability_outputs,
)


DEFAULT_OUTPUT_DIR = Path(r"C:\Users\zyf37\Desktop\FormulaicAlpha_Data\research_output\ml_dataset\factor_applicability")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export alpha_001-alpha_010 ML factor applicability summary.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-name", type=str, default="alpha_001_010_ml_factor_applicability")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    paths = write_ml_factor_applicability_outputs(
        output_dir=args.output_dir,
        output_name=args.output_name,
    )
    df = build_ml_factor_applicability_table()

    print("saved ML factor applicability files:")
    for name, path in paths.items():
        print(f"{name}: {path}")

    print("\nML factor applicability:")
    print(df.to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
