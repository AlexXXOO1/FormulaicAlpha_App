from __future__ import annotations

import argparse
from pathlib import Path

from alpha_engine.pipeline.factor_builder import build_formulaic_alpha
from core.paths import DAILY_BARS_BY_SYMBOL, FORMULAIC_ALPHAS_BY_SYMBOL


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", required=True, help="Registered alpha name, e.g. alpha_001")
    parser.add_argument("--input-dir", default=str(DAILY_BARS_BY_SYMBOL))
    parser.add_argument("--output-dir", default=str(FORMULAIC_ALPHAS_BY_SYMBOL))
    args = parser.parse_args()

    build_formulaic_alpha(
        alpha_name=args.alpha,
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        show_progress=True,
    )


if __name__ == "__main__":
    main()
