
from __future__ import annotations

import argparse
from pathlib import Path

from research.candidate_pool.full_market_formulaic_candidate_pool import (
    build_full_market_formulaic_candidate_pool,
    parse_combinations,
    parse_factor_list,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build full-market Formulaic Alpha candidate pool from manual override rules."
    )
    parser.add_argument("--market-dir", required=True)
    parser.add_argument("--factor-dir", required=True)
    parser.add_argument("--research-root", required=True)
    parser.add_argument("--manual-summary-path", required=True)
    parser.add_argument("--regime-path", required=True)
    parser.add_argument("--target-date", required=True)
    parser.add_argument("--output-path", required=True)

    parser.add_argument(
        "--factors",
        default=None,
        help="Comma-separated factor list. Example: alpha_005,alpha_002,alpha_006",
    )
    parser.add_argument(
        "--combinations",
        default=None,
        help="Semicolon-separated combinations. Example: alpha_005;alpha_005+alpha_002",
    )
    parser.add_argument("--min-amount", type=float, default=0.0)
    parser.add_argument("--include-holdout", action="store_true")
    parser.add_argument("--include-one-word-limit-up", action="store_true")

    args = parser.parse_args()

    build_full_market_formulaic_candidate_pool(
        market_dir=Path(args.market_dir),
        factor_dir=Path(args.factor_dir),
        research_root=Path(args.research_root),
        manual_summary_path=Path(args.manual_summary_path),
        regime_path=Path(args.regime_path),
        target_date=args.target_date,
        output_path=Path(args.output_path),
        factor_names=parse_factor_list(args.factors),
        combinations=parse_combinations(args.combinations),
        min_amount=args.min_amount,
        include_holdout=args.include_holdout,
        exclude_one_word_limit_up=not args.include_one_word_limit_up,
    )


if __name__ == "__main__":
    main()
