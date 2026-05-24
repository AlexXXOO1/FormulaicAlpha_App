
from __future__ import annotations

import argparse
from pathlib import Path

from research.candidate_pool.full_market_formulaic_combination_test import (
    build_full_market_formulaic_combination_test,
    parse_combinations,
    parse_factor_names,
    parse_horizons,
    parse_regimes,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run full-market Formulaic Alpha multi-factor combination test."
    )
    parser.add_argument("--market-dir", required=True)
    parser.add_argument("--factor-dir", required=True)
    parser.add_argument("--research-root", required=True)
    parser.add_argument("--manual-summary-path", required=True)
    parser.add_argument("--regime-path", required=True)
    parser.add_argument("--allowed-regimes", default=None)
    parser.add_argument("--excluded-regimes", default=None)
    parser.add_argument("--output-dir", required=True)

    parser.add_argument("--factors", default="alpha_005,alpha_002,alpha_006")
    parser.add_argument(
        "--combinations",
        default=None,
        help="Example: alpha_005;alpha_005+alpha_002;alpha_005+alpha_006",
    )
    parser.add_argument("--horizons", default="3,4")
    parser.add_argument("--start-date", default="2021-01-01")
    parser.add_argument("--end-date", default="2026-12-31")
    parser.add_argument("--min-amount", type=float, default=0.0)
    parser.add_argument("--include-holdout", action="store_true")
    parser.add_argument("--include-one-word-limit-up", action="store_true")
    parser.add_argument("--output-prefix", default="full_market_formulaic_combination_test")
    parser.add_argument("--recent-trading-days", type=int, default=120)
    parser.add_argument("--min-candidate-count", type=int, default=10)
    parser.add_argument("--max-candidate-count", type=int, default=80)

    args = parser.parse_args()

    build_full_market_formulaic_combination_test(
        market_dir=Path(args.market_dir),
        factor_dir=Path(args.factor_dir),
        research_root=Path(args.research_root),
        manual_summary_path=Path(args.manual_summary_path),
        regime_path=Path(args.regime_path),
        output_dir=Path(args.output_dir),
        factor_names=parse_factor_names(args.factors),
        combinations=parse_combinations(args.combinations),
        horizons=parse_horizons(args.horizons),
        start_date=args.start_date,
        end_date=args.end_date,
        min_amount=args.min_amount,
        include_holdout=args.include_holdout,
        exclude_one_word_limit_up=not args.include_one_word_limit_up,
        output_prefix=args.output_prefix,
        recent_trading_days=args.recent_trading_days,
        min_candidate_count=args.min_candidate_count,
        max_candidate_count=args.max_candidate_count,
        allowed_regimes=parse_regimes(args.allowed_regimes),
        excluded_regimes=parse_regimes(args.excluded_regimes),
    )


if __name__ == "__main__":
    main()
