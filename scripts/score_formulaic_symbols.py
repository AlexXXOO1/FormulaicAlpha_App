
from __future__ import annotations

import argparse
from pathlib import Path

from research.candidate_pool.formulaic_symbol_score import (
    parse_factor_list,
    parse_symbol_list,
    score_formulaic_symbols,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score specific symbols against validated Formulaic Alpha bucket rules."
    )

    parser.add_argument("--symbols", required=True, help="Comma-separated symbols. Example: SH#600051,SZ#002057")
    parser.add_argument("--target-date", required=True)

    parser.add_argument("--market-dir", required=True)
    parser.add_argument("--factor-dir", required=True)
    parser.add_argument("--research-root", required=True)
    parser.add_argument("--manual-summary-path", required=True)
    parser.add_argument("--regime-path", required=True)
    parser.add_argument("--output-path", required=True)

    parser.add_argument("--factors", default="alpha_002,alpha_006")
    parser.add_argument("--combination-name", default="alpha_002_and_alpha_006")
    parser.add_argument("--rule-state", default="PAPER_WATCH")
    parser.add_argument("--min-amount", type=float, default=0.0)
    parser.add_argument("--include-holdout", action="store_true")
    parser.add_argument("--ignore-regime-gate", action="store_true")

    args = parser.parse_args()

    out = score_formulaic_symbols(
        symbols=parse_symbol_list(args.symbols),
        target_date=args.target_date,
        market_dir=Path(args.market_dir),
        factor_dir=Path(args.factor_dir),
        research_root=Path(args.research_root),
        manual_summary_path=Path(args.manual_summary_path),
        regime_path=Path(args.regime_path),
        output_path=Path(args.output_path),
        factor_names=parse_factor_list(args.factors),
        include_holdout=args.include_holdout,
        ignore_regime_gate=args.ignore_regime_gate,
        min_amount=args.min_amount,
        combination_name=args.combination_name,
        rule_state=args.rule_state,
    )

    print("\n========== FORMULAIC SYMBOL SCORE ==========")
    show_cols = [
        "symbol",
        "date",
        "market_regime",
        "rule_state",
        "allow_new_entry",
        "combo_name",
        "combo_pass",
        "score",
        "status",
        "reject_reason",
        "close",
        "amount",
        "alpha_002",
        "alpha_002_train_defined_bucket",
        "alpha_002_pass",
        "alpha_006",
        "alpha_006_train_defined_bucket",
        "alpha_006_pass",
    ]
    show_cols = [c for c in show_cols if c in out.columns]
    print(out[show_cols].to_string(index=False))


if __name__ == "__main__":
    main()
