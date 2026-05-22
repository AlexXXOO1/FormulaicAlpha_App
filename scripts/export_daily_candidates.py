from __future__ import annotations

import argparse
from pathlib import Path

from core.paths import DATA_ROOT, DAILY_CANDIDATES, FORMULAIC_ALPHAS_BY_SYMBOL
from research.portfolio_analysis.daily_candidate_exporter import export_daily_candidates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="latest")
    parser.add_argument("--factor-dir", default=str(FORMULAIC_ALPHAS_BY_SYMBOL))
    parser.add_argument(
        "--custom-market-regime",
        default=str(DATA_ROOT / "market_regime" / "custom_market_regime.csv"),
    )
    parser.add_argument("--output-dir", default=str(DAILY_CANDIDATES))
    parser.add_argument("--bucket-count", type=int, default=10)

    args = parser.parse_args()

    result = export_daily_candidates(
        factor_dir=Path(args.factor_dir),
        regime_path=Path(args.custom_market_regime),
        output_dir=Path(args.output_dir),
        target_date=args.date,
        bucket_count=args.bucket_count,
    )

    print("target_date:", result.target_date.date())
    print("market_regime:", result.market_regime)
    print("wide_count:", result.wide_count)
    print("precision_count:", result.precision_count)
    print("wide_path:", result.wide_path)
    print("precision_path:", result.precision_path)
    print("summary_path:", result.summary_path)


if __name__ == "__main__":
    main()
