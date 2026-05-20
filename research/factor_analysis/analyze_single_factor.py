from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()

    if suffix == ".parquet":
        return pd.read_parquet(path)

    if suffix == ".csv":
        return pd.read_csv(path)

    raise ValueError(f"Unsupported file type: {suffix}")


def add_forward_returns(
    market: pd.DataFrame,
    *,
    symbol_col: str,
    date_col: str,
    close_col: str,
    horizons: list[int],
) -> pd.DataFrame:
    df = market[[symbol_col, date_col, close_col]].copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce").dt.normalize()
    df[close_col] = pd.to_numeric(df[close_col], errors="coerce")

    df = df.dropna(subset=[symbol_col, date_col, close_col])
    df = df.sort_values([symbol_col, date_col], kind="mergesort").reset_index(drop=True)

    g = df.groupby(symbol_col, sort=False)

    for h in horizons:
        future_close = g[close_col].shift(-h)
        df[f"fwd_return_pct_T{h}"] = (future_close / df[close_col] - 1.0) * 100.0

    return df


def assign_global_quantile_buckets(
    df: pd.DataFrame,
    *,
    factor_col: str,
    bucket_count: int,
) -> pd.Series:
    valid = df[factor_col].dropna()

    if valid.empty or valid.nunique() < 2:
        return pd.Series(np.nan, index=df.index)

    q = min(bucket_count, valid.nunique(), len(valid))

    ranked = df[factor_col].rank(method="first")
    buckets = pd.qcut(
        ranked,
        q=q,
        labels=False,
        duplicates="drop",
    )

    return buckets.astype("float64") + 1.0


def analyze_target(
    df: pd.DataFrame,
    *,
    factor_col: str,
    bucket_col: str,
    target_col: str,
    date_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = df[[date_col, factor_col, bucket_col, target_col]].copy()
    work = work.dropna(subset=[factor_col, bucket_col, target_col])

    if work.empty:
        return pd.DataFrame(), pd.DataFrame()

    bucket_summary = (
        work.groupby(bucket_col, dropna=True)
        .agg(
            sample_count=(target_col, "size"),
            mean_return_pct=(target_col, "mean"),
            median_return_pct=(target_col, "median"),
            up_ratio=(target_col, lambda x: float((x > 0).mean())),
            min_factor=(factor_col, "min"),
            max_factor=(factor_col, "max"),
        )
        .reset_index()
        .sort_values(bucket_col)
    )

    ic_rows = []

    for date, part in work.groupby(date_col, sort=True):
        if part[factor_col].nunique() < 2 or part[target_col].nunique() < 2:
            continue

        ic_rows.append({
            "date": date,
            "spearman_ic": part[factor_col].corr(part[target_col], method="spearman"),
            "pearson_ic": part[factor_col].corr(part[target_col], method="pearson"),
            "sample_count": len(part),
        })

    daily_ic = pd.DataFrame(ic_rows)

    return bucket_summary, daily_ic


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--market", required=True)
    parser.add_argument("--factor", required=True)
    parser.add_argument("--factor-col", default="alpha_001")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--horizons", default="1,2,3,5,10")
    parser.add_argument("--bucket-count", type=int, default=5)
    parser.add_argument("--symbol-col", default="symbol")
    parser.add_argument("--date-col", default="date")
    parser.add_argument("--close-col", default="close")
    args = parser.parse_args()

    horizons = [int(x.strip()) for x in args.horizons.split(",") if x.strip()]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("loading market:", args.market)
    market = read_table(Path(args.market))

    print("loading factor:", args.factor)
    factor = read_table(Path(args.factor))

    market_ret = add_forward_returns(
        market,
        symbol_col=args.symbol_col,
        date_col=args.date_col,
        close_col=args.close_col,
        horizons=horizons,
    )

    factor = factor[[args.symbol_col, args.date_col, args.factor_col]].copy()
    factor[args.date_col] = pd.to_datetime(factor[args.date_col], errors="coerce").dt.normalize()
    factor[args.factor_col] = pd.to_numeric(factor[args.factor_col], errors="coerce")

    merged = market_ret.merge(
        factor,
        on=[args.symbol_col, args.date_col],
        how="inner",
        validate="one_to_one",
    )

    bucket_col = f"{args.factor_col}_bucket"
    merged[bucket_col] = assign_global_quantile_buckets(
        merged,
        factor_col=args.factor_col,
        bucket_count=args.bucket_count,
    )

    merged.to_parquet(output_dir / "analysis_input.parquet", index=False)

    all_summary = []
    all_ic = []

    for h in horizons:
        target_col = f"fwd_return_pct_T{h}"

        summary, daily_ic = analyze_target(
            merged,
            factor_col=args.factor_col,
            bucket_col=bucket_col,
            target_col=target_col,
            date_col=args.date_col,
        )

        if not summary.empty:
            summary.insert(0, "target", target_col)
            all_summary.append(summary)

        if not daily_ic.empty:
            daily_ic.insert(0, "target", target_col)
            all_ic.append(daily_ic)

    summary_out = pd.concat(all_summary, ignore_index=True) if all_summary else pd.DataFrame()
    ic_out = pd.concat(all_ic, ignore_index=True) if all_ic else pd.DataFrame()

    summary_out.to_csv(output_dir / "bucket_summary.csv", index=False, encoding="utf-8-sig")
    ic_out.to_csv(output_dir / "daily_ic.csv", index=False, encoding="utf-8-sig")

    print("saved:", output_dir)
    print("rows:", len(merged))
    print("factor_non_null:", merged[args.factor_col].notna().sum())
    print("bucket_summary_rows:", len(summary_out))
    print("daily_ic_rows:", len(ic_out))

    if not summary_out.empty:
        print()
        print(summary_out.to_string(index=False))


if __name__ == "__main__":
    main()
