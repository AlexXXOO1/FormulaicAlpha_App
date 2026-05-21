from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def print_progress(label: str, current: int, total: int, width: int = 30) -> None:
    if total <= 0:
        return

    ratio = min(max(current / total, 0.0), 1.0)
    done = int(width * ratio)
    bar = "#" * done + "." * (width - done)

    end = "\n" if current >= total else "\r"
    print(f"{label}: [{bar}] {current}/{total} {ratio * 100:6.2f}%", end=end, flush=True)


def read_table(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    suffix = path.suffix.lower()

    if suffix == ".parquet":
        return pd.read_parquet(path, columns=columns)

    if suffix == ".csv":
        if columns is None:
            return pd.read_csv(path)
        return pd.read_csv(path, usecols=columns)

    raise ValueError(f"Unsupported file type: {suffix}")


def list_parquet_files(path: Path) -> list[Path]:
    files = sorted(Path(path).glob("*.parquet"))

    if not files:
        raise FileNotFoundError(f"No parquet files found under: {path}")

    return files


def read_parquet_path(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    path = Path(path)

    if path.is_file():
        return read_table(path, columns=columns)

    if path.is_dir():
        files = list_parquet_files(path)
        parts = []

        for i, file in enumerate(files, start=1):
            print_progress("loading parquet files", i, len(files))
            parts.append(pd.read_parquet(file, columns=columns))

        return pd.concat(parts, ignore_index=True)

    raise FileNotFoundError(path)


def load_market_data(
    path: Path,
    close_col: str | None = None,
    require_open: bool = False,
) -> pd.DataFrame:
    if close_col is None:
        close_col = "adjusted_close"

    columns = ["symbol", "date", close_col]

    if require_open:
        columns.append("adjusted_open")

    df = read_parquet_path(path, columns=columns)

    required = ["symbol", "date", close_col]
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"Market data missing columns: {missing}")

    use_cols = required.copy()
    if "adjusted_open" in df.columns:
        use_cols.append("adjusted_open")

    out = df[use_cols].copy()
    out = out.rename(columns={close_col: "close", "adjusted_open": "open"})

    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    if "open" in out.columns:
        out["open"] = pd.to_numeric(out["open"], errors="coerce")

    out = out.dropna(subset=["symbol", "date", "close"])
    out = out.sort_values(["symbol", "date"], kind="mergesort").reset_index(drop=True)

    return out


def load_factor_data(path: Path, factor_col: str) -> pd.DataFrame:
    df = read_parquet_path(path, columns=["symbol", "date", factor_col])

    required = ["symbol", "date", factor_col]
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"Factor data missing columns: {missing}")

    out = df[required].copy()

    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    out[factor_col] = pd.to_numeric(out[factor_col], errors="coerce")

    out = out.dropna(subset=["symbol", "date"])
    out = out.sort_values(["symbol", "date"], kind="mergesort").reset_index(drop=True)

    return out


def add_forward_returns(
    market: pd.DataFrame,
    *,
    symbol_col: str,
    date_col: str,
    close_col: str,
    horizons: list[int],
    return_mode: str = "t0_close_to_tn_close",
) -> pd.DataFrame:
    base_cols = [symbol_col, date_col, close_col]
    if return_mode == "t1_open_to_tn_close":
        if "open" not in market.columns:
            raise ValueError("return_mode=t1_open_to_tn_close requires open column.")
        base_cols.append("open")

    df = market[base_cols].copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce").dt.normalize()
    df[close_col] = pd.to_numeric(df[close_col], errors="coerce")

    if "open" in df.columns:
        df["open"] = pd.to_numeric(df["open"], errors="coerce")

    df = df.dropna(subset=[symbol_col, date_col, close_col])
    df = df.sort_values([symbol_col, date_col], kind="mergesort").reset_index(drop=True)

    g = df.groupby(symbol_col, sort=False)

    if return_mode == "t0_close_to_tn_close":
        for h in horizons:
            future_close = g[close_col].shift(-h)
            df[f"fwd_return_pct_T{h}"] = (future_close / df[close_col] - 1.0) * 100.0

    elif return_mode == "t1_open_to_tn_close":
        t1_open = g["open"].shift(-1)

        for h in horizons:
            if h <= 1:
                raise ValueError("T+1 open entry requires horizon >= 2 because A-share positions bought at T+1 open cannot be sold on T+1.")

            exit_close = g[close_col].shift(-h)
            df[f"fwd_return_pct_T{h}"] = (exit_close / t1_open - 1.0) * 100.0

    else:
        raise ValueError(f"Unsupported return_mode: {return_mode}")

    return df


def build_analysis_frame_from_symbol_dirs(
    *,
    market_dir: Path,
    factor_dir: Path,
    factor_col: str,
    horizons: list[int],
    close_col: str | None = None,
    return_mode: str = "t0_close_to_tn_close",
) -> tuple[pd.DataFrame, int, int]:
    market_files = {p.stem: p for p in list_parquet_files(market_dir)}
    factor_files = {p.stem: p for p in list_parquet_files(factor_dir)}

    missing_factor = sorted(set(market_files) - set(factor_files))
    if missing_factor:
        raise FileNotFoundError(f"Missing factor parquet files for symbols: {missing_factor[:20]}")

    extra_factor = sorted(set(factor_files) - set(market_files))
    if extra_factor:
        print(f"[WARNING] Extra factor parquet files ignored: {extra_factor[:20]}")

    symbols = sorted(market_files)
    parts = []
    market_rows = 0
    factor_rows = 0

    for i, symbol in enumerate(symbols, start=1):
        print_progress("building analysis frame by symbol", i, len(symbols))

        market = load_market_data(
            market_files[symbol],
            close_col=close_col,
            require_open=return_mode == "t1_open_to_tn_close",
        )
        factor = load_factor_data(factor_files[symbol], factor_col=factor_col)

        market_rows += len(market)
        factor_rows += len(factor)

        market_fwd = add_forward_returns(
            market,
            symbol_col="symbol",
            date_col="date",
            close_col="close",
            horizons=horizons,
            return_mode=return_mode,
        )

        part = market_fwd.merge(
            factor,
            on=["symbol", "date"],
            how="inner",
            validate="one_to_one",
        )

        parts.append(part)

    if not parts:
        raise ValueError("No analysis rows generated.")

    work = pd.concat(parts, ignore_index=True)

    work["date"] = pd.to_datetime(work["date"], errors="coerce").dt.normalize()
    work["symbol"] = work["symbol"].astype("category")

    work = work.sort_values(["date", "symbol"], kind="mergesort").reset_index(drop=True)

    return work, market_rows, factor_rows


def build_analysis_frame(
    *,
    market_path: Path,
    factor_path: Path,
    factor_col: str,
    horizons: list[int],
    close_col: str | None = None,
    return_mode: str = "t0_close_to_tn_close",
) -> tuple[pd.DataFrame, int, int]:
    market_path = Path(market_path)
    factor_path = Path(factor_path)

    if market_path.is_dir() and factor_path.is_dir():
        return build_analysis_frame_from_symbol_dirs(
            market_dir=market_path,
            factor_dir=factor_path,
            factor_col=factor_col,
            horizons=horizons,
            close_col=close_col,
            return_mode=return_mode,
        )

    market = load_market_data(
        market_path,
        close_col=close_col,
        require_open=return_mode == "t1_open_to_tn_close",
    )
    factor = load_factor_data(factor_path, factor_col=factor_col)

    market_fwd = add_forward_returns(
        market,
        symbol_col="symbol",
        date_col="date",
        close_col="close",
        horizons=horizons,
        return_mode=return_mode,
    )

    work = market_fwd.merge(
        factor,
        on=["symbol", "date"],
        how="inner",
        validate="one_to_one",
    )

    work["symbol"] = work["symbol"].astype("category")

    return work, len(market), len(factor)


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

    if q < 2:
        return pd.Series(np.nan, index=df.index)

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
    work = df[["symbol", date_col, factor_col, bucket_col, target_col]].copy()
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

    bucket_summary.insert(0, "target", target_col)

    ic_rows = []

    for date, part in work.groupby(date_col, sort=True):
        if part[factor_col].nunique() < 2 or part[target_col].nunique() < 2:
            continue

        ic_rows.append({
            "date": date,
            "target": target_col,
            "spearman_ic": part[factor_col].corr(part[target_col], method="spearman"),
            "pearson_ic": part[factor_col].corr(part[target_col], method="pearson"),
            "sample_count": len(part),
        })

    daily_ic = pd.DataFrame(ic_rows)

    return bucket_summary, daily_ic



def analyze_target_by_year(
    df: pd.DataFrame,
    *,
    factor_col: str,
    bucket_col: str,
    target_col: str,
    date_col: str,
) -> pd.DataFrame:
    work = df[[date_col, factor_col, bucket_col, target_col]].copy()
    work = work.dropna(subset=[factor_col, bucket_col, target_col])

    if work.empty:
        return pd.DataFrame()

    work["year"] = pd.to_datetime(work[date_col], errors="coerce").dt.year
    work = work.dropna(subset=["year"])
    work["year"] = work["year"].astype(int)

    out = (
        work.groupby(["year", bucket_col], dropna=True)
        .agg(
            sample_count=(target_col, "size"),
            mean_return_pct=(target_col, "mean"),
            median_return_pct=(target_col, "median"),
            up_ratio=(target_col, lambda x: float((x > 0).mean())),
            min_factor=(factor_col, "min"),
            max_factor=(factor_col, "max"),
        )
        .reset_index()
        .sort_values(["year", bucket_col])
    )

    out.insert(0, "target", target_col)

    return out


def run_single_factor_analysis(
    *,
    market_path: Path,
    factor_path: Path,
    factor_col: str,
    output_dir: Path,
    horizons: list[int],
    bucket_count: int,
    close_col: str | None = None,
    write_member_detail: bool = True,
    return_mode: str = "t0_close_to_tn_close",
) -> dict[str, object]:
    print("step 1/5: loading and merging market/factor data")
    work, market_rows, factor_rows = build_analysis_frame(
        market_path=market_path,
        factor_path=factor_path,
        factor_col=factor_col,
        horizons=horizons,
        close_col=close_col,
        return_mode=return_mode,
    )

    print("step 2/5: assigning global quantile buckets")
    work[f"{factor_col}_bucket"] = assign_global_quantile_buckets(
        work,
        factor_col=factor_col,
        bucket_count=bucket_count,
    )

    print("step 3/5: analyzing horizons")
    bucket_parts = []
    yearly_bucket_parts = []
    ic_parts = []

    for i, h in enumerate(horizons, start=1):
        print_progress("analyzing horizons", i, len(horizons))

        target_col = f"fwd_return_pct_T{h}"

        bucket_summary, daily_ic = analyze_target(
            work,
            factor_col=factor_col,
            bucket_col=f"{factor_col}_bucket",
            target_col=target_col,
            date_col="date",
        )

        yearly_bucket_summary = analyze_target_by_year(
            work,
            factor_col=factor_col,
            bucket_col=f"{factor_col}_bucket",
            target_col=target_col,
            date_col="date",
        )

        if not bucket_summary.empty:
            bucket_parts.append(bucket_summary)

        if not yearly_bucket_summary.empty:
            yearly_bucket_parts.append(yearly_bucket_summary)

        if not daily_ic.empty:
            ic_parts.append(daily_ic)

    output_dir.mkdir(parents=True, exist_ok=True)

    bucket_all = pd.concat(bucket_parts, ignore_index=True) if bucket_parts else pd.DataFrame()
    yearly_bucket_all = (
        pd.concat(yearly_bucket_parts, ignore_index=True)
        if yearly_bucket_parts
        else pd.DataFrame()
    )
    ic_all = pd.concat(ic_parts, ignore_index=True) if ic_parts else pd.DataFrame()

    bucket_path = output_dir / "bucket_summary.csv"
    yearly_bucket_path = output_dir / "yearly_bucket_summary.csv"
    ic_path = output_dir / "daily_ic.csv"
    member_path = output_dir / "factor_member_detail.parquet"

    print("step 4/5: writing summary outputs")
    bucket_all.to_csv(bucket_path, index=False, encoding="utf-8-sig")
    yearly_bucket_all.to_csv(yearly_bucket_path, index=False, encoding="utf-8-sig")
    ic_all.to_csv(ic_path, index=False, encoding="utf-8-sig")

    if write_member_detail:
        print("step 5/5: writing member detail parquet")
        work.to_parquet(member_path, index=False)
        member_path_value = str(member_path)
    else:
        print("step 5/5: skipped member detail parquet")
        member_path_value = None

    return {
        "market_rows": market_rows,
        "factor_rows": factor_rows,
        "merged_rows": len(work),
        "factor_non_null": int(work[factor_col].notna().sum()),
        "bucket_summary_rows": len(bucket_all),
        "yearly_bucket_summary_rows": len(yearly_bucket_all),
        "daily_ic_rows": len(ic_all),
        "bucket_path": str(bucket_path),
        "yearly_bucket_path": str(yearly_bucket_path),
        "daily_ic_path": str(ic_path),
        "member_path": member_path_value,
    }


def parse_horizons(value: str) -> list[int]:
    horizons = [int(x.strip()) for x in value.split(",") if x.strip()]
    horizons = sorted(set(horizons))

    if not horizons:
        raise ValueError("At least one horizon is required.")

    if any(h <= 0 for h in horizons):
        raise ValueError(f"Horizons must be positive integers: {horizons}")

    return horizons


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--market", required=True)
    parser.add_argument("--factor", required=True)
    parser.add_argument("--factor-col", default="alpha_001")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--horizons", default="1,2,3,5,10")
    parser.add_argument("--bucket-count", type=int, default=5)
    parser.add_argument("--close-col", default=None)
    parser.add_argument(
        "--return-mode",
        default="t0_close_to_tn_close",
        choices=["t0_close_to_tn_close", "t1_open_to_tn_close"],
    )
    parser.add_argument(
        "--skip-member-detail",
        action="store_true",
        help="Do not write factor_member_detail.parquet. Use this for faster full-universe runs.",
    )
    args = parser.parse_args()

    result = run_single_factor_analysis(
        market_path=Path(args.market),
        factor_path=Path(args.factor),
        factor_col=args.factor_col,
        output_dir=Path(args.output_dir),
        horizons=parse_horizons(args.horizons),
        bucket_count=args.bucket_count,
        close_col=args.close_col,
        write_member_detail=not args.skip_member_detail,
        return_mode=args.return_mode,
    )

    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
