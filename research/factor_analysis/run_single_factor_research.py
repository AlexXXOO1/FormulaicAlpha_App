from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from research.factor_analysis.analyze_single_factor import (
    analyze_target,
    analyze_target_by_year,
    assign_global_quantile_buckets,
    run_single_factor_analysis,
)
from research.factor_analysis.diagnose_single_factor_result import (
    diagnose_single_factor_result,
    infer_bucket_col,
)


GROUP_BINS = [0, 3, 7, 10]
GROUP_LABELS = ["low_1_3", "middle_4_7", "high_8_10"]


def parse_horizons(value: str) -> list[int]:
    horizons = sorted({int(x.strip()) for x in value.split(",") if x.strip()})
    if not horizons:
        raise ValueError("At least one horizon is required.")
    return horizons


def step_file(output_dir: Path, step: int, factor_name: str, desc: str, suffix: str = "csv") -> Path:
    return output_dir / f"step{step}_{factor_name}_{desc}.{suffix}"


def move_output(src: Path, dst: Path) -> None:
    if dst.exists():
        dst.unlink()
    if src.exists():
        shutil.move(str(src), str(dst))



def read_factor_dir(factor_dir: Path, factor_col: str) -> pd.DataFrame:
    files = sorted(factor_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No factor parquet files found: {factor_dir}")

    parts = []

    for i, p in enumerate(files, start=1):
        if i % 500 == 0 or i == len(files):
            print(f"loading factor {i}/{len(files)}")

        try:
            df = pd.read_parquet(p, columns=["symbol", "date", factor_col])
        except Exception:
            df = pd.read_parquet(p, columns=["date", factor_col])
            df.insert(0, "symbol", p.stem)

        if "symbol" not in df.columns:
            df.insert(0, "symbol", p.stem)

        df["symbol"] = df["symbol"].fillna(p.stem).astype(str)
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
        df[factor_col] = pd.to_numeric(df[factor_col], errors="coerce")

        parts.append(df[["symbol", "date", factor_col]])

    return pd.concat(parts, ignore_index=True)


def write_step1_input_validation(
    *,
    market_dir: Path,
    factor_dir: Path,
    factor_name: str,
    output_dir: Path,
) -> None:
    market_files = sorted(market_dir.glob("*.parquet"))
    factor_files = sorted(factor_dir.glob("*.parquet"))

    market_symbols = {p.stem for p in market_files}
    factor_symbols = {p.stem for p in factor_files}
    all_symbols = sorted(market_symbols | factor_symbols)

    alignment = pd.DataFrame({
        "symbol": all_symbols,
        "market_file_exists": [s in market_symbols for s in all_symbols],
        "factor_file_exists": [s in factor_symbols for s in all_symbols],
    })

    validation = pd.DataFrame([
        {"metric": "market_file_count", "value": len(market_files)},
        {"metric": "factor_file_count", "value": len(factor_files)},
        {"metric": "missing_factor_file_count", "value": len(market_symbols - factor_symbols)},
        {"metric": "extra_factor_file_count", "value": len(factor_symbols - market_symbols)},
        {"metric": "symbol_alignment_pass", "value": market_symbols == factor_symbols},
    ])

    validation.to_csv(
        step_file(output_dir, 1, factor_name, "input_validation"),
        index=False,
        encoding="utf-8-sig",
    )
    alignment.to_csv(
        step_file(output_dir, 1, factor_name, "symbol_alignment"),
        index=False,
        encoding="utf-8-sig",
    )


def write_step2_factor_distribution(
    *,
    factor_df: pd.DataFrame,
    factor_col: str,
    factor_name: str,
    output_dir: Path,
) -> None:
    s = factor_df[factor_col]

    distribution = pd.DataFrame([{
        "row_count": len(factor_df),
        "non_null_count": int(s.notna().sum()),
        "null_count": int(s.isna().sum()),
        "unique_count": int(s.nunique(dropna=True)),
        "min": float(s.min()) if s.notna().any() else np.nan,
        "max": float(s.max()) if s.notna().any() else np.nan,
        "mean": float(s.mean()) if s.notna().any() else np.nan,
        "std": float(s.std()) if s.notna().any() else np.nan,
        "p01": float(s.quantile(0.01)) if s.notna().any() else np.nan,
        "p05": float(s.quantile(0.05)) if s.notna().any() else np.nan,
        "p10": float(s.quantile(0.10)) if s.notna().any() else np.nan,
        "p25": float(s.quantile(0.25)) if s.notna().any() else np.nan,
        "p50": float(s.quantile(0.50)) if s.notna().any() else np.nan,
        "p75": float(s.quantile(0.75)) if s.notna().any() else np.nan,
        "p90": float(s.quantile(0.90)) if s.notna().any() else np.nan,
        "p95": float(s.quantile(0.95)) if s.notna().any() else np.nan,
        "p99": float(s.quantile(0.99)) if s.notna().any() else np.nan,
    }])

    daily = (
        factor_df.groupby("date")
        .agg(
            row_count=(factor_col, "size"),
            non_null_count=(factor_col, lambda x: int(x.notna().sum())),
        )
        .reset_index()
        .sort_values("date")
    )
    daily["coverage_ratio"] = daily["non_null_count"] / daily["row_count"]

    distribution.to_csv(
        step_file(output_dir, 2, factor_name, "factor_distribution"),
        index=False,
        encoding="utf-8-sig",
    )
    daily.to_csv(
        step_file(output_dir, 2, factor_name, "daily_coverage"),
        index=False,
        encoding="utf-8-sig",
    )


def run_analysis_and_rename(
    *,
    market_dir: Path,
    factor_dir: Path,
    factor_col: str,
    factor_name: str,
    output_dir: Path,
    step: int,
    prefix: str,
    horizons: list[int],
    bucket_count: int,
    return_mode: str,
    write_member_detail: bool,
) -> dict[str, Path]:
    run_single_factor_analysis(
        market_path=market_dir,
        factor_path=factor_dir,
        factor_col=factor_col,
        output_dir=output_dir,
        horizons=horizons,
        bucket_count=bucket_count,
        return_mode=return_mode,
        write_member_detail=write_member_detail,
    )

    bucket_path = step_file(output_dir, step, factor_name, f"{prefix}_bucket_summary")
    daily_ic_path = step_file(output_dir, step, factor_name, f"{prefix}_daily_ic")
    yearly_bucket_path = step_file(output_dir, step, factor_name, f"{prefix}_yearly_bucket_summary")

    move_output(output_dir / "bucket_summary.csv", bucket_path)
    move_output(output_dir / "daily_ic.csv", daily_ic_path)
    move_output(output_dir / "yearly_bucket_summary.csv", yearly_bucket_path)

    result = {
        "bucket_summary": bucket_path,
        "daily_ic": daily_ic_path,
        "yearly_bucket_summary": yearly_bucket_path,
    }

    if write_member_detail:
        member_path = step_file(output_dir, step, factor_name, f"{prefix}_member_detail", "parquet")
        move_output(output_dir / "factor_member_detail.parquet", member_path)
        result["member_detail"] = member_path

    return result


def write_bucket_group_outputs(
    *,
    bucket_summary_path: Path,
    daily_ic_path: Path,
    yearly_bucket_path: Path,
    factor_name: str,
    output_dir: Path,
) -> None:
    pattern_path = step_file(output_dir, 5, factor_name, "bucket_pattern_summary")
    diagnose_single_factor_result(
        bucket_summary_path=bucket_summary_path,
        daily_ic_path=daily_ic_path,
        output_path=pattern_path,
    )

    yearly = pd.read_csv(yearly_bucket_path)
    bucket_col = infer_bucket_col(yearly)

    yearly["bucket_group"] = pd.cut(
        yearly[bucket_col],
        bins=GROUP_BINS,
        labels=GROUP_LABELS,
    )

    grouped = (
        yearly.dropna(subset=["bucket_group"])
        .groupby(["target", "year", "bucket_group"], observed=True)
        .apply(
            lambda x: pd.Series({
                "sample_count": int(x["sample_count"].sum()),
                "mean_return_pct": float((x["mean_return_pct"] * x["sample_count"]).sum() / x["sample_count"].sum()),
                "up_ratio": float((x["up_ratio"] * x["sample_count"]).sum() / x["sample_count"].sum()),
            }),
            include_groups=False,
        )
        .reset_index()
    )

    grouped.to_csv(
        step_file(output_dir, 5, factor_name, "bucket_group_summary"),
        index=False,
        encoding="utf-8-sig",
    )

    rows = []
    for (target, year), part in grouped.groupby(["target", "year"], sort=True):
        values = dict(zip(part["bucket_group"], part["mean_return_pct"]))
        if not set(GROUP_LABELS).issubset(values):
            continue

        low = float(values["low_1_3"])
        mid = float(values["middle_4_7"])
        high = float(values["high_8_10"])

        rows.append({
            "target": target,
            "year": int(year),
            "middle_minus_low_pct": mid - low,
            "middle_minus_high_pct": mid - high,
            "middle_beats_low": mid > low,
            "middle_beats_high": mid > high,
            "middle_beats_both": mid > low and mid > high,
            "best_group": max({"low_1_3": low, "middle_4_7": mid, "high_8_10": high}, key={"low_1_3": low, "middle_4_7": mid, "high_8_10": high}.get),
        })

    detail = pd.DataFrame(rows)
    win = (
        detail.groupby("target")
        .agg(
            year_count=("year", "count"),
            middle_beats_low_count=("middle_beats_low", "sum"),
            middle_beats_high_count=("middle_beats_high", "sum"),
            middle_beats_both_count=("middle_beats_both", "sum"),
            avg_middle_minus_low_pct=("middle_minus_low_pct", "mean"),
            avg_middle_minus_high_pct=("middle_minus_high_pct", "mean"),
        )
        .reset_index()
    )

    win.to_csv(
        step_file(output_dir, 5, factor_name, "bucket_group_win_summary"),
        index=False,
        encoding="utf-8-sig",
    )


def make_train_edges(member: pd.DataFrame, factor_col: str, bucket_count: int, train_start: int, train_end: int) -> np.ndarray:
    train = member[
        member["year"].between(train_start, train_end)
        & member[factor_col].notna()
    ].copy()

    if train.empty:
        raise ValueError("No train rows available for bucket edge calculation.")

    quantiles = [i / bucket_count for i in range(bucket_count + 1)]
    raw_edges = train[factor_col].quantile(quantiles).to_numpy(copy=True)

    edges = raw_edges.copy()
    edges[0] = -float("inf")
    edges[-1] = float("inf")

    return raw_edges, edges


def assign_train_bucket(series: pd.Series, edges: np.ndarray, bucket_count: int) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")

    bins = (
        pd.Series(edges, dtype="float64")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    if len(bins) < 2:
        return pd.Series(np.nan, index=series.index, dtype="float64")

    bins[0] = -float("inf")
    bins[-1] = float("inf")

    labels = list(range(1, len(bins)))

    return pd.cut(
        values,
        bins=bins,
        labels=labels,
        include_lowest=True,
    ).astype("float64")


def write_step6_train_test(
    *,
    member_path: Path,
    factor_col: str,
    factor_name: str,
    output_dir: Path,
    bucket_count: int,
    train_start: int,
    train_end: int,
    test_start: int,
    test_end: int,
    targets: list[str],
) -> tuple[pd.DataFrame, np.ndarray]:
    member = pd.read_parquet(member_path)
    member["date"] = pd.to_datetime(member["date"]).dt.normalize()
    member["year"] = member["date"].dt.year

    raw_edges, edges = make_train_edges(member, factor_col, bucket_count, train_start, train_end)

    edge_table = pd.DataFrame({
        "bucket": list(range(1, bucket_count + 1)),
        "lower_edge": raw_edges[:-1],
        "upper_edge": raw_edges[1:],
    })
    edge_table.to_csv(
        step_file(output_dir, 6, factor_name, "train_bucket_edges"),
        index=False,
        encoding="utf-8-sig",
    )

    member["train_defined_bucket"] = assign_train_bucket(member[factor_col], edges, bucket_count)
    member["bucket_group"] = pd.cut(
        member["train_defined_bucket"],
        bins=GROUP_BINS,
        labels=GROUP_LABELS,
    )

    rows = []
    datasets = [
        ("train", member[member["year"].between(train_start, train_end)]),
        ("test", member[member["year"].between(test_start, test_end)]),
    ]

    for dataset_name, part in datasets:
        for target in targets:
            work = part.dropna(subset=["bucket_group", target])
            if work.empty:
                continue

            summary = (
                work.groupby("bucket_group", observed=True)
                .agg(
                    sample_count=(target, "size"),
                    mean_return_pct=(target, "mean"),
                    median_return_pct=(target, "median"),
                    up_ratio=(target, lambda x: float((x > 0).mean())),
                )
                .reset_index()
            )
            summary.insert(0, "target", target)
            summary.insert(0, "dataset", dataset_name)
            rows.append(summary)

    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    out.to_csv(
        step_file(output_dir, 6, factor_name, "train_test_bucket_check"),
        index=False,
        encoding="utf-8-sig",
    )

    return member, edges


def write_step7_candidate_counts(
    *,
    member: pd.DataFrame,
    factor_name: str,
    output_dir: Path,
) -> None:
    bucket_col = f"{factor_name}_bucket"
    if bucket_col not in member.columns:
        raise ValueError(f"Missing expected bucket column: {bucket_col}")

    work = member.dropna(subset=[bucket_col]).copy()
    work["all_sample_bucket_group"] = pd.cut(
        work[bucket_col],
        bins=GROUP_BINS,
        labels=GROUP_LABELS,
    )

    daily = (
        work.groupby("date")
        .agg(
            universe_count=("symbol", "size"),
            middle_4_7_count=("all_sample_bucket_group", lambda x: int((x == "middle_4_7").sum())),
            low_1_3_count=("all_sample_bucket_group", lambda x: int((x == "low_1_3").sum())),
            high_8_10_count=("all_sample_bucket_group", lambda x: int((x == "high_8_10").sum())),
        )
        .reset_index()
    )
    daily["middle_4_7_ratio"] = daily["middle_4_7_count"] / daily["universe_count"]

    daily.to_csv(
        step_file(output_dir, 7, factor_name, "daily_candidate_count"),
        index=False,
        encoding="utf-8-sig",
    )

    train_daily = (
        member.dropna(subset=["bucket_group"])
        .groupby("date")
        .agg(
            universe_count=("symbol", "size"),
            middle_4_7_count=("bucket_group", lambda x: int((x == "middle_4_7").sum())),
            low_1_3_count=("bucket_group", lambda x: int((x == "low_1_3").sum())),
            high_8_10_count=("bucket_group", lambda x: int((x == "high_8_10").sum())),
        )
        .reset_index()
    )
    train_daily["middle_4_7_ratio"] = train_daily["middle_4_7_count"] / train_daily["universe_count"]

    train_daily.to_csv(
        step_file(output_dir, 7, factor_name, "train_defined_daily_candidate_count"),
        index=False,
        encoding="utf-8-sig",
    )


def write_step8_custom_regime(
    *,
    member: pd.DataFrame,
    custom_market_regime: Path,
    factor_name: str,
    output_dir: Path,
    targets: list[str],
) -> None:
    regime = pd.read_csv(custom_market_regime)
    regime["date"] = pd.to_datetime(regime["date"]).dt.normalize()

    keep_cols = [
        c for c in ["date", "market_regime", "risk_score", "trend_score", "repair_score"]
        if c in regime.columns
    ]
    regime = regime[keep_cols].drop_duplicates("date", keep="last")

    rows = []

    for target in targets:
        valid = member.dropna(subset=[target, "bucket_group"]).copy()

        universe_daily = (
            valid.groupby("date")
            .agg(
                universe_count=("symbol", "size"),
                universe_return_pct=(target, "mean"),
            )
            .reset_index()
        )

        group_daily = (
            valid.groupby(["date", "bucket_group"], observed=True)
            .agg(
                group_count=("symbol", "size"),
                group_return_pct=(target, "mean"),
            )
            .reset_index()
            .rename(columns={"bucket_group": "alpha001_group"})
        )

        merged = group_daily.merge(universe_daily, on="date", how="left")
        merged = merged.merge(regime, on="date", how="inner")
        merged["target"] = target
        merged["year"] = merged["date"].dt.year
        merged["excess_return_pct"] = merged["group_return_pct"] - merged["universe_return_pct"]

        rows.append(merged)

    detail = pd.concat(rows, ignore_index=True)

    summary = (
        detail.groupby(["target", "market_regime", "alpha001_group"], observed=True)
        .agg(
            trading_days=("date", "nunique"),
            avg_group_count=("group_count", "mean"),
            avg_universe_count=("universe_count", "mean"),
            mean_group_return_pct=("group_return_pct", "mean"),
            mean_universe_return_pct=("universe_return_pct", "mean"),
            mean_excess_return_pct=("excess_return_pct", "mean"),
            median_excess_return_pct=("excess_return_pct", "median"),
            excess_win_ratio=("excess_return_pct", lambda x: float((x > 0).mean())),
        )
        .reset_index()
    )

    middle_yearly = (
        detail[detail["alpha001_group"].eq("middle_4_7")]
        .groupby(["target", "market_regime", "year"], observed=True)
        .agg(
            trading_days=("date", "nunique"),
            avg_group_count=("group_count", "mean"),
            mean_group_return_pct=("group_return_pct", "mean"),
            mean_universe_return_pct=("universe_return_pct", "mean"),
            mean_excess_return_pct=("excess_return_pct", "mean"),
            median_excess_return_pct=("excess_return_pct", "median"),
            excess_win_ratio=("excess_return_pct", lambda x: float((x > 0).mean())),
        )
        .reset_index()
    )

    stability = (
        middle_yearly.groupby(["target", "market_regime"], observed=True)
        .agg(
            year_count=("year", "count"),
            positive_excess_years=("mean_excess_return_pct", lambda x: int((x > 0).sum())),
            avg_excess_pct=("mean_excess_return_pct", "mean"),
            median_excess_pct=("mean_excess_return_pct", "median"),
            avg_win_ratio=("excess_win_ratio", "mean"),
            min_excess_pct=("mean_excess_return_pct", "min"),
            max_excess_pct=("mean_excess_return_pct", "max"),
        )
        .reset_index()
    )

    summary.to_csv(
        step_file(output_dir, 8, factor_name, "custom_regime_group_summary"),
        index=False,
        encoding="utf-8-sig",
    )
    stability.to_csv(
        step_file(output_dir, 8, factor_name, "custom_regime_yearly_stability"),
        index=False,
        encoding="utf-8-sig",
    )
    middle_yearly.to_csv(
        step_file(output_dir, 8, factor_name, "custom_regime_middle_yearly"),
        index=False,
        encoding="utf-8-sig",
    )




def build_shared_analysis_frame(
    *,
    market_dir: Path,
    factor_dir: Path,
    factor_col: str,
    research_horizons: list[int],
    trade_horizons: list[int],
) -> pd.DataFrame:
    market_files = {p.stem: p for p in sorted(market_dir.glob("*.parquet"))}
    factor_files = {p.stem: p for p in sorted(factor_dir.glob("*.parquet"))}

    common_symbols = sorted(set(market_files) & set(factor_files))

    if not common_symbols:
        raise ValueError("No overlapping market/factor parquet symbols.")

    for h in trade_horizons:
        if h <= 1:
            raise ValueError(
                "T+1 open entry requires trade horizon >= 2 because A-share positions "
                "bought at T+1 open cannot be sold on T+1."
            )

    parts = []

    for i, symbol in enumerate(common_symbols, start=1):
        if i % 200 == 0 or i == len(common_symbols):
            pct = i / len(common_symbols) * 100.0
            print(f"building shared analysis frame: {i}/{len(common_symbols)} {pct:6.2f}%")

        market = pd.read_parquet(
            market_files[symbol],
            columns=["date", "adjusted_open", "adjusted_close"],
        )

        factor = pd.read_parquet(
            factor_files[symbol],
            columns=["date", factor_col],
        )

        market["date"] = pd.to_datetime(market["date"], errors="coerce").dt.normalize()
        factor["date"] = pd.to_datetime(factor["date"], errors="coerce").dt.normalize()

        market["adjusted_open"] = pd.to_numeric(market["adjusted_open"], errors="coerce")
        market["adjusted_close"] = pd.to_numeric(market["adjusted_close"], errors="coerce")
        factor[factor_col] = pd.to_numeric(factor[factor_col], errors="coerce")

        market = market.dropna(subset=["date", "adjusted_open", "adjusted_close"])
        factor = factor.dropna(subset=["date"])

        market = market.drop_duplicates(subset=["date"], keep="last")
        factor = factor.drop_duplicates(subset=["date"], keep="last")

        part = market.merge(
            factor,
            on="date",
            how="inner",
            validate="one_to_one",
        )

        part["symbol"] = symbol

        part = part.sort_values("date", kind="mergesort").reset_index(drop=True)

        for h in research_horizons:
            future_close = part["adjusted_close"].shift(-h)
            part[f"research_fwd_return_pct_T{h}"] = (
                future_close / part["adjusted_close"] - 1.0
            ) * 100.0

        t1_open = part["adjusted_open"].shift(-1)

        for h in trade_horizons:
            exit_close = part["adjusted_close"].shift(-h)
            part[f"trade_fwd_return_pct_T{h}"] = (
                exit_close / t1_open - 1.0
            ) * 100.0

        value_cols = (
            ["date", factor_col]
            + [f"research_fwd_return_pct_T{h}" for h in research_horizons]
            + [f"trade_fwd_return_pct_T{h}" for h in trade_horizons]
        )

        selected = part[value_cols].copy()
        selected.insert(0, "symbol", symbol)

        parts.append(selected)

    out = pd.concat(parts, ignore_index=True)

    if "symbol" not in out.columns:
        raise RuntimeError(f"shared analysis frame missing symbol column. columns={out.columns.tolist()}")

    out = out.sort_values(["symbol", "date"], kind="mergesort").reset_index(drop=True)

    return out


def write_mode_outputs_from_shared_frame(
    *,
    shared: pd.DataFrame,
    factor_col: str,
    factor_name: str,
    output_dir: Path,
    step: int,
    prefix: str,
    mode_prefix: str,
    horizons: list[int],
    write_member_detail: bool,
) -> dict[str, Path]:
    bucket_col = f"{factor_col}_bucket"

    bucket_parts = []
    yearly_bucket_parts = []
    ic_parts = []

    for h in horizons:
        source_col = f"{mode_prefix}_fwd_return_pct_T{h}"
        target_col = f"fwd_return_pct_T{h}"

        work = shared[["symbol", "date", factor_col, bucket_col, source_col]].rename(
            columns={source_col: target_col}
        )

        bucket_summary, daily_ic = analyze_target(
            work,
            factor_col=factor_col,
            bucket_col=bucket_col,
            target_col=target_col,
            date_col="date",
        )

        yearly_bucket = analyze_target_by_year(
            work,
            factor_col=factor_col,
            bucket_col=bucket_col,
            target_col=target_col,
            date_col="date",
        )

        if not bucket_summary.empty:
            if "target" not in bucket_summary.columns:
                bucket_summary.insert(0, "target", target_col)
            else:
                bucket_summary["target"] = target_col
            bucket_parts.append(bucket_summary)

        if not daily_ic.empty:
            if "target" not in daily_ic.columns:
                daily_ic.insert(0, "target", target_col)
            else:
                daily_ic["target"] = target_col
            ic_parts.append(daily_ic)

        if not yearly_bucket.empty:
            if "target" not in yearly_bucket.columns:
                yearly_bucket.insert(0, "target", target_col)
            else:
                yearly_bucket["target"] = target_col
            yearly_bucket_parts.append(yearly_bucket)

    bucket_all = pd.concat(bucket_parts, ignore_index=True) if bucket_parts else pd.DataFrame()
    ic_all = pd.concat(ic_parts, ignore_index=True) if ic_parts else pd.DataFrame()
    yearly_bucket_all = (
        pd.concat(yearly_bucket_parts, ignore_index=True)
        if yearly_bucket_parts
        else pd.DataFrame()
    )

    bucket_path = step_file(output_dir, step, factor_name, f"{prefix}_bucket_summary")
    daily_ic_path = step_file(output_dir, step, factor_name, f"{prefix}_daily_ic")
    yearly_bucket_path = step_file(output_dir, step, factor_name, f"{prefix}_yearly_bucket_summary")

    bucket_all.to_csv(bucket_path, index=False, encoding="utf-8-sig")
    ic_all.to_csv(daily_ic_path, index=False, encoding="utf-8-sig")
    yearly_bucket_all.to_csv(yearly_bucket_path, index=False, encoding="utf-8-sig")

    result = {
        "bucket_summary": bucket_path,
        "daily_ic": daily_ic_path,
        "yearly_bucket_summary": yearly_bucket_path,
    }

    if write_member_detail:
        target_cols = [f"{mode_prefix}_fwd_return_pct_T{h}" for h in horizons]
        rename_map = {
            f"{mode_prefix}_fwd_return_pct_T{h}": f"fwd_return_pct_T{h}"
            for h in horizons
        }

        member = shared[["symbol", "date", factor_col, bucket_col] + target_cols].rename(
            columns=rename_map
        )

        member_path = step_file(output_dir, step, factor_name, f"{prefix}_member_detail", "parquet")
        member.to_parquet(member_path, index=False)
        result["member_detail"] = member_path

    return result



def write_step9_train_defined_bucket_regime_check(
    *,
    member: pd.DataFrame,
    custom_market_regime: Path,
    factor_name: str,
    output_dir: Path,
    targets: list[str],
) -> None:
    if "train_defined_bucket" not in member.columns:
        raise ValueError("Step9 requires train_defined_bucket. Run Step6 before Step9.")

    regime = pd.read_csv(custom_market_regime)
    regime["date"] = pd.to_datetime(regime["date"], errors="coerce").dt.normalize()

    if "market_regime" not in regime.columns:
        raise ValueError("custom_market_regime must contain market_regime column.")

    regime = regime[["date", "market_regime"]].dropna(subset=["date", "market_regime"])
    regime = regime.drop_duplicates(subset=["date"], keep="last")

    df = member.merge(regime, on="date", how="left")

    bucket_groups = {
        "bucket_4": [4.0],
        "bucket_4_5": [4.0, 5.0],
        "middle_4_7": [4.0, 5.0, 6.0, 7.0],
    }

    rows = []

    for target in targets:
        valid = df.dropna(subset=[target, "train_defined_bucket", "market_regime"]).copy()

        universe_daily = (
            valid.groupby("date", as_index=False)
            .agg(universe_return_pct=(target, "mean"))
        )

        for bucket_group, buckets in bucket_groups.items():
            group_df = valid[valid["train_defined_bucket"].isin(buckets)].copy()

            if group_df.empty:
                continue

            daily = (
                group_df.groupby(["date", "market_regime"], as_index=False)
                .agg(
                    group_count=("symbol", "size"),
                    group_return_pct=(target, "mean"),
                )
                .merge(universe_daily, on="date", how="left")
            )

            daily["excess_return_pct"] = daily["group_return_pct"] - daily["universe_return_pct"]

            summary = (
                daily.groupby("market_regime", as_index=False)
                .agg(
                    trading_days=("date", "nunique"),
                    avg_group_count=("group_count", "mean"),
                    mean_group_return_pct=("group_return_pct", "mean"),
                    mean_universe_return_pct=("universe_return_pct", "mean"),
                    mean_excess_return_pct=("excess_return_pct", "mean"),
                    median_excess_return_pct=("excess_return_pct", "median"),
                    excess_win_ratio=("excess_return_pct", lambda x: float((x > 0).mean())),
                )
            )

            summary.insert(0, "bucket_group", bucket_group)
            summary.insert(0, "target", target)

            rows.append(summary)

    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    out.to_csv(
        step_file(output_dir, 9, factor_name, "train_defined_bucket_regime_check"),
        index=False,
        encoding="utf-8-sig",
    )






def bucket_rule_from_group(bucket_group: str) -> str:
    mapping = {
        "low_1_3": "train_defined_bucket in [1,2,3]",
        "middle_4_7": "train_defined_bucket in [4,5,6,7]",
        "high_8_10": "train_defined_bucket in [8,9,10]",
        "bucket_4": "train_defined_bucket in [4]",
        "bucket_4_5": "train_defined_bucket in [4,5]",
    }
    return mapping.get(bucket_group, f"bucket_group = {bucket_group}")


def write_step10_factor_conclusion(
    *,
    factor_name: str,
    output_dir: Path,
) -> None:
    step9_path = step_file(output_dir, 9, factor_name, "train_defined_bucket_regime_check")

    if not step9_path.exists():
        raise FileNotFoundError(f"Missing Step9 file: {step9_path}")

    df = pd.read_csv(step9_path)

    required_cols = [
        "target",
        "bucket_group",
        "market_regime",
        "trading_days",
        "avg_group_count",
        "mean_excess_return_pct",
        "median_excess_return_pct",
        "excess_win_ratio",
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Step9 file missing required columns: {missing}")

    available_targets = sorted(df["target"].dropna().unique().tolist())
    focus_targets = [x for x in ["fwd_return_pct_T3", "fwd_return_pct_T4"] if x in available_targets]

    if not focus_targets:
        focus_targets = available_targets

    focus_all = df[df["target"].isin(focus_targets)].copy()

    group_score = (
        focus_all.groupby("bucket_group", observed=True)
        .agg(
            row_count=("mean_excess_return_pct", "size"),
            positive_row_count=("mean_excess_return_pct", lambda x: int((x > 0).sum())),
            avg_excess_pct=("mean_excess_return_pct", "mean"),
            median_excess_pct=("mean_excess_return_pct", "median"),
            avg_win_ratio=("excess_win_ratio", "mean"),
            min_excess_pct=("mean_excess_return_pct", "min"),
            max_excess_pct=("mean_excess_return_pct", "max"),
            avg_group_count=("avg_group_count", "mean"),
        )
        .reset_index()
    )

    group_score["positive_row_ratio"] = group_score["positive_row_count"] / group_score["row_count"]

    group_score = group_score.sort_values(
        ["positive_row_ratio", "avg_excess_pct", "avg_win_ratio"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    if group_score.empty:
        raise ValueError("No bucket group score available for Step10.")

    best_bucket_group = str(group_score.loc[0, "bucket_group"])
    best_group_score = group_score.loc[0].to_dict()

    focus = focus_all[focus_all["bucket_group"].eq(best_bucket_group)].copy()

    regime_check = (
        focus.groupby("market_regime", observed=True)
        .agg(
            target_count=("target", "nunique"),
            positive_target_count=("mean_excess_return_pct", lambda x: int((x > 0).sum())),
            avg_excess_pct=("mean_excess_return_pct", "mean"),
            avg_win_ratio=("excess_win_ratio", "mean"),
            min_excess_pct=("mean_excess_return_pct", "min"),
            max_excess_pct=("mean_excess_return_pct", "max"),
            avg_group_count=("avg_group_count", "mean"),
        )
        .reset_index()
    )

    valid_regimes = regime_check[
        (regime_check["target_count"] > 0)
        & (regime_check["positive_target_count"] == regime_check["target_count"])
    ]["market_regime"].tolist()

    avoid_regimes = regime_check[
        regime_check["positive_target_count"] < regime_check["target_count"]
    ]["market_regime"].tolist()

    target_score = (
        focus.groupby("target", observed=True)
        .agg(
            avg_excess_pct=("mean_excess_return_pct", "mean"),
            avg_win_ratio=("excess_win_ratio", "mean"),
        )
        .reset_index()
        .sort_values(["avg_excess_pct", "avg_win_ratio"], ascending=[False, False])
    )

    main_target = str(target_score.iloc[0]["target"]) if not target_score.empty else ""
    secondary_target = str(target_score.iloc[1]["target"]) if len(target_score) > 1 else ""

    main_horizon = main_target.replace("fwd_return_pct_", "") if main_target else ""
    secondary_horizon = secondary_target.replace("fwd_return_pct_", "") if secondary_target else ""

    avg_excess = float(best_group_score.get("avg_excess_pct", 0.0))
    avg_win_ratio = float(best_group_score.get("avg_win_ratio", 0.0))
    avg_group_count = float(best_group_score.get("avg_group_count", 0.0))

    if not valid_regimes or avg_excess <= 0:
        factor_type = "reject_factor"
        standalone_tradable = False
        is_filter_factor = False
        is_scoring_factor = False
        recommended_usage = "Reject for now. Do not use this factor in strategy construction."
        rejection_reason = "No stable positive excess return after train-defined bucket and regime validation."
        next_action = "Move to the next factor or revisit after more data/features are available."
    elif avoid_regimes:
        factor_type = "regime_aware_filter"
        standalone_tradable = False
        is_filter_factor = True
        is_scoring_factor = False
        recommended_usage = "Use this factor only under valid regimes as a broad filter before combining with stronger precision factors."
        rejection_reason = "The factor effect is regime-dependent and not sufficient as a standalone stock-selection signal."
        next_action = "Keep this factor as a regime-aware filter and combine with additional factors."
    elif avg_group_count > 500:
        factor_type = "broad_filter"
        standalone_tradable = False
        is_filter_factor = True
        is_scoring_factor = False
        recommended_usage = "Use this factor as a broad universe filter before combining with stronger precision factors."
        rejection_reason = "Candidate pool remains too wide for standalone stock selection."
        next_action = "Combine with additional factors to improve precision."
    elif avg_win_ratio >= 0.55:
        factor_type = "candidate_scoring_factor"
        standalone_tradable = False
        is_filter_factor = True
        is_scoring_factor = True
        recommended_usage = "Use this factor as one component in a multi-factor scoring model."
        rejection_reason = "Single-factor validation is not enough to approve standalone trading."
        next_action = "Test this factor inside a multi-factor combination."
    else:
        factor_type = "weak_filter"
        standalone_tradable = False
        is_filter_factor = True
        is_scoring_factor = False
        recommended_usage = "Use cautiously as a weak filter only if it improves a multi-factor combination."
        rejection_reason = "Positive excess exists but the win ratio is not strong enough for standalone use."
        next_action = "Validate contribution in multi-factor tests."

    conclusion = {
        "factor_name": factor_name,
        "factor_source": "101 Formulaic Alphas",
        "factor_type": factor_type,
        "formula_summary": f"See alpha_engine/formulaic_alphas/{factor_name}.py",
        "return_mode": "T0 signal, T+1 open buy, T+2/T+3/T+4 close evaluation",
        "bucket_method": "train-defined global quantile buckets",
        "best_bucket_group": best_bucket_group,
        "bucket_rule": bucket_rule_from_group(best_bucket_group),
        "valid_regime": ",".join(valid_regimes),
        "avoid_regime": ",".join(avoid_regimes),
        "main_horizon": main_horizon,
        "secondary_horizon": secondary_horizon,
        "standalone_tradable": standalone_tradable,
        "is_filter_factor": is_filter_factor,
        "is_scoring_factor": is_scoring_factor,
        "avg_excess_pct": avg_excess,
        "avg_win_ratio": avg_win_ratio,
        "avg_group_count": avg_group_count,
        "recommended_usage": recommended_usage,
        "rejection_reason_as_standalone": rejection_reason,
        "next_action": next_action,
    }

    out_csv = step_file(output_dir, 10, factor_name, "factor_conclusion")
    out_md = step_file(output_dir, 10, factor_name, "factor_conclusion", suffix="md")

    pd.DataFrame([conclusion]).to_csv(out_csv, index=False, encoding="utf-8-sig")

    md = f"""# {factor_name} Factor Conclusion

## Final Classification

| Field | Value |
|---|---|
| factor_name | {factor_name} |
| factor_source | 101 Formulaic Alphas |
| factor_type | {factor_type} |
| return_mode | {conclusion["return_mode"]} |
| bucket_method | {conclusion["bucket_method"]} |
| best_bucket_group | {best_bucket_group} |
| bucket_rule | {conclusion["bucket_rule"]} |
| valid_regime | {conclusion["valid_regime"]} |
| avoid_regime | {conclusion["avoid_regime"]} |
| main_horizon | {main_horizon} |
| secondary_horizon | {secondary_horizon} |
| standalone_tradable | {standalone_tradable} |
| is_filter_factor | {is_filter_factor} |
| is_scoring_factor | {is_scoring_factor} |

## Recommended Usage

{recommended_usage}

## Rejection Reason As Standalone

{rejection_reason}

## Next Action

{next_action}

## Bucket Group Score

{group_score.to_string(index=False)}

## Core Evidence: Selected Bucket Group

{focus.to_string(index=False)}

## Regime Summary

{regime_check.to_string(index=False)}
"""

    out_md.write_text(md, encoding="utf-8")



def run_single_factor_research(
    *,
    market_dir: Path,
    factor_dir: Path,
    factor_col: str,
    output_dir: Path,
    custom_market_regime: Path,
    bucket_count: int = 10,
    research_horizons: list[int] | None = None,
    trade_horizons: list[int] | None = None,
    train_start: int = 2021,
    train_end: int = 2024,
    test_start: int = 2025,
    test_end: int = 2026,
) -> None:
    factor_name = factor_col
    research_horizons = research_horizons or [1, 2, 3, 4, 5, 6, 7]
    trade_horizons = trade_horizons or [2, 3, 4]

    output_dir.mkdir(parents=True, exist_ok=True)

    print("step 1/8: input validation")
    write_step1_input_validation(
        market_dir=market_dir,
        factor_dir=factor_dir,
        factor_name=factor_name,
        output_dir=output_dir,
    )

    print("step 2/8: factor distribution")
    factor_df = read_factor_dir(factor_dir, factor_col)
    write_step2_factor_distribution(
        factor_df=factor_df,
        factor_col=factor_col,
        factor_name=factor_name,
        output_dir=output_dir,
    )
    del factor_df

    print("step 3-4/8: shared research/trade analysis frame")
    shared = build_shared_analysis_frame(
        market_dir=market_dir,
        factor_dir=factor_dir,
        factor_col=factor_col,
        research_horizons=research_horizons,
        trade_horizons=trade_horizons,
    )

    shared[f"{factor_col}_bucket"] = assign_global_quantile_buckets(
        shared,
        factor_col=factor_col,
        bucket_count=bucket_count,
    )

    print("step 3/8: research return analysis")
    research = write_mode_outputs_from_shared_frame(
        shared=shared,
        factor_col=factor_col,
        factor_name=factor_name,
        output_dir=output_dir,
        step=3,
        prefix="research",
        mode_prefix="research",
        horizons=research_horizons,
        write_member_detail=False,
    )

    diagnose_single_factor_result(
        bucket_summary_path=research["bucket_summary"],
        daily_ic_path=research["daily_ic"],
        output_path=step_file(output_dir, 3, factor_name, "research_factor_diagnosis"),
    )

    print("step 4/8: trade return analysis")
    trade = write_mode_outputs_from_shared_frame(
        shared=shared,
        factor_col=factor_col,
        factor_name=factor_name,
        output_dir=output_dir,
        step=4,
        prefix="trade",
        mode_prefix="trade",
        horizons=trade_horizons,
        write_member_detail=True,
    )

    del shared

    print("step 5/8: bucket pattern")
    write_bucket_group_outputs(
        bucket_summary_path=trade["bucket_summary"],
        daily_ic_path=trade["daily_ic"],
        yearly_bucket_path=trade["yearly_bucket_summary"],
        factor_name=factor_name,
        output_dir=output_dir,
    )

    print("step 6/8: train/test bucket check")
    trade_targets = [f"fwd_return_pct_T{x}" for x in trade_horizons]
    member, _ = write_step6_train_test(
        member_path=trade["member_detail"],
        factor_col=factor_col,
        factor_name=factor_name,
        output_dir=output_dir,
        bucket_count=bucket_count,
        train_start=train_start,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
        targets=trade_targets,
    )

    print("step 7/8: candidate count")
    write_step7_candidate_counts(
        member=member,
        factor_name=factor_name,
        output_dir=output_dir,
    )

    print("step 8/10: custom market regime")
    write_step8_custom_regime(
        member=member,
        custom_market_regime=custom_market_regime,
        factor_name=factor_name,
        output_dir=output_dir,
        targets=trade_targets,
    )

    print("step 9/10: train-defined bucket regime check")
    write_step9_train_defined_bucket_regime_check(
        member=member,
        custom_market_regime=custom_market_regime,
        factor_name=factor_name,
        output_dir=output_dir,
        targets=trade_targets,
    )

    print("step 10/10: factor conclusion")
    write_step10_factor_conclusion(
        factor_name=factor_name,
        output_dir=output_dir,
    )

    print("single factor research: DONE")
    print("output_dir:", output_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--market-dir", required=True)
    parser.add_argument("--factor-dir", required=True)
    parser.add_argument("--factor-col", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--custom-market-regime", required=True)
    parser.add_argument("--bucket-count", type=int, default=10)
    parser.add_argument("--research-horizons", default="1,2,3,4,5,6,7")
    parser.add_argument("--trade-horizons", default="2,3,4")
    parser.add_argument("--train-start", type=int, default=2021)
    parser.add_argument("--train-end", type=int, default=2024)
    parser.add_argument("--test-start", type=int, default=2025)
    parser.add_argument("--test-end", type=int, default=2026)
    args = parser.parse_args()

    run_single_factor_research(
        market_dir=Path(args.market_dir),
        factor_dir=Path(args.factor_dir),
        factor_col=args.factor_col,
        output_dir=Path(args.output_dir),
        custom_market_regime=Path(args.custom_market_regime),
        bucket_count=args.bucket_count,
        research_horizons=parse_horizons(args.research_horizons),
        trade_horizons=parse_horizons(args.trade_horizons),
        train_start=args.train_start,
        train_end=args.train_end,
        test_start=args.test_start,
        test_end=args.test_end,
    )


if __name__ == "__main__":
    main()
