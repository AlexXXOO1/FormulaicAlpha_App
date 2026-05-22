from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_ALPHA_COLS = ("alpha_001", "alpha_002", "alpha_005")
DEFAULT_VALID_REGIMES = ("range_bound", "risk_off")


@dataclass(frozen=True)
class DailyCandidateExportResult:
    target_date: pd.Timestamp
    market_regime: str | None
    wide_path: Path
    precision_path: Path
    summary_path: Path
    wide_count: int
    precision_count: int


def read_formulaic_alpha_features(
    factor_dir: Path,
    alpha_cols: tuple[str, ...] = DEFAULT_ALPHA_COLS,
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []

    for path in sorted(Path(factor_dir).glob("*.parquet")):
        columns = pd.read_parquet(path, columns=None).columns.tolist()
        use_cols = ["symbol", "date"] + [col for col in alpha_cols if col in columns]

        if len(use_cols) <= 2:
            continue

        part = pd.read_parquet(path, columns=use_cols)
        part["date"] = pd.to_datetime(part["date"], errors="coerce").dt.normalize()
        parts.append(part)

    if not parts:
        raise ValueError(f"No usable formulaic alpha parquet files found under: {factor_dir}")

    df = pd.concat(parts, ignore_index=True)

    missing = [col for col in alpha_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required alpha columns: {missing}")

    return df.dropna(subset=["symbol", "date"]).copy()


def read_market_regime(regime_path: Path) -> pd.DataFrame:
    regime = pd.read_csv(regime_path)
    regime.columns = [str(col).strip() for col in regime.columns]

    date_col = "date" if "date" in regime.columns else regime.columns[0]

    if "market_regime" in regime.columns:
        regime_col = "market_regime"
    elif "regime" in regime.columns:
        regime_col = "regime"
    else:
        raise ValueError(f"Cannot find market regime column in: {regime_path}")

    regime[date_col] = pd.to_datetime(regime[date_col], errors="coerce").dt.normalize()

    return (
        regime[[date_col, regime_col]]
        .rename(columns={date_col: "date", regime_col: "market_regime"})
        .dropna(subset=["date", "market_regime"])
        .drop_duplicates(subset=["date"], keep="last")
        .sort_values("date")
        .reset_index(drop=True)
    )


def assign_train_defined_bucket(
    df: pd.DataFrame,
    factor_col: str,
    *,
    bucket_count: int = 10,
    train_start: str = "2021-01-01",
    train_end: str = "2024-12-31",
) -> pd.Series:
    train = df.loc[
        df["date"].between(pd.Timestamp(train_start), pd.Timestamp(train_end)),
        factor_col,
    ].dropna()

    if train.empty:
        raise ValueError(f"Empty train sample for {factor_col}")

    _, bins = pd.qcut(train, q=bucket_count, retbins=True, duplicates="drop")
    bins = np.unique(bins)

    if len(bins) < 2:
        raise ValueError(f"Cannot create train-defined buckets for {factor_col}")

    bins[0] = -np.inf
    bins[-1] = np.inf

    labels = list(range(1, len(bins)))

    return pd.cut(
        df[factor_col],
        bins=bins,
        labels=labels,
        include_lowest=True,
    ).astype("Int64")


def build_daily_candidate_frame(
    *,
    factor_dir: Path,
    regime_path: Path,
    target_date: str | pd.Timestamp | None = None,
    bucket_count: int = 10,
    valid_regimes: tuple[str, ...] = DEFAULT_VALID_REGIMES,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Timestamp, str | None]:
    features = read_formulaic_alpha_features(Path(factor_dir))
    regime = read_market_regime(Path(regime_path))

    df = features.merge(regime, on="date", how="left")

    for col in DEFAULT_ALPHA_COLS:
        df[f"{col}_bucket"] = assign_train_defined_bucket(
            df,
            col,
            bucket_count=bucket_count,
        )

    if target_date is None or str(target_date).lower() == "latest":
        selected_date = df["date"].max()
    else:
        selected_date = pd.Timestamp(target_date).normalize()

    latest = df[df["date"].eq(selected_date)].copy()

    valid_regime_mask = latest["market_regime"].isin(valid_regimes)

    wide = latest[
        valid_regime_mask
        & latest["alpha_005_bucket"].isin([4, 5])
    ].copy()

    precision = latest[
        valid_regime_mask
        & latest["alpha_001_bucket"].isin([4, 5, 6, 7])
        & latest["alpha_002_bucket"].isin([4])
        & latest["alpha_005_bucket"].isin([4, 5])
    ].copy()

    detail_cols = [
        "symbol",
        "date",
        "market_regime",
        "alpha_001_bucket",
        "alpha_002_bucket",
        "alpha_005_bucket",
        "alpha_001",
        "alpha_002",
        "alpha_005",
        "pool_name",
    ]

    wide["pool_name"] = "wide_pool"
    precision["pool_name"] = "precision_pool"

    wide = wide[detail_cols].sort_values(["alpha_005_bucket", "symbol"]).reset_index(drop=True)
    precision = precision[detail_cols].sort_values(
        ["alpha_005_bucket", "alpha_001_bucket", "alpha_002_bucket", "symbol"]
    ).reset_index(drop=True)

    market_regime = None
    if not latest.empty and latest["market_regime"].notna().any():
        market_regime = str(latest["market_regime"].dropna().iloc[0])

    summary = pd.DataFrame(
        [
            {
                "date": selected_date,
                "market_regime": market_regime,
                "pool_name": "wide_pool",
                "candidate_count": int(len(wide)),
                "rule": "market_regime in [range_bound,risk_off] AND alpha_005_bucket in [4,5]",
            },
            {
                "date": selected_date,
                "market_regime": market_regime,
                "pool_name": "precision_pool",
                "candidate_count": int(len(precision)),
                "rule": "market_regime in [range_bound,risk_off] AND alpha_001_bucket in [4,5,6,7] AND alpha_002_bucket in [4] AND alpha_005_bucket in [4,5]",
            },
        ]
    )

    return wide, precision, summary, selected_date, market_regime


def export_daily_candidates(
    *,
    factor_dir: Path,
    regime_path: Path,
    output_dir: Path,
    target_date: str | pd.Timestamp | None = None,
    bucket_count: int = 10,
) -> DailyCandidateExportResult:
    wide, precision, summary, selected_date, market_regime = build_daily_candidate_frame(
        factor_dir=Path(factor_dir),
        regime_path=Path(regime_path),
        target_date=target_date,
        bucket_count=bucket_count,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    date_tag = selected_date.strftime("%Y-%m-%d")
    wide_path = output_dir / f"daily_candidates_{date_tag}_wide.csv"
    precision_path = output_dir / f"daily_candidates_{date_tag}_precision.csv"
    summary_path = output_dir / f"daily_candidates_{date_tag}_summary.csv"

    wide.to_csv(wide_path, index=False, encoding="utf-8-sig")
    precision.to_csv(precision_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    return DailyCandidateExportResult(
        target_date=selected_date,
        market_regime=market_regime,
        wide_path=wide_path,
        precision_path=precision_path,
        summary_path=summary_path,
        wide_count=int(len(wide)),
        precision_count=int(len(precision)),
    )
