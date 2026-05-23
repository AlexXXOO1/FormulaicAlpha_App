from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

STABLE_ALPHA_COLS: tuple[str, ...] = (
    "alpha_001",
    "alpha_002",
    "alpha_005",
    "alpha_006",
    "alpha_008",
    "alpha_010",
)
MODEL_HOLDOUT_ALPHA_COLS: tuple[str, ...] = ("alpha_003", "alpha_004")
REJECTED_ALPHA_COLS: tuple[str, ...] = ("alpha_007", "alpha_009")
DEFAULT_ALPHA_COLS: tuple[str, ...] = STABLE_ALPHA_COLS
DEFAULT_TRADE_HORIZON = 3


def list_parquet_files(path: Path) -> list[Path]:
    files = sorted(Path(path).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found under: {path}")
    return files


def _normalize_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.normalize()


def _as_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _validate_factor_cols(factor_cols: Sequence[str]) -> list[str]:
    cols = [str(c) for c in factor_cols]
    if not cols:
        raise ValueError("At least one factor column is required.")
    duplicated = sorted({c for c in cols if cols.count(c) > 1})
    if duplicated:
        raise ValueError(f"Duplicated factor columns: {duplicated}")
    return cols


def load_market_frame(path: str | Path) -> pd.DataFrame:
    path = _as_path(path)
    df = pd.read_parquet(path)
    required = ["symbol", "date", "adjusted_open", "adjusted_close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Market data missing columns in {path.name}: {missing}")

    out = df[required].copy()
    out["date"] = _normalize_date(out["date"])
    out["adjusted_open"] = pd.to_numeric(out["adjusted_open"], errors="coerce")
    out["adjusted_close"] = pd.to_numeric(out["adjusted_close"], errors="coerce")
    out = out.dropna(subset=["symbol", "date", "adjusted_open", "adjusted_close"])
    out = out.sort_values(["symbol", "date"], kind="mergesort").reset_index(drop=True)
    return out


def load_feature_frame(path: str | Path, factor_cols: Sequence[str]) -> pd.DataFrame:
    path = _as_path(path)
    cols = _validate_factor_cols(factor_cols)
    df = pd.read_parquet(path)

    required = ["symbol", "date", *cols]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Feature data missing columns in {path.name}: {missing}")

    out = df[required].copy()
    out["date"] = _normalize_date(out["date"])
    for col in cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["symbol", "date"])
    out = out.sort_values(["symbol", "date"], kind="mergesort").reset_index(drop=True)
    return out


def add_trade_return_label(
    market: pd.DataFrame,
    *,
    horizon: int = DEFAULT_TRADE_HORIZON,
    label_col: str = "label_return_pct",
) -> pd.DataFrame:
    if horizon < 2:
        raise ValueError("A-share T+1 open entry evaluation requires horizon >= 2.")

    required = ["symbol", "date", "adjusted_open", "adjusted_close"]
    missing = [c for c in required if c not in market.columns]
    if missing:
        raise ValueError(f"Market frame missing columns: {missing}")

    work = market[required].copy()
    work["date"] = _normalize_date(work["date"])
    work["adjusted_open"] = pd.to_numeric(work["adjusted_open"], errors="coerce")
    work["adjusted_close"] = pd.to_numeric(work["adjusted_close"], errors="coerce")
    work = work.dropna(subset=required)
    work = work.sort_values(["symbol", "date"], kind="mergesort").reset_index(drop=True)

    grouped = work.groupby("symbol", sort=False)
    entry_open = grouped["adjusted_open"].shift(-1)
    exit_close = grouped["adjusted_close"].shift(-horizon)

    work[label_col] = (exit_close / entry_open - 1.0) * 100.0
    work["label_up"] = (work[label_col] > 0).where(work[label_col].notna())
    work["target_horizon"] = int(horizon)

    return work[["symbol", "date", label_col, "label_up", "target_horizon"]]


def load_market_regime(path: str | Path) -> pd.DataFrame:
    path = _as_path(path)
    suffix = path.suffix.lower()

    if suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix == ".json":
        df = pd.read_json(path)
    else:
        raise ValueError(f"Unsupported market regime file type: {suffix}")

    if "date" not in df.columns:
        raise ValueError(f"Market regime file missing date column: {path}")

    out = df.copy()
    out["date"] = _normalize_date(out["date"])
    out = out.dropna(subset=["date"])
    out = out.drop_duplicates(subset=["date"], keep="last")
    return out


def assign_time_split(
    dates: pd.Series,
    *,
    train_start: int = 2021,
    train_end: int = 2024,
    test_start: int = 2025,
    test_end: int = 2026,
) -> pd.Series:
    years = pd.to_datetime(dates, errors="coerce").dt.year
    split = np.full(len(years), "ignore", dtype=object)

    train_mask = years.between(train_start, train_end, inclusive="both")
    test_mask = years.between(test_start, test_end, inclusive="both")

    split[train_mask.fillna(False).to_numpy()] = "train"
    split[test_mask.fillna(False).to_numpy()] = "test"
    return pd.Series(split, index=dates.index, name="split")


def build_ml_dataset_from_symbol_dirs(
    *,
    market_dir: str | Path,
    factor_dir: str | Path,
    factor_cols: Sequence[str] = DEFAULT_ALPHA_COLS,
    horizon: int = DEFAULT_TRADE_HORIZON,
    custom_market_regime: str | Path | None = None,
    train_start: int = 2021,
    train_end: int = 2024,
    test_start: int = 2025,
    test_end: int = 2026,
    drop_missing_label: bool = True,
) -> pd.DataFrame:
    if horizon < 2:
        raise ValueError("A-share T+1 open entry evaluation requires horizon >= 2.")

    cols = _validate_factor_cols(factor_cols)
    market_dir = _as_path(market_dir)
    factor_dir = _as_path(factor_dir)

    market_files = {p.stem: p for p in list_parquet_files(market_dir)}
    factor_files = {p.stem: p for p in list_parquet_files(factor_dir)}

    missing_factor = sorted(set(market_files) - set(factor_files))
    if missing_factor:
        raise FileNotFoundError(f"Missing factor parquet files for symbols: {missing_factor[:20]}")

    extra_factor = sorted(set(factor_files) - set(market_files))
    if extra_factor:
        print(f"[WARNING] Extra factor parquet files ignored: {extra_factor[:20]}")

    parts: list[pd.DataFrame] = []
    for symbol in sorted(market_files):
        market = load_market_frame(market_files[symbol])
        feature = load_feature_frame(factor_files[symbol], cols)
        label = add_trade_return_label(market, horizon=horizon)

        part = feature.merge(
            label,
            on=["symbol", "date"],
            how="inner",
            validate="one_to_one",
        )
        parts.append(part)

    if not parts:
        raise ValueError("No ML dataset rows generated.")

    out = pd.concat(parts, ignore_index=True)
    out["date"] = _normalize_date(out["date"])

    if custom_market_regime is not None:
        regime = load_market_regime(custom_market_regime)
        regime_cols = [c for c in regime.columns if c != "date"]
        out = out.merge(regime[["date", *regime_cols]], on="date", how="left", validate="many_to_one")

    out["split"] = assign_time_split(
        out["date"],
        train_start=train_start,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
    )

    if drop_missing_label:
        out = out.dropna(subset=["label_return_pct"])

    ordered_cols = [
        "symbol",
        "date",
        "split",
        *cols,
        "label_return_pct",
        "label_up",
        "target_horizon",
    ]
    regime_extra_cols = [c for c in out.columns if c not in ordered_cols]
    out = out[ordered_cols + regime_extra_cols]
    out = out.sort_values(["date", "symbol"], kind="mergesort").reset_index(drop=True)
    return out


def summarize_ml_dataset(df: pd.DataFrame, *, factor_cols: Sequence[str]) -> pd.DataFrame:
    cols = _validate_factor_cols(factor_cols)
    required = ["split", "label_return_pct", *cols]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"ML dataset missing columns: {missing}")

    rows = []
    for split, part in df.groupby("split", dropna=False):
        rows.append(
            {
                "split": split,
                "rows": len(part),
                "dates": int(part["date"].nunique()) if "date" in part.columns else np.nan,
                "symbols": int(part["symbol"].nunique()) if "symbol" in part.columns else np.nan,
                "label_non_null": int(part["label_return_pct"].notna().sum()),
                "label_mean_pct": float(part["label_return_pct"].mean()),
                "label_median_pct": float(part["label_return_pct"].median()),
                "feature_complete_rows": int(part[cols].notna().all(axis=1).sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("split").reset_index(drop=True)
