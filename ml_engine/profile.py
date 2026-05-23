from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from ml_engine.dataset import DEFAULT_ALPHA_COLS


DEFAULT_REGIME_COL_CANDIDATES: tuple[str, ...] = (
    "market_regime",
    "regime",
    "custom_market_regime",
)


def _validate_factor_cols(df: pd.DataFrame, factor_cols: Sequence[str]) -> list[str]:
    cols = [str(c) for c in factor_cols]
    if not cols:
        raise ValueError("At least one factor column is required.")
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing factor columns: {missing}")
    return cols


def _require_columns(df: pd.DataFrame, required: Sequence[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")


def infer_regime_col(df: pd.DataFrame, explicit: str | None = None) -> str | None:
    if explicit:
        if explicit not in df.columns:
            raise ValueError(f"Requested regime column not found: {explicit}")
        return explicit

    for col in DEFAULT_REGIME_COL_CANDIDATES:
        if col in df.columns:
            return col
    return None


def load_ml_dataset(path: str | Path) -> pd.DataFrame:
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"ML dataset not found: {path}")

    df = pd.read_parquet(path)
    _require_columns(df, ["symbol", "date", "split", "label_return_pct"])

    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    out["label_return_pct"] = pd.to_numeric(out["label_return_pct"], errors="coerce")
    return out


def factor_missing_summary(df: pd.DataFrame, factor_cols: Sequence[str]) -> pd.DataFrame:
    cols = _validate_factor_cols(df, factor_cols)
    rows: list[dict[str, object]] = []

    for split, part in df.groupby("split", dropna=False):
        total = len(part)
        for col in cols:
            missing_count = int(part[col].isna().sum())
            rows.append(
                {
                    "split": split,
                    "factor": col,
                    "rows": total,
                    "missing_count": missing_count,
                    "missing_rate": missing_count / total if total else np.nan,
                    "non_null_count": int(part[col].notna().sum()),
                    "unique_count": int(part[col].nunique(dropna=True)),
                }
            )

    return pd.DataFrame(rows).sort_values(["split", "factor"]).reset_index(drop=True)


def _describe_one(series: pd.Series) -> Mapping[str, float]:
    x = pd.to_numeric(series, errors="coerce").dropna()
    if x.empty:
        return {
            "count": 0,
            "mean": np.nan,
            "std": np.nan,
            "p01": np.nan,
            "p05": np.nan,
            "p25": np.nan,
            "median": np.nan,
            "p75": np.nan,
            "p95": np.nan,
            "p99": np.nan,
        }

    q = x.quantile([0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99])
    return {
        "count": int(x.count()),
        "mean": float(x.mean()),
        "std": float(x.std(ddof=1)),
        "p01": float(q.loc[0.01]),
        "p05": float(q.loc[0.05]),
        "p25": float(q.loc[0.25]),
        "median": float(q.loc[0.50]),
        "p75": float(q.loc[0.75]),
        "p95": float(q.loc[0.95]),
        "p99": float(q.loc[0.99]),
    }


def _population_stability_index(train: pd.Series, test: pd.Series, *, bucket_count: int = 10) -> float:
    train_x = pd.to_numeric(train, errors="coerce").dropna()
    test_x = pd.to_numeric(test, errors="coerce").dropna()

    if train_x.empty or test_x.empty:
        return np.nan

    edges = train_x.quantile(np.linspace(0.0, 1.0, bucket_count + 1)).to_numpy()
    edges = np.unique(edges)

    if len(edges) < 3:
        return np.nan

    edges[0] = -np.inf
    edges[-1] = np.inf

    train_bins = pd.cut(train_x, bins=edges, include_lowest=True)
    test_bins = pd.cut(test_x, bins=edges, include_lowest=True)

    train_pct = train_bins.value_counts(sort=False, normalize=True)
    test_pct = test_bins.value_counts(sort=False, normalize=True)

    aligned = pd.DataFrame({"train": train_pct, "test": test_pct}).fillna(0.0)
    aligned = aligned.clip(lower=1e-6)

    psi = ((aligned["test"] - aligned["train"]) * np.log(aligned["test"] / aligned["train"])).sum()
    return float(psi)


def factor_distribution_drift(
    df: pd.DataFrame,
    factor_cols: Sequence[str],
    *,
    train_split: str = "train",
    test_split: str = "test",
) -> pd.DataFrame:
    cols = _validate_factor_cols(df, factor_cols)

    train = df[df["split"].eq(train_split)]
    test = df[df["split"].eq(test_split)]

    rows: list[dict[str, object]] = []
    for col in cols:
        train_desc = _describe_one(train[col])
        test_desc = _describe_one(test[col])

        train_mean = train_desc["mean"]
        test_mean = test_desc["mean"]
        train_std = train_desc["std"]
        test_std = test_desc["std"]

        rows.append(
            {
                "factor": col,
                "train_count": train_desc["count"],
                "test_count": test_desc["count"],
                "train_mean": train_mean,
                "test_mean": test_mean,
                "mean_diff": test_mean - train_mean if pd.notna(train_mean) and pd.notna(test_mean) else np.nan,
                "train_std": train_std,
                "test_std": test_std,
                "std_ratio": test_std / train_std if pd.notna(train_std) and train_std not in (0, 0.0) else np.nan,
                "train_median": train_desc["median"],
                "test_median": test_desc["median"],
                "median_diff": test_desc["median"] - train_desc["median"]
                if pd.notna(train_desc["median"]) and pd.notna(test_desc["median"])
                else np.nan,
                "psi_train_to_test": _population_stability_index(train[col], test[col]),
                "train_p01": train_desc["p01"],
                "train_p99": train_desc["p99"],
                "test_p01": test_desc["p01"],
                "test_p99": test_desc["p99"],
            }
        )

    return pd.DataFrame(rows).sort_values("factor").reset_index(drop=True)


def factor_correlation_matrix(df: pd.DataFrame, factor_cols: Sequence[str], *, split: str | None = "train") -> pd.DataFrame:
    cols = _validate_factor_cols(df, factor_cols)
    work = df if split is None else df[df["split"].eq(split)]
    corr = work[cols].apply(pd.to_numeric, errors="coerce").corr(method="pearson")
    corr.index.name = "factor"
    return corr.reset_index()


def label_by_year(df: pd.DataFrame, factor_cols: Sequence[str]) -> pd.DataFrame:
    cols = _validate_factor_cols(df, factor_cols)
    _require_columns(df, ["date", "split", "label_return_pct"])

    work = df.copy()
    work["year"] = pd.to_datetime(work["date"], errors="coerce").dt.year

    rows: list[dict[str, object]] = []
    for (split, year), part in work.groupby(["split", "year"], dropna=False):
        label = pd.to_numeric(part["label_return_pct"], errors="coerce")
        rows.append(
            {
                "split": split,
                "year": year,
                "rows": len(part),
                "symbols": int(part["symbol"].nunique()) if "symbol" in part.columns else np.nan,
                "dates": int(part["date"].nunique()),
                "label_mean_pct": float(label.mean()),
                "label_median_pct": float(label.median()),
                "label_win_rate": float((label > 0).mean()),
                "feature_complete_rows": int(part[cols].notna().all(axis=1).sum()),
            }
        )

    return pd.DataFrame(rows).sort_values(["split", "year"]).reset_index(drop=True)


def label_by_regime(df: pd.DataFrame, factor_cols: Sequence[str], *, regime_col: str | None = None) -> pd.DataFrame:
    cols = _validate_factor_cols(df, factor_cols)
    regime = infer_regime_col(df, regime_col)

    if regime is None:
        return pd.DataFrame(
            columns=[
                "split",
                "regime_col",
                "regime",
                "rows",
                "dates",
                "symbols",
                "label_mean_pct",
                "label_median_pct",
                "label_win_rate",
                "feature_complete_rows",
            ]
        )

    rows: list[dict[str, object]] = []
    for (split, regime_value), part in df.groupby(["split", regime], dropna=False):
        label = pd.to_numeric(part["label_return_pct"], errors="coerce")
        rows.append(
            {
                "split": split,
                "regime_col": regime,
                "regime": regime_value,
                "rows": len(part),
                "dates": int(part["date"].nunique()) if "date" in part.columns else np.nan,
                "symbols": int(part["symbol"].nunique()) if "symbol" in part.columns else np.nan,
                "label_mean_pct": float(label.mean()),
                "label_median_pct": float(label.median()),
                "label_win_rate": float((label > 0).mean()),
                "feature_complete_rows": int(part[cols].notna().all(axis=1).sum()),
            }
        )

    return pd.DataFrame(rows).sort_values(["split", "regime"]).reset_index(drop=True)


def daily_sample_stability(df: pd.DataFrame, factor_cols: Sequence[str]) -> pd.DataFrame:
    cols = _validate_factor_cols(df, factor_cols)
    _require_columns(df, ["date", "split", "label_return_pct"])

    rows: list[dict[str, object]] = []
    for (split, date), part in df.groupby(["split", "date"], dropna=False):
        label = pd.to_numeric(part["label_return_pct"], errors="coerce")
        rows.append(
            {
                "split": split,
                "date": date,
                "rows": len(part),
                "symbols": int(part["symbol"].nunique()) if "symbol" in part.columns else np.nan,
                "label_mean_pct": float(label.mean()),
                "label_median_pct": float(label.median()),
                "label_win_rate": float((label > 0).mean()),
                "feature_complete_rows": int(part[cols].notna().all(axis=1).sum()),
                "feature_complete_rate": float(part[cols].notna().all(axis=1).mean()),
            }
        )

    return pd.DataFrame(rows).sort_values(["split", "date"]).reset_index(drop=True)


def missing_pattern_summary(df: pd.DataFrame, factor_cols: Sequence[str], *, top_n: int = 50) -> pd.DataFrame:
    cols = _validate_factor_cols(df, factor_cols)

    miss = df[cols].isna()
    any_missing = miss.any(axis=1)
    work = df.loc[any_missing, ["split", "date", "symbol", *cols]].copy()

    if work.empty:
        return pd.DataFrame(columns=["split", "missing_pattern", "rows", "dates", "symbols"])

    work["missing_pattern"] = miss.loc[any_missing].apply(
        lambda row: ",".join([col for col, is_missing in row.items() if is_missing]),
        axis=1,
    )

    rows: list[dict[str, object]] = []
    for (split, pattern), part in work.groupby(["split", "missing_pattern"], dropna=False):
        rows.append(
            {
                "split": split,
                "missing_pattern": pattern,
                "rows": len(part),
                "dates": int(part["date"].nunique()),
                "symbols": int(part["symbol"].nunique()),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(["rows", "split", "missing_pattern"], ascending=[False, True, True])
        .head(top_n)
        .reset_index(drop=True)
    )


def build_ml_dataset_profile(
    df: pd.DataFrame,
    *,
    factor_cols: Sequence[str] = DEFAULT_ALPHA_COLS,
    regime_col: str | None = None,
) -> dict[str, pd.DataFrame]:
    cols = _validate_factor_cols(df, factor_cols)

    return {
        "factor_missing": factor_missing_summary(df, cols),
        "factor_distribution_drift": factor_distribution_drift(df, cols),
        "factor_correlation_train": factor_correlation_matrix(df, cols, split="train"),
        "factor_correlation_all": factor_correlation_matrix(df, cols, split=None),
        "label_by_year": label_by_year(df, cols),
        "label_by_regime": label_by_regime(df, cols, regime_col=regime_col),
        "daily_sample_stability": daily_sample_stability(df, cols),
        "missing_pattern": missing_pattern_summary(df, cols),
    }


def write_ml_dataset_profile(
    df: pd.DataFrame,
    *,
    output_dir: str | Path,
    output_name: str = "ml_dataset_v0_profile",
    factor_cols: Sequence[str] = DEFAULT_ALPHA_COLS,
    regime_col: str | None = None,
) -> dict[str, Path]:
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    profile = build_ml_dataset_profile(df, factor_cols=factor_cols, regime_col=regime_col)

    paths: dict[str, Path] = {}
    for name, frame in profile.items():
        path = output_dir / f"{output_name}_{name}.csv"
        frame.to_csv(path, index=False, encoding="utf-8-sig")
        paths[name] = path

    return paths
