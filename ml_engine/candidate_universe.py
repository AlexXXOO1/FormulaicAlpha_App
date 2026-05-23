from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from ml_engine.dataset import summarize_ml_dataset


DEFAULT_CANDIDATE_BUCKET_FACTOR = "alpha_005"
DEFAULT_CANDIDATE_BUCKETS: tuple[int, ...] = (4, 5, 6, 7)
DEFAULT_BUCKET_COUNT = 10
DEFAULT_CANDIDATE_FEATURE_COLS: tuple[str, ...] = (
    "alpha_001",
    "alpha_002",
    "alpha_006",
    "alpha_008",
    "alpha_010",
)


def _normalize_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.normalize()


def _validate_required_columns(df: pd.DataFrame, cols: Sequence[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")


def parse_bucket_numbers(raw: str | Sequence[int] | None) -> list[int]:
    if raw is None:
        values = list(DEFAULT_CANDIDATE_BUCKETS)
    elif isinstance(raw, str):
        values = [int(x.strip()) for x in raw.split(",") if x.strip()]
    else:
        values = [int(x) for x in raw]

    if not values:
        raise ValueError("At least one candidate bucket is required.")
    if any(v <= 0 for v in values):
        raise ValueError(f"Candidate buckets must be positive: {values}")

    return sorted(set(values))


def parse_feature_cols(raw: str | Sequence[str] | None) -> list[str]:
    if raw is None:
        cols = list(DEFAULT_CANDIDATE_FEATURE_COLS)
    elif isinstance(raw, str):
        cols = [x.strip() for x in raw.split(",") if x.strip()]
    else:
        cols = [str(x) for x in raw]

    if not cols:
        raise ValueError("At least one feature column is required.")

    duplicated = sorted({c for c in cols if cols.count(c) > 1})
    if duplicated:
        raise ValueError(f"Duplicated feature columns: {duplicated}")

    return cols


def build_train_defined_bucket_edges(
    train_values: pd.Series,
    *,
    bucket_count: int = DEFAULT_BUCKET_COUNT,
) -> np.ndarray:
    if bucket_count < 2:
        raise ValueError("bucket_count must be >= 2.")

    x = pd.to_numeric(train_values, errors="coerce").dropna()
    if x.empty:
        raise ValueError("Training bucket factor has no valid numeric values.")

    edges = x.quantile(np.linspace(0.0, 1.0, bucket_count + 1)).to_numpy()
    edges = np.unique(edges)

    if len(edges) < 3:
        raise ValueError("Training bucket factor has too few unique values for bucketing.")

    edges = edges.astype(float)
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def assign_train_defined_buckets(
    values: pd.Series,
    edges: np.ndarray,
) -> pd.Series:
    x = pd.to_numeric(values, errors="coerce")
    bucket = pd.cut(x, bins=edges, labels=False, include_lowest=True)
    return (bucket + 1).astype("Int64")


def build_candidate_universe_dataset(
    df: pd.DataFrame,
    *,
    bucket_factor: str = DEFAULT_CANDIDATE_BUCKET_FACTOR,
    candidate_buckets: Sequence[int] = DEFAULT_CANDIDATE_BUCKETS,
    bucket_count: int = DEFAULT_BUCKET_COUNT,
    feature_cols: Sequence[str] = DEFAULT_CANDIDATE_FEATURE_COLS,
    label_col: str = "label_return_pct",
    train_split: str = "train",
    split_col: str = "split",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    features = parse_feature_cols(feature_cols)
    buckets = parse_bucket_numbers(candidate_buckets)

    required = ["symbol", "date", split_col, bucket_factor, label_col, *features]
    _validate_required_columns(df, required)

    work = df.copy()
    work["date"] = _normalize_date(work["date"])
    work[label_col] = pd.to_numeric(work[label_col], errors="coerce")

    train = work[work[split_col].eq(train_split)]
    if train.empty:
        raise ValueError(f"No rows found for train split: {train_split}")

    edges = build_train_defined_bucket_edges(train[bucket_factor], bucket_count=bucket_count)
    bucket_col = f"{bucket_factor}_train_bucket"
    work[bucket_col] = assign_train_defined_buckets(work[bucket_factor], edges)

    selected = work[work[bucket_col].isin(buckets)].copy()
    selected["candidate_bucket_factor"] = bucket_factor
    selected["candidate_bucket_rule"] = ",".join(str(x) for x in buckets)
    selected["candidate_bucket_count"] = int(bucket_count)

    trace_cols = [
        "symbol",
        "date",
        split_col,
        *features,
        bucket_factor,
        bucket_col,
        "candidate_bucket_factor",
        "candidate_bucket_rule",
        "candidate_bucket_count",
        label_col,
    ]

    optional_cols = [
        "label_up",
        "target_horizon",
        "market_regime",
    ]
    for col in optional_cols:
        if col in selected.columns and col not in trace_cols:
            trace_cols.append(col)

    extra_cols = [c for c in selected.columns if c not in trace_cols]
    selected = selected[trace_cols + extra_cols]
    selected = selected.sort_values(["date", "symbol"], kind="mergesort").reset_index(drop=True)

    edge_rows = []
    finite_edges = edges.copy()
    for idx in range(1, len(edges)):
        edge_rows.append(
            {
                "bucket_factor": bucket_factor,
                "bucket": idx,
                "left_edge": finite_edges[idx - 1],
                "right_edge": finite_edges[idx],
                "selected": idx in buckets,
                "bucket_count_requested": int(bucket_count),
                "bucket_count_actual": int(len(edges) - 1),
            }
        )
    edge_df = pd.DataFrame(edge_rows)

    return selected, edge_df


def summarize_candidate_universe_dataset(
    candidate_df: pd.DataFrame,
    *,
    feature_cols: Sequence[str] = DEFAULT_CANDIDATE_FEATURE_COLS,
    label_col: str = "label_return_pct",
) -> pd.DataFrame:
    features = parse_feature_cols(feature_cols)
    _validate_required_columns(candidate_df, ["symbol", "date", "split", label_col, *features])

    return summarize_ml_dataset(candidate_df, factor_cols=features)


def daily_candidate_count(candidate_df: pd.DataFrame) -> pd.DataFrame:
    _validate_required_columns(candidate_df, ["symbol", "date", "split"])

    work = candidate_df.copy()
    work["date"] = _normalize_date(work["date"])

    rows = []
    for (split, date), part in work.groupby(["split", "date"], sort=True):
        rows.append(
            {
                "split": split,
                "date": date,
                "rows": int(len(part)),
                "symbols": int(part["symbol"].nunique()),
            }
        )

    return pd.DataFrame(rows).sort_values(["split", "date"]).reset_index(drop=True)


def write_candidate_universe_outputs(
    *,
    candidate_df: pd.DataFrame,
    bucket_edges: pd.DataFrame,
    output_dir: str | Path,
    output_name: str,
    feature_cols: Sequence[str] = DEFAULT_CANDIDATE_FEATURE_COLS,
) -> dict[str, Path]:
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = output_dir / f"{output_name}.parquet"
    summary_path = output_dir / f"{output_name}_summary.csv"
    bucket_edges_path = output_dir / f"{output_name}_bucket_edges.csv"
    daily_count_path = output_dir / f"{output_name}_daily_count.csv"

    candidate_df.to_parquet(dataset_path, index=False)

    summary = summarize_candidate_universe_dataset(candidate_df, feature_cols=feature_cols)
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    bucket_edges.to_csv(bucket_edges_path, index=False, encoding="utf-8-sig")

    daily_count = daily_candidate_count(candidate_df)
    daily_count.to_csv(daily_count_path, index=False, encoding="utf-8-sig")

    return {
        "dataset": dataset_path,
        "summary": summary_path,
        "bucket_edges": bucket_edges_path,
        "daily_count": daily_count_path,
    }
