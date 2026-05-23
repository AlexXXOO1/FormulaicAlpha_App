from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from ml_engine.evaluator import DEFAULT_TOP_K, parse_top_k


DEFAULT_BUCKET_FACTORS: tuple[str, ...] = ("alpha_001", "alpha_005")
DEFAULT_MIDDLE_BUCKETS: tuple[int, ...] = (4, 5, 6, 7)


def _normalize_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.normalize()


def _require_columns(df: pd.DataFrame, cols: Sequence[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Data missing required columns: {missing}")


def infer_regime_col(df: pd.DataFrame, explicit: str | None = None) -> str | None:
    if explicit:
        if explicit not in df.columns:
            raise ValueError(f"Requested regime column not found: {explicit}")
        return explicit
    return "market_regime" if "market_regime" in df.columns else None


def _regime_value(part: pd.DataFrame, regime_col: str | None) -> object:
    if regime_col is None or regime_col not in part.columns:
        return None
    non_null = part[regime_col].dropna()
    return non_null.iloc[0] if not non_null.empty else None


def _daily_metrics_from_selection(
    selected: pd.DataFrame,
    *,
    benchmark_name: str,
    benchmark_type: str,
    label_col: str,
    regime_col: str | None,
    top_k: int | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    if selected.empty:
        return pd.DataFrame(
            columns=[
                "benchmark_name",
                "benchmark_type",
                "date",
                "top_k",
                "selected_count",
                "return_mean_pct",
                "return_median_pct",
                "symbol_win_rate",
                "regime_col",
                "regime",
            ]
        )

    for date, part in selected.groupby("date", sort=True):
        ret = pd.to_numeric(part[label_col], errors="coerce").dropna()
        if ret.empty:
            continue

        rows.append(
            {
                "benchmark_name": benchmark_name,
                "benchmark_type": benchmark_type,
                "date": date,
                "top_k": top_k,
                "selected_count": int(len(ret)),
                "return_mean_pct": float(ret.mean()),
                "return_median_pct": float(ret.median()),
                "symbol_win_rate": float((ret > 0).mean()),
                "regime_col": regime_col,
                "regime": _regime_value(part, regime_col),
            }
        )

    return pd.DataFrame(rows).sort_values(["benchmark_name", "date"]).reset_index(drop=True)


def load_ml_topk_daily(path: str | Path) -> pd.DataFrame:
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"ML top-k daily file not found: {path}")

    df = pd.read_csv(path)
    _require_columns(df, ["date", "top_k", "selected_count", "return_mean_pct"])

    out = df.copy()
    out["date"] = _normalize_date(out["date"])
    out["benchmark_name"] = out["top_k"].apply(lambda k: f"ml_top_{int(k)}")
    out["benchmark_type"] = "ml_topk"

    keep_cols = [
        "benchmark_name",
        "benchmark_type",
        "date",
        "top_k",
        "selected_count",
        "return_mean_pct",
    ]

    if "return_median_pct" not in out.columns:
        out["return_median_pct"] = np.nan
    if "symbol_win_rate" not in out.columns:
        out["symbol_win_rate"] = np.nan
    if "regime_col" not in out.columns:
        out["regime_col"] = None
    if "regime" not in out.columns:
        out["regime"] = None

    keep_cols += ["return_median_pct", "symbol_win_rate", "regime_col", "regime"]
    return out[keep_cols].sort_values(["benchmark_name", "date"]).reset_index(drop=True)


def build_market_daily_benchmark(
    dataset: pd.DataFrame,
    *,
    label_col: str = "label_return_pct",
    split: str = "test",
    regime_col: str | None = None,
) -> pd.DataFrame:
    _require_columns(dataset, ["symbol", "date", "split", label_col])
    regime = infer_regime_col(dataset, regime_col)

    work = dataset[dataset["split"].eq(split)].copy()
    work["date"] = _normalize_date(work["date"])
    work[label_col] = pd.to_numeric(work[label_col], errors="coerce")
    work = work.dropna(subset=["date", label_col])

    return _daily_metrics_from_selection(
        work,
        benchmark_name="market_all",
        benchmark_type="market",
        label_col=label_col,
        regime_col=regime,
        top_k=None,
    )


def build_random_topk_benchmark(
    dataset: pd.DataFrame,
    *,
    top_k: Sequence[int] = DEFAULT_TOP_K,
    label_col: str = "label_return_pct",
    split: str = "test",
    random_state: int = 42,
    regime_col: str | None = None,
) -> pd.DataFrame:
    _require_columns(dataset, ["symbol", "date", "split", label_col])
    k_values = parse_top_k(top_k)
    regime = infer_regime_col(dataset, regime_col)

    work = dataset[dataset["split"].eq(split)].copy()
    work["date"] = _normalize_date(work["date"])
    work[label_col] = pd.to_numeric(work[label_col], errors="coerce")
    work = work.dropna(subset=["date", label_col]).reset_index(drop=True)

    rng = np.random.default_rng(random_state)
    work["_random_score"] = rng.random(len(work))
    ranked = work.sort_values(["date", "_random_score", "symbol"], ascending=[True, False, True], kind="mergesort")

    parts: list[pd.DataFrame] = []
    for k in k_values:
        selected = ranked.groupby("date", sort=False).head(k).copy()
        parts.append(
            _daily_metrics_from_selection(
                selected,
                benchmark_name=f"random_top_{k}",
                benchmark_type="random_topk",
                label_col=label_col,
                regime_col=regime,
                top_k=k,
            )
        )

    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _train_defined_bucket(
    train_values: pd.Series,
    target_values: pd.Series,
    *,
    bucket_count: int = 10,
) -> pd.Series:
    train = pd.to_numeric(train_values, errors="coerce").dropna()
    target = pd.to_numeric(target_values, errors="coerce")

    if train.empty:
        return pd.Series(pd.NA, index=target_values.index, dtype="Int64")

    edges = train.quantile(np.linspace(0.0, 1.0, bucket_count + 1)).to_numpy()
    edges = np.unique(edges)

    if len(edges) < 3:
        return pd.Series(pd.NA, index=target_values.index, dtype="Int64")

    edges[0] = -np.inf
    edges[-1] = np.inf

    bucket = pd.cut(target, bins=edges, labels=False, include_lowest=True)
    return (bucket + 1).astype("Int64")


def build_middle_bucket_benchmark(
    dataset: pd.DataFrame,
    *,
    factor_cols: Sequence[str] = DEFAULT_BUCKET_FACTORS,
    bucket_numbers: Sequence[int] = DEFAULT_MIDDLE_BUCKETS,
    bucket_count: int = 10,
    label_col: str = "label_return_pct",
    train_split: str = "train",
    test_split: str = "test",
    regime_col: str | None = None,
    include_combined_and: bool = True,
) -> pd.DataFrame:
    factor_cols = [str(c) for c in factor_cols]
    if not factor_cols:
        raise ValueError("At least one bucket factor is required.")

    _require_columns(dataset, ["symbol", "date", "split", label_col, *factor_cols])
    regime = infer_regime_col(dataset, regime_col)

    train = dataset[dataset["split"].eq(train_split)].copy()
    test = dataset[dataset["split"].eq(test_split)].copy()
    test["date"] = _normalize_date(test["date"])
    test[label_col] = pd.to_numeric(test[label_col], errors="coerce")
    test = test.dropna(subset=["date", label_col]).copy()

    bucket_numbers = sorted({int(x) for x in bucket_numbers})
    if not bucket_numbers:
        raise ValueError("At least one bucket number is required.")

    parts: list[pd.DataFrame] = []
    bucket_cols: list[str] = []

    for factor in factor_cols:
        bucket_col = f"_{factor}_bucket"
        bucket_cols.append(bucket_col)
        test[bucket_col] = _train_defined_bucket(
            train[factor],
            test[factor],
            bucket_count=bucket_count,
        )

        selected = test[test[bucket_col].isin(bucket_numbers)].copy()
        bucket_label = f"{min(bucket_numbers)}_{max(bucket_numbers)}"
        parts.append(
            _daily_metrics_from_selection(
                selected,
                benchmark_name=f"{factor}_bucket_{bucket_label}",
                benchmark_type="middle_bucket",
                label_col=label_col,
                regime_col=regime,
                top_k=None,
            )
        )

    if include_combined_and and len(factor_cols) >= 2:
        mask = pd.Series(True, index=test.index)
        for bucket_col in bucket_cols:
            mask &= test[bucket_col].isin(bucket_numbers)

        selected = test[mask].copy()
        bucket_label = f"{min(bucket_numbers)}_{max(bucket_numbers)}"
        combined_name = "_".join(factor_cols) + f"_bucket_{bucket_label}_AND"

        parts.append(
            _daily_metrics_from_selection(
                selected,
                benchmark_name=combined_name,
                benchmark_type="middle_bucket_and",
                label_col=label_col,
                regime_col=regime,
                top_k=None,
            )
        )

    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def summarize_benchmark_daily(daily: pd.DataFrame) -> pd.DataFrame:
    _require_columns(daily, ["benchmark_name", "benchmark_type", "return_mean_pct", "selected_count"])

    rows: list[dict[str, object]] = []
    for (name, btype), part in daily.groupby(["benchmark_name", "benchmark_type"], sort=True):
        ret = pd.to_numeric(part["return_mean_pct"], errors="coerce").dropna()
        if ret.empty:
            continue

        compound = float((np.prod(1.0 + ret / 100.0) - 1.0) * 100.0)

        rows.append(
            {
                "benchmark_name": name,
                "benchmark_type": btype,
                "days": int(len(ret)),
                "mean_daily_return_pct": float(ret.mean()),
                "median_daily_return_pct": float(ret.median()),
                "daily_win_rate": float((ret > 0).mean()),
                "compound_return_pct": compound,
                "avg_selected_count": float(part["selected_count"].mean()),
                "min_selected_count": int(part["selected_count"].min()),
                "max_selected_count": int(part["selected_count"].max()),
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["mean_daily_return_pct", "daily_win_rate", "benchmark_name"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def summarize_benchmark_by_year(daily: pd.DataFrame) -> pd.DataFrame:
    _require_columns(daily, ["benchmark_name", "benchmark_type", "date", "return_mean_pct", "selected_count"])

    work = daily.copy()
    work["year"] = _normalize_date(work["date"]).dt.year

    rows: list[dict[str, object]] = []
    for (name, btype, year), part in work.groupby(["benchmark_name", "benchmark_type", "year"], sort=True):
        ret = pd.to_numeric(part["return_mean_pct"], errors="coerce").dropna()
        if ret.empty:
            continue
        rows.append(
            {
                "benchmark_name": name,
                "benchmark_type": btype,
                "year": int(year),
                "days": int(len(ret)),
                "mean_daily_return_pct": float(ret.mean()),
                "median_daily_return_pct": float(ret.median()),
                "daily_win_rate": float((ret > 0).mean()),
                "avg_selected_count": float(part["selected_count"].mean()),
            }
        )

    return pd.DataFrame(rows).sort_values(["benchmark_name", "year"]).reset_index(drop=True)


def summarize_benchmark_by_regime(daily: pd.DataFrame) -> pd.DataFrame:
    if "regime" not in daily.columns:
        return pd.DataFrame()

    work = daily.dropna(subset=["regime"]).copy()
    if work.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for (name, btype, regime), part in work.groupby(["benchmark_name", "benchmark_type", "regime"], sort=True):
        ret = pd.to_numeric(part["return_mean_pct"], errors="coerce").dropna()
        if ret.empty:
            continue
        rows.append(
            {
                "benchmark_name": name,
                "benchmark_type": btype,
                "regime": regime,
                "days": int(len(ret)),
                "mean_daily_return_pct": float(ret.mean()),
                "median_daily_return_pct": float(ret.median()),
                "daily_win_rate": float((ret > 0).mean()),
                "avg_selected_count": float(part["selected_count"].mean()),
            }
        )

    return pd.DataFrame(rows).sort_values(["benchmark_name", "regime"]).reset_index(drop=True)


def build_benchmark_comparison(
    dataset: pd.DataFrame,
    *,
    ml_topk_daily: pd.DataFrame | None = None,
    top_k: Sequence[int] = DEFAULT_TOP_K,
    bucket_factors: Sequence[str] = DEFAULT_BUCKET_FACTORS,
    bucket_numbers: Sequence[int] = DEFAULT_MIDDLE_BUCKETS,
    label_col: str = "label_return_pct",
    train_split: str = "train",
    test_split: str = "test",
    random_state: int = 42,
    regime_col: str | None = None,
) -> dict[str, pd.DataFrame]:
    parts: list[pd.DataFrame] = []

    if ml_topk_daily is not None and not ml_topk_daily.empty:
        parts.append(ml_topk_daily.copy())

    parts.append(
        build_market_daily_benchmark(
            dataset,
            label_col=label_col,
            split=test_split,
            regime_col=regime_col,
        )
    )
    parts.append(
        build_random_topk_benchmark(
            dataset,
            top_k=top_k,
            label_col=label_col,
            split=test_split,
            random_state=random_state,
            regime_col=regime_col,
        )
    )
    parts.append(
        build_middle_bucket_benchmark(
            dataset,
            factor_cols=bucket_factors,
            bucket_numbers=bucket_numbers,
            label_col=label_col,
            train_split=train_split,
            test_split=test_split,
            regime_col=regime_col,
        )
    )

    daily = pd.concat(parts, ignore_index=True)
    daily["date"] = _normalize_date(daily["date"])

    return {
        "benchmark_daily": daily.sort_values(["benchmark_name", "date"]).reset_index(drop=True),
        "benchmark_summary": summarize_benchmark_daily(daily),
        "benchmark_by_year": summarize_benchmark_by_year(daily),
        "benchmark_by_regime": summarize_benchmark_by_regime(daily),
    }


def write_benchmark_comparison(
    comparison: dict[str, pd.DataFrame],
    *,
    output_dir: str | Path,
    output_name: str,
) -> dict[str, Path]:
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}
    for name, frame in comparison.items():
        path = output_dir / f"{output_name}_{name}.csv"
        frame.to_csv(path, index=False, encoding="utf-8-sig")
        paths[name] = path

    return paths
