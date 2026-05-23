from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


DEFAULT_PREDICTION_COL = "prediction"
DEFAULT_LABEL_COL = "label_return_pct"
DEFAULT_BUCKET_COUNT = 10
DEFAULT_TOPK_VALUES: tuple[int, ...] = (10, 20, 50)


def _normalize_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.normalize()


def _require_columns(df: pd.DataFrame, cols: Sequence[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Data missing required columns: {missing}")


def build_prediction_bucket_detail(
    predictions: pd.DataFrame,
    *,
    prediction_col: str = DEFAULT_PREDICTION_COL,
    label_col: str = DEFAULT_LABEL_COL,
    bucket_count: int = DEFAULT_BUCKET_COUNT,
    min_daily_rows: int = 100,
) -> pd.DataFrame:
    if bucket_count < 2:
        raise ValueError("bucket_count must be >= 2.")

    _require_columns(predictions, ["symbol", "date", prediction_col, label_col])

    work = predictions.copy()
    work["date"] = _normalize_date(work["date"])
    work[prediction_col] = pd.to_numeric(work[prediction_col], errors="coerce")
    work[label_col] = pd.to_numeric(work[label_col], errors="coerce")
    work = work.dropna(subset=["symbol", "date", prediction_col, label_col])

    rows: list[dict[str, object]] = []

    for date, daily in work.groupby("date", sort=True):
        if len(daily) < min_daily_rows:
            continue

        daily = daily.copy()
        try:
            daily["prediction_bucket"] = (
                pd.qcut(
                    daily[prediction_col],
                    q=bucket_count,
                    labels=False,
                    duplicates="drop",
                )
                + 1
            )
        except ValueError:
            continue

        daily = daily.dropna(subset=["prediction_bucket"])
        if daily.empty:
            continue

        daily["prediction_bucket"] = daily["prediction_bucket"].astype(int)

        for bucket, part in daily.groupby("prediction_bucket", sort=True):
            returns = pd.to_numeric(part[label_col], errors="coerce").dropna()
            if returns.empty:
                continue

            rows.append(
                {
                    "date": date,
                    "prediction_bucket": int(bucket),
                    "rows": int(len(part)),
                    "mean_return_pct": float(returns.mean()),
                    "median_return_pct": float(returns.median()),
                    "win_rate": float((returns > 0).mean()),
                    "avg_prediction": float(part[prediction_col].mean()),
                    "min_prediction": float(part[prediction_col].min()),
                    "max_prediction": float(part[prediction_col].max()),
                }
            )

    return pd.DataFrame(rows).sort_values(["date", "prediction_bucket"]).reset_index(drop=True)


def summarize_prediction_buckets(bucket_detail: pd.DataFrame) -> pd.DataFrame:
    _require_columns(
        bucket_detail,
        [
            "date",
            "prediction_bucket",
            "rows",
            "mean_return_pct",
            "median_return_pct",
            "win_rate",
            "avg_prediction",
        ],
    )

    rows: list[dict[str, object]] = []

    for bucket, part in bucket_detail.groupby("prediction_bucket", sort=True):
        rows.append(
            {
                "prediction_bucket": int(bucket),
                "days": int(part["date"].nunique()),
                "avg_rows": float(part["rows"].mean()),
                "mean_return_pct": float(part["mean_return_pct"].mean()),
                "median_return_pct": float(part["median_return_pct"].mean()),
                "win_rate": float(part["win_rate"].mean()),
                "avg_prediction": float(part["avg_prediction"].mean()),
                "min_prediction": float(part["min_prediction"].min()) if "min_prediction" in part.columns else np.nan,
                "max_prediction": float(part["max_prediction"].max()) if "max_prediction" in part.columns else np.nan,
            }
        )

    return pd.DataFrame(rows).sort_values("prediction_bucket").reset_index(drop=True)


def load_predictions(path: str | Path) -> pd.DataFrame:
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Predictions parquet not found: {path}")

    df = pd.read_parquet(path)
    _require_columns(df, ["symbol", "date", DEFAULT_PREDICTION_COL, DEFAULT_LABEL_COL])

    df = df.copy()
    df["date"] = _normalize_date(df["date"])
    return df


def load_optional_csv(path: str | Path | None) -> pd.DataFrame | None:
    if path is None:
        return None

    path = Path(path).expanduser().resolve()
    if not path.exists():
        return None

    return pd.read_csv(path)


def _extract_topk_returns(
    topk_summary: pd.DataFrame | None,
    benchmark_summary: pd.DataFrame | None,
    *,
    topk_values: Sequence[int] = DEFAULT_TOPK_VALUES,
) -> dict[int, float]:
    out: dict[int, float] = {}

    if topk_summary is not None and not topk_summary.empty:
        if {"top_k", "mean_daily_return_pct"}.issubset(topk_summary.columns):
            for _, row in topk_summary.iterrows():
                k = int(row["top_k"])
                if k in topk_values:
                    out[k] = float(row["mean_daily_return_pct"])

    if benchmark_summary is not None and not benchmark_summary.empty:
        if {"benchmark_name", "mean_daily_return_pct"}.issubset(benchmark_summary.columns):
            for k in topk_values:
                name = f"ml_top_{k}"
                hit = benchmark_summary[benchmark_summary["benchmark_name"].eq(name)]
                if not hit.empty:
                    out[k] = float(hit.iloc[0]["mean_daily_return_pct"])

    return out


def _extract_candidate_mean(
    benchmark_summary: pd.DataFrame | None,
    *,
    candidate_benchmark_name: str = "market_all",
) -> float | None:
    if benchmark_summary is None or benchmark_summary.empty:
        return None

    if {"benchmark_name", "mean_daily_return_pct"}.issubset(benchmark_summary.columns):
        hit = benchmark_summary[benchmark_summary["benchmark_name"].eq(candidate_benchmark_name)]
        if not hit.empty:
            return float(hit.iloc[0]["mean_daily_return_pct"])

    return None


def diagnose_prediction_gate(
    bucket_summary: pd.DataFrame,
    *,
    topk_summary: pd.DataFrame | None = None,
    benchmark_summary: pd.DataFrame | None = None,
    topk_values: Sequence[int] = DEFAULT_TOPK_VALUES,
    candidate_benchmark_name: str = "market_all",
) -> pd.DataFrame:
    _require_columns(bucket_summary, ["prediction_bucket", "mean_return_pct"])

    summary = bucket_summary.sort_values("prediction_bucket").copy()
    if summary.empty:
        raise ValueError("bucket_summary is empty.")

    bottom_bucket = int(summary["prediction_bucket"].min())
    top_bucket = int(summary["prediction_bucket"].max())

    bottom_return = float(summary.loc[summary["prediction_bucket"].eq(bottom_bucket), "mean_return_pct"].iloc[0])
    top_return = float(summary.loc[summary["prediction_bucket"].eq(top_bucket), "mean_return_pct"].iloc[0])
    top_bottom_spread = top_return - bottom_return

    low_group = summary[summary["prediction_bucket"].between(1, 3)]
    high_group = summary[summary["prediction_bucket"].between(8, 10)]

    low_group_return = float(low_group["mean_return_pct"].mean()) if not low_group.empty else np.nan
    high_group_return = float(high_group["mean_return_pct"].mean()) if not high_group.empty else np.nan
    high_low_spread = high_group_return - low_group_return if pd.notna(high_group_return) and pd.notna(low_group_return) else np.nan

    bucket_values = summary["mean_return_pct"].to_numpy(dtype=float)
    bucket_diffs = np.diff(bucket_values)
    all_bucket_increasing = bool(np.all(bucket_diffs >= 0)) if len(bucket_diffs) > 0 else False

    topk_returns = _extract_topk_returns(
        topk_summary,
        benchmark_summary,
        topk_values=topk_values,
    )
    candidate_mean = _extract_candidate_mean(
        benchmark_summary,
        candidate_benchmark_name=candidate_benchmark_name,
    )

    top10 = topk_returns.get(10)
    top20 = topk_returns.get(20)
    top50 = topk_returns.get(50)

    has_topk_10_20_50 = top10 is not None and top20 is not None and top50 is not None
    topk_monotonic = bool(top10 >= top20 >= top50) if has_topk_10_20_50 else False
    top10_beats_candidate = bool(top10 >= candidate_mean) if top10 is not None and candidate_mean is not None else False
    topk_gate_pass = topk_monotonic or top10_beats_candidate

    top_bucket_beats_bottom = bool(top_return > bottom_return)
    high_group_beats_low_group = bool(high_group_return > low_group_return) if pd.notna(high_low_spread) else False

    passed = bool(top_bucket_beats_bottom and high_group_beats_low_group and topk_gate_pass)

    if passed:
        reason = "passed"
        model_status = "passed"
    elif not top_bucket_beats_bottom or not high_group_beats_low_group:
        reason = "no_monotonic_prediction_edge"
        model_status = "failed"
    elif not topk_gate_pass:
        reason = "topk_gate_failed"
        model_status = "failed"
    else:
        reason = "failed_unknown"
        model_status = "failed"

    return pd.DataFrame(
        [
            {
                "model_status": model_status,
                "reason": reason,
                "bottom_bucket": bottom_bucket,
                "top_bucket": top_bucket,
                "bottom_bucket_return_pct": bottom_return,
                "top_bucket_return_pct": top_return,
                "top_bottom_spread_pct": top_bottom_spread,
                "low_1_3_mean_return_pct": low_group_return,
                "high_8_10_mean_return_pct": high_group_return,
                "high_low_spread_pct": high_low_spread,
                "all_bucket_increasing": all_bucket_increasing,
                "top10_return_pct": top10,
                "top20_return_pct": top20,
                "top50_return_pct": top50,
                "candidate_mean_return_pct": candidate_mean,
                "topk_monotonic_10_20_50": topk_monotonic,
                "top10_beats_candidate": top10_beats_candidate,
                "top_bucket_beats_bottom": top_bucket_beats_bottom,
                "high_group_beats_low_group": high_group_beats_low_group,
                "topk_gate_pass": topk_gate_pass,
            }
        ]
    )


def build_prediction_diagnostics(
    predictions: pd.DataFrame,
    *,
    topk_summary: pd.DataFrame | None = None,
    benchmark_summary: pd.DataFrame | None = None,
    prediction_col: str = DEFAULT_PREDICTION_COL,
    label_col: str = DEFAULT_LABEL_COL,
    bucket_count: int = DEFAULT_BUCKET_COUNT,
    min_daily_rows: int = 100,
    candidate_benchmark_name: str = "market_all",
) -> dict[str, pd.DataFrame]:
    detail = build_prediction_bucket_detail(
        predictions,
        prediction_col=prediction_col,
        label_col=label_col,
        bucket_count=bucket_count,
        min_daily_rows=min_daily_rows,
    )
    summary = summarize_prediction_buckets(detail)
    gate = diagnose_prediction_gate(
        summary,
        topk_summary=topk_summary,
        benchmark_summary=benchmark_summary,
        candidate_benchmark_name=candidate_benchmark_name,
    )

    return {
        "prediction_bucket_detail": detail,
        "prediction_bucket_summary": summary,
        "prediction_bucket_gate": gate,
    }


def write_prediction_diagnostics(
    diagnostics: dict[str, pd.DataFrame],
    *,
    output_dir: str | Path,
    output_name: str,
) -> dict[str, Path]:
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}
    for name, frame in diagnostics.items():
        path = output_dir / f"{output_name}_{name}.csv"
        frame.to_csv(path, index=False, encoding="utf-8-sig")
        paths[name] = path

    return paths
