from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def infer_bucket_col(bucket_summary: pd.DataFrame) -> str:
    candidates = [c for c in bucket_summary.columns if c.endswith("_bucket")]

    if not candidates:
        raise ValueError("Cannot infer bucket column. Expected a column ending with '_bucket'.")

    if len(candidates) > 1:
        raise ValueError(f"Multiple bucket columns found: {candidates}")

    return candidates[0]


def is_non_decreasing(values: list[float]) -> bool:
    return all(a <= b for a, b in zip(values, values[1:]))


def is_non_increasing(values: list[float]) -> bool:
    return all(a >= b for a, b in zip(values, values[1:]))


def diagnose_one_target(
    *,
    bucket_part: pd.DataFrame,
    ic_part: pd.DataFrame,
    bucket_col: str,
    target: str,
) -> dict[str, object]:
    part = bucket_part.sort_values(bucket_col).copy()

    bucket_values = part[bucket_col].tolist()
    mean_values = part["mean_return_pct"].tolist()

    bottom_row = part.iloc[0]
    top_row = part.iloc[-1]
    best_row = part.loc[part["mean_return_pct"].idxmax()]
    worst_row = part.loc[part["mean_return_pct"].idxmin()]

    non_decreasing = is_non_decreasing(mean_values)
    non_increasing = is_non_increasing(mean_values)

    if non_decreasing and not non_increasing:
        monotonic_pattern = "increasing"
    elif non_increasing and not non_decreasing:
        monotonic_pattern = "decreasing"
    elif non_decreasing and non_increasing:
        monotonic_pattern = "flat"
    else:
        monotonic_pattern = "non_monotonic"

    max_bucket = int(max(bucket_values))
    if max_bucket >= 6:
        middle_bucket_set = {4.0, 5.0, 6.0}
    else:
        midpoint = (min(bucket_values) + max(bucket_values)) / 2.0
        middle_bucket_set = {
            b for b in bucket_values
            if abs(b - midpoint) <= 1.0
        }

    edge_bucket_set = {min(bucket_values), max(bucket_values)}

    middle = part[part[bucket_col].isin(middle_bucket_set)]
    edges = part[part[bucket_col].isin(edge_bucket_set)]

    middle_mean = float(middle["mean_return_pct"].mean()) if not middle.empty else None
    edge_mean = float(edges["mean_return_pct"].mean()) if not edges.empty else None

    middle_minus_edges = (
        middle_mean - edge_mean
        if middle_mean is not None and edge_mean is not None
        else None
    )

    best_bucket = float(best_row[bucket_col])
    top_bottom_spread = float(top_row["mean_return_pct"] - bottom_row["mean_return_pct"])
    best_worst_spread = float(best_row["mean_return_pct"] - worst_row["mean_return_pct"])

    if monotonic_pattern == "increasing" and top_bottom_spread > 0:
        conclusion = "prefer_high"
    elif monotonic_pattern == "decreasing" and top_bottom_spread < 0:
        conclusion = "prefer_low"
    elif best_bucket in middle_bucket_set and middle_minus_edges is not None and middle_minus_edges > 0:
        conclusion = "prefer_middle"
    else:
        conclusion = "no_clear_pattern"

    return {
        "target": target,
        "bucket_count": int(part[bucket_col].nunique()),
        "sample_count": int(part["sample_count"].sum()),

        "ic_count": int(len(ic_part)),
        "spearman_ic_mean": float(ic_part["spearman_ic"].mean()) if not ic_part.empty else None,
        "spearman_ic_median": float(ic_part["spearman_ic"].median()) if not ic_part.empty else None,
        "spearman_ic_std": float(ic_part["spearman_ic"].std()) if not ic_part.empty else None,
        "pearson_ic_mean": float(ic_part["pearson_ic"].mean()) if not ic_part.empty else None,
        "pearson_ic_median": float(ic_part["pearson_ic"].median()) if not ic_part.empty else None,
        "pearson_ic_std": float(ic_part["pearson_ic"].std()) if not ic_part.empty else None,
        "daily_sample_count_mean": float(ic_part["sample_count"].mean()) if not ic_part.empty else None,

        "bottom_bucket": float(bottom_row[bucket_col]),
        "bottom_bucket_mean_return_pct": float(bottom_row["mean_return_pct"]),
        "top_bucket": float(top_row[bucket_col]),
        "top_bucket_mean_return_pct": float(top_row["mean_return_pct"]),
        "top_bottom_spread_pct": top_bottom_spread,

        "best_bucket": best_bucket,
        "best_bucket_mean_return_pct": float(best_row["mean_return_pct"]),
        "worst_bucket": float(worst_row[bucket_col]),
        "worst_bucket_mean_return_pct": float(worst_row["mean_return_pct"]),
        "best_worst_spread_pct": best_worst_spread,

        "middle_bucket_mean_return_pct": middle_mean,
        "edge_bucket_mean_return_pct": edge_mean,
        "middle_minus_edges_pct": middle_minus_edges,

        "monotonic_pattern": monotonic_pattern,
        "conclusion": conclusion,
    }


def diagnose_single_factor_result(
    *,
    bucket_summary_path: Path,
    daily_ic_path: Path,
    output_path: Path,
) -> pd.DataFrame:
    bucket_summary = pd.read_csv(bucket_summary_path)
    daily_ic = pd.read_csv(daily_ic_path)

    required_bucket_cols = {"target", "sample_count", "mean_return_pct"}
    missing_bucket = required_bucket_cols - set(bucket_summary.columns)
    if missing_bucket:
        raise ValueError(f"bucket_summary missing columns: {sorted(missing_bucket)}")

    required_ic_cols = {"target", "spearman_ic", "pearson_ic", "sample_count"}
    missing_ic = required_ic_cols - set(daily_ic.columns)
    if missing_ic:
        raise ValueError(f"daily_ic missing columns: {sorted(missing_ic)}")

    bucket_col = infer_bucket_col(bucket_summary)

    rows = []

    for target, bucket_part in bucket_summary.groupby("target", sort=True):
        ic_part = daily_ic[daily_ic["target"].eq(target)].copy()

        rows.append(
            diagnose_one_target(
                bucket_part=bucket_part,
                ic_part=ic_part,
                bucket_col=bucket_col,
                target=target,
            )
        )

    out = pd.DataFrame(rows)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False, encoding="utf-8-sig")

    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket-summary", required=True)
    parser.add_argument("--daily-ic", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    out = diagnose_single_factor_result(
        bucket_summary_path=Path(args.bucket_summary),
        daily_ic_path=Path(args.daily_ic),
        output_path=Path(args.output),
    )

    print(out.to_string(index=False))
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
