
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from research.candidate_pool.full_market_formulaic_candidate_pool import (
    DEFAULT_ROLE_ALLOWLIST,
    RuleCombination,
    apply_base_tradability_filters,
    assign_train_defined_bucket,
    build_rule_combinations,
    load_train_defined_edges,
)


@dataclass(frozen=True)
class ManualFactorRule:
    factor_name: str
    buckets: tuple[int, ...]
    bucket_rule: str
    valid_regimes: tuple[str, ...]
    weak_regimes: tuple[str, ...]
    factor_type_manual: str
    ml_baseline_role: str


def _bool_like(value) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _split_csv_like(value) -> tuple[str, ...]:
    text = "" if pd.isna(value) else str(value)
    items = [x.strip() for x in text.split(",") if x.strip()]
    return tuple(x for x in items if x.lower() not in {"none", "nan", "null"})


def _parse_bucket_rule(rule: str) -> tuple[int, ...]:
    text = "" if pd.isna(rule) else str(rule)
    bracket = re.search(r"\[(.*?)\]", text)
    source = bracket.group(1) if bracket else text
    values = [int(x) for x in re.findall(r"\d+", source)]
    return tuple(sorted(set(values)))


def _parse_csv_list(value: str | None) -> list[str] | None:
    if value is None or str(value).strip() == "":
        return None
    return [x.strip().lower() for x in str(value).split(",") if x.strip()]


def _parse_horizons(value: str | None) -> tuple[int, ...]:
    if value is None or str(value).strip() == "":
        return (3, 4)
    horizons = tuple(int(x.strip()) for x in str(value).split(",") if x.strip())
    if not horizons:
        raise ValueError("empty horizons")
    if any(h < 2 for h in horizons):
        raise ValueError("trade horizons must be >= 2 for T+1 open entry")
    return horizons


def parse_combinations(value: str | None) -> list[RuleCombination] | None:
    if value is None or str(value).strip() == "":
        return None

    combos: list[RuleCombination] = []
    for raw_part in str(value).split(";"):
        part = raw_part.strip()
        if not part:
            continue

        if ":" in part:
            raw_name, raw_factors = part.split(":", 1)
            factors = tuple(x.strip().lower() for x in re.split(r"[,+]", raw_factors) if x.strip())
            name = raw_name.strip()
        else:
            factors = tuple(x.strip().lower() for x in re.split(r"[,+]", part) if x.strip())
            name = "_and_".join(factors)

        if not factors:
            raise ValueError(f"empty combination definition: {raw_part}")

        combos.append(RuleCombination(name=name, factors=factors))

    return combos or None


def load_manual_filter_rules(
    manual_summary_path: Path,
    *,
    factor_names: list[str] | None = None,
    include_holdout: bool = False,
) -> tuple[list[ManualFactorRule], pd.DataFrame]:
    df = pd.read_csv(manual_summary_path)

    required = {
        "factor_name",
        "factor_type_manual",
        "bucket_rule",
        "valid_regime",
        "weak_regime",
        "is_filter_factor",
        "manual_override",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"manual summary missing required columns: {missing}")

    df = df.copy()
    df["factor_name"] = df["factor_name"].astype(str).str.strip().str.lower()

    requested = set(factor_names or [])
    if requested:
        missing_requested = sorted(requested - set(df["factor_name"]))
        if missing_requested:
            raise ValueError(f"requested factors not found in manual summary: {missing_requested}")
        df = df[df["factor_name"].isin(requested)].copy()

    rules: list[ManualFactorRule] = []
    audit_rows: list[dict] = []

    for _, row in df.iterrows():
        factor = str(row["factor_name"]).strip().lower()
        manual_type = str(row.get("factor_type_manual", "")).strip()
        bucket_rule = str(row.get("bucket_rule", "")).strip()
        buckets = _parse_bucket_rule(bucket_rule)
        valid_regimes = _split_csv_like(row.get("valid_regime", ""))
        weak_regimes = _split_csv_like(row.get("weak_regime", ""))
        role = str(row.get("ml_baseline_role", "")).strip()

        active = False
        reason = "active"

        if not _bool_like(row.get("manual_override", "")):
            reason = "manual_override is not True"
        elif manual_type == "reject":
            reason = "manual conclusion is reject"
        elif not _bool_like(row.get("is_filter_factor", "")):
            reason = "manual conclusion is not filter factor"
        elif not buckets:
            reason = "bucket_rule has no usable bucket"
        elif not valid_regimes:
            reason = "valid_regime is empty"
        elif not include_holdout and role not in DEFAULT_ROLE_ALLOWLIST:
            reason = f"ml_baseline_role {role} is not in default role allowlist"
        else:
            active = True
            rules.append(
                ManualFactorRule(
                    factor_name=factor,
                    buckets=buckets,
                    bucket_rule=bucket_rule,
                    valid_regimes=valid_regimes,
                    weak_regimes=weak_regimes,
                    factor_type_manual=manual_type,
                    ml_baseline_role=role,
                )
            )

        audit_rows.append(
            {
                "factor_name": factor,
                "factor_type_manual": manual_type,
                "bucket_rule": bucket_rule,
                "valid_regime": ",".join(valid_regimes),
                "weak_regime": ",".join(weak_regimes),
                "ml_baseline_role": role,
                "active": active,
                "reason": reason,
            }
        )

    audit = pd.DataFrame(audit_rows)

    if not rules:
        raise ValueError("No active manual filter rules.\n" + audit.to_string(index=False))

    return rules, audit


def load_regime_frame(regime_path: Path) -> pd.DataFrame:
    df = pd.read_csv(regime_path)
    if df.empty:
        raise ValueError(f"empty regime file: {regime_path}")

    date_col = "date" if "date" in df.columns else df.columns[0]
    regime_col = "market_regime" if "market_regime" in df.columns else "regime"
    if regime_col not in df.columns:
        raise ValueError(f"regime file missing market_regime/regime column: {regime_path}")

    out = df[[date_col, regime_col]].copy()
    out.columns = ["date", "market_regime"]
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    out = out.dropna(subset=["date", "market_regime"]).drop_duplicates("date", keep="last")
    return out


def _find_feature_file(factor_dir: Path, market_file: Path) -> Path | None:
    candidates = [
        factor_dir / market_file.name,
        factor_dir / f"{market_file.stem}.parquet",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _add_adjusted_aliases(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    mapping = {
        "adjusted_open": "open",
        "adjusted_high": "high",
        "adjusted_low": "low",
        "adjusted_close": "close",
        "adjusted_vwap": "vwap",
    }
    for source, target in mapping.items():
        if target not in out.columns and source in out.columns:
            out[target] = out[source]
    return out


def _add_trade_forward_returns(
    market: pd.DataFrame,
    *,
    horizons: tuple[int, ...],
    symbol_col: str = "symbol",
) -> pd.DataFrame:
    work = _add_adjusted_aliases(market)
    work = work.sort_values([symbol_col, "date"], kind="stable").reset_index(drop=True)

    for col in ["open", "close"]:
        if col not in work.columns:
            raise ValueError(f"market frame missing required price column after aliasing: {col}")
        work[col] = pd.to_numeric(work[col], errors="coerce")

    grouped = work.groupby(symbol_col, sort=False)
    entry_open = grouped["open"].shift(-1)

    for h in horizons:
        exit_close = grouped["close"].shift(-h)
        work[f"fwd_return_pct_T{h}"] = (exit_close / entry_open - 1.0) * 100.0

    return work


def load_full_market_analysis_frame(
    *,
    market_dir: Path,
    factor_dir: Path,
    factors: list[str],
    horizons: tuple[int, ...],
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    market_files = sorted(market_dir.glob("*.parquet"))
    parts: list[pd.DataFrame] = []

    start_ts = pd.Timestamp(start_date).normalize() if start_date else None
    end_ts = pd.Timestamp(end_date).normalize() if end_date else None

    for idx, market_file in enumerate(market_files, start=1):
        if idx == 1 or idx % 200 == 0 or idx == len(market_files):
            print(f"read combination input {idx}/{len(market_files)}")

        feature_file = _find_feature_file(factor_dir, market_file)
        if feature_file is None:
            continue

        market = pd.read_parquet(market_file)
        if "date" not in market.columns:
            continue

        market = market.copy()
        market["date"] = pd.to_datetime(market["date"], errors="coerce").dt.normalize()
        if "symbol" not in market.columns:
            market["symbol"] = market_file.stem

        market = _add_trade_forward_returns(market, horizons=horizons)

        feature = pd.read_parquet(feature_file)
        if "date" not in feature.columns:
            continue

        feature = feature.copy()
        feature["date"] = pd.to_datetime(feature["date"], errors="coerce").dt.normalize()
        if "symbol" not in feature.columns:
            feature["symbol"] = market_file.stem

        for factor in factors:
            if factor not in feature.columns:
                feature[factor] = pd.NA

        merged = market.merge(
            feature[["symbol", "date"] + factors],
            on=["symbol", "date"],
            how="left",
        )

        if start_ts is not None:
            merged = merged[merged["date"].ge(start_ts)]
        if end_ts is not None:
            merged = merged[merged["date"].le(end_ts)]

        if not merged.empty:
            parts.append(merged)

    if not parts:
        return pd.DataFrame(columns=["symbol", "date"] + factors)

    return pd.concat(parts, ignore_index=True)


def _dataset_from_year(year: pd.Series) -> pd.Series:
    return pd.Series(
        pd.NA,
        index=year.index,
        dtype="object",
    ).mask(
        year.between(2021, 2024),
        "train",
    ).mask(
        year.between(2025, 2026),
        "test",
    )


def _combo_valid_regimes(combo: RuleCombination, rule_by_factor: dict[str, ManualFactorRule]) -> tuple[str, ...]:
    regime_sets = [set(rule_by_factor[f].valid_regimes) for f in combo.factors]
    if not regime_sets:
        return tuple()
    return tuple(sorted(set.intersection(*regime_sets)))


def build_daily_combo_returns(
    universe: pd.DataFrame,
    *,
    combos: list[RuleCombination],
    rule_by_factor: dict[str, ManualFactorRule],
    targets: list[str],
) -> pd.DataFrame:
    base = universe[
        universe["is_tradable_base"]
        & universe["dataset"].isin(["train", "test"])
    ].copy()

    daily_parts: list[pd.DataFrame] = []

    for combo in combos:
        valid_regimes = _combo_valid_regimes(combo, rule_by_factor)

        combo_mask = base["market_regime"].isin(valid_regimes)
        for factor in combo.factors:
            combo_mask = combo_mask & base[f"{factor}_pass"].fillna(False)

        selected = base[combo_mask].copy()

        for target in targets:
            valid_base = base.dropna(subset=[target])
            valid_selected = selected.dropna(subset=[target])

            if valid_base.empty:
                continue

            group_cols = ["date", "dataset", "year", "market_regime"]

            universe_daily = (
                valid_base.groupby(group_cols, dropna=False)[target]
                .mean()
                .rename("universe_return_pct")
                .reset_index()
            )

            if valid_selected.empty:
                continue

            group_daily = (
                valid_selected.groupby(group_cols, dropna=False)
                .agg(
                    group_return_pct=(target, "mean"),
                    group_median_return_pct=(target, "median"),
                    group_up_ratio=(target, lambda s: (s > 0).mean()),
                    candidate_count=("symbol", "nunique"),
                )
                .reset_index()
            )

            daily = group_daily.merge(universe_daily, on=group_cols, how="left")
            daily["excess_return_pct"] = daily["group_return_pct"] - daily["universe_return_pct"]
            daily["target"] = target
            daily["combined_rule_name"] = combo.name
            daily["combined_rule_factors"] = ",".join(combo.factors)
            daily["valid_regime_intersection"] = ",".join(valid_regimes)

            daily_parts.append(daily)

    if not daily_parts:
        return pd.DataFrame()

    return pd.concat(daily_parts, ignore_index=True)


def summarize_daily_returns(daily: pd.DataFrame, by_cols: list[str]) -> pd.DataFrame:
    columns = [
        "target",
        "combined_rule_name",
        *by_cols,
        "trading_days",
        "avg_candidate_count",
        "mean_return_pct",
        "median_return_pct",
        "mean_universe_return_pct",
        "mean_excess_return_pct",
        "median_excess_return_pct",
        "excess_win_ratio",
        "avg_up_ratio",
    ]

    if daily.empty:
        return pd.DataFrame(columns=columns)

    out = (
        daily.groupby(["target", "combined_rule_name"] + by_cols, dropna=False)
        .agg(
            trading_days=("date", "nunique"),
            avg_candidate_count=("candidate_count", "mean"),
            mean_return_pct=("group_return_pct", "mean"),
            median_return_pct=("group_median_return_pct", "median"),
            mean_universe_return_pct=("universe_return_pct", "mean"),
            mean_excess_return_pct=("excess_return_pct", "mean"),
            median_excess_return_pct=("excess_return_pct", "median"),
            excess_win_ratio=("excess_return_pct", lambda s: (s > 0).mean()),
            avg_up_ratio=("group_up_ratio", "mean"),
        )
        .reset_index()
    )
    return out



def build_recency_period_daily(
    daily: pd.DataFrame,
    *,
    recent_trading_days: int = 120,
) -> pd.DataFrame:
    if daily.empty:
        return pd.DataFrame(columns=list(daily.columns) + ["period"])

    work = daily.copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce").dt.normalize()
    work["year"] = pd.to_numeric(work["year"], errors="coerce")

    parts: list[pd.DataFrame] = []

    ytd_2026 = work[work["year"].eq(2026)].copy()
    if not ytd_2026.empty:
        ytd_2026["period"] = "2026_ytd"
        parts.append(ytd_2026)

    unique_dates = (
        work["date"]
        .dropna()
        .drop_duplicates()
        .sort_values()
    )

    if recent_trading_days <= 0:
        raise ValueError("recent_trading_days must be positive")

    recent_dates = set(unique_dates.tail(recent_trading_days))
    recent = work[work["date"].isin(recent_dates)].copy()
    if not recent.empty:
        recent["period"] = f"recent_{recent_trading_days}d"
        parts.append(recent)

    if not parts:
        return pd.DataFrame(columns=list(work.columns) + ["period"])

    return pd.concat(parts, ignore_index=True)





def build_market_recent_period_daily(
    daily: pd.DataFrame,
    *,
    market_dates: pd.Series,
    windows: tuple[int, ...] = (20, 40, 60, 80, 120),
) -> pd.DataFrame:
    if daily.empty:
        return pd.DataFrame(columns=list(daily.columns) + ["period"])

    unique_market_dates = (
        pd.to_datetime(market_dates, errors="coerce")
        .dropna()
        .dt.normalize()
        .drop_duplicates()
        .sort_values()
    )

    if unique_market_dates.empty:
        return pd.DataFrame(columns=list(daily.columns) + ["period"])

    if any(w <= 0 for w in windows):
        raise ValueError("market recent windows must be positive")

    work = daily.copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce").dt.normalize()

    parts: list[pd.DataFrame] = []

    for window in windows:
        recent_dates = set(unique_market_dates.tail(window))
        part = work[work["date"].isin(recent_dates)].copy()
        if part.empty:
            continue
        part["period"] = f"market_recent_{window}d"
        parts.append(part)

    if not parts:
        return pd.DataFrame(columns=list(work.columns) + ["period"])

    return pd.concat(parts, ignore_index=True)

def _metric_value(
    df: pd.DataFrame,
    *,
    combined_rule_name: str,
    target: str,
    filters: dict[str, object],
    metric_col: str,
) -> float:
    if df.empty or metric_col not in df.columns:
        return float("nan")

    mask = (
        df["combined_rule_name"].astype(str).eq(combined_rule_name)
        & df["target"].astype(str).eq(target)
    )

    for col, value in filters.items():
        if col not in df.columns:
            return float("nan")
        mask = mask & df[col].astype(str).eq(str(value))

    hit = df.loc[mask, metric_col]
    if hit.empty:
        return float("nan")

    return float(pd.to_numeric(hit, errors="coerce").mean())


def _is_pos(value: float) -> bool:
    return pd.notna(value) and value > 0.0


def _is_non_negative(value: float) -> bool:
    return pd.notna(value) and value >= 0.0


def _is_win_ok(value: float) -> bool:
    return pd.notna(value) and value >= 0.50


def build_formulaic_combination_conclusion_summary(
    *,
    train_test_summary: pd.DataFrame,
    recency_period_summary: pd.DataFrame,
    market_recent_period_summary: pd.DataFrame | None = None,
    recent_trading_days: int = 120,
    market_recent_windows: tuple[int, ...] = (20, 40, 60, 80, 120),
    min_candidate_count: int = 10,
    max_candidate_count: int = 80,
    clear_negative_mean_excess_pct: float = -0.03,
) -> pd.DataFrame:
    recent_period = f"recent_{recent_trading_days}d"
    market_recent_period_summary = (
        pd.DataFrame()
        if market_recent_period_summary is None
        else market_recent_period_summary
    )
    short_gate_windows = {20, 40}

    combo_names = sorted(
        set(train_test_summary.get("combined_rule_name", pd.Series(dtype="object")).astype(str))
        | set(recency_period_summary.get("combined_rule_name", pd.Series(dtype="object")).astype(str))
        | set(market_recent_period_summary.get("combined_rule_name", pd.Series(dtype="object")).astype(str))
    )
    combo_names = [x for x in combo_names if x and x.lower() != "nan"]

    targets = sorted(
        set(train_test_summary.get("target", pd.Series(dtype="object")).astype(str))
        | set(recency_period_summary.get("target", pd.Series(dtype="object")).astype(str))
        | set(market_recent_period_summary.get("target", pd.Series(dtype="object")).astype(str))
    )
    targets = [t for t in targets if t in {"fwd_return_pct_T3", "fwd_return_pct_T4"}]

    if not targets:
        targets = ["fwd_return_pct_T3", "fwd_return_pct_T4"]

    rows: list[dict] = []

    for combo in combo_names:
        row: dict[str, object] = {
            "combined_rule_name": combo,
            "target_set": ",".join(targets),
            "recent_period": recent_period,
            "min_candidate_count": min_candidate_count,
            "max_candidate_count": max_candidate_count,
            "clear_negative_mean_excess_pct": clear_negative_mean_excess_pct,
            "market_recent_windows": ",".join(str(x) for x in market_recent_windows),
            "short_gate_windows": ",".join(str(x) for x in sorted(short_gate_windows)),
        }

        train_test_mean_pass = 0
        ytd_mean_pass = 0
        recent_mean_pass = 0
        recent_median_pass = 0
        recent_win_pass = 0

        recent_mean_values: list[float] = []
        ytd_mean_values: list[float] = []
        recent_candidate_counts: list[float] = []
        short_window_clear_negative_hits: list[str] = []

        for target in targets:
            suffix = target.replace("fwd_return_pct_", "")

            train_mean = _metric_value(
                train_test_summary,
                combined_rule_name=combo,
                target=target,
                filters={"dataset": "train"},
                metric_col="mean_excess_return_pct",
            )
            test_mean = _metric_value(
                train_test_summary,
                combined_rule_name=combo,
                target=target,
                filters={"dataset": "test"},
                metric_col="mean_excess_return_pct",
            )
            ytd_mean = _metric_value(
                recency_period_summary,
                combined_rule_name=combo,
                target=target,
                filters={"period": "2026_ytd"},
                metric_col="mean_excess_return_pct",
            )
            recent_mean = _metric_value(
                recency_period_summary,
                combined_rule_name=combo,
                target=target,
                filters={"period": recent_period},
                metric_col="mean_excess_return_pct",
            )
            recent_median = _metric_value(
                recency_period_summary,
                combined_rule_name=combo,
                target=target,
                filters={"period": recent_period},
                metric_col="median_excess_return_pct",
            )
            recent_win = _metric_value(
                recency_period_summary,
                combined_rule_name=combo,
                target=target,
                filters={"period": recent_period},
                metric_col="excess_win_ratio",
            )
            recent_count = _metric_value(
                recency_period_summary,
                combined_rule_name=combo,
                target=target,
                filters={"period": recent_period},
                metric_col="avg_candidate_count",
            )

            row[f"train_{suffix}_mean_excess"] = train_mean
            row[f"test_{suffix}_mean_excess"] = test_mean
            row[f"ytd_2026_{suffix}_mean_excess"] = ytd_mean
            row[f"{recent_period}_{suffix}_mean_excess"] = recent_mean
            row[f"{recent_period}_{suffix}_median_excess"] = recent_median
            row[f"{recent_period}_{suffix}_win_ratio"] = recent_win
            row[f"{recent_period}_{suffix}_avg_candidate_count"] = recent_count

            for window in market_recent_windows:
                market_period = f"market_recent_{window}d"

                market_recent_mean = _metric_value(
                    market_recent_period_summary,
                    combined_rule_name=combo,
                    target=target,
                    filters={"period": market_period},
                    metric_col="mean_excess_return_pct",
                )
                market_recent_median = _metric_value(
                    market_recent_period_summary,
                    combined_rule_name=combo,
                    target=target,
                    filters={"period": market_period},
                    metric_col="median_excess_return_pct",
                )
                market_recent_win = _metric_value(
                    market_recent_period_summary,
                    combined_rule_name=combo,
                    target=target,
                    filters={"period": market_period},
                    metric_col="excess_win_ratio",
                )
                market_recent_count = _metric_value(
                    market_recent_period_summary,
                    combined_rule_name=combo,
                    target=target,
                    filters={"period": market_period},
                    metric_col="avg_candidate_count",
                )

                row[f"{market_period}_{suffix}_mean_excess"] = market_recent_mean
                row[f"{market_period}_{suffix}_median_excess"] = market_recent_median
                row[f"{market_period}_{suffix}_win_ratio"] = market_recent_win
                row[f"{market_period}_{suffix}_avg_candidate_count"] = market_recent_count

                if (
                    window in short_gate_windows
                    and pd.notna(market_recent_mean)
                    and market_recent_mean <= clear_negative_mean_excess_pct
                ):
                    short_window_clear_negative_hits.append(
                        f"{market_period}_{suffix}_mean_excess={market_recent_mean:.6f}"
                    )

            if _is_pos(train_mean) and _is_pos(test_mean):
                train_test_mean_pass += 1
            if _is_pos(ytd_mean):
                ytd_mean_pass += 1
            if _is_pos(recent_mean):
                recent_mean_pass += 1
            if _is_non_negative(recent_median):
                recent_median_pass += 1
            if _is_win_ok(recent_win):
                recent_win_pass += 1

            if pd.notna(recent_mean):
                recent_mean_values.append(recent_mean)
            if pd.notna(ytd_mean):
                ytd_mean_values.append(ytd_mean)
            if pd.notna(recent_count):
                recent_candidate_counts.append(recent_count)

        target_count = len(targets)

        candidate_count_mean = (
            sum(recent_candidate_counts) / len(recent_candidate_counts)
            if recent_candidate_counts
            else float("nan")
        )

        row["target_count"] = target_count
        row["train_test_mean_pass_count"] = train_test_mean_pass
        row["ytd_2026_mean_pass_count"] = ytd_mean_pass
        row[f"{recent_period}_mean_pass_count"] = recent_mean_pass
        row[f"{recent_period}_median_pass_count"] = recent_median_pass
        row[f"{recent_period}_win_pass_count"] = recent_win_pass
        row[f"{recent_period}_candidate_count_mean"] = candidate_count_mean

        candidate_count_ok = (
            pd.notna(candidate_count_mean)
            and candidate_count_mean >= min_candidate_count
            and candidate_count_mean <= max_candidate_count
        )

        recent_clear_negative = (
            len(recent_mean_values) == target_count
            and (
                all(x <= 0 for x in recent_mean_values)
                or any(x <= clear_negative_mean_excess_pct for x in recent_mean_values)
            )
        )
        ytd_clear_negative = (
            len(ytd_mean_values) == target_count
            and (
                all(x <= 0 for x in ytd_mean_values)
                or any(x <= clear_negative_mean_excess_pct for x in ytd_mean_values)
            )
        )

        short_window_clear_negative = len(short_window_clear_negative_hits) > 0
        row["short_window_clear_negative"] = short_window_clear_negative
        row["short_window_clear_negative_detail"] = (
            ";".join(short_window_clear_negative_hits)
            if short_window_clear_negative_hits
            else "none"
        )

        pass_ok = (
            train_test_mean_pass == target_count
            and ytd_mean_pass == target_count
            and recent_mean_pass == target_count
            and recent_median_pass == target_count
            and recent_win_pass == target_count
            and candidate_count_ok
            and not short_window_clear_negative
        )

        reasons: list[str] = []

        if not candidate_count_ok:
            reasons.append("candidate_count_out_of_bounds")
        if train_test_mean_pass < target_count:
            reasons.append("train_test_mean_not_all_positive")
        if ytd_mean_pass < target_count:
            reasons.append("2026_ytd_mean_not_all_positive")
        if recent_mean_pass < target_count:
            reasons.append(f"{recent_period}_mean_not_all_positive")
        if recent_median_pass < target_count:
            reasons.append(f"{recent_period}_median_negative")
        if recent_win_pass < target_count:
            reasons.append(f"{recent_period}_win_ratio_below_0_50")
        if recent_clear_negative:
            reasons.append(f"{recent_period}_clear_negative")
        if ytd_clear_negative:
            reasons.append("2026_ytd_clear_negative")
        if short_window_clear_negative:
            reasons.append("market_recent_20d_40d_clear_negative")

        if pass_ok:
            conclusion = "PASS"
        elif (
            candidate_count_ok
            and train_test_mean_pass == target_count
            and ytd_mean_pass == target_count
            and recent_mean_pass >= target_count - 1
            and not recent_clear_negative
            and not ytd_clear_negative
            and not short_window_clear_negative
        ):
            conclusion = "WATCH"
        else:
            conclusion = "REJECT"

        row["auto_conclusion"] = conclusion
        row["is_live_eligible"] = conclusion == "PASS"
        row["is_watchlist_eligible"] = conclusion in {"PASS", "WATCH"}
        row["reject_or_watch_reasons"] = ";".join(reasons) if reasons else "none"

        rows.append(row)

    return pd.DataFrame(rows).sort_values("combined_rule_name").reset_index(drop=True)


def build_full_market_formulaic_combination_test(
    *,
    market_dir: Path,
    factor_dir: Path,
    research_root: Path,
    manual_summary_path: Path,
    regime_path: Path,
    output_dir: Path,
    factor_names: list[str] | None = None,
    combinations: list[RuleCombination] | None = None,
    horizons: tuple[int, ...] = (3, 4),
    start_date: str | None = "2021-01-01",
    end_date: str | None = "2026-12-31",
    min_amount: float = 0.0,
    include_holdout: bool = False,
    exclude_one_word_limit_up: bool = True,
    output_prefix: str = "full_market_formulaic_combination_test",
    recent_trading_days: int = 120,
    min_candidate_count: int = 10,
    max_candidate_count: int = 80,
) -> dict[str, pd.DataFrame]:
    rules, rule_audit = load_manual_filter_rules(
        manual_summary_path,
        factor_names=factor_names,
        include_holdout=include_holdout,
    )

    active_factors = [r.factor_name for r in rules]
    rule_by_factor = {r.factor_name: r for r in rules}

    combos = build_rule_combinations(active_factors, combinations)

    missing = sorted({f for combo in combos for f in combo.factors} - set(active_factors))
    if missing:
        raise ValueError(f"combination references inactive/missing factors: {missing}")

    frame = load_full_market_analysis_frame(
        market_dir=market_dir,
        factor_dir=factor_dir,
        factors=active_factors,
        horizons=horizons,
        start_date=start_date,
        end_date=end_date,
    )

    if frame.empty:
        raise ValueError("empty analysis frame")

    regime = load_regime_frame(regime_path)
    frame = frame.merge(regime, on="date", how="left")
    frame["market_regime"] = frame["market_regime"].fillna("unknown")

    frame = apply_base_tradability_filters(
        frame,
        min_amount=min_amount,
        exclude_one_word_limit_up=exclude_one_word_limit_up,
    )

    frame["year"] = frame["date"].dt.year
    frame["dataset"] = _dataset_from_year(frame["year"])

    edges_by_factor = {
        factor: load_train_defined_edges(research_root, factor)
        for factor in active_factors
    }

    for factor in active_factors:
        bucket_col = f"{factor}_train_defined_bucket"
        pass_col = f"{factor}_pass"

        frame[bucket_col] = assign_train_defined_bucket(frame[factor], edges_by_factor[factor])
        frame[pass_col] = frame[bucket_col].isin(rule_by_factor[factor].buckets)

    targets = [f"fwd_return_pct_T{h}" for h in horizons]

    daily = build_daily_combo_returns(
        frame,
        combos=combos,
        rule_by_factor=rule_by_factor,
        targets=targets,
    )

    train_test_summary = summarize_daily_returns(daily, ["dataset"])
    train_test_regime_summary = summarize_daily_returns(daily, ["dataset", "market_regime"])
    yearly_summary = summarize_daily_returns(daily, ["dataset", "year"])

    recency_daily = build_recency_period_daily(
        daily,
        recent_trading_days=recent_trading_days,
    )
    recency_period_summary = summarize_daily_returns(recency_daily, ["period"])
    recency_period_regime_summary = summarize_daily_returns(
        recency_daily,
        ["period", "market_regime"],
    )

    market_recent_daily = build_market_recent_period_daily(
        daily,
        market_dates=frame["date"],
        windows=(20, 40, 60, 80, 120),
    )
    market_recent_period_summary = summarize_daily_returns(market_recent_daily, ["period"])
    market_recent_period_regime_summary = summarize_daily_returns(
        market_recent_daily,
        ["period", "market_regime"],
    )

    combination_conclusion_summary = build_formulaic_combination_conclusion_summary(
        train_test_summary=train_test_summary,
        recency_period_summary=recency_period_summary,
        market_recent_period_summary=market_recent_period_summary,
        recent_trading_days=recent_trading_days,
        market_recent_windows=(20, 40, 60, 80, 120),
        min_candidate_count=min_candidate_count,
        max_candidate_count=max_candidate_count,
    )

    if daily.empty:
        candidate_count = pd.DataFrame()
    else:
        candidate_count = (
            daily.groupby(["date", "market_regime", "combined_rule_name"], dropna=False)
            .agg(avg_candidate_count=("candidate_count", "mean"))
            .reset_index()
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "rule_audit": output_dir / f"{output_prefix}_rule_audit.csv",
        "daily": output_dir / f"{output_prefix}_daily.parquet",
        "train_test_summary": output_dir / f"{output_prefix}_train_test_summary.csv",
        "train_test_regime_summary": output_dir / f"{output_prefix}_train_test_regime_summary.csv",
        "yearly_summary": output_dir / f"{output_prefix}_yearly_summary.csv",
        "candidate_daily_count": output_dir / f"{output_prefix}_candidate_daily_count.csv",
        "recency_period_summary": output_dir / f"{output_prefix}_recency_period_summary.csv",
        "recency_period_regime_summary": output_dir / f"{output_prefix}_recency_period_regime_summary.csv",
        "market_recent_period_summary": output_dir / f"{output_prefix}_market_recent_period_summary.csv",
        "market_recent_period_regime_summary": output_dir / f"{output_prefix}_market_recent_period_regime_summary.csv",
        "combination_conclusion_summary": output_dir / "manual_formulaic_combination_conclusion_summary.csv",
    }

    rule_audit.to_csv(paths["rule_audit"], index=False, encoding="utf-8-sig")
    daily.to_parquet(paths["daily"], index=False)
    train_test_summary.to_csv(paths["train_test_summary"], index=False, encoding="utf-8-sig")
    train_test_regime_summary.to_csv(paths["train_test_regime_summary"], index=False, encoding="utf-8-sig")
    yearly_summary.to_csv(paths["yearly_summary"], index=False, encoding="utf-8-sig")
    candidate_count.to_csv(paths["candidate_daily_count"], index=False, encoding="utf-8-sig")
    recency_period_summary.to_csv(paths["recency_period_summary"], index=False, encoding="utf-8-sig")
    recency_period_regime_summary.to_csv(paths["recency_period_regime_summary"], index=False, encoding="utf-8-sig")
    market_recent_period_summary.to_csv(paths["market_recent_period_summary"], index=False, encoding="utf-8-sig")
    market_recent_period_regime_summary.to_csv(paths["market_recent_period_regime_summary"], index=False, encoding="utf-8-sig")
    combination_conclusion_summary.to_csv(paths["combination_conclusion_summary"], index=False, encoding="utf-8-sig")

    print("saved:", paths["rule_audit"])
    print("saved:", paths["daily"])
    print("saved:", paths["train_test_summary"])
    print("saved:", paths["train_test_regime_summary"])
    print("saved:", paths["yearly_summary"])
    print("saved:", paths["candidate_daily_count"])
    print("saved:", paths["recency_period_summary"])
    print("saved:", paths["recency_period_regime_summary"])
    print("saved:", paths["market_recent_period_summary"])
    print("saved:", paths["market_recent_period_regime_summary"])
    print("saved:", paths["combination_conclusion_summary"])
    print("active_factors:", active_factors)
    print("combinations:", [c.name for c in combos])
    print("horizons:", horizons)
    print("daily_rows:", len(daily))

    return {
        "rule_audit": rule_audit,
        "daily": daily,
        "train_test_summary": train_test_summary,
        "train_test_regime_summary": train_test_regime_summary,
        "yearly_summary": yearly_summary,
        "candidate_daily_count": candidate_count,
        "recency_period_summary": recency_period_summary,
        "recency_period_regime_summary": recency_period_regime_summary,
        "market_recent_period_summary": market_recent_period_summary,
        "market_recent_period_regime_summary": market_recent_period_regime_summary,
        "combination_conclusion_summary": combination_conclusion_summary,
    }


def parse_factor_names(value: str | None) -> list[str] | None:
    return _parse_csv_list(value)


def parse_horizons(value: str | None) -> tuple[int, ...]:
    return _parse_horizons(value)
