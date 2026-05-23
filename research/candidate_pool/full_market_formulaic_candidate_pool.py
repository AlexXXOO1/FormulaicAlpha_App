
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


DEFAULT_AUTO_COMBINATIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("alpha_005_only", ("alpha_005",)),
    ("alpha_005_and_alpha_002", ("alpha_005", "alpha_002")),
    ("alpha_005_and_alpha_006", ("alpha_005", "alpha_006")),
    ("alpha_005_and_alpha_002_and_alpha_006", ("alpha_005", "alpha_002", "alpha_006")),
)

DEFAULT_ROLE_ALLOWLIST = {
    "eligible_filter_candidate",
    "eligible_low_confidence_filter",
}


@dataclass(frozen=True)
class FactorBucketRule:
    factor_name: str
    buckets: tuple[int, ...]
    bucket_rule: str
    valid_regimes: tuple[str, ...]
    weak_regimes: tuple[str, ...]
    factor_type_manual: str
    ml_baseline_role: str


@dataclass(frozen=True)
class RuleCombination:
    name: str
    factors: tuple[str, ...]


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


def _normalise_factor_name(name: str) -> str:
    return str(name).strip().lower()


def _normalise_combo_name(factors: Iterable[str]) -> str:
    return "_and_".join(_normalise_factor_name(x) for x in factors)


def parse_factor_list(value: str | None) -> list[str] | None:
    if value is None or str(value).strip() == "":
        return None
    return [_normalise_factor_name(x) for x in str(value).split(",") if x.strip()]


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
            factors = tuple(_normalise_factor_name(x) for x in re.split(r"[,+]", raw_factors) if x.strip())
            name = raw_name.strip()
        else:
            factors = tuple(_normalise_factor_name(x) for x in re.split(r"[,+]", part) if x.strip())
            name = _normalise_combo_name(factors)

        if not factors:
            raise ValueError(f"empty combination definition: {raw_part}")

        combos.append(RuleCombination(name=name, factors=factors))

    if not combos:
        raise ValueError("no usable combinations parsed")
    return combos


def load_target_regime(regime_path: Path, target_date: str) -> str:
    df = pd.read_csv(regime_path)
    if df.empty:
        raise ValueError(f"empty regime file: {regime_path}")

    date_col = "date" if "date" in df.columns else df.columns[0]
    regime_col = "market_regime" if "market_regime" in df.columns else "regime"

    if regime_col not in df.columns:
        raise ValueError(f"regime file missing market_regime/regime column: {regime_path}")

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce").dt.normalize()
    target_ts = pd.Timestamp(target_date).normalize()

    hit = df[df[date_col].eq(target_ts)]
    if hit.empty:
        raise ValueError(f"target_date not found in regime file: {target_date}")

    return str(hit.iloc[-1][regime_col]).strip()


def load_manual_factor_rules(
    manual_summary_path: Path,
    *,
    target_regime: str,
    factor_names: list[str] | None = None,
    include_holdout: bool = False,
) -> tuple[list[FactorBucketRule], pd.DataFrame]:
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
    df["factor_name"] = df["factor_name"].map(_normalise_factor_name)

    requested = set(factor_names or [])
    if requested:
        missing_requested = sorted(requested - set(df["factor_name"]))
        if missing_requested:
            raise ValueError(f"requested factors not found in manual summary: {missing_requested}")
        df = df[df["factor_name"].isin(requested)].copy()

    rules: list[FactorBucketRule] = []
    audit_rows: list[dict] = []

    for _, row in df.iterrows():
        factor = _normalise_factor_name(row["factor_name"])
        manual_type = str(row.get("factor_type_manual", "")).strip()
        bucket_rule = str(row.get("bucket_rule", "")).strip()
        valid_regimes = _split_csv_like(row.get("valid_regime", ""))
        weak_regimes = _split_csv_like(row.get("weak_regime", ""))
        buckets = _parse_bucket_rule(bucket_rule)
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
        elif target_regime not in valid_regimes:
            reason = f"target_regime {target_regime} not in valid_regime"
        elif not include_holdout and role not in DEFAULT_ROLE_ALLOWLIST:
            reason = f"ml_baseline_role {role} is not in default role allowlist"
        else:
            active = True
            rules.append(
                FactorBucketRule(
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
                "target_regime": target_regime,
                "active": active,
                "reason": reason,
            }
        )

    audit = pd.DataFrame(audit_rows)

    if not rules:
        raise ValueError(
            "No active factor rules for this target_regime. "
            f"target_regime={target_regime}\n{audit.to_string(index=False)}"
        )

    return rules, audit


def load_train_defined_edges(research_root: Path, factor_name: str) -> pd.DataFrame:
    factor_name = _normalise_factor_name(factor_name)
    path = research_root / f"{factor_name}_research" / f"step6_{factor_name}_train_bucket_edges.csv"

    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"empty train-defined bucket edge file: {path}")

    colmap = {c.lower(): c for c in df.columns}

    bucket_col = None
    for candidate in ["bucket", f"{factor_name}_bucket", "train_defined_bucket"]:
        if candidate in colmap:
            bucket_col = colmap[candidate]
            break

    left_col = right_col = None
    for left_name, right_name in [
        ("left_edge", "right_edge"),
        ("bucket_left", "bucket_right"),
        ("lower_edge", "upper_edge"),
        ("lower", "upper"),
        ("min_factor", "max_factor"),
        ("min_value", "max_value"),
        ("factor_min", "factor_max"),
    ]:
        if left_name in colmap and right_name in colmap:
            left_col = colmap[left_name]
            right_col = colmap[right_name]
            break

    if bucket_col is None or left_col is None or right_col is None:
        raise ValueError(
            f"Unsupported train bucket edge schema for {factor_name}. "
            f"columns={list(df.columns)}"
        )

    edges = df[[bucket_col, left_col, right_col]].copy()
    edges.columns = ["bucket", "left_edge", "right_edge"]

    edges["bucket"] = pd.to_numeric(edges["bucket"], errors="coerce")
    edges["left_edge"] = pd.to_numeric(edges["left_edge"], errors="coerce")
    edges["right_edge"] = pd.to_numeric(edges["right_edge"], errors="coerce")
    edges = edges.dropna(subset=["bucket"]).sort_values("bucket").reset_index(drop=True)
    edges["bucket"] = edges["bucket"].astype(int)

    if edges.empty:
        raise ValueError(f"no usable bucket edges for {factor_name}: {path}")

    edges.loc[edges.index[0], "left_edge"] = float("-inf")
    edges.loc[edges.index[-1], "right_edge"] = float("inf")

    return edges


def assign_train_defined_bucket(values: pd.Series, edges: pd.DataFrame) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    out = pd.Series(pd.NA, index=values.index, dtype="Float64")

    for i, row in edges.iterrows():
        left = float(row["left_edge"])
        right = float(row["right_edge"])
        bucket = int(row["bucket"])

        if i == 0:
            mask = numeric.ge(left) & numeric.le(right)
        else:
            mask = numeric.gt(left) & numeric.le(right)

        out.loc[mask] = float(bucket)

    return out.astype("float")


def _find_feature_file(factor_dir: Path, market_file: Path) -> Path | None:
    candidates = [
        factor_dir / market_file.name,
        factor_dir / f"{market_file.stem}.parquet",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _add_adjusted_price_aliases(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    alias_map = {
        "adjusted_open": "open",
        "adjusted_high": "high",
        "adjusted_low": "low",
        "adjusted_close": "close",
        "adjusted_vwap": "vwap",
    }

    for source, alias in alias_map.items():
        if alias not in out.columns and source in out.columns:
            out[alias] = out[source]

    return out


def _append_reason(reason: pd.Series, mask: pd.Series, text: str) -> pd.Series:
    if not mask.any():
        return reason

    existing = reason.loc[mask].fillna("")
    reason.loc[mask] = existing.where(existing.eq(""), existing + ";") + text
    return reason


def apply_base_tradability_filters(
    df: pd.DataFrame,
    *,
    min_amount: float = 0.0,
    exclude_one_word_limit_up: bool = True,
) -> pd.DataFrame:
    work = _add_adjusted_price_aliases(df)

    reason = pd.Series("", index=work.index, dtype="object")

    for col in ["open", "high", "low", "close", "volume", "amount", "returns", "daily_return_pct"]:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")

    if "close" in work.columns:
        reason = _append_reason(reason, work["close"].isna() | work["close"].le(0), "invalid_close")

    if "volume" in work.columns:
        reason = _append_reason(reason, work["volume"].isna() | work["volume"].le(0), "invalid_volume")

    if "amount" in work.columns:
        reason = _append_reason(reason, work["amount"].isna() | work["amount"].le(0), "invalid_amount")
        if min_amount > 0:
            reason = _append_reason(reason, work["amount"].lt(min_amount), "below_min_amount")

    if "is_st" in work.columns:
        reason = _append_reason(reason, work["is_st"].map(_bool_like), "st_stock")

    if "stock_name" in work.columns:
        st_mask = work["stock_name"].astype(str).str.upper().str.contains("ST", na=False)
        reason = _append_reason(reason, st_mask, "st_stock")

    if exclude_one_word_limit_up:
        price_cols = {"open", "high", "low", "close"}
        if price_cols.issubset(work.columns):
            one_word = (
                work["open"].eq(work["high"])
                & work["high"].eq(work["low"])
                & work["low"].eq(work["close"])
            )

            if "daily_return_pct" in work.columns:
                limit_up = work["daily_return_pct"].ge(9.5)
            elif "returns" in work.columns:
                limit_up = work["returns"].ge(0.095)
            else:
                limit_up = pd.Series(False, index=work.index)

            reason = _append_reason(reason, one_word & limit_up, "one_word_limit_up")

    work["base_reject_reason"] = reason.mask(reason.eq(""), "pass")
    work["is_tradable_base"] = work["base_reject_reason"].eq("pass")

    return work


def load_universe_for_date(
    *,
    market_dir: Path,
    factor_dir: Path,
    target_date: str,
    active_factors: list[str],
) -> pd.DataFrame:
    target_ts = pd.Timestamp(target_date).normalize()
    market_files = sorted(market_dir.glob("*.parquet"))

    parts: list[pd.DataFrame] = []

    for idx, market_file in enumerate(market_files, start=1):
        if idx == 1 or idx % 200 == 0 or idx == len(market_files):
            print(f"read candidate universe {idx}/{len(market_files)}")

        feature_file = _find_feature_file(factor_dir, market_file)
        if feature_file is None:
            continue

        market = pd.read_parquet(market_file)
        if "date" not in market.columns:
            continue

        market = market.copy()
        market["date"] = pd.to_datetime(market["date"], errors="coerce").dt.normalize()
        market = market[market["date"].eq(target_ts)].copy()

        if market.empty:
            continue

        if "symbol" not in market.columns:
            market["symbol"] = market_file.stem

        feature = pd.read_parquet(feature_file)
        if "date" not in feature.columns:
            continue

        feature = feature.copy()
        feature["date"] = pd.to_datetime(feature["date"], errors="coerce").dt.normalize()
        feature = feature[feature["date"].eq(target_ts)].copy()

        if feature.empty:
            continue

        for factor in active_factors:
            if factor not in feature.columns:
                feature[factor] = pd.NA

        keep_cols = ["date"] + active_factors
        merge_cols = ["date"]

        if "symbol" in feature.columns:
            keep_cols = ["symbol"] + keep_cols
            merge_cols = ["symbol", "date"]

        merged = market.merge(
            feature[keep_cols],
            on=merge_cols,
            how="left",
        )
        parts.append(merged)

    if not parts:
        return pd.DataFrame(columns=["symbol", "date"] + active_factors)

    out = pd.concat(parts, ignore_index=True)
    out = _add_adjusted_price_aliases(out)
    return out


def build_rule_combinations(
    active_factors: list[str],
    requested_combinations: list[RuleCombination] | None = None,
) -> list[RuleCombination]:
    active = set(active_factors)

    if requested_combinations is not None:
        missing = sorted({f for combo in requested_combinations for f in combo.factors} - active)
        if missing:
            raise ValueError(f"combination references inactive/missing factors: {missing}")
        return requested_combinations

    combos: list[RuleCombination] = []
    for name, factors in DEFAULT_AUTO_COMBINATIONS:
        if set(factors).issubset(active):
            combos.append(RuleCombination(name=name, factors=factors))

    if combos:
        return combos

    return [RuleCombination(name=f"{factor}_only", factors=(factor,)) for factor in active_factors]


def _save_outputs(
    *,
    candidate: pd.DataFrame,
    universe: pd.DataFrame,
    rule_audit: pd.DataFrame,
    rule_summary: pd.DataFrame,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    candidate.to_parquet(output_path, index=False)
    candidate.to_csv(output_path.with_suffix(".csv"), index=False, encoding="utf-8-sig")

    stem = output_path.with_suffix("")
    daily_summary_path = Path(str(stem) + "_daily_summary.csv")
    rule_summary_path = Path(str(stem) + "_rule_summary.csv")
    reject_summary_path = Path(str(stem) + "_reject_reason_summary.csv")
    rule_audit_path = Path(str(stem) + "_rule_audit.csv")

    if candidate.empty:
        daily_summary = pd.DataFrame(
            columns=[
                "date",
                "market_regime",
                "combined_rule_name",
                "candidate_count",
                "unique_symbol_count",
            ]
        )
    else:
        daily_summary = (
            candidate.groupby(["date", "market_regime", "combined_rule_name"], dropna=False)
            .agg(
                candidate_count=("symbol", "size"),
                unique_symbol_count=("symbol", "nunique"),
            )
            .reset_index()
        )

    daily_summary.to_csv(daily_summary_path, index=False, encoding="utf-8-sig")
    rule_summary.to_csv(rule_summary_path, index=False, encoding="utf-8-sig")

    reject_summary = (
        universe.groupby("base_reject_reason", dropna=False)
        .size()
        .reset_index(name="row_count")
        .sort_values("row_count", ascending=False)
    )
    reject_summary.to_csv(reject_summary_path, index=False, encoding="utf-8-sig")

    rule_audit.to_csv(rule_audit_path, index=False, encoding="utf-8-sig")

    print("saved:", output_path)
    print("saved:", output_path.with_suffix(".csv"))
    print("saved:", daily_summary_path)
    print("saved:", rule_summary_path)
    print("saved:", reject_summary_path)
    print("saved:", rule_audit_path)


def build_full_market_formulaic_candidate_pool(
    *,
    market_dir: Path,
    factor_dir: Path,
    research_root: Path,
    manual_summary_path: Path,
    regime_path: Path,
    target_date: str,
    output_path: Path,
    factor_names: list[str] | None = None,
    combinations: list[RuleCombination] | None = None,
    min_amount: float = 0.0,
    include_holdout: bool = False,
    exclude_one_word_limit_up: bool = True,
) -> pd.DataFrame:
    target_ts = pd.Timestamp(target_date).normalize()
    target_regime = load_target_regime(regime_path, target_date)

    rules, rule_audit = load_manual_factor_rules(
        manual_summary_path,
        target_regime=target_regime,
        factor_names=factor_names,
        include_holdout=include_holdout,
    )

    active_factors = [rule.factor_name for rule in rules]
    rule_by_factor = {rule.factor_name: rule for rule in rules}
    combos = build_rule_combinations(active_factors, combinations)

    edges_by_factor = {
        factor: load_train_defined_edges(research_root, factor)
        for factor in active_factors
    }

    universe = load_universe_for_date(
        market_dir=market_dir,
        factor_dir=factor_dir,
        target_date=target_date,
        active_factors=active_factors,
    )

    universe = apply_base_tradability_filters(
        universe,
        min_amount=min_amount,
        exclude_one_word_limit_up=exclude_one_word_limit_up,
    )

    universe["target_date"] = target_ts
    universe["market_regime"] = target_regime

    for factor in active_factors:
        bucket_col = f"{factor}_train_defined_bucket"
        pass_col = f"{factor}_pass"

        if factor not in universe.columns:
            universe[factor] = pd.NA

        universe[bucket_col] = assign_train_defined_bucket(universe[factor], edges_by_factor[factor])
        universe[pass_col] = universe[bucket_col].isin(rule_by_factor[factor].buckets)

    candidate_parts: list[pd.DataFrame] = []
    rule_summary_rows: list[dict] = []

    tradable = universe[universe["is_tradable_base"]].copy()

    for combo in combos:
        pass_cols = [f"{factor}_pass" for factor in combo.factors]
        combo_mask = tradable[pass_cols].all(axis=1) if pass_cols else pd.Series(False, index=tradable.index)

        selected = tradable[combo_mask].copy()
        selected["combined_rule_name"] = combo.name
        selected["combined_rule_factors"] = ",".join(combo.factors)
        selected["combined_pass"] = True

        candidate_parts.append(selected)

        rule_summary_rows.append(
            {
                "target_date": target_ts.date().isoformat(),
                "market_regime": target_regime,
                "combined_rule_name": combo.name,
                "combined_rule_factors": ",".join(combo.factors),
                "universe_count": int(len(universe)),
                "tradable_count": int(len(tradable)),
                "candidate_count": int(len(selected)),
            }
        )

    if candidate_parts:
        candidate = pd.concat(candidate_parts, ignore_index=True)
    else:
        candidate = tradable.iloc[0:0].copy()
        candidate["combined_rule_name"] = pd.Series(dtype="object")
        candidate["combined_rule_factors"] = pd.Series(dtype="object")
        candidate["combined_pass"] = pd.Series(dtype="bool")

    if candidate.empty:
        candidate["candidate_count_by_date"] = pd.Series(dtype="int64")
    else:
        candidate["candidate_count_by_date"] = (
            candidate.groupby(["date", "combined_rule_name"])["symbol"].transform("size").astype(int)
        )

    preferred_cols = [
        "symbol",
        "date",
        "target_date",
        "market_regime",
        "is_tradable_base",
        "base_reject_reason",
        "combined_rule_name",
        "combined_rule_factors",
        "combined_pass",
        "candidate_count_by_date",
        "open",
        "high",
        "low",
        "close",
        "vwap",
        "volume",
        "amount",
        "returns",
    ]

    factor_cols: list[str] = []
    for factor in active_factors:
        factor_cols.extend([factor, f"{factor}_train_defined_bucket", f"{factor}_pass"])

    ordered = [c for c in preferred_cols + factor_cols if c in candidate.columns]
    extras = [c for c in candidate.columns if c not in ordered]

    candidate = candidate[ordered + extras].sort_values(
        ["date", "combined_rule_name", "symbol"],
        kind="stable",
    ).reset_index(drop=True)

    rule_summary = pd.DataFrame(rule_summary_rows)

    _save_outputs(
        candidate=candidate,
        universe=universe,
        rule_audit=rule_audit,
        rule_summary=rule_summary,
        output_path=output_path,
    )

    print("target_date:", target_ts.date())
    print("market_regime:", target_regime)
    print("active_factors:", active_factors)
    print("combinations:", [combo.name for combo in combos])
    print("candidate_rows:", len(candidate))

    return candidate
