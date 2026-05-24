
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from research.candidate_pool.full_market_formulaic_candidate_pool import (
    apply_base_tradability_filters,
    assign_train_defined_bucket,
    load_target_regime,
    load_train_defined_edges,
)


@dataclass(frozen=True)
class SymbolScoreRule:
    factor_name: str
    buckets: tuple[int, ...]
    bucket_rule: str
    valid_regimes: tuple[str, ...]
    weak_regimes: tuple[str, ...]
    factor_type_manual: str
    ml_baseline_role: str


def normalize_symbol(symbol: str) -> str:
    raw = str(symbol).strip().upper()
    if "#" in raw:
        prefix, code = raw.split("#", 1)
        return f"{prefix}#{code.zfill(6)}"

    digits = re.sub(r"\D", "", raw)
    if len(digits) != 6:
        raise ValueError(
            f"Unsupported symbol format: {symbol}. Use SH#600000, SZ#000001, or 6-digit code."
        )

    if digits.startswith(("6", "9")):
        return f"SH#{digits}"
    if digits.startswith(("0", "2", "3")):
        return f"SZ#{digits}"

    raise ValueError(f"Cannot infer exchange for symbol: {symbol}")


def parse_symbol_list(value: str) -> list[str]:
    symbols = [normalize_symbol(x) for x in str(value).split(",") if x.strip()]
    if not symbols:
        raise ValueError("empty symbols")
    return symbols


def parse_factor_list(value: str | None) -> list[str]:
    if value is None or str(value).strip() == "":
        return ["alpha_002", "alpha_006"]
    factors = [x.strip().lower() for x in str(value).split(",") if x.strip()]
    if not factors:
        raise ValueError("empty factors")
    return factors


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


def load_symbol_score_rules(
    manual_summary_path: Path,
    *,
    factor_names: list[str],
    include_holdout: bool = False,
) -> list[SymbolScoreRule]:
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

    requested = set(factor_names)
    missing_requested = sorted(requested - set(df["factor_name"]))
    if missing_requested:
        raise ValueError(f"requested factors not found in manual summary: {missing_requested}")

    df = df[df["factor_name"].isin(requested)].copy()

    rules: list[SymbolScoreRule] = []
    inactive_rows: list[dict] = []

    for _, row in df.iterrows():
        factor = str(row["factor_name"]).strip().lower()
        manual_type = str(row.get("factor_type_manual", "")).strip()
        bucket_rule = str(row.get("bucket_rule", "")).strip()
        buckets = _parse_bucket_rule(bucket_rule)
        valid_regimes = _split_csv_like(row.get("valid_regime", ""))
        weak_regimes = _split_csv_like(row.get("weak_regime", ""))
        role = str(row.get("ml_baseline_role", "")).strip()

        active = True
        reason = "active"

        if not _bool_like(row.get("manual_override", "")):
            active = False
            reason = "manual_override is not True"
        elif manual_type == "reject":
            active = False
            reason = "manual conclusion is reject"
        elif not _bool_like(row.get("is_filter_factor", "")):
            active = False
            reason = "manual conclusion is not filter factor"
        elif not buckets:
            active = False
            reason = "bucket_rule has no usable bucket"
        elif not include_holdout and role == "holdout_redundant_factor":
            active = False
            reason = "holdout factor excluded"

        if active:
            rules.append(
                SymbolScoreRule(
                    factor_name=factor,
                    buckets=buckets,
                    bucket_rule=bucket_rule,
                    valid_regimes=valid_regimes,
                    weak_regimes=weak_regimes,
                    factor_type_manual=manual_type,
                    ml_baseline_role=role,
                )
            )
        else:
            inactive_rows.append(
                {
                    "factor_name": factor,
                    "reason": reason,
                    "factor_type_manual": manual_type,
                    "bucket_rule": bucket_rule,
                    "ml_baseline_role": role,
                }
            )

    if not rules:
        raise ValueError(f"No active score rules. inactive_rows={inactive_rows}")

    rule_factors = {r.factor_name for r in rules}
    missing_active = sorted(requested - rule_factors)
    if missing_active:
        raise ValueError(f"requested factors are not active score rules: {missing_active}")

    return sorted(rules, key=lambda x: x.factor_name)


def _read_symbol_row(path: Path, target_date: str) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_parquet(path)
    if "date" not in df.columns:
        return pd.DataFrame()

    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    target_ts = pd.Timestamp(target_date).normalize()

    out = out[out["date"].eq(target_ts)].copy()

    if out.empty:
        return out

    if "symbol" not in out.columns:
        out["symbol"] = path.stem

    return out


def _merge_symbol_market_and_features(
    *,
    market_dir: Path,
    factor_dir: Path,
    symbol: str,
    target_date: str,
    factors: list[str],
) -> pd.DataFrame:
    market_path = market_dir / f"{symbol}.parquet"
    feature_path = factor_dir / f"{symbol}.parquet"

    market = _read_symbol_row(market_path, target_date)
    if market.empty:
        return pd.DataFrame(
            {
                "symbol": [symbol],
                "date": [pd.Timestamp(target_date).normalize()],
                "missing_reason": ["missing_market_row"],
            }
        )

    feature = _read_symbol_row(feature_path, target_date)
    if feature.empty:
        out = market.copy()
        out["missing_reason"] = "missing_feature_row"
        for factor in factors:
            out[factor] = pd.NA
        return out

    for factor in factors:
        if factor not in feature.columns:
            feature[factor] = pd.NA

    keep = ["symbol", "date"] + factors
    merged = market.merge(feature[keep], on=["symbol", "date"], how="left")
    merged["missing_reason"] = "none"

    return merged


def score_formulaic_symbols(
    *,
    symbols: list[str],
    target_date: str,
    market_dir: Path,
    factor_dir: Path,
    research_root: Path,
    manual_summary_path: Path,
    regime_path: Path,
    output_path: Path | None = None,
    factor_names: list[str] | None = None,
    include_holdout: bool = False,
    ignore_regime_gate: bool = False,
    min_amount: float = 0.0,
    combination_name: str | None = None,
    rule_state: str = "PAPER_WATCH",
) -> pd.DataFrame:
    factors = parse_factor_list(",".join(factor_names) if factor_names else None)
    rules = load_symbol_score_rules(
        manual_summary_path,
        factor_names=factors,
        include_holdout=include_holdout,
    )

    factor_names = [r.factor_name for r in rules]
    rule_by_factor = {r.factor_name: r for r in rules}
    target_regime = load_target_regime(regime_path, target_date)

    edges_by_factor = {
        factor: load_train_defined_edges(research_root, factor)
        for factor in factor_names
    }

    rows: list[pd.DataFrame] = []

    for symbol in symbols:
        norm_symbol = normalize_symbol(symbol)
        merged = _merge_symbol_market_and_features(
            market_dir=market_dir,
            factor_dir=factor_dir,
            symbol=norm_symbol,
            target_date=target_date,
            factors=factor_names,
        )
        rows.append(merged)

    if rows:
        work = pd.concat(rows, ignore_index=True)
    else:
        work = pd.DataFrame()

    if work.empty:
        raise ValueError("no symbol rows to score")

    work = apply_base_tradability_filters(work, min_amount=min_amount)

    work["market_regime"] = target_regime
    work["rule_state"] = rule_state
    work["allow_new_entry"] = False

    factor_weight = 80.0 / len(factor_names)
    base_weight = 10.0
    regime_weight = 10.0

    for factor in factor_names:
        rule = rule_by_factor[factor]
        bucket_col = f"{factor}_train_defined_bucket"
        pass_col = f"{factor}_pass"
        regime_col = f"{factor}_regime_allowed"
        bucket_rule_col = f"{factor}_bucket_rule"

        if factor not in work.columns:
            work[factor] = pd.NA

        work[bucket_col] = assign_train_defined_bucket(work[factor], edges_by_factor[factor])
        work[pass_col] = work[bucket_col].isin(rule.buckets)
        work[regime_col] = target_regime in rule.valid_regimes
        work[bucket_rule_col] = rule.bucket_rule

    factor_pass_cols = [f"{factor}_pass" for factor in factor_names]
    factor_regime_cols = [f"{factor}_regime_allowed" for factor in factor_names]

    work["combo_name"] = combination_name or "_and_".join(factor_names)
    work["combo_factors"] = ",".join(factor_names)
    work["combo_pass"] = work[factor_pass_cols].all(axis=1)
    work["regime_allowed"] = work[factor_regime_cols].all(axis=1) | ignore_regime_gate

    work["base_score"] = work["is_tradable_base"].astype(float) * base_weight
    work["regime_score"] = work["regime_allowed"].astype(float) * regime_weight

    factor_score_cols = []
    for factor in factor_names:
        score_col = f"{factor}_score"
        work[score_col] = work[f"{factor}_pass"].astype(float) * factor_weight
        factor_score_cols.append(score_col)

    work["score"] = work["base_score"] + work["regime_score"] + work[factor_score_cols].sum(axis=1)
    work["score"] = work["score"].round(2)

    reject_reasons: list[str] = []

    for _, row in work.iterrows():
        reasons: list[str] = []

        if str(row.get("missing_reason", "none")) != "none":
            reasons.append(str(row.get("missing_reason")))

        if not bool(row.get("is_tradable_base", False)):
            reasons.append(str(row.get("base_reject_reason", "base_not_tradable")))

        if not bool(row.get("regime_allowed", False)):
            reasons.append(f"regime_not_allowed:{target_regime}")

        for factor in factor_names:
            if pd.isna(row.get(factor)):
                reasons.append(f"{factor}_missing_value")
            elif not bool(row.get(f"{factor}_pass", False)):
                bucket = row.get(f"{factor}_train_defined_bucket")
                reasons.append(f"{factor}_bucket_not_pass:{bucket}")

        reject_reasons.append(";".join(reasons) if reasons else "none")

    work["reject_reason"] = reject_reasons

    def _status(row: pd.Series) -> str:
        if str(row.get("missing_reason", "none")) != "none":
            return "REJECT_MISSING_DATA"
        if not bool(row.get("is_tradable_base", False)):
            return "REJECT_BASE_TRADABILITY"
        if not bool(row.get("regime_allowed", False)):
            return "FORCE_REJECT_BY_REGIME"
        if bool(row.get("combo_pass", False)) and float(row.get("score", 0.0)) >= 90.0:
            return "PAPER_WATCH_PASS"
        if float(row.get("score", 0.0)) >= 40.0:
            return "PARTIAL_MATCH"
        return "REJECT"

    work["status"] = work.apply(_status, axis=1)

    preferred_cols = [
        "symbol",
        "date",
        "market_regime",
        "rule_state",
        "allow_new_entry",
        "combo_name",
        "combo_factors",
        "combo_pass",
        "score",
        "status",
        "reject_reason",
        "is_tradable_base",
        "base_reject_reason",
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
    for factor in factor_names:
        factor_cols.extend(
            [
                factor,
                f"{factor}_train_defined_bucket",
                f"{factor}_pass",
                f"{factor}_regime_allowed",
                f"{factor}_bucket_rule",
                f"{factor}_score",
            ]
        )

    ordered = [c for c in preferred_cols + factor_cols if c in work.columns]
    extras = [c for c in work.columns if c not in ordered]
    out = work[ordered + extras].sort_values(["score", "symbol"], ascending=[False, True]).reset_index(drop=True)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(output_path, index=False, encoding="utf-8-sig")
        print("saved:", output_path)

    return out
