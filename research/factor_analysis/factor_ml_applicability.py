from __future__ import annotations

from pathlib import Path

import pandas as pd


ML_FACTOR_APPLICABILITY_ROWS: list[dict[str, object]] = [
    {
        "factor": "alpha_001",
        "manual_factor_type": "regime_aware_filter",
        "bucket_rule": "train_defined_bucket in [4,5,6,7]",
        "valid_regime": "range_bound,risk_off,strong_repair",
        "avoid_regime": "strong_trend",
        "usable_as_filter": True,
        "usable_as_model_feature": True,
        "candidate_universe_filter": False,
        "holdout_redundant": False,
        "rejected": False,
        "ml_role": "auxiliary_feature",
        "reason": "Valid as broad regime-aware middle-bucket filter; not standalone tradable.",
        "next_action": "Keep as candidate-model feature, but do not use as standalone selector.",
    },
    {
        "factor": "alpha_002",
        "manual_factor_type": "regime_aware_filter",
        "bucket_rule": "train_defined_bucket in [4]",
        "valid_regime": "range_bound,risk_off,strong_repair",
        "avoid_regime": "strong_trend",
        "usable_as_filter": True,
        "usable_as_model_feature": True,
        "candidate_universe_filter": False,
        "holdout_redundant": False,
        "rejected": False,
        "ml_role": "auxiliary_feature",
        "reason": "Usable filter factor with regime dependence; not standalone tradable.",
        "next_action": "Keep as candidate-model feature and monitor regime stability.",
    },
    {
        "factor": "alpha_003",
        "manual_factor_type": "broad_filter",
        "bucket_rule": "train_defined_bucket in [4,5,6,7]",
        "valid_regime": "range_bound,risk_off,strong_repair,strong_trend",
        "avoid_regime": "",
        "usable_as_filter": True,
        "usable_as_model_feature": False,
        "candidate_universe_filter": False,
        "holdout_redundant": True,
        "rejected": False,
        "ml_role": "holdout_redundant_factor",
        "reason": "Currently effective, but redundant with alpha_001/alpha_005 middle-bucket filtering behavior.",
        "next_action": "Exclude from current ML baseline; only add later for ablation test.",
    },
    {
        "factor": "alpha_004",
        "manual_factor_type": "broad_filter",
        "bucket_rule": "train_defined_bucket in [4,5,6,7]",
        "valid_regime": "range_bound,risk_off,strong_repair,strong_trend",
        "avoid_regime": "",
        "usable_as_filter": True,
        "usable_as_model_feature": False,
        "candidate_universe_filter": False,
        "holdout_redundant": True,
        "rejected": False,
        "ml_role": "holdout_redundant_factor",
        "reason": "Currently effective, but redundant with alpha_001/alpha_005 middle-bucket filtering behavior.",
        "next_action": "Exclude from current ML baseline; only add later for ablation test.",
    },
    {
        "factor": "alpha_005",
        "manual_factor_type": "broad_filter",
        "bucket_rule": "train_defined_bucket in [4,5,6,7] for current ML candidate universe",
        "valid_regime": "range_bound,risk_off,strong_repair,strong_trend",
        "avoid_regime": "",
        "usable_as_filter": True,
        "usable_as_model_feature": False,
        "candidate_universe_filter": True,
        "holdout_redundant": False,
        "rejected": False,
        "ml_role": "candidate_universe_filter",
        "reason": "Best current broad universe filter; using it as universe filter means it should not also be used as an in-universe model feature.",
        "next_action": "Use alpha_005 bucket 4-7 as candidate universe; train ranking models only inside this universe.",
    },
    {
        "factor": "alpha_006",
        "manual_factor_type": "market_state_filter",
        "bucket_rule": "train_defined_bucket in [4,5]",
        "valid_regime": "range_bound,risk_off,strong_repair",
        "avoid_regime": "strong_trend",
        "usable_as_filter": True,
        "usable_as_model_feature": True,
        "candidate_universe_filter": False,
        "holdout_redundant": False,
        "rejected": False,
        "ml_role": "auxiliary_feature",
        "reason": "Usable as market-state filter; not standalone selector.",
        "next_action": "Keep as candidate-model feature and evaluate by regime.",
    },
    {
        "factor": "alpha_007",
        "manual_factor_type": "rejected",
        "bucket_rule": "",
        "valid_regime": "",
        "avoid_regime": "",
        "usable_as_filter": False,
        "usable_as_model_feature": False,
        "candidate_universe_filter": False,
        "holdout_redundant": False,
        "rejected": True,
        "ml_role": "excluded",
        "reason": "Rejected by single-factor validation.",
        "next_action": "Do not use in ML baseline or candidate universe.",
    },
    {
        "factor": "alpha_008",
        "manual_factor_type": "conservative_filter",
        "bucket_rule": "train_defined_bucket in [4]",
        "valid_regime": "range_bound first",
        "avoid_regime": "",
        "usable_as_filter": True,
        "usable_as_model_feature": True,
        "candidate_universe_filter": False,
        "holdout_redundant": False,
        "rejected": False,
        "ml_role": "auxiliary_feature",
        "reason": "Conservative filter candidate; useful but should be monitored by regime.",
        "next_action": "Keep as candidate-model feature, especially for range_bound tests.",
    },
    {
        "factor": "alpha_009",
        "manual_factor_type": "rejected_train_test_instability",
        "bucket_rule": "",
        "valid_regime": "",
        "avoid_regime": "",
        "usable_as_filter": False,
        "usable_as_model_feature": False,
        "candidate_universe_filter": False,
        "holdout_redundant": False,
        "rejected": True,
        "ml_role": "excluded",
        "reason": "Rejected or held out due to train/test instability.",
        "next_action": "Do not use in current ML baseline; revisit only after more data or formula review.",
    },
    {
        "factor": "alpha_010",
        "manual_factor_type": "broad_filter",
        "bucket_rule": "train_defined_bucket in [4,5,6,7]",
        "valid_regime": "range_bound,risk_off,strong_repair,strong_trend",
        "avoid_regime": "",
        "usable_as_filter": True,
        "usable_as_model_feature": True,
        "candidate_universe_filter": False,
        "holdout_redundant": False,
        "rejected": False,
        "ml_role": "auxiliary_feature",
        "reason": "Broad filter; not standalone tradable.",
        "next_action": "Keep as candidate-model feature and evaluate by monotonic prediction diagnostics.",
    },
]


def build_ml_factor_applicability_table() -> pd.DataFrame:
    df = pd.DataFrame(ML_FACTOR_APPLICABILITY_ROWS)

    expected_factors = [f"alpha_{i:03d}" for i in range(1, 11)]
    if df["factor"].tolist() != expected_factors:
        raise ValueError("ML applicability table must cover alpha_001 through alpha_010 in order.")

    bool_cols = [
        "usable_as_filter",
        "usable_as_model_feature",
        "candidate_universe_filter",
        "holdout_redundant",
        "rejected",
    ]
    for col in bool_cols:
        df[col] = df[col].astype(bool)

    return df


def build_ml_factor_applicability_markdown(df: pd.DataFrame) -> str:
    cols = [
        "factor",
        "manual_factor_type",
        "ml_role",
        "usable_as_filter",
        "usable_as_model_feature",
        "candidate_universe_filter",
        "holdout_redundant",
        "rejected",
        "next_action",
    ]
    view = df[cols].copy()

    lines = [
        "# Formulaic Alpha ML Applicability Summary",
        "",
        "This table summarizes how alpha_001 through alpha_010 should be used in the current ML workflow.",
        "",
        view.to_markdown(index=False),
        "",
    ]
    return "\n".join(lines)


def write_ml_factor_applicability_outputs(
    *,
    output_dir: str | Path,
    output_name: str = "alpha_001_010_ml_factor_applicability",
) -> dict[str, Path]:
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    df = build_ml_factor_applicability_table()

    csv_path = output_dir / f"{output_name}.csv"
    md_path = output_dir / f"{output_name}.md"

    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    md_path.write_text(build_ml_factor_applicability_markdown(df), encoding="utf-8")

    return {
        "csv": csv_path,
        "markdown": md_path,
    }
