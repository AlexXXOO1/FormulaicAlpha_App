from __future__ import annotations

import pandas as pd

from research.factor_analysis.run_single_factor_research import write_step10_factor_conclusion


def test_step10_handles_empty_step9_file(tmp_path):
    factor_name = "alpha_999"
    step9_path = tmp_path / f"step9_{factor_name}_train_defined_bucket_regime_check.csv"
    step9_path.write_text("", encoding="utf-8")

    write_step10_factor_conclusion(factor_name=factor_name, output_dir=tmp_path)

    out_csv = tmp_path / f"step10_{factor_name}_factor_conclusion.csv"
    out_md = tmp_path / f"step10_{factor_name}_factor_conclusion.md"

    assert out_csv.exists()
    assert out_md.exists()

    df = pd.read_csv(out_csv)
    row = df.iloc[0]

    assert row["factor_name"] == factor_name
    assert row["factor_type"] == "reject"
    assert row["bucket_rule"] == "none"
    assert bool(row["standalone_tradable"]) is False
    assert "Step9 train-defined bucket regime check is empty" in row["rejection_reason_as_standalone"]
