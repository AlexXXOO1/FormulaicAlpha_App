from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

from research.factor_analysis.factor_ml_applicability import (
    build_ml_factor_applicability_markdown,
    build_ml_factor_applicability_table,
    write_ml_factor_applicability_outputs,
)


def test_build_ml_factor_applicability_table_covers_alpha_001_to_010():
    df = build_ml_factor_applicability_table()

    assert df["factor"].tolist() == [f"alpha_{i:03d}" for i in range(1, 11)]
    assert len(df) == 10

    rejected = set(df[df["rejected"]]["factor"])
    holdout = set(df[df["holdout_redundant"]]["factor"])
    model_features = set(df[df["usable_as_model_feature"]]["factor"])
    candidate_filters = set(df[df["candidate_universe_filter"]]["factor"])

    assert rejected == {"alpha_007", "alpha_009"}
    assert holdout == {"alpha_003", "alpha_004"}
    assert "alpha_005" in candidate_filters
    assert "alpha_005" not in model_features

    assert {"alpha_001", "alpha_002", "alpha_006", "alpha_008", "alpha_010"}.issubset(model_features)


def test_build_markdown_contains_core_table():
    df = build_ml_factor_applicability_table()
    md = build_ml_factor_applicability_markdown(df)

    assert "# Formulaic Alpha ML Applicability Summary" in md
    assert "alpha_005" in md
    assert "candidate_universe_filter" in md


def test_write_ml_factor_applicability_outputs(tmp_path: Path):
    paths = write_ml_factor_applicability_outputs(output_dir=tmp_path, output_name="unit_applicability")

    assert paths["csv"].exists()
    assert paths["markdown"].exists()

    df = pd.read_csv(paths["csv"])
    assert len(df) == 10
    assert "alpha_003" in set(df[df["holdout_redundant"]]["factor"])


def test_export_ml_factor_applicability_script(tmp_path: Path):
    cmd = [
        sys.executable,
        "scripts/export_ml_factor_applicability.py",
        "--output-dir",
        str(tmp_path),
        "--output-name",
        "unit_cli_applicability",
    ]

    result = subprocess.run(cmd, check=True, capture_output=True, text=True)

    assert "saved ML factor applicability files:" in result.stdout
    assert "ML factor applicability:" in result.stdout
    assert (tmp_path / "unit_cli_applicability.csv").exists()
    assert (tmp_path / "unit_cli_applicability.md").exists()
