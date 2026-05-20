from __future__ import annotations

from pathlib import Path

from data_engine.raw_import.tdx_txt_importer import parse_tdx_txt_file


def test_parse_tdx_txt_file_skips_non_data_lines(tmp_path: Path):
    path = tmp_path / "SH#600000.txt"
    path.write_text(
        """
garbled header should be ignored
date,open,high,low,close,volume,amount
02/08/2021,7.57,7.78,7.51,7.67,45713350,416533728.00
03/08/2021,7.67,7.68,7.58,7.66,33014050,300047776.00
footer should be ignored
""".strip(),
        encoding="utf-8",
    )

    df = parse_tdx_txt_file(path)

    assert len(df) == 2
    assert list(df.columns) == [
        "symbol",
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
        "vwap",
        "returns",
        "source_file",
        "source_encoding",
    ]
    assert df.loc[0, "symbol"] == "SH#600000"
    assert df.loc[0, "open"] == 7.57
    assert df.loc[1, "close"] == 7.66
    assert df.loc[0, "returns"] != df.loc[0, "returns"]
    assert df.loc[1, "returns"] < 0
