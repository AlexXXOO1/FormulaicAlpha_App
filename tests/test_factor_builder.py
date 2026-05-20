from __future__ import annotations

import pandas as pd

from alpha_engine.pipeline.factor_builder import build_alpha_input_frame


def test_build_alpha_input_frame_maps_adjusted_columns_to_paper_columns():
    market = pd.DataFrame(
        [
            {
                "symbol": "SZ#000001",
                "date": "2021-08-02",
                "adjusted_open": 15.56,
                "adjusted_high": 16.12,
                "adjusted_low": 15.43,
                "adjusted_close": 15.93,
                "volume": 1500000,
                "adjusted_vwap": 15.6783,
                "returns": None,
            }
        ]
    )

    out = build_alpha_input_frame(market)

    assert list(out.columns) == [
        "symbol",
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "vwap",
        "returns",
    ]

    assert out.loc[0, "open"] == 15.56
    assert out.loc[0, "high"] == 16.12
    assert out.loc[0, "low"] == 15.43
    assert out.loc[0, "close"] == 15.93
    assert out.loc[0, "vwap"] == 15.6783
