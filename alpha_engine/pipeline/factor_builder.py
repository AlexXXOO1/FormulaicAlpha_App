from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

from alpha_engine.formulaic_alphas.registry import get_formulaic_alpha
from core.paths import DAILY_BARS_BY_SYMBOL, FORMULAIC_ALPHAS_BY_SYMBOL, ensure_data_dirs


ALPHA_INPUT_COLUMNS = [
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


MARKET_TO_ALPHA_COLUMN_MAP = {
    "adjusted_open": "open",
    "adjusted_high": "high",
    "adjusted_low": "low",
    "adjusted_close": "close",
    "volume": "volume",
    "adjusted_vwap": "vwap",
    "returns": "returns",
}


REQUIRED_MARKET_COLUMNS = [
    "symbol",
    "date",
    "adjusted_open",
    "adjusted_high",
    "adjusted_low",
    "adjusted_close",
    "volume",
    "adjusted_vwap",
    "returns",
]


def _print_progress(
    *,
    label: str,
    current: int,
    total: int,
    enabled: bool,
) -> None:
    if not enabled or total <= 0:
        return

    current = min(max(current, 0), total)
    width = 30
    pct = current / total
    filled = int(width * pct)
    bar = "#" * filled + "-" * (width - filled)

    end = "\n" if current >= total else ""
    sys.stderr.write(f"\r{label} [{bar}] {current}/{total} {pct:6.2%}")
    sys.stderr.flush()

    if end:
        sys.stderr.write(end)
        sys.stderr.flush()


def load_market_parquet_dir(
    input_dir: Path = DAILY_BARS_BY_SYMBOL,
    *,
    show_progress: bool = False,
) -> pd.DataFrame:
    files = sorted(Path(input_dir).glob("*.parquet"))

    if not files:
        raise FileNotFoundError(f"No market parquet files found under: {input_dir}")

    parts = []

    total_files = len(files)

    for idx, file in enumerate(files, start=1):
        df = pd.read_parquet(file)

        missing = [c for c in REQUIRED_MARKET_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"{file} missing required market columns: {missing}")

        parts.append(df[REQUIRED_MARKET_COLUMNS].copy())
        _print_progress(
            label="read market",
            current=idx,
            total=total_files,
            enabled=show_progress,
        )

    out = pd.concat(parts, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()

    out = out.dropna(subset=["symbol", "date"])
    out = out.sort_values(["symbol", "date"], kind="mergesort").reset_index(drop=True)

    return out


def build_alpha_input_frame(market_df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED_MARKET_COLUMNS if c not in market_df.columns]
    if missing:
        raise ValueError(f"Missing required market columns: {missing}")

    out = market_df[["symbol", "date", *MARKET_TO_ALPHA_COLUMN_MAP.keys()]].copy()
    out = out.rename(columns=MARKET_TO_ALPHA_COLUMN_MAP)

    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()

    for col in ["open", "high", "low", "close", "volume", "vwap", "returns"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out[ALPHA_INPUT_COLUMNS].copy()
    out = out.sort_values(["symbol", "date"], kind="mergesort").reset_index(drop=True)

    return out


def load_alpha_input_frame(
    input_dir: Path = DAILY_BARS_BY_SYMBOL,
    *,
    show_progress: bool = False,
) -> pd.DataFrame:
    market_df = load_market_parquet_dir(input_dir, show_progress=show_progress)
    return build_alpha_input_frame(market_df)


def merge_alpha_into_symbol_file(
    *,
    part: pd.DataFrame,
    output_path: Path,
    alpha_col: str,
) -> dict[str, object]:
    new_part = part[["symbol", "date", alpha_col]].copy()
    new_part["date"] = pd.to_datetime(new_part["date"], errors="coerce").dt.normalize()
    new_part = new_part.sort_values(["symbol", "date"], kind="mergesort").reset_index(drop=True)

    old_rows = 0

    if output_path.exists():
        old = pd.read_parquet(output_path)
        old["date"] = pd.to_datetime(old["date"], errors="coerce").dt.normalize()
        old_rows = len(old)

        if alpha_col in old.columns:
            old = old.drop(columns=[alpha_col])

        merged = old.merge(
            new_part,
            on=["symbol", "date"],
            how="outer",
            validate="one_to_one",
        )
    else:
        merged = new_part

    merged = (
        merged.sort_values(["symbol", "date"], kind="mergesort")
        .drop_duplicates(subset=["symbol", "date"], keep="last")
        .reset_index(drop=True)
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(output_path, index=False)

    return {
        "symbol": str(new_part["symbol"].iloc[0]),
        "output_path": str(output_path),
        "old_rows": old_rows,
        "new_rows": len(new_part),
        "final_rows": len(merged),
        "non_null": int(merged[alpha_col].notna().sum()),
    }


def save_alpha_by_symbol(
    alpha_df: pd.DataFrame,
    output_dir: Path,
    alpha_col: str,
    *,
    show_progress: bool = False,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)

    report_rows = []

    grouped = alpha_df.groupby("symbol", sort=True)
    total_symbols = grouped.ngroups

    for idx, (_, part) in enumerate(grouped, start=1):
        symbol = str(part["symbol"].iloc[0])
        output_path = output_dir / f"{symbol}.parquet"

        report = merge_alpha_into_symbol_file(
            part=part,
            output_path=output_path,
            alpha_col=alpha_col,
        )

        report_rows.append(report)
        _print_progress(
            label=f"write {alpha_col}",
            current=idx,
            total=total_symbols,
            enabled=show_progress,
        )

    return pd.DataFrame(report_rows)


def build_formulaic_alpha(
    *,
    alpha_name: str,
    input_dir: Path = DAILY_BARS_BY_SYMBOL,
    output_dir: Path = FORMULAIC_ALPHAS_BY_SYMBOL,
    show_progress: bool = False,
) -> pd.DataFrame:
    ensure_data_dirs()

    alpha_name = alpha_name.strip().lower()
    compute_alpha = get_formulaic_alpha(alpha_name)

    alpha_input = load_alpha_input_frame(Path(input_dir), show_progress=show_progress)
    alpha_df = compute_alpha(alpha_input)

    if alpha_name not in alpha_df.columns:
        raise ValueError(f"Computed alpha output missing expected column: {alpha_name}")

    return save_alpha_by_symbol(
        alpha_df=alpha_df,
        output_dir=Path(output_dir),
        alpha_col=alpha_name,
        show_progress=show_progress,
    )
