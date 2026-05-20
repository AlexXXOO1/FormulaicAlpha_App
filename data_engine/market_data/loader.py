from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from core.paths import (
    RAW_DATA_ADJUSTED_OHLC,
    RAW_DATA_UNADJUSTED_OHLCV,
    DAILY_BARS_BY_SYMBOL,
    ensure_data_dirs,
)


DATE_LINE_RE = re.compile(r"^\d{2}/\d{2}/\d{4},")
SYMBOL_RE = re.compile(r"(SH|SZ|BJ)#\d{6}", re.IGNORECASE)
NORMALIZED_MARKER = "# normalized_by=formulaic_alpha_loader_v1"

FINAL_COLUMNS = [
    "symbol",
    "date",
    "adjusted_open",
    "adjusted_high",
    "adjusted_low",
    "adjusted_close",
    "unadjusted_open",
    "unadjusted_high",
    "unadjusted_low",
    "unadjusted_close",
    "volume",
    "amount",
    "unadjusted_vwap",
    "adjustment_factor",
    "adjusted_vwap",
    "returns",
    "source_adjusted_ohlc_file",
    "source_adjusted_ohlc_encoding",
    "source_unadjusted_ohlcv_file",
    "source_unadjusted_ohlcv_encoding",
]


def read_text_with_fallback(path: Path) -> tuple[str, str]:
    raw = path.read_bytes()

    for encoding in ["utf-8-sig", "utf-8", "gb18030", "gbk", "big5"]:
        try:
            return raw.decode(encoding), encoding
        except UnicodeDecodeError:
            continue

    return raw.decode("latin1", errors="ignore"), "latin1-ignore"


def infer_symbol_from_path(path: Path) -> str:
    name = path.stem.upper()

    match = SYMBOL_RE.search(name)
    if match:
        return match.group(0).upper()

    digits = re.search(r"\d{6}", name)
    if digits:
        code = digits.group(0)
        if code.startswith(("0", "3")):
            return f"SZ#{code}"
        if code.startswith(("6", "9")):
            return f"SH#{code}"
        return code

    raise ValueError(f"Cannot infer symbol from file name: {path.name}")


def normalize_tdx_txt_file(path: Path, price_basis: str) -> dict[str, object]:
    path = Path(path)
    symbol = infer_symbol_from_path(path)
    text, encoding = read_text_with_fallback(path)

    data_rows = sum(1 for line in text.splitlines() if DATE_LINE_RE.match(line.strip()))

    if NORMALIZED_MARKER in text:
        return {
            "path": str(path),
            "symbol": symbol,
            "price_basis": price_basis,
            "encoding_before": encoding,
            "rewritten": False,
            "data_rows": data_rows,
        }

    data_lines: list[str] = []

    for line in text.splitlines():
        line = line.strip().lstrip("\ufeff")

        if not DATE_LINE_RE.match(line):
            continue

        parts = [x.strip() for x in line.split(",")]

        if len(parts) >= 7:
            data_lines.append(",".join(parts[:7]))

    if not data_lines:
        raise ValueError(f"No valid data rows found while normalizing: {path}")

    normalized_text = "\n".join(
        [
            NORMALIZED_MARKER,
            f"# symbol={symbol}",
            "# source_format=tdx_daily_txt",
            f"# price_basis={price_basis}",
            f"# original_encoding={encoding}",
            "date,open,high,low,close,volume,amount",
            *data_lines,
            "# source=tdx",
            "",
        ]
    )

    path.write_text(normalized_text, encoding="utf-8")

    return {
        "path": str(path),
        "symbol": symbol,
        "price_basis": price_basis,
        "encoding_before": encoding,
        "rewritten": True,
        "data_rows": len(data_lines),
    }


def collect_txt_files(input_path: Path) -> list[Path]:
    input_path = Path(input_path)

    if input_path.is_file():
        return [input_path]

    if input_path.is_dir():
        files = sorted(input_path.rglob("*.txt"))
        if not files:
            raise FileNotFoundError(f"No txt files found under: {input_path}")
        return files

    raise FileNotFoundError(input_path)


def build_symbol_file_map(input_path: Path) -> dict[str, Path]:
    files = collect_txt_files(input_path)
    out: dict[str, Path] = {}

    for file in files:
        symbol = infer_symbol_from_path(file)

        if symbol in out:
            raise ValueError(f"Duplicate txt file for symbol {symbol}: {out[symbol]} and {file}")

        out[symbol] = file

    return out


def parse_tdx_txt_file(path: Path) -> pd.DataFrame:
    path = Path(path)
    symbol = infer_symbol_from_path(path)
    text, encoding = read_text_with_fallback(path)

    rows = []

    for line_no, line in enumerate(text.splitlines(), start=1):
        line = line.strip().lstrip("\ufeff")

        if not DATE_LINE_RE.match(line):
            continue

        parts = [x.strip() for x in line.split(",")]

        if len(parts) < 7:
            continue

        try:
            rows.append({
                "symbol": symbol,
                "date": pd.to_datetime(parts[0], format="%d/%m/%Y", errors="raise"),
                "open": float(parts[1]),
                "high": float(parts[2]),
                "low": float(parts[3]),
                "close": float(parts[4]),
                "volume": float(parts[5]),
                "amount": float(parts[6]),
                "source_file": path.name,
                "source_encoding": encoding,
                "_line_no": line_no,
            })
        except Exception as exc:
            raise ValueError(f"Failed to parse {path} line {line_no}: {line}") from exc

    if not rows:
        raise ValueError(f"No valid daily bar rows found in: {path}")

    df = pd.DataFrame(rows)

    df = (
        df.sort_values(["symbol", "date", "_line_no"], kind="mergesort")
        .drop_duplicates(subset=["symbol", "date"], keep="last")
        .drop(columns=["_line_no"])
        .reset_index(drop=True)
    )

    bad_rows = (
        (df["open"] <= 0)
        | (df["high"] <= 0)
        | (df["low"] <= 0)
        | (df["close"] <= 0)
        | (df["high"] < df[["open", "low", "close"]].max(axis=1))
        | (df["low"] > df[["open", "high", "close"]].min(axis=1))
        | (df["volume"] < 0)
        | (df["amount"] < 0)
    )

    if bad_rows.any():
        sample = df.loc[bad_rows].head(10)
        raise ValueError(f"Invalid OHLCV rows in {path}:\n{sample.to_string(index=False)}")

    return df


def merge_adjusted_ohlc_and_unadjusted_ohlcv(
    adjusted_df: pd.DataFrame,
    unadjusted_df: pd.DataFrame,
) -> pd.DataFrame:
    adjusted = adjusted_df[
        [
            "symbol",
            "date",
            "open",
            "high",
            "low",
            "close",
            "source_file",
            "source_encoding",
        ]
    ].rename(
        columns={
            "open": "adjusted_open",
            "high": "adjusted_high",
            "low": "adjusted_low",
            "close": "adjusted_close",
            "source_file": "source_adjusted_ohlc_file",
            "source_encoding": "source_adjusted_ohlc_encoding",
        }
    )

    unadjusted = unadjusted_df[
        [
            "symbol",
            "date",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "amount",
            "source_file",
            "source_encoding",
        ]
    ].rename(
        columns={
            "open": "unadjusted_open",
            "high": "unadjusted_high",
            "low": "unadjusted_low",
            "close": "unadjusted_close",
            "source_file": "source_unadjusted_ohlcv_file",
            "source_encoding": "source_unadjusted_ohlcv_encoding",
        }
    )

    merged = adjusted.merge(
        unadjusted,
        on=["symbol", "date"],
        how="left",
        validate="one_to_one",
    )

    missing_unadjusted = merged["unadjusted_close"].isna()
    if missing_unadjusted.any():
        sample = merged.loc[missing_unadjusted, ["symbol", "date"]].head(10)
        raise ValueError(f"Missing unadjusted OHLCV rows:\n{sample.to_string(index=False)}")

    merged["unadjusted_vwap"] = merged["amount"] / merged["volume"].where(merged["volume"] != 0)
    merged["adjustment_factor"] = merged["adjusted_close"] / merged["unadjusted_close"]
    merged["adjusted_vwap"] = merged["unadjusted_vwap"] * merged["adjustment_factor"]

    merged = merged.sort_values(["symbol", "date"], kind="mergesort").reset_index(drop=True)
    merged["returns"] = merged.groupby("symbol", sort=False)["adjusted_close"].pct_change()

    return merged[FINAL_COLUMNS].copy()


def merge_with_existing_parquet(new_df: pd.DataFrame, output_path: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    if not output_path.exists():
        merged = new_df.copy()
        merged = merged.sort_values(["symbol", "date"], kind="mergesort").reset_index(drop=True)
        merged["returns"] = merged.groupby("symbol", sort=False)["adjusted_close"].pct_change()

        return merged[FINAL_COLUMNS].copy(), {
            "old_rows": 0,
            "new_rows": len(new_df),
            "final_rows": len(merged),
            "added_rows": len(new_df),
            "overlap_rows": 0,
        }

    old_df = pd.read_parquet(output_path)

    missing = [c for c in FINAL_COLUMNS if c not in old_df.columns]
    if missing:
        raise ValueError(
            f"Existing parquet has incompatible schema: {output_path}. "
            f"Run with --full-refresh. Missing={missing}"
        )

    old_df = old_df[FINAL_COLUMNS].copy()

    old_keys = set(zip(old_df["symbol"].astype(str), pd.to_datetime(old_df["date"]).dt.normalize()))
    new_keys = set(zip(new_df["symbol"].astype(str), pd.to_datetime(new_df["date"]).dt.normalize()))

    overlap_rows = len(old_keys & new_keys)
    added_rows = len(new_keys - old_keys)

    merged = pd.concat([old_df, new_df], ignore_index=True)
    merged["date"] = pd.to_datetime(merged["date"], errors="coerce").dt.normalize()

    merged = (
        merged.sort_values(["symbol", "date"], kind="mergesort")
        .drop_duplicates(subset=["symbol", "date"], keep="last")
        .reset_index(drop=True)
    )

    merged["returns"] = merged.groupby("symbol", sort=False)["adjusted_close"].pct_change()

    return merged[FINAL_COLUMNS].copy(), {
        "old_rows": len(old_df),
        "new_rows": len(new_df),
        "final_rows": len(merged),
        "added_rows": added_rows,
        "overlap_rows": overlap_rows,
    }


def atomic_write_parquet(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = output_path.with_name(output_path.name + ".tmp.parquet")
    df.to_parquet(tmp_path, index=False)
    tmp_path.replace(output_path)


def import_tdx_txt_to_daily_parquet(
    adjusted_input: Path = RAW_DATA_ADJUSTED_OHLC,
    unadjusted_input: Path = RAW_DATA_UNADJUSTED_OHLCV,
    output_dir: Path = DAILY_BARS_BY_SYMBOL,
    normalize_raw: bool = True,
    incremental: bool = True,
) -> pd.DataFrame:
    ensure_data_dirs()

    adjusted_map = build_symbol_file_map(adjusted_input)
    unadjusted_map = build_symbol_file_map(unadjusted_input)

    report_rows = []
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, (symbol, adjusted_file) in enumerate(sorted(adjusted_map.items()), start=1):
        print(f"parsing {i}/{len(adjusted_map)}: {symbol}")

        try:
            if symbol not in unadjusted_map:
                raise FileNotFoundError(f"Missing unadjusted OHLCV txt file for symbol: {symbol}")

            unadjusted_file = unadjusted_map[symbol]

            adjusted_norm = {}
            unadjusted_norm = {}

            if normalize_raw:
                adjusted_norm = normalize_tdx_txt_file(adjusted_file, "adjusted_ohlc")
                unadjusted_norm = normalize_tdx_txt_file(unadjusted_file, "unadjusted_ohlcv")

            adjusted_df = parse_tdx_txt_file(adjusted_file)
            unadjusted_df = parse_tdx_txt_file(unadjusted_file)

            new_df = merge_adjusted_ohlc_and_unadjusted_ohlcv(adjusted_df, unadjusted_df)

            output_path = output_dir / f"{symbol}.parquet"

            if incremental:
                final_df, merge_stats = merge_with_existing_parquet(new_df, output_path)
            else:
                final_df = new_df
                merge_stats = {
                    "old_rows": 0,
                    "new_rows": len(new_df),
                    "final_rows": len(final_df),
                    "added_rows": len(new_df),
                    "overlap_rows": 0,
                }

            atomic_write_parquet(final_df, output_path)

            report_rows.append({
                "symbol": symbol,
                "adjusted_file": str(adjusted_file),
                "unadjusted_file": str(unadjusted_file),
                "raw_normalized": bool(normalize_raw),
                "adjusted_rewritten": bool(adjusted_norm.get("rewritten", False)),
                "unadjusted_rewritten": bool(unadjusted_norm.get("rewritten", False)),
                "incremental": bool(incremental),
                "old_rows": merge_stats["old_rows"],
                "new_rows": merge_stats["new_rows"],
                "final_rows": merge_stats["final_rows"],
                "added_rows": merge_stats["added_rows"],
                "overlap_rows": merge_stats["overlap_rows"],
                "date_min": final_df["date"].min(),
                "date_max": final_df["date"].max(),
                "output_path": str(output_path),
                "status": "ok",
                "error": "",
            })

            print(
                f"saved: {output_path} "
                f"old={merge_stats['old_rows']} "
                f"new={merge_stats['new_rows']} "
                f"added={merge_stats['added_rows']} "
                f"overlap={merge_stats['overlap_rows']} "
                f"final={merge_stats['final_rows']}"
            )

        except Exception as exc:
            report_rows.append({
                "symbol": symbol,
                "adjusted_file": str(adjusted_file),
                "unadjusted_file": str(unadjusted_map.get(symbol, "")),
                "raw_normalized": bool(normalize_raw),
                "adjusted_rewritten": False,
                "unadjusted_rewritten": False,
                "incremental": bool(incremental),
                "old_rows": 0,
                "new_rows": 0,
                "final_rows": 0,
                "added_rows": 0,
                "overlap_rows": 0,
                "date_min": "",
                "date_max": "",
                "output_path": "",
                "status": "failed",
                "error": repr(exc),
            })
            print(f"[FAILED] {symbol}: {exc}")

    report = pd.DataFrame(report_rows)
    report_path = output_dir.parent / "import_report.csv"
    report.to_csv(report_path, index=False, encoding="utf-8-sig")

    print("report:", report_path)
    print("ok_symbols:", int((report["status"] == "ok").sum()))
    print("failed_symbols:", int((report["status"] == "failed").sum()))

    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adjusted-input", default=str(RAW_DATA_ADJUSTED_OHLC))
    parser.add_argument("--unadjusted-input", default=str(RAW_DATA_UNADJUSTED_OHLCV))
    parser.add_argument("--output-dir", default=str(DAILY_BARS_BY_SYMBOL))
    parser.add_argument("--no-normalize-raw", action="store_true")
    parser.add_argument("--full-refresh", action="store_true")
    args = parser.parse_args()

    import_tdx_txt_to_daily_parquet(
        adjusted_input=Path(args.adjusted_input),
        unadjusted_input=Path(args.unadjusted_input),
        output_dir=Path(args.output_dir),
        normalize_raw=not args.no_normalize_raw,
        incremental=not args.full_refresh,
    )


if __name__ == "__main__":
    main()
