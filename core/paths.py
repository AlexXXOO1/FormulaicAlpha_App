from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT = Path(r"C:\Users\zyf37\Desktop\FormulaicAlpha_Data")

# Forward-adjusted OHLC txt files.
RAW_DATA_ADJUSTED_OHLC = DATA_ROOT / "raw_data_adjusted_ohlc"

# Unadjusted OHLCV txt files.
RAW_DATA_UNADJUSTED_OHLCV = DATA_ROOT / "raw_data_unadjusted_ohlcv"

MARKET_DATA = DATA_ROOT / "market_data"
DAILY_BARS_BY_SYMBOL = MARKET_DATA / "daily_bars_by_symbol"

FEATURE_DATA = DATA_ROOT / "feature_data"
FORMULAIC_ALPHAS_BY_SYMBOL = FEATURE_DATA / "formulaic_alphas_by_symbol"

RESEARCH_OUTPUT = DATA_ROOT / "research_output"
BACKTEST_OUTPUT = DATA_ROOT / "backtest_output"
LOGS = DATA_ROOT / "logs"


def ensure_data_dirs() -> None:
    for path in [
        RAW_DATA_ADJUSTED_OHLC,
        RAW_DATA_UNADJUSTED_OHLCV,
        DAILY_BARS_BY_SYMBOL,
        FORMULAIC_ALPHAS_BY_SYMBOL,
        RESEARCH_OUTPUT,
        BACKTEST_OUTPUT,
        LOGS,
    ]:
        path.mkdir(parents=True, exist_ok=True)
