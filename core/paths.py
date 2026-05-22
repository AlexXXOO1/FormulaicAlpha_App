from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = APP_ROOT.parent / "FormulaicAlpha_Data"

RAW_DATA_ADJUSTED_OHLC = DATA_ROOT / "raw_data_adjusted_ohlc"
RAW_DATA_UNADJUSTED_OHLCV = DATA_ROOT / "raw_data_unadjusted_ohlcv"

MARKET_DATA = DATA_ROOT / "market_data"
DAILY_BARS_BY_SYMBOL = MARKET_DATA / "daily_bars_by_symbol"
IMPORT_REPORT = MARKET_DATA / "import_report.csv"

FEATURE_DATA = DATA_ROOT / "feature_data"
FORMULAIC_ALPHAS_BY_SYMBOL = FEATURE_DATA / "formulaic_alphas_by_symbol"

MARKET_REGIME = DATA_ROOT / "market_regime"
RESEARCH_OUTPUT = DATA_ROOT / "research_output"
CANDIDATE_OUTPUT = DATA_ROOT / "candidate_output"
DAILY_CANDIDATES = CANDIDATE_OUTPUT / "daily_candidates"
BACKTEST_OUTPUT = DATA_ROOT / "backtest_output"
LOGS_DIR = DATA_ROOT / "logs"

def ensure_data_dirs() -> None:
    """Create external data directories required by the application."""
    dirs = [
        DATA_ROOT,
        RAW_DATA_ADJUSTED_OHLC,
        RAW_DATA_UNADJUSTED_OHLCV,
        MARKET_DATA,
        DAILY_BARS_BY_SYMBOL,
        FEATURE_DATA,
        FORMULAIC_ALPHAS_BY_SYMBOL,
        MARKET_REGIME,
        RESEARCH_OUTPUT,
        CANDIDATE_OUTPUT,
        DAILY_CANDIDATES,
        BACKTEST_OUTPUT,
        LOGS_DIR,
    ]

    for path in dirs:
        path.mkdir(parents=True, exist_ok=True)

