"""Configuration constants for the paper trading engine."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

DB_PATH = os.environ.get(
    "TRADING_DB_PATH",
    str(PROJECT_ROOT / "paper_portfolio.db"),
)

RULES_PATH = os.environ.get(
    "TRADING_RULES_PATH",
    str(PROJECT_ROOT / "rules.json"),
)

BROKER_PROFILES_DIR = PROJECT_ROOT / "broker_profiles"

DEFAULT_BROKER = os.environ.get("TRADING_BROKER", "etoro")

DEFAULT_SESSION_ID = os.environ.get("TRADING_SESSION", "default")

DEFAULT_INITIAL_CAPITAL = float(os.environ.get("TRADING_CAPITAL", "1000.0"))

WATCHLIST = [
    "SPY",   # S&P 500
    "QQQ",   # Nasdaq 100
    "IWM",   # Russell 2000
    "DIA",   # Dow Jones
    "XLF",   # Financials
    "XLK",   # Technology
    "GLD",   # Gold
    "TLT",   # 20+ Year Treasury
]

INDICATOR_PERIODS = {
    "rsi_14": 14,
    "rsi_3": 3,
    "ema_8": 8,
    "ema_20": 20,
    "ema_50": 50,
    "ema_200": 200,
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    "bb_period": 20,
    "bb_std": 2.0,
    "atr_period": 14,
}

HISTORY_DAYS = 200

MAX_POSITION_PCT = 0.10
MAX_DAILY_LOSS_PCT = 0.02

MARKET_OPEN_HOUR_ET = 9
MARKET_OPEN_MIN_ET = 30
MARKET_CLOSE_HOUR_ET = 16
MARKET_CLOSE_MIN_ET = 0
