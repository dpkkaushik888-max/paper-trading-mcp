"""S18 frozen configuration.

All values below are LOCKED per S18 spec decision D8. Do not modify mid-test.
Any change requires a new spec and restarts the 90-day clock.
"""

from __future__ import annotations

# ── Capital & sizing (S20) ──────────────────────────────────────────────────
STARTING_CAPITAL = 10_000.0
POS_SIZE_PCT = 0.15
MAX_CONCURRENT = 6

# ── Honest cost model (S16) ─────────────────────────────────────────────────
COST_PCT = 0.0020          # 20 bps per side
SLIPPAGE_BPS = 0.0005      # 5 bps per fill
SL_SLIPPAGE_BPS = 0.0010   # extra 10 bps on stop-loss fills

# ── Universe (S19 — 20 liquid crypto, pre-committed) ────────────────────────
# Paper-forward uses Binance spot symbols directly (no -USD suffix mapping).
UNIVERSE = [
    "BTCUSDT",  "ETHUSDT",  "SOLUSDT",  "AVAXUSDT", "LINKUSDT", "MATICUSDT",
    "DOGEUSDT", "XRPUSDT",  "ADAUSDT",  "DOTUSDT",  "ATOMUSDT", "NEARUSDT",
    "LTCUSDT",  "TRXUSDT",  "BCHUSDT",  "APTUSDT",  "UNIUSDT",  "ARBUSDT",
    "OPUSDT",   "SUIUSDT",
]

# ── Test duration ───────────────────────────────────────────────────────────
TEST_DAYS = 90

# ── Early-termination triggers (S18 D10) ────────────────────────────────────
MAX_DRAWDOWN_PCT = 0.15           # >15% DD → halt
MAX_CONSECUTIVE_LOSSES = 10       # 10 straight losers → halt
MAX_NO_SIGNAL_DAYS = 30           # 30 days no trades → halt
MAX_SKIPPED_DAYS = 5              # >5 cron failures → halt

# ── Data source ─────────────────────────────────────────────────────────────
BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"
BINANCE_FALLBACK_URL = "https://api.binance.us/api/v3/klines"
KLINE_INTERVAL = "1d"
KLINE_LIMIT = 300  # Need ≥200 for SMA(200) + ≥14 for ADX + buffer

# ── Paths ───────────────────────────────────────────────────────────────────
JOURNAL_PATH = "state/journal.json"
DOCS_DIR = "docs"
DASHBOARD_HTML = "docs/index.html"
DASHBOARD_DATA = "docs/data.json"
