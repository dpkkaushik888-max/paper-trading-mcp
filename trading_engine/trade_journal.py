"""Persistent trade journal with outcome tracking for the learning loop.

Stores every trade + its outcome (P&L, holding period, max adverse excursion)
in SQLite so the model can learn from its own history.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

from .config import PROJECT_ROOT

JOURNAL_DB_PATH = str(PROJECT_ROOT / "trade_journal.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS journal_trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    market TEXT NOT NULL,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL CHECK(side IN ('long', 'short')),
    entry_date TEXT NOT NULL,
    entry_price REAL NOT NULL,
    entry_confidence REAL NOT NULL,
    entry_cost REAL NOT NULL DEFAULT 0,
    shares REAL NOT NULL,
    exit_date TEXT,
    exit_price REAL,
    exit_reason TEXT,
    gross_pnl REAL,
    net_pnl REAL,
    pnl_pct REAL,
    holding_days INTEGER,
    max_adverse_excursion REAL,
    max_favorable_excursion REAL,
    status TEXT NOT NULL CHECK(status IN ('open', 'closed')) DEFAULT 'open',
    entry_features_json TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS model_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    market TEXT NOT NULL,
    snapshot_date TEXT NOT NULL,
    train_samples INTEGER NOT NULL,
    feature_count INTEGER NOT NULL,
    model_path TEXT,
    calibration_json TEXT,
    performance_json TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS daily_snapshots_tm (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    market TEXT NOT NULL,
    date TEXT NOT NULL,
    cash REAL NOT NULL,
    long_value REAL NOT NULL DEFAULT 0,
    short_value REAL NOT NULL DEFAULT 0,
    total_value REAL NOT NULL,
    daily_pnl REAL NOT NULL DEFAULT 0,
    open_longs INTEGER NOT NULL DEFAULT 0,
    open_shorts INTEGER NOT NULL DEFAULT 0,
    trades_today INTEGER NOT NULL DEFAULT 0,
    confidence_threshold REAL NOT NULL,
    model_calibration REAL,
    UNIQUE(session_id, market, date)
);

CREATE INDEX IF NOT EXISTS idx_journal_session
    ON journal_trades(session_id, market);
CREATE INDEX IF NOT EXISTS idx_journal_symbol
    ON journal_trades(session_id, symbol, status);
CREATE INDEX IF NOT EXISTS idx_snapshots_session
    ON model_snapshots(session_id, market);
CREATE INDEX IF NOT EXISTS idx_daily_session
    ON daily_snapshots_tm(session_id, market, date);
"""


class TradeJournal:
    """SQLite-backed trade journal with outcome tracking."""

    def __init__(self, db_path: str = JOURNAL_DB_PATH, session_id: str = "default"):
        self.db_path = db_path
        self.session_id = session_id
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()

    def _init_db(self):
        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def _ensure_conn(self):
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path)
            self._conn.row_factory = sqlite3.Row

    def open_trade(
        self,
        market: str,
        symbol: str,
        side: str,
        entry_date: str,
        entry_price: float,
        entry_confidence: float,
        shares: float,
        entry_cost: float = 0.0,
        entry_features: Optional[dict] = None,
    ) -> int:
        """Record a new trade opening. Returns trade ID."""
        self._ensure_conn()
        cur = self._conn.execute(
            """INSERT INTO journal_trades
               (session_id, market, symbol, side, entry_date, entry_price,
                entry_confidence, entry_cost, shares, status, entry_features_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'open', ?)""",
            (
                self.session_id, market, symbol, side, entry_date,
                entry_price, entry_confidence, entry_cost, shares,
                json.dumps(entry_features) if entry_features else None,
            ),
        )
        self._conn.commit()
        return cur.lastrowid

    def close_trade(
        self,
        trade_id: int,
        exit_date: str,
        exit_price: float,
        exit_reason: str,
        gross_pnl: float,
        net_pnl: float,
        max_adverse_excursion: float = 0.0,
        max_favorable_excursion: float = 0.0,
    ):
        """Record trade closing with outcome metrics."""
        self._ensure_conn()
        row = self._conn.execute(
            "SELECT entry_date, entry_price, shares FROM journal_trades WHERE id = ?",
            (trade_id,),
        ).fetchone()
        if row is None:
            return

        from datetime import datetime as dt
        entry_dt = dt.strptime(row["entry_date"][:10], "%Y-%m-%d")
        exit_dt = dt.strptime(exit_date[:10], "%Y-%m-%d")
        holding_days = (exit_dt - entry_dt).days
        pnl_pct = net_pnl / (row["entry_price"] * row["shares"]) * 100 if row["entry_price"] * row["shares"] > 0 else 0

        self._conn.execute(
            """UPDATE journal_trades SET
               exit_date = ?, exit_price = ?, exit_reason = ?,
               gross_pnl = ?, net_pnl = ?, pnl_pct = ?,
               holding_days = ?, max_adverse_excursion = ?,
               max_favorable_excursion = ?, status = 'closed'
               WHERE id = ?""",
            (
                exit_date, exit_price, exit_reason,
                gross_pnl, net_pnl, round(pnl_pct, 3),
                holding_days, max_adverse_excursion,
                max_favorable_excursion, trade_id,
            ),
        )
        self._conn.commit()

    def get_open_trades(self, market: str) -> list[dict]:
        """Get all open trades for a market."""
        self._ensure_conn()
        rows = self._conn.execute(
            """SELECT * FROM journal_trades
               WHERE session_id = ? AND market = ? AND status = 'open'""",
            (self.session_id, market),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_closed_trades(
        self, market: str, symbol: Optional[str] = None, limit: int = 500
    ) -> list[dict]:
        """Get closed trades, optionally filtered by symbol."""
        self._ensure_conn()
        if symbol:
            rows = self._conn.execute(
                """SELECT * FROM journal_trades
                   WHERE session_id = ? AND market = ? AND symbol = ? AND status = 'closed'
                   ORDER BY exit_date DESC LIMIT ?""",
                (self.session_id, market, symbol, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                """SELECT * FROM journal_trades
                   WHERE session_id = ? AND market = ? AND status = 'closed'
                   ORDER BY exit_date DESC LIMIT ?""",
                (self.session_id, market, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_symbol_stats(self, market: str, symbol: str) -> dict:
        """Get trading stats for a specific symbol."""
        self._ensure_conn()
        closed = self.get_closed_trades(market, symbol)
        if not closed:
            return {"trades": 0, "win_rate": 0, "avg_pnl_pct": 0, "avg_holding_days": 0}

        wins = sum(1 for t in closed if (t.get("net_pnl") or 0) > 0)
        total = len(closed)
        avg_pnl = sum(t.get("pnl_pct", 0) or 0 for t in closed) / total
        avg_hold = sum(t.get("holding_days", 0) or 0 for t in closed) / total

        return {
            "trades": total,
            "win_rate": round(wins / total * 100, 1) if total > 0 else 0,
            "avg_pnl_pct": round(avg_pnl, 3),
            "avg_holding_days": round(avg_hold, 1),
            "last_pnl": closed[0].get("net_pnl", 0) if closed else 0,
        }

    def get_recent_calibration_data(
        self, market: str, lookback_trades: int = 100
    ) -> list[dict]:
        """Get recent trades with confidence + outcome for calibration."""
        self._ensure_conn()
        rows = self._conn.execute(
            """SELECT entry_confidence, net_pnl, pnl_pct, side
               FROM journal_trades
               WHERE session_id = ? AND market = ? AND status = 'closed'
               ORDER BY exit_date DESC LIMIT ?""",
            (self.session_id, market, lookback_trades),
        ).fetchall()
        return [dict(r) for r in rows]

    def save_model_snapshot(
        self,
        market: str,
        snapshot_date: str,
        train_samples: int,
        feature_count: int,
        model_path: Optional[str] = None,
        calibration: Optional[dict] = None,
        performance: Optional[dict] = None,
    ):
        """Record a model training snapshot."""
        self._ensure_conn()
        self._conn.execute(
            """INSERT INTO model_snapshots
               (session_id, market, snapshot_date, train_samples, feature_count,
                model_path, calibration_json, performance_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                self.session_id, market, snapshot_date, train_samples,
                feature_count, model_path,
                json.dumps(calibration) if calibration else None,
                json.dumps(performance) if performance else None,
            ),
        )
        self._conn.commit()

    def save_daily_snapshot(
        self,
        market: str,
        date_str: str,
        cash: float,
        long_value: float,
        short_value: float,
        total_value: float,
        daily_pnl: float,
        open_longs: int,
        open_shorts: int,
        trades_today: int,
        confidence_threshold: float,
        model_calibration: Optional[float] = None,
    ):
        """Record daily portfolio snapshot."""
        self._ensure_conn()
        self._conn.execute(
            """INSERT OR REPLACE INTO daily_snapshots_tm
               (session_id, market, date, cash, long_value, short_value,
                total_value, daily_pnl, open_longs, open_shorts,
                trades_today, confidence_threshold, model_calibration)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                self.session_id, market, date_str, cash, long_value,
                short_value, total_value, daily_pnl, open_longs,
                open_shorts, trades_today, confidence_threshold,
                model_calibration,
            ),
        )
        self._conn.commit()

    def get_daily_snapshots(self, market: str) -> list[dict]:
        """Get all daily snapshots for a market session."""
        self._ensure_conn()
        rows = self._conn.execute(
            """SELECT * FROM daily_snapshots_tm
               WHERE session_id = ? AND market = ?
               ORDER BY date""",
            (self.session_id, market),
        ).fetchall()
        return [dict(r) for r in rows]

    def clear_session(self, market: str):
        """Clear all data for a session+market (for fresh backtest runs)."""
        self._ensure_conn()
        for table in ["journal_trades", "model_snapshots", "daily_snapshots_tm"]:
            self._conn.execute(
                f"DELETE FROM {table} WHERE session_id = ? AND market = ?",
                (self.session_id, market),
            )
        self._conn.commit()

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None
