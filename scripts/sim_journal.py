"""Extended trade journal for ADAPTIVE strategy backtests.

Extends the base TradeJournal with:
- Prediction logging (every bar's model output, even skipped ones)
- Regime tracking per bar
- Funding rate integration
- Sharpe / Sortino / loss analysis reporting
"""

from __future__ import annotations

import json
import math
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

DB_PATH = str(Path(__file__).resolve().parent.parent / "adaptive_journal.db")

SCHEMA_V2 = """
CREATE TABLE IF NOT EXISTS sim_trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    mode TEXT NOT NULL,
    date TEXT NOT NULL,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    entry_time TEXT NOT NULL,
    entry_price REAL NOT NULL,
    entry_confidence REAL NOT NULL,
    exit_time TEXT,
    exit_price REAL,
    exit_reason TEXT,
    pnl REAL,
    pnl_pct REAL,
    hold_bars INTEGER,
    regime_at_entry TEXT,
    funding_rate_at_entry REAL,
    atr_pct_at_entry REAL,
    up_prob REAL,
    down_prob REAL,
    win INTEGER DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS sim_predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    mode TEXT NOT NULL,
    date TEXT NOT NULL,
    bar_time TEXT NOT NULL,
    symbol TEXT NOT NULL,
    up_prob REAL NOT NULL,
    down_prob REAL NOT NULL,
    regime TEXT,
    atr_pct REAL,
    funding_rate REAL,
    action TEXT NOT NULL,
    block_reason TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS sim_daily (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    mode TEXT NOT NULL,
    date TEXT NOT NULL,
    return_pct REAL NOT NULL,
    wins INTEGER NOT NULL DEFAULT 0,
    losses INTEGER NOT NULL DEFAULT 0,
    entries INTEGER NOT NULL DEFAULT 0,
    exits INTEGER NOT NULL DEFAULT 0,
    regime_blocks INTEGER NOT NULL DEFAULT 0,
    max_dd REAL NOT NULL DEFAULT 0,
    final_value REAL NOT NULL DEFAULT 1000,
    sharpe_daily REAL,
    avg_win_pnl REAL,
    avg_loss_pnl REAL,
    profit_factor REAL,
    dominant_regime TEXT,
    UNIQUE(run_id, mode, date)
);

CREATE TABLE IF NOT EXISTS sim_model_state (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    date TEXT NOT NULL,
    train_samples INTEGER NOT NULL,
    feature_count INTEGER NOT NULL,
    train_time_sec REAL,
    top_features_json TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS sim_funding_rates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    funding_rate REAL NOT NULL,
    UNIQUE(symbol, timestamp)
);

CREATE INDEX IF NOT EXISTS idx_trades_run ON sim_trades(run_id, mode, date);
CREATE INDEX IF NOT EXISTS idx_trades_symbol ON sim_trades(run_id, symbol, side);
CREATE INDEX IF NOT EXISTS idx_preds_run ON sim_predictions(run_id, mode, date);
CREATE INDEX IF NOT EXISTS idx_daily_run ON sim_daily(run_id, mode, date);
CREATE INDEX IF NOT EXISTS idx_funding ON sim_funding_rates(symbol, timestamp);
"""


class SimJournal:
    """SQLite-backed journal for ADAPTIVE strategy simulation."""

    def __init__(self, db_path: str = DB_PATH, run_id: str = "default"):
        self.db_path = db_path
        self.run_id = run_id
        self._conn: Optional[sqlite3.Connection] = None
        self._pred_buffer: list[tuple] = []
        self._trade_buffer: list[tuple] = []
        self._init_db()

    def _init_db(self):
        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(SCHEMA_V2)
        self._conn.commit()

    def _ensure_conn(self):
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path)
            self._conn.row_factory = sqlite3.Row

    # ── Trade Recording ───────────────────────────────────────────────────

    def record_trade(
        self,
        mode: str,
        date: str,
        symbol: str,
        side: str,
        entry_time: str,
        entry_price: float,
        entry_confidence: float,
        exit_time: str,
        exit_price: float,
        exit_reason: str,
        pnl: float,
        pnl_pct: float,
        hold_bars: int,
        regime: str = "",
        funding_rate: float = 0.0,
        atr_pct: float = 0.0,
        up_prob: float = 0.0,
        down_prob: float = 0.0,
    ):
        """Record a completed trade."""
        self._ensure_conn()
        win = 1 if pnl > 0 else 0
        self._conn.execute(
            """INSERT INTO sim_trades
               (run_id, mode, date, symbol, side, entry_time, entry_price,
                entry_confidence, exit_time, exit_price, exit_reason,
                pnl, pnl_pct, hold_bars, regime_at_entry, funding_rate_at_entry,
                atr_pct_at_entry, up_prob, down_prob, win)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                self.run_id, mode, date, symbol, side,
                entry_time, entry_price, entry_confidence,
                exit_time, exit_price, exit_reason,
                round(pnl, 6), round(pnl_pct, 4), hold_bars,
                regime, funding_rate, atr_pct,
                round(up_prob, 4), round(down_prob, 4), win,
            ),
        )
        self._conn.commit()

    # ── Prediction Logging (buffered for performance) ─────────────────────

    def buffer_prediction(
        self,
        mode: str,
        date: str,
        bar_time: str,
        symbol: str,
        up_prob: float,
        down_prob: float,
        regime: str = "",
        atr_pct: float = 0.0,
        funding_rate: float = 0.0,
        action: str = "skip",
        block_reason: str = "",
    ):
        """Buffer a prediction for batch insert (called every bar)."""
        self._pred_buffer.append((
            self.run_id, mode, date, bar_time, symbol,
            round(up_prob, 4), round(down_prob, 4),
            regime, atr_pct, funding_rate, action, block_reason,
        ))
        if len(self._pred_buffer) >= 500:
            self.flush_predictions()

    def flush_predictions(self):
        """Batch insert buffered predictions."""
        if not self._pred_buffer:
            return
        self._ensure_conn()
        self._conn.executemany(
            """INSERT INTO sim_predictions
               (run_id, mode, date, bar_time, symbol, up_prob, down_prob,
                regime, atr_pct, funding_rate, action, block_reason)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            self._pred_buffer,
        )
        self._conn.commit()
        self._pred_buffer.clear()

    # ── Daily Summary ─────────────────────────────────────────────────────

    def record_daily(
        self,
        mode: str,
        date: str,
        return_pct: float,
        wins: int,
        losses: int,
        entries: int,
        exits: int,
        regime_blocks: int,
        max_dd: float,
        final_value: float,
        dominant_regime: str = "",
        avg_win_pnl: float = 0.0,
        avg_loss_pnl: float = 0.0,
        profit_factor: float = 0.0,
    ):
        """Record daily summary."""
        self._ensure_conn()
        self._conn.execute(
            """INSERT OR REPLACE INTO sim_daily
               (run_id, mode, date, return_pct, wins, losses, entries, exits,
                regime_blocks, max_dd, final_value, dominant_regime,
                avg_win_pnl, avg_loss_pnl, profit_factor)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                self.run_id, mode, date, round(return_pct, 4),
                wins, losses, entries, exits,
                regime_blocks, round(max_dd, 4), round(final_value, 4),
                dominant_regime, round(avg_win_pnl, 6),
                round(avg_loss_pnl, 6), round(profit_factor, 4),
            ),
        )
        self._conn.commit()

    # ── Model State ───────────────────────────────────────────────────────

    def record_model_state(
        self,
        date: str,
        train_samples: int,
        feature_count: int,
        train_time_sec: float = 0.0,
        top_features: Optional[list] = None,
    ):
        """Record model training snapshot."""
        self._ensure_conn()
        self._conn.execute(
            """INSERT INTO sim_model_state
               (run_id, date, train_samples, feature_count, train_time_sec,
                top_features_json)
               VALUES (?,?,?,?,?,?)""",
            (
                self.run_id, date, train_samples, feature_count,
                round(train_time_sec, 3),
                json.dumps(top_features) if top_features else None,
            ),
        )
        self._conn.commit()

    # ── Funding Rates ─────────────────────────────────────────────────────

    def store_funding_rates(self, symbol: str, rates: list[tuple[str, float]]):
        """Bulk store funding rates [(timestamp, rate), ...]."""
        self._ensure_conn()
        self._conn.executemany(
            """INSERT OR IGNORE INTO sim_funding_rates
               (symbol, timestamp, funding_rate) VALUES (?,?,?)""",
            [(symbol, ts, rate) for ts, rate in rates],
        )
        self._conn.commit()

    def get_funding_rate(self, symbol: str, before_time: str) -> Optional[float]:
        """Get latest funding rate before a given time."""
        self._ensure_conn()
        row = self._conn.execute(
            """SELECT funding_rate FROM sim_funding_rates
               WHERE symbol = ? AND timestamp <= ?
               ORDER BY timestamp DESC LIMIT 1""",
            (symbol, before_time),
        ).fetchone()
        return row["funding_rate"] if row else None

    # ── Analysis Queries ──────────────────────────────────────────────────

    def loss_analysis(self, mode: str = "ADAPTIVE") -> dict:
        """Detailed analysis of losing trades."""
        self._ensure_conn()
        rows = self._conn.execute(
            """SELECT * FROM sim_trades
               WHERE run_id = ? AND mode = ? AND pnl < 0
               ORDER BY pnl ASC""",
            (self.run_id, mode),
        ).fetchall()
        if not rows:
            return {"total_losses": 0}

        losses = [dict(r) for r in rows]
        total_loss = sum(t["pnl"] for t in losses)
        by_reason = {}
        by_regime = {}
        by_symbol = {}
        by_hour = {}

        for t in losses:
            reason = t.get("exit_reason", "unknown")
            by_reason[reason] = by_reason.get(reason, 0) + t["pnl"]

            regime = t.get("regime_at_entry", "unknown")
            by_regime[regime] = by_regime.get(regime, 0) + t["pnl"]

            sym = t["symbol"]
            by_symbol[sym] = by_symbol.get(sym, 0) + t["pnl"]

            hour = t["entry_time"][11:13] if len(t["entry_time"]) > 13 else "?"
            by_hour[hour] = by_hour.get(hour, 0) + t["pnl"]

        return {
            "total_losses": len(losses),
            "total_loss_pnl": round(total_loss, 4),
            "avg_loss": round(total_loss / len(losses), 4),
            "worst_trade": losses[0],
            "by_exit_reason": dict(sorted(by_reason.items(), key=lambda x: x[1])),
            "by_regime": dict(sorted(by_regime.items(), key=lambda x: x[1])),
            "by_symbol": dict(sorted(by_symbol.items(), key=lambda x: x[1])),
            "by_hour": dict(sorted(by_hour.items(), key=lambda x: x[1])),
        }

    def sharpe_sortino(self, mode: str = "ADAPTIVE") -> dict:
        """Calculate Sharpe and Sortino ratios from daily returns."""
        self._ensure_conn()
        rows = self._conn.execute(
            """SELECT return_pct FROM sim_daily
               WHERE run_id = ? AND mode = ?
               ORDER BY date""",
            (self.run_id, mode),
        ).fetchall()
        if len(rows) < 2:
            r = [r["return_pct"] for r in rows]
            return {
                "days": len(rows), "total_return": round(sum(r), 4),
                "mean_daily": round(sum(r) / max(len(r), 1), 4), "std_daily": 0,
                "sharpe_annual": 0, "sortino_annual": 0,
                "max_gain": round(max(r), 4) if r else 0,
                "max_loss": round(min(r), 4) if r else 0,
                "positive_days": sum(1 for x in r if x > 0),
                "zero_days": sum(1 for x in r if x == 0),
                "negative_days": sum(1 for x in r if x < 0),
            }

        returns = [r["return_pct"] for r in rows]
        n = len(returns)
        mean = sum(returns) / n
        variance = sum((r - mean) ** 2 for r in returns) / (n - 1)
        std = math.sqrt(variance) if variance > 0 else 0.0001

        sharpe_daily = mean / std
        sharpe_annual = sharpe_daily * math.sqrt(365)

        downside = [r for r in returns if r < 0]
        downside_var = sum(r ** 2 for r in downside) / n if downside else 0.0001
        downside_std = math.sqrt(downside_var)
        sortino_daily = mean / downside_std if downside_std > 0 else 0
        sortino_annual = sortino_daily * math.sqrt(365)

        positive = sum(1 for r in returns if r > 0)
        negative = sum(1 for r in returns if r < 0)
        zero = sum(1 for r in returns if r == 0)

        return {
            "days": n,
            "total_return": round(sum(returns), 4),
            "mean_daily": round(mean, 4),
            "std_daily": round(std, 4),
            "sharpe_annual": round(sharpe_annual, 2),
            "sortino_annual": round(sortino_annual, 2),
            "max_gain": round(max(returns), 4),
            "max_loss": round(min(returns), 4),
            "positive_days": positive,
            "zero_days": zero,
            "negative_days": negative,
        }

    def regime_accuracy(self, mode: str = "ADAPTIVE") -> dict:
        """How accurate is the regime detection? BULL should mean price goes up."""
        self._ensure_conn()
        rows = self._conn.execute(
            """SELECT regime_at_entry, side, pnl, pnl_pct
               FROM sim_trades
               WHERE run_id = ? AND mode = ?""",
            (self.run_id, mode),
        ).fetchall()
        if not rows:
            return {}

        regime_stats = {}
        for r in rows:
            reg = r["regime_at_entry"] or "unknown"
            if reg not in regime_stats:
                regime_stats[reg] = {"trades": 0, "wins": 0, "total_pnl": 0}
            regime_stats[reg]["trades"] += 1
            if r["pnl"] > 0:
                regime_stats[reg]["wins"] += 1
            regime_stats[reg]["total_pnl"] += r["pnl"]

        for reg, s in regime_stats.items():
            s["win_rate"] = round(s["wins"] / s["trades"] * 100, 1) if s["trades"] > 0 else 0
            s["total_pnl"] = round(s["total_pnl"], 4)

        return regime_stats

    def print_report(self, mode: str = "ADAPTIVE"):
        """Print comprehensive analysis report."""
        print(f"\n{'='*70}")
        print(f"  ANALYSIS REPORT — {mode} (run: {self.run_id})")
        print(f"{'='*70}")

        # Sharpe / Sortino
        ss = self.sharpe_sortino(mode)
        print(f"\n  RISK METRICS ({ss['days']} days)")
        print(f"  {'─'*40}")
        print(f"  Total Return:      {ss['total_return']:+.3f}%")
        print(f"  Mean Daily:        {ss['mean_daily']:+.4f}%")
        print(f"  Std Dev (daily):   {ss['std_daily']:.4f}%")
        print(f"  Sharpe (annual):   {ss['sharpe_annual']:+.2f}")
        print(f"  Sortino (annual):  {ss['sortino_annual']:+.2f}")
        print(f"  Best Day:          {ss['max_gain']:+.3f}%")
        print(f"  Worst Day:         {ss['max_loss']:+.3f}%")
        print(f"  Days +/0/-:        {ss['positive_days']}/{ss['zero_days']}/{ss['negative_days']}")

        # Regime accuracy
        ra = self.regime_accuracy(mode)
        if ra:
            print(f"\n  REGIME ACCURACY")
            print(f"  {'─'*40}")
            for reg, s in sorted(ra.items()):
                print(f"  {reg:10s}: {s['trades']:3d} trades | "
                      f"{s['win_rate']:5.1f}% WR | PnL: {s['total_pnl']:+.4f}")

        # Loss analysis
        la = self.loss_analysis(mode)
        if la["total_losses"] > 0:
            print(f"\n  LOSS ANALYSIS ({la['total_losses']} losing trades)")
            print(f"  {'─'*40}")
            print(f"  Total Loss:     ${la['total_loss_pnl']:+.4f}")
            print(f"  Avg Loss:       ${la['avg_loss']:+.4f}")

            print(f"\n  By Exit Reason:")
            for reason, pnl in la["by_exit_reason"].items():
                print(f"    {reason:6s}: ${pnl:+.4f}")

            print(f"\n  By Regime at Entry:")
            for reg, pnl in la["by_regime"].items():
                print(f"    {reg:10s}: ${pnl:+.4f}")

            print(f"\n  By Symbol:")
            for sym, pnl in la["by_symbol"].items():
                print(f"    {sym:12s}: ${pnl:+.4f}")

            print(f"\n  By Hour:")
            for hour, pnl in sorted(la["by_hour"].items()):
                print(f"    {hour}:00 : ${pnl:+.4f}")

        print(f"\n{'='*70}")

    # ── Cleanup ───────────────────────────────────────────────────────────

    def clear_run(self):
        """Clear all data for this run_id."""
        self._ensure_conn()
        for table in ["sim_trades", "sim_predictions", "sim_daily", "sim_model_state"]:
            self._conn.execute(
                f"DELETE FROM {table} WHERE run_id = ?", (self.run_id,)
            )
        self._conn.commit()

    def close(self):
        self.flush_predictions()
        if self._conn:
            self._conn.close()
            self._conn = None
