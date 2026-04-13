"""SQLite paper portfolio — positions, trades, P&L tracking."""

from __future__ import annotations

import sqlite3
from datetime import datetime, date
from typing import Optional

from .config import DB_PATH, DEFAULT_SESSION_ID, DEFAULT_INITIAL_CAPITAL, DEFAULT_BROKER
from .models import (
    CostBreakdown,
    DailySnapshot,
    PortfolioSummary,
    Position,
    Trade,
)


SCHEMA = """
CREATE TABLE IF NOT EXISTS portfolio (
    id INTEGER PRIMARY KEY,
    session_id TEXT NOT NULL DEFAULT 'default',
    broker_profile TEXT NOT NULL DEFAULT 'etoro',
    initial_capital REAL NOT NULL,
    cash REAL NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(session_id)
);

CREATE TABLE IF NOT EXISTS positions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL DEFAULT 'default',
    symbol TEXT NOT NULL,
    side TEXT NOT NULL CHECK(side IN ('long', 'short')),
    shares REAL NOT NULL,
    entry_price REAL NOT NULL,
    entry_cost REAL NOT NULL,
    exit_price REAL,
    exit_proceeds REAL,
    pnl REAL,
    status TEXT NOT NULL CHECK(status IN ('open', 'closed')),
    entry_time TEXT NOT NULL,
    exit_time TEXT,
    entry_reason TEXT,
    exit_reason TEXT
);

CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL DEFAULT 'default',
    position_id INTEGER REFERENCES positions(id),
    symbol TEXT NOT NULL,
    action TEXT NOT NULL CHECK(action IN ('buy', 'sell')),
    shares REAL NOT NULL,
    price REAL NOT NULL,
    spread_cost REAL NOT NULL,
    slippage_cost REAL NOT NULL,
    fx_cost REAL NOT NULL DEFAULT 0,
    total_cost REAL NOT NULL,
    gross_value REAL NOT NULL,
    net_value REAL NOT NULL,
    timestamp TEXT NOT NULL,
    reason TEXT
);

CREATE TABLE IF NOT EXISTS daily_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL DEFAULT 'default',
    date TEXT NOT NULL,
    cash REAL NOT NULL,
    positions_value REAL NOT NULL,
    total_value REAL NOT NULL,
    daily_pnl_gross REAL NOT NULL,
    daily_pnl_net REAL NOT NULL,
    cumulative_pnl_gross REAL NOT NULL,
    cumulative_pnl_net REAL NOT NULL,
    total_spread_cost REAL NOT NULL,
    total_slippage_cost REAL NOT NULL,
    total_fx_cost REAL NOT NULL,
    estimated_tax REAL NOT NULL,
    daily_pnl_after_tax REAL NOT NULL,
    cost_drag_pct REAL NOT NULL,
    trades_today INTEGER NOT NULL,
    wins_today INTEGER NOT NULL,
    losses_today INTEGER NOT NULL,
    UNIQUE(session_id, date)
);
"""


class Portfolio:
    """SQLite-backed paper trading portfolio."""

    def __init__(
        self,
        db_path: str = DB_PATH,
        session_id: str = DEFAULT_SESSION_ID,
        broker: str = DEFAULT_BROKER,
        initial_capital: float = DEFAULT_INITIAL_CAPITAL,
    ):
        self.db_path = db_path
        self.session_id = session_id
        self.broker = broker
        self.initial_capital = initial_capital
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self):
        conn = self._get_conn()
        try:
            conn.executescript(SCHEMA)
            existing = conn.execute(
                "SELECT id FROM portfolio WHERE session_id = ?",
                (self.session_id,),
            ).fetchone()
            if not existing:
                conn.execute(
                    "INSERT INTO portfolio (session_id, broker_profile, initial_capital, cash, created_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (
                        self.session_id,
                        self.broker,
                        self.initial_capital,
                        self.initial_capital,
                        datetime.now().isoformat(),
                    ),
                )
            conn.commit()
        finally:
            conn.close()

    def get_cash(self) -> float:
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT cash FROM portfolio WHERE session_id = ?",
                (self.session_id,),
            ).fetchone()
            return float(row["cash"]) if row else 0.0
        finally:
            conn.close()

    def _update_cash(self, conn: sqlite3.Connection, amount: float):
        conn.execute(
            "UPDATE portfolio SET cash = cash + ? WHERE session_id = ?",
            (amount, self.session_id),
        )

    def open_position(
        self,
        symbol: str,
        shares: float,
        price: float,
        costs: CostBreakdown,
        reason: str = "",
    ) -> dict:
        """Open a new long position. Returns position + trade dicts."""
        if shares <= 0:
            return {"error": "Cannot open position with 0 shares. Price too high for position limit."}
        now = datetime.now().isoformat()
        conn = self._get_conn()
        try:
            cash = float(
                conn.execute(
                    "SELECT cash FROM portfolio WHERE session_id = ?",
                    (self.session_id,),
                ).fetchone()["cash"]
            )
            if costs.net_value > cash:
                return {
                    "error": f"Insufficient cash. Need ${costs.net_value:.2f}, have ${cash:.2f}"
                }

            cursor = conn.execute(
                "INSERT INTO positions "
                "(session_id, symbol, side, shares, entry_price, entry_cost, status, entry_time, entry_reason) "
                "VALUES (?, ?, 'long', ?, ?, ?, 'open', ?, ?)",
                (
                    self.session_id,
                    symbol,
                    shares,
                    price,
                    costs.net_value,
                    now,
                    reason,
                ),
            )
            position_id = cursor.lastrowid

            conn.execute(
                "INSERT INTO trades "
                "(session_id, position_id, symbol, action, shares, price, "
                "spread_cost, slippage_cost, fx_cost, total_cost, gross_value, net_value, "
                "timestamp, reason) "
                "VALUES (?, ?, ?, 'buy', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    self.session_id,
                    position_id,
                    symbol,
                    shares,
                    price,
                    costs.spread_cost,
                    costs.slippage_cost,
                    costs.fx_cost,
                    costs.total_cost,
                    costs.gross_value,
                    costs.net_value,
                    now,
                    reason,
                ),
            )

            self._update_cash(conn, -costs.net_value)
            conn.commit()

            return {
                "status": "opened",
                "position_id": position_id,
                "symbol": symbol,
                "shares": shares,
                "entry_price": round(price, 4),
                "costs": costs.to_dict(),
                "cash_remaining": round(cash - costs.net_value, 2),
            }
        finally:
            conn.close()

    def close_position(
        self,
        position_id: int,
        price: float,
        costs: CostBreakdown,
        reason: str = "",
    ) -> dict:
        """Close an open position. Returns P&L breakdown."""
        now = datetime.now().isoformat()
        conn = self._get_conn()
        try:
            pos = conn.execute(
                "SELECT * FROM positions WHERE id = ? AND session_id = ? AND status = 'open'",
                (position_id, self.session_id),
            ).fetchone()
            if not pos:
                return {"error": f"No open position with id {position_id}"}

            gross_pnl = (price - pos["entry_price"]) * pos["shares"]
            entry_costs_row = conn.execute(
                "SELECT SUM(total_cost) as entry_total_cost FROM trades "
                "WHERE position_id = ? AND action = 'buy'",
                (position_id,),
            ).fetchone()
            entry_total_cost = float(entry_costs_row["entry_total_cost"]) if entry_costs_row else 0.0
            net_pnl = gross_pnl - entry_total_cost - costs.total_cost

            conn.execute(
                "UPDATE positions SET exit_price = ?, exit_proceeds = ?, pnl = ?, "
                "status = 'closed', exit_time = ?, exit_reason = ? WHERE id = ?",
                (price, costs.net_value, net_pnl, now, reason, position_id),
            )

            conn.execute(
                "INSERT INTO trades "
                "(session_id, position_id, symbol, action, shares, price, "
                "spread_cost, slippage_cost, fx_cost, total_cost, gross_value, net_value, "
                "timestamp, reason) "
                "VALUES (?, ?, ?, 'sell', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    self.session_id,
                    position_id,
                    pos["symbol"],
                    pos["shares"],
                    price,
                    costs.spread_cost,
                    costs.slippage_cost,
                    costs.fx_cost,
                    costs.total_cost,
                    costs.gross_value,
                    costs.net_value,
                    now,
                    reason,
                ),
            )

            self._update_cash(conn, costs.net_value)
            conn.commit()

            return {
                "status": "closed",
                "position_id": position_id,
                "symbol": pos["symbol"],
                "shares": float(pos["shares"]),
                "entry_price": round(float(pos["entry_price"]), 4),
                "exit_price": round(price, 4),
                "gross_pnl": round(gross_pnl, 4),
                "entry_costs": round(entry_total_cost, 4),
                "exit_costs": costs.to_dict(),
                "net_pnl": round(net_pnl, 4),
                "outcome": "win" if net_pnl > 0 else "loss",
            }
        finally:
            conn.close()

    def get_open_positions(self) -> list[dict]:
        conn = self._get_conn()
        try:
            rows = conn.execute(
                "SELECT * FROM positions WHERE session_id = ? AND status = 'open'",
                (self.session_id,),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def get_closed_positions(self, limit: int = 50) -> list[dict]:
        conn = self._get_conn()
        try:
            rows = conn.execute(
                "SELECT * FROM positions WHERE session_id = ? AND status = 'closed' "
                "ORDER BY exit_time DESC LIMIT ?",
                (self.session_id, limit),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def get_position_for_symbol(self, symbol: str) -> Optional[dict]:
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT * FROM positions WHERE session_id = ? AND symbol = ? AND status = 'open'",
                (self.session_id, symbol),
            ).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def get_trades(
        self,
        limit: int = 50,
        symbol: Optional[str] = None,
    ) -> list[dict]:
        conn = self._get_conn()
        try:
            if symbol:
                rows = conn.execute(
                    "SELECT * FROM trades WHERE session_id = ? AND symbol = ? "
                    "ORDER BY timestamp DESC LIMIT ?",
                    (self.session_id, symbol, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM trades WHERE session_id = ? "
                    "ORDER BY timestamp DESC LIMIT ?",
                    (self.session_id, limit),
                ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def get_cumulative_costs(self) -> dict:
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT "
                "  COALESCE(SUM(spread_cost), 0) as total_spread, "
                "  COALESCE(SUM(slippage_cost), 0) as total_slippage, "
                "  COALESCE(SUM(fx_cost), 0) as total_fx, "
                "  COALESCE(SUM(total_cost), 0) as total_all, "
                "  COUNT(*) as trade_count "
                "FROM trades WHERE session_id = ?",
                (self.session_id,),
            ).fetchone()

            realized_row = conn.execute(
                "SELECT COALESCE(SUM(pnl), 0) as realized_pnl "
                "FROM positions WHERE session_id = ? AND status = 'closed'",
                (self.session_id,),
            ).fetchone()
            realized_pnl = float(realized_row["realized_pnl"])

            total_costs = float(row["total_all"])
            gross_pnl = realized_pnl + total_costs

            return {
                "total_spread_cost": round(float(row["total_spread"]), 4),
                "total_slippage_cost": round(float(row["total_slippage"]), 4),
                "total_fx_cost": round(float(row["total_fx"]), 4),
                "total_all_costs": round(total_costs, 4),
                "trade_count": row["trade_count"],
                "realized_pnl_net": round(realized_pnl, 4),
                "realized_pnl_gross": round(gross_pnl, 4),
                "cost_drag_pct": round(
                    (total_costs / gross_pnl * 100) if gross_pnl > 0 else 0, 2
                ),
            }
        finally:
            conn.close()

    def get_portfolio_summary(
        self,
        current_prices: Optional[dict[str, float]] = None,
    ) -> PortfolioSummary:
        """Get full portfolio summary with unrealized P&L."""
        conn = self._get_conn()
        try:
            pf = conn.execute(
                "SELECT * FROM portfolio WHERE session_id = ?",
                (self.session_id,),
            ).fetchone()

            positions = self.get_open_positions()
            positions_value = 0.0
            unrealized_pnl = 0.0

            enriched_positions = []
            for pos in positions:
                current_price = (
                    current_prices.get(pos["symbol"], pos["entry_price"])
                    if current_prices
                    else pos["entry_price"]
                )
                value = current_price * pos["shares"]
                pnl = (current_price - pos["entry_price"]) * pos["shares"]
                positions_value += value
                unrealized_pnl += pnl
                enriched_positions.append({
                    **pos,
                    "current_price": round(current_price, 4),
                    "market_value": round(value, 2),
                    "unrealized_pnl": round(pnl, 2),
                })

            cash = float(pf["cash"])
            costs = self.get_cumulative_costs()

            realized_row = conn.execute(
                "SELECT COALESCE(SUM(pnl), 0) as realized "
                "FROM positions WHERE session_id = ? AND status = 'closed'",
                (self.session_id,),
            ).fetchone()
            realized_pnl = float(realized_row["realized"])

            return PortfolioSummary(
                session_id=self.session_id,
                broker_profile=pf["broker_profile"],
                initial_capital=float(pf["initial_capital"]),
                cash=cash,
                positions=enriched_positions,
                positions_value=positions_value,
                total_value=cash + positions_value,
                unrealized_pnl=unrealized_pnl,
                realized_pnl=realized_pnl,
                total_costs=costs["total_all_costs"],
                cost_breakdown=costs,
            )
        finally:
            conn.close()

    def get_daily_trades_count(self, trade_date: Optional[str] = None) -> int:
        trade_date = trade_date or date.today().isoformat()
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT COUNT(*) as cnt FROM trades "
                "WHERE session_id = ? AND date(timestamp) = ?",
                (self.session_id, trade_date),
            ).fetchone()
            return row["cnt"] if row else 0
        finally:
            conn.close()

    def get_daily_pnl(self, trade_date: Optional[str] = None) -> float:
        trade_date = trade_date or date.today().isoformat()
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT COALESCE(SUM(pnl), 0) as daily_pnl "
                "FROM positions WHERE session_id = ? AND status = 'closed' "
                "AND date(exit_time) = ?",
                (self.session_id, trade_date),
            ).fetchone()
            return float(row["daily_pnl"]) if row else 0.0
        finally:
            conn.close()

    def save_daily_snapshot(self, snapshot: DailySnapshot):
        conn = self._get_conn()
        try:
            conn.execute(
                "INSERT OR REPLACE INTO daily_snapshots "
                "(session_id, date, cash, positions_value, total_value, "
                "daily_pnl_gross, daily_pnl_net, cumulative_pnl_gross, cumulative_pnl_net, "
                "total_spread_cost, total_slippage_cost, total_fx_cost, "
                "estimated_tax, daily_pnl_after_tax, cost_drag_pct, "
                "trades_today, wins_today, losses_today) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    snapshot.session_id,
                    snapshot.date,
                    snapshot.cash,
                    snapshot.positions_value,
                    snapshot.total_value,
                    snapshot.daily_pnl_gross,
                    snapshot.daily_pnl_net,
                    snapshot.cumulative_pnl_gross,
                    snapshot.cumulative_pnl_net,
                    snapshot.total_spread_cost,
                    snapshot.total_slippage_cost,
                    snapshot.total_fx_cost,
                    snapshot.estimated_tax,
                    snapshot.daily_pnl_after_tax,
                    snapshot.cost_drag_pct,
                    snapshot.trades_today,
                    snapshot.wins_today,
                    snapshot.losses_today,
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def get_snapshots(self, limit: int = 30) -> list[dict]:
        conn = self._get_conn()
        try:
            rows = conn.execute(
                "SELECT * FROM daily_snapshots WHERE session_id = ? "
                "ORDER BY date DESC LIMIT ?",
                (self.session_id, limit),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()
