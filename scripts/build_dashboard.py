"""Generate static dashboard for S18 paper-forward from state/journal.json.

Outputs:
  docs/index.html   — single-page dashboard (Chart.js from CDN)
  docs/data.json    — machine-readable snapshot for the page JS

GitHub Pages serves docs/ automatically (configured in repo settings).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from trading_engine.paper.config import (
    JOURNAL_PATH, DOCS_DIR, TEST_DAYS, STARTING_CAPITAL,
)
from trading_engine.paper.journal import Journal
from trading_engine.paper.shadow import SHADOW_VARIANTS
from trading_engine.paper.pool import dashboard as pool_dash

POOL_PLACEHOLDER = "<!--S35-POOL-->\n"


# S20 backtest baselines for side-by-side comparison (from specs/S20)
BACKTEST_BASELINE = {
    "cagr_pct": 13.39,
    "sharpe": 1.064,
    "max_dd_pct": 7.68,
    "win_rate_pct": 70.5,
    "profit_factor": 2.40,
    "trades": 44,
    "holdout_days": 326,
}


def compute_benchmark_metrics(j: Journal, key: str) -> dict:
    """Compute days-elapsed metrics for a benchmark (bh_btc_value / bh_basket_value)."""
    days = j.days
    starting = j.starting_capital
    if not days:
        return {"value": starting, "total_return_pct": 0.0, "cagr_pct": 0.0, "sharpe": 0.0, "max_dd_pct": 0.0, "calmar": None}
    # Defensively drop non-positive values (bad/missing days) so they cannot
    # produce a fake 100% drawdown or negative Sharpe.
    vals = [v for v in (d.get(key, starting) for d in days) if v and v > 0]
    if not vals:
        vals = [starting]
    current = vals[-1]
    n_days = len(vals)
    total_ret = (current / starting - 1.0)
    cagr = ((current / starting) ** (365 / max(n_days, 1)) - 1.0)
    rets = [(vals[i] / vals[i - 1] - 1.0) for i in range(1, len(vals)) if vals[i - 1] > 0]
    if len(rets) >= 2:
        import statistics
        m = statistics.mean(rets)
        s = statistics.pstdev(rets)
        sharpe = (m / s * (365 ** 0.5)) if s > 0 else 0.0
    else:
        sharpe = 0.0
    # Max DD
    peak = vals[0]
    max_dd = 0.0
    for v in vals:
        peak = max(peak, v)
        dd = (peak - v) / peak if peak > 0 else 0.0
        max_dd = max(max_dd, dd)
    return {
        "value": current,
        "total_return_pct": total_ret * 100,
        "cagr_pct": cagr * 100,
        "sharpe": sharpe,
        "max_dd_pct": max_dd * 100,
        "calmar": calmar(cagr * 100, max_dd * 100),
    }


def calmar(cagr_pct: float, max_dd_pct: float):
    """Return per unit of drawdown (CAGR % / max DD %); None if there is no drawdown."""
    return (cagr_pct / max_dd_pct) if max_dd_pct and max_dd_pct > 0 else None


def compute_live_metrics(j: Journal) -> dict:
    days = j.days
    closed = j.closed_trades
    starting = j.starting_capital
    current = j.last_portfolio_value()

    # Duration
    n_days = len(days)
    total_return = (current / starting - 1.0) if starting > 0 else 0.0
    cagr = ((current / starting) ** (365 / max(n_days, 1)) - 1.0) if n_days > 0 else 0.0

    # Win rate / PF
    wins = [t for t in closed if t.pnl > 0]
    losses = [t for t in closed if t.pnl <= 0]
    win_rate = (len(wins) / len(closed)) if closed else 0.0
    total_win = sum(t.pnl for t in wins)
    total_loss = -sum(t.pnl for t in losses)
    pf = (total_win / total_loss) if total_loss > 0 else float("inf") if wins else 0.0

    # Daily returns for Sharpe
    vals = [d["portfolio_value"] for d in days]
    rets = [(vals[i] / vals[i - 1] - 1.0) for i in range(1, len(vals))]
    if len(rets) >= 2:
        import statistics
        mean_r = statistics.mean(rets)
        std_r = statistics.pstdev(rets)
        sharpe = (mean_r / std_r * (365 ** 0.5)) if std_r > 0 else 0.0
    else:
        sharpe = 0.0

    return {
        "days_elapsed": n_days,
        "total_return_pct": total_return * 100,
        "cagr_pct": cagr * 100,
        "sharpe": sharpe,
        "max_dd_pct": j.max_drawdown() * 100,
        "calmar": calmar(cagr * 100, j.max_drawdown() * 100),
        "win_rate_pct": win_rate * 100,
        "profit_factor": pf if pf != float("inf") else None,
        "trades": len(closed),
        "wins": len(wins),
        "losses": len(losses),
        "portfolio_value": current,
        "cash": j.cash,
        "n_open": len(j.open_positions),
    }


def compute_utilisation(j: Journal) -> dict:
    """Time-weighted capital use: mean over days of (portfolio_value - cash) / portfolio_value,
    and the number of days with no open position. Skipped / non-positive days are ignored."""
    days = [d for d in j.days if d.get("portfolio_value") and d["portfolio_value"] > 0
            and not d.get("skipped")]
    if not days:
        return {"invested_share_avg": None, "days_fully_in_cash": 0}
    shares = [(d["portfolio_value"] - d["cash"]) / d["portfolio_value"] for d in days]
    return {
        "invested_share_avg": sum(shares) / len(shares),
        "days_fully_in_cash": sum(1 for d in days if d.get("n_open", 0) == 0),
    }


def series_metrics(vals: list[float]) -> dict:
    """Return % / Sharpe / max DD % for a value series whose first element is the baseline.
    Sharpe is None with fewer than 2 daily returns (rendered as an em dash)."""
    import statistics
    vals = [v for v in vals if v and v > 0]
    if len(vals) < 2:
        return {"return_pct": None, "sharpe": None, "max_dd_pct": None}
    rets = [vals[i] / vals[i - 1] - 1.0 for i in range(1, len(vals))]
    sharpe = None
    if len(rets) >= 2:
        sd = statistics.pstdev(rets)
        sharpe = (statistics.mean(rets) / sd * (365 ** 0.5)) if sd > 0 else 0.0
    peak, max_dd = vals[0], 0.0
    for v in vals:
        peak = max(peak, v)
        max_dd = max(max_dd, (peak - v) / peak)
    return {"return_pct": (vals[-1] / vals[0] - 1.0) * 100, "sharpe": sharpe, "max_dd_pct": max_dd * 100}


def compute_shadow_rows(j: Journal) -> list[dict]:
    """One row per shadow series with >= 1 day of data, plus Connors over the same window.
    Both series are measured from their value just before the window (shadow: its
    starting capital; Connors: previous day's portfolio value, or starting capital)."""
    rows = []
    days = j.days
    last_errors = (days[-1].get("shadow_errors") or {}) if days else {}
    for v in SHADOW_VARIANTS:
        key = f"{v.name}_value"
        idx = [i for i, d in enumerate(days) if (d.get(key) or 0) > 0]
        error = last_errors.get(v.name)
        if not idx:
            if error:  # failing since its first run: still show it, flagged
                rows.append({"name": v.name, "start_date": None, "days": 0, "value": None,
                             "return_pct": None, "sharpe": None, "max_dd_pct": None,
                             "closed_trades": None, "connors_return_pct": None,
                             "connors_sharpe": None, "connors_max_dd_pct": None, "error": error})
            continue
        first = idx[0]
        book = j.get_benchmark(v.name)
        base = float(book.get("starting_capital") or STARTING_CAPITAL)
        window = days[first:]
        shadow_vals = [base] + [d[key] for d in window if (d.get(key) or 0) > 0]
        connors_base = days[first - 1]["portfolio_value"] if first > 0 else j.starting_capital
        connors_vals = [connors_base] + [d["portfolio_value"] for d in window]
        sm, cm = series_metrics(shadow_vals), series_metrics(connors_vals)
        rows.append({
            "name": v.name,
            "start_date": days[first]["date"],
            "days": len(window),
            "value": shadow_vals[-1],
            "return_pct": sm["return_pct"],
            "sharpe": sm["sharpe"],
            "max_dd_pct": sm["max_dd_pct"],
            "closed_trades": len(book.get("closed_trades") or []),
            "connors_return_pct": cm["return_pct"],
            "connors_sharpe": cm["sharpe"],
            "connors_max_dd_pct": cm["max_dd_pct"],
            "error": error,
        })
    return rows


def build_data_json(j: Journal) -> dict:
    metrics = compute_live_metrics(j)
    metrics.update(compute_utilisation(j))
    bh_btc_metrics = compute_benchmark_metrics(j, "bh_btc_value")
    bh_basket_metrics = compute_benchmark_metrics(j, "bh_basket_value")

    equity_curve = [
        {
            "date": d["date"],
            "connors": d["portfolio_value"],
            "bh_btc": d.get("bh_btc_value", STARTING_CAPITAL),
            "bh_basket": d.get("bh_basket_value", STARTING_CAPITAL),
        }
        for d in j.days
    ]
    # Prepend day-0 (before first run).
    if j.days:
        equity_curve.insert(0, {
            "date": j.start_date, "connors": STARTING_CAPITAL,
            "bh_btc": STARTING_CAPITAL, "bh_basket": STARTING_CAPITAL,
        })

    open_positions = []
    latest_closes = j.days[-1]["closes"] if j.days else {}
    for p in j.open_positions:
        current_price = latest_closes.get(p.symbol, p.entry_price)
        unrealized = (current_price - p.entry_price) * p.shares
        open_positions.append({
            "symbol": p.symbol,
            "entry_date": p.entry_date,
            "entry_price": p.entry_price,
            "current_price": current_price,
            "shares": p.shares,
            "unrealized_pnl": unrealized,
            "unrealized_pct": (current_price / p.entry_price - 1) * 100,
        })

    closed_trades = [
        {
            "symbol": t.symbol, "entry_date": t.entry_date, "exit_date": t.exit_date,
            "entry_price": t.entry_price, "exit_price": t.exit_price,
            "pnl": t.pnl, "pnl_pct": t.pnl_pct * 100,
            "hold_days": t.hold_days, "reason": t.reason,
        }
        for t in j.closed_trades
    ]

    # Per-symbol attribution
    per_symbol: dict[str, dict] = {}
    for t in closed_trades:
        s = per_symbol.setdefault(t["symbol"], {"trades": 0, "wins": 0, "pnl": 0.0})
        s["trades"] += 1
        s["pnl"] += t["pnl"]
        if t["pnl"] > 0:
            s["wins"] += 1

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "start_date": j.start_date,
        "starting_capital": j.starting_capital,
        "test_days": TEST_DAYS,
        "metrics": metrics,
        "bh_btc_metrics": bh_btc_metrics,
        "bh_basket_metrics": bh_basket_metrics,
        "backtest_baseline": {**BACKTEST_BASELINE,
                              "calmar": calmar(BACKTEST_BASELINE["cagr_pct"], BACKTEST_BASELINE["max_dd_pct"])},
        "equity_curve": equity_curve,
        "open_positions": open_positions,
        "closed_trades": closed_trades,
        "per_symbol": per_symbol,
        "shadows": compute_shadow_rows(j),
    }


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>S18 Paper-Forward Dashboard</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  * { box-sizing: border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    margin: 0; padding: 1rem; background: #0d1117; color: #e6edf3;
    max-width: 1200px; margin-left: auto; margin-right: auto;
  }
  h1, h2 { margin: 1rem 0 0.5rem; }
  h1 { font-size: 1.5rem; border-bottom: 1px solid #30363d; padding-bottom: 0.5rem; }
  .muted { color: #7d8590; font-size: 0.9rem; }
  .kpi-grid {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 0.75rem; margin: 1rem 0;
  }
  .kpi {
    background: #161b22; border: 1px solid #30363d; border-radius: 6px;
    padding: 0.75rem;
  }
  .kpi-label { font-size: 0.75rem; text-transform: uppercase; color: #7d8590; }
  .kpi-value { font-size: 1.3rem; font-weight: 600; margin-top: 0.25rem; }
  .pos { color: #3fb950; }
  .neg { color: #f85149; }
  table { width: 100%; border-collapse: collapse; margin-top: 0.5rem; }
  th, td { padding: 0.4rem 0.6rem; text-align: left; border-bottom: 1px solid #30363d; }
  th { background: #161b22; font-size: 0.8rem; text-transform: uppercase; color: #7d8590; }
  td.num { text-align: right; font-variant-numeric: tabular-nums; }
  .card { background: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 1rem; margin-top: 1rem; }
  canvas { max-height: 300px; }
  .progress { background: #30363d; border-radius: 4px; height: 8px; overflow: hidden; margin-top: 0.5rem; }
  .progress > div { height: 100%; background: linear-gradient(90deg, #1f6feb, #3fb950); }
  footer { color: #7d8590; font-size: 0.8rem; margin-top: 2rem; text-align: center; }
  a { color: #58a6ff; }
</style>
</head>
<body>
<h1>📊 S18 Paper-Forward Dashboard</h1>
<div class="muted" id="subtitle">Loading…</div>
<div class="progress"><div id="progress-bar"></div></div>

<div class="kpi-grid" id="kpis"></div>

<div class="card">
  <h2>Equity Curve — Connors vs Buy &amp; Hold</h2>
  <canvas id="equity"></canvas>
  <div class="muted" style="margin-top:0.5rem;font-size:0.8rem">
    Blue = Connors active strategy · Orange = 100% BTC · Green = equal-weight 20-coin basket
  </div>
</div>

<div class="card">
  <h2>Active (Connors) vs Passive (Buy &amp; Hold)</h2>
  <table id="vsbh">
    <thead><tr><th>Metric</th><th class="num">Connors</th><th class="num">BH BTC</th><th class="num">BH Basket</th><th>Verdict</th></tr></thead>
    <tbody></tbody>
  </table>
  <div class="muted" style="margin-top:0.5rem;font-size:0.8rem">
    <div id="gate" style="font-weight:600;font-size:1rem;margin-bottom:0.25rem"></div>
    S18 PASS gate: Connors must beat both benchmarks on Sharpe and on CAGR / Max DD (risk-adjusted). CAGR is shown for information.
  </div>
</div>

<div class="card">
  <h2>Shadow variants (paper-only, not part of the S18 gate)</h2>
  <table id="shadows">
    <thead><tr><th>Series</th><th>Start</th><th class="num">Value</th><th class="num">Return</th><th class="num">Sharpe</th><th class="num">Max DD</th><th class="num">Closed trades</th><th class="num">Connors return</th><th class="num">Connors Sharpe</th><th class="num">Connors Max DD</th></tr></thead>
    <tbody></tbody>
  </table>
  <div class="muted" style="margin-top:0.5rem;font-size:0.8rem">
    Each shadow starts at $10,000 on its first run and differs from Connors in one rule (combo: all three). Connors columns cover the same window.
  </div>
</div>
<!--S35-POOL-->

<div class="card">
  <h2>S20 Backtest vs S18 Live</h2>
  <table id="compare">
    <thead><tr><th>Metric</th><th class="num">S20 Backtest</th><th class="num">S18 Live</th><th class="num">Delta</th></tr></thead>
    <tbody></tbody>
  </table>
</div>

<div class="card">
  <h2>Open Positions</h2>
  <table id="open">
    <thead><tr><th>Symbol</th><th>Entry Date</th><th class="num">Entry</th><th class="num">Current</th><th class="num">Unrealized</th><th class="num">%</th></tr></thead>
    <tbody></tbody>
  </table>
</div>

<div class="card">
  <h2>Closed Trades</h2>
  <table id="trades">
    <thead><tr><th>Symbol</th><th>Entry</th><th>Exit</th><th class="num">Entry $</th><th class="num">Exit $</th><th class="num">P&L</th><th class="num">%</th><th>Hold</th><th>Reason</th></tr></thead>
    <tbody></tbody>
  </table>
</div>

<div class="card">
  <h2>Per-Symbol Attribution</h2>
  <table id="symtab">
    <thead><tr><th>Symbol</th><th class="num">Trades</th><th class="num">Wins</th><th class="num">WR</th><th class="num">Total P&L</th></tr></thead>
    <tbody></tbody>
  </table>
</div>

<footer>
  Generated <span id="gen"></span> · Read-only · No real orders ·
  <a href="https://github.com/dpkkaushik888-max/paper-trading-mcp">Source</a>
</footer>

<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<script>
(async () => {
  const d = await fetch('data.json', {cache: 'no-store'}).then(r => r.json());
  const m = d.metrics, b = d.backtest_baseline;

  document.getElementById('subtitle').textContent =
    `Day ${m.days_elapsed} of ${d.test_days} · Started ${d.start_date} · Starting capital $${d.starting_capital.toFixed(0)}`;
  document.getElementById('progress-bar').style.width =
    Math.min(100, (m.days_elapsed / d.test_days) * 100) + '%';

  const fmt = (v, suf='', dp=2) => v == null ? '—' :
    (typeof v === 'number' ? v.toFixed(dp) + suf : v);
  const cls = v => v > 0 ? 'pos' : v < 0 ? 'neg' : '';

  const sh = d.shadows || [];
  const shBody = document.querySelector('#shadows tbody');
  shBody.innerHTML = sh.length === 0
    ? '<tr><td colspan="10" class="muted">no shadow data yet; starts at next daily run</td></tr>'
    : sh.map(r => `<tr><td>${r.name}${r.error ? ' <span class="neg" title="' + String(r.error).replace(/"/g, '&quot;') + '">⚠ error on last run</span>' : ''}</td><td>${r.start_date || '—'}</td><td class="num">${typeof r.value === 'number' ? '$' + r.value.toFixed(2) : '—'}</td>` +
        `<td class="num ${cls(r.return_pct)}">${fmt(r.return_pct, '%')}</td><td class="num">${fmt(r.sharpe)}</td><td class="num">${fmt(r.max_dd_pct, '%')}</td>` +
        `<td class="num">${r.closed_trades ?? '—'}</td><td class="num ${cls(r.connors_return_pct)}">${fmt(r.connors_return_pct, '%')}</td>` +
        `<td class="num">${fmt(r.connors_sharpe)}</td><td class="num">${fmt(r.connors_max_dd_pct, '%')}</td></tr>`).join('');

  const kpis = [
    ['Portfolio', `$${m.portfolio_value.toFixed(2)}`, cls(m.portfolio_value - d.starting_capital)],
    ['Total Return', fmt(m.total_return_pct, '%'), cls(m.total_return_pct)],
    ['CAGR (ann.)', fmt(m.cagr_pct, '%'), cls(m.cagr_pct)],
    ['Sharpe', fmt(m.sharpe, '', 2), cls(m.sharpe)],
    ['Max DD', fmt(m.max_dd_pct, '%'), m.max_dd_pct > 10 ? 'neg' : ''],
    ['Win Rate', fmt(m.win_rate_pct, '%'), cls(m.win_rate_pct - 50)],
    ['Trades', m.trades, ''],
    ['Open', m.n_open + ' / 6', ''],
    ['Avg invested share (time-weighted)', typeof m.invested_share_avg === 'number' ? fmt(m.invested_share_avg * 100, '%', 1) : '—', ''],
    ['Days fully in cash (skipped days excluded)', m.days_fully_in_cash ?? '—', ''],
  ];
  document.getElementById('kpis').innerHTML = kpis.map(([l, v, c]) =>
    `<div class="kpi"><div class="kpi-label">${l}</div><div class="kpi-value ${c}">${v}</div></div>`
  ).join('');

  // Equity curve — 3 lines
  if (d.equity_curve.length) {
    new Chart(document.getElementById('equity'), {
      type: 'line',
      data: {
        labels: d.equity_curve.map(p => p.date),
        datasets: [
          {
            label: 'Connors (active)',
            data: d.equity_curve.map(p => p.connors),
            borderColor: '#58a6ff', backgroundColor: 'rgba(88,166,255,0.08)',
            fill: false, tension: 0.1, borderWidth: 2,
          },
          {
            label: 'BH BTC (100% BTC)',
            data: d.equity_curve.map(p => p.bh_btc),
            borderColor: '#f78166', backgroundColor: 'rgba(247,129,102,0.08)',
            fill: false, tension: 0.1, borderWidth: 2, borderDash: [4, 4],
          },
          {
            label: 'BH Basket (eq-wt 20)',
            data: d.equity_curve.map(p => p.bh_basket),
            borderColor: '#3fb950', backgroundColor: 'rgba(63,185,80,0.08)',
            fill: false, tension: 0.1, borderWidth: 2, borderDash: [2, 2],
          },
        ],
      },
      options: {
        scales: {
          x: {ticks: {color: '#7d8590'}, grid: {color: '#30363d'}},
          y: {ticks: {color: '#7d8590'}, grid: {color: '#30363d'}},
        },
        plugins: {legend: {labels: {color: '#e6edf3'}}},
      },
    });
  }

  // Connors vs Buy & Hold table
  const cn = m, bh1 = d.bh_btc_metrics, bh2 = d.bh_basket_metrics;
  // null on any side = no verdict (same rule as beatsBoth below).
  const verdictOf = (cnv, b1v, b2v) =>
    (cnv == null || b1v == null || b2v == null) ? '<span class="muted">—</span>' :
    cnv > b1v && cnv > b2v ? '<span class="pos">Beats both</span>' :
    cnv > b1v || cnv > b2v ? '<span class="muted">Beats one</span>' :
    '<span class="neg">Trails both</span>';
  const vsRows = [
    ['Total Return', cn.total_return_pct, bh1.total_return_pct, bh2.total_return_pct, '%'],
    ['CAGR (ann.)', cn.cagr_pct, bh1.cagr_pct, bh2.cagr_pct, '%'],
    ['Sharpe',      cn.sharpe,   bh1.sharpe,   bh2.sharpe,   ''],
    ['Max DD',      cn.max_dd_pct, bh1.max_dd_pct, bh2.max_dd_pct, '%'],
    ['CAGR / Max DD', cn.calmar, bh1.calmar, bh2.calmar, ''],
  ];
  // Gate: beat BOTH benchmarks on Sharpe AND CAGR / Max DD (null calmar never beats).
  const beatsBoth = (cnv, b1v, b2v) => cnv != null && b1v != null && b2v != null && cnv > b1v && cnv > b2v;
  const gatePass = beatsBoth(cn.sharpe, bh1.sharpe, bh2.sharpe) && beatsBoth(cn.calmar, bh1.calmar, bh2.calmar);
  const gateEl = document.getElementById('gate');
  gateEl.textContent = 'S18 gate: ' + (gatePass ? 'PASS' : 'FAIL');
  gateEl.style.color = gatePass ? '#3fb950' : '#f85149';
  document.querySelector('#vsbh tbody').innerHTML = vsRows.map(([lab, cv, b1v, b2v, suf]) => {
    const verdict = lab === 'Max DD'
      ? (cv < b1v && cv < b2v ? '<span class="pos">Lower DD</span>' : '<span class="muted">—</span>')
      : verdictOf(cv, b1v, b2v) + (lab === 'CAGR (ann.)' ? ' <span class="muted">(info)</span>' : '');
    return `<tr><td>${lab}</td><td class="num ${cls(cv)}">${fmt(cv, suf)}</td><td class="num ${cls(b1v)}">${fmt(b1v, suf)}</td><td class="num ${cls(b2v)}">${fmt(b2v, suf)}</td><td>${verdict}</td></tr>`;
  }).join('');

  // Comparison table
  const cmpRows = [
    ['CAGR', b.cagr_pct, m.cagr_pct, '%'],
    ['Sharpe', b.sharpe, m.sharpe, ''],
    ['MaxDD', b.max_dd_pct, m.max_dd_pct, '%'],
    ['CAGR / Max DD', b.calmar, m.calmar, ''],
    ['Win Rate', b.win_rate_pct, m.win_rate_pct, '%'],
    ['Trades (90d est)', Math.round(b.trades * 90 / b.holdout_days), m.trades, ''],
  ];
  document.querySelector('#compare tbody').innerHTML = cmpRows.map(([lab, bv, lv, suf]) => {
    const hasDelta = bv != null && lv != null;
    const delta = hasDelta ? lv - bv : null;
    const dcell = hasDelta ? `${delta >= 0 ? '+' : ''}${fmt(delta, suf)}` : '—';
    return `<tr><td>${lab}</td><td class="num">${fmt(bv, suf)}</td><td class="num">${fmt(lv, suf)}</td><td class="num ${hasDelta ? cls(delta) : ''}">${dcell}</td></tr>`;
  }).join('');

  // Open positions
  const openRows = d.open_positions.map(p =>
    `<tr><td>${p.symbol}</td><td>${p.entry_date}</td><td class="num">$${p.entry_price.toFixed(2)}</td><td class="num">$${p.current_price.toFixed(2)}</td><td class="num ${cls(p.unrealized_pnl)}">$${p.unrealized_pnl.toFixed(2)}</td><td class="num ${cls(p.unrealized_pct)}">${p.unrealized_pct >= 0 ? '+' : ''}${p.unrealized_pct.toFixed(2)}%</td></tr>`
  ).join('') || '<tr><td colspan="6" class="muted">No open positions.</td></tr>';
  document.querySelector('#open tbody').innerHTML = openRows;

  // Closed trades
  const tradeRows = d.closed_trades.slice().reverse().map(t =>
    `<tr><td>${t.symbol}</td><td>${t.entry_date}</td><td>${t.exit_date}</td><td class="num">$${t.entry_price.toFixed(2)}</td><td class="num">$${t.exit_price.toFixed(2)}</td><td class="num ${cls(t.pnl)}">$${t.pnl.toFixed(2)}</td><td class="num ${cls(t.pnl_pct)}">${t.pnl_pct >= 0 ? '+' : ''}${t.pnl_pct.toFixed(2)}%</td><td class="num">${t.hold_days}d</td><td>${t.reason}</td></tr>`
  ).join('') || '<tr><td colspan="9" class="muted">No closed trades yet.</td></tr>';
  document.querySelector('#trades tbody').innerHTML = tradeRows;

  // Per-symbol
  const syms = Object.entries(d.per_symbol).sort((a, b) => b[1].pnl - a[1].pnl);
  const symRows = syms.map(([sym, s]) =>
    `<tr><td>${sym}</td><td class="num">${s.trades}</td><td class="num">${s.wins}</td><td class="num">${(s.wins / s.trades * 100).toFixed(0)}%</td><td class="num ${cls(s.pnl)}">$${s.pnl.toFixed(2)}</td></tr>`
  ).join('') || '<tr><td colspan="5" class="muted">No trades yet.</td></tr>';
  document.querySelector('#symtab tbody').innerHTML = symRows;

  document.getElementById('gen').textContent = d.generated_at;
})();
</script>
</body>
</html>
"""


def render_index(pool_html: str) -> str:
    """INDEX_HTML with the server-rendered S35 pool block in place of the placeholder (live parts untouched)."""
    return INDEX_HTML.replace(POOL_PLACEHOLDER, pool_html)


def build_pool(journal: Journal, journal_path: Path, report_path: Path | None, baselines: Path | None = None,
               lines=None, B: int | None = None) -> tuple[str, dict]:
    """(pool HTML block, data.json ``pool`` aggregates). Never raises (S35 §9)."""
    try:
        jd = journal.to_dict()
        rep, why = pool_dash.load_report(report_path, jd, baselines or journal_path.parent / "pool_baselines",
                                         journal_path.parent / "pool_context_history.json", lines=lines, B=B)
        return pool_dash.render_pool_section(rep, why, jd), pool_dash.pool_summary(rep, why)
    except Exception as e:  # noqa: BLE001
        why = f"{type(e).__name__}: {e}"
        return pool_dash.unavailable_box(why), {"available": False, "reason": why}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build the static paper-forward dashboard")
    ap.add_argument("--journal", default=JOURNAL_PATH)
    ap.add_argument("--docs-dir", default=DOCS_DIR)
    ap.add_argument("--pool-report", default=None, help="default: <journal dir>/pool_report.json")
    ap.add_argument("--baselines", default=None, help="default: <journal dir>/pool_baselines")
    a = ap.parse_args(argv)
    jp = Path(a.journal)
    journal = Journal.load(jp)
    docs = Path(a.docs_dir)
    docs.mkdir(parents=True, exist_ok=True)

    data = build_data_json(journal)
    pool_html, data["pool"] = build_pool(journal, jp, Path(a.pool_report) if a.pool_report else jp.parent / "pool_report.json",
                                         Path(a.baselines) if a.baselines else None)
    (docs / "data.json").write_text(json.dumps(data, indent=2))
    (docs / "index.html").write_text(render_index(pool_html))
    print(f"Dashboard written to {docs}/")
    print(f"  Days elapsed:   {data['metrics']['days_elapsed']}/{data['test_days']}")
    print(f"  Portfolio:      ${data['metrics']['portfolio_value']:.2f}")
    print(f"  CAGR:           {data['metrics']['cagr_pct']:+.2f}%")
    print(f"  Trades:         {data['metrics']['trades']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
