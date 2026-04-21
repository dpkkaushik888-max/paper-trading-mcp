"""Generate static dashboard for S18 paper-forward from state/journal.json.

Outputs:
  docs/index.html   — single-page dashboard (Chart.js from CDN)
  docs/data.json    — machine-readable snapshot for the page JS

GitHub Pages serves docs/ automatically (configured in repo settings).
"""

from __future__ import annotations

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
        "win_rate_pct": win_rate * 100,
        "profit_factor": pf if pf != float("inf") else None,
        "trades": len(closed),
        "wins": len(wins),
        "losses": len(losses),
        "portfolio_value": current,
        "cash": j.cash,
        "n_open": len(j.open_positions),
    }


def build_data_json(j: Journal) -> dict:
    metrics = compute_live_metrics(j)
    equity_curve = [
        {"date": d["date"], "value": d["portfolio_value"]}
        for d in j.days
    ]
    # Prepend day-0 (before first run).
    if j.days:
        equity_curve.insert(0, {"date": j.start_date, "value": STARTING_CAPITAL})

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
        "backtest_baseline": BACKTEST_BASELINE,
        "equity_curve": equity_curve,
        "open_positions": open_positions,
        "closed_trades": closed_trades,
        "per_symbol": per_symbol,
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
  <h2>Equity Curve</h2>
  <canvas id="equity"></canvas>
</div>

<div class="card">
  <h2>Backtest vs Live</h2>
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

  const kpis = [
    ['Portfolio', `$${m.portfolio_value.toFixed(2)}`, cls(m.portfolio_value - d.starting_capital)],
    ['Total Return', fmt(m.total_return_pct, '%'), cls(m.total_return_pct)],
    ['CAGR (ann.)', fmt(m.cagr_pct, '%'), cls(m.cagr_pct)],
    ['Sharpe', fmt(m.sharpe, '', 2), cls(m.sharpe)],
    ['Max DD', fmt(m.max_dd_pct, '%'), m.max_dd_pct > 10 ? 'neg' : ''],
    ['Win Rate', fmt(m.win_rate_pct, '%'), cls(m.win_rate_pct - 50)],
    ['Trades', m.trades, ''],
    ['Open', m.n_open + ' / 6', ''],
  ];
  document.getElementById('kpis').innerHTML = kpis.map(([l, v, c]) =>
    `<div class="kpi"><div class="kpi-label">${l}</div><div class="kpi-value ${c}">${v}</div></div>`
  ).join('');

  // Equity curve
  if (d.equity_curve.length) {
    new Chart(document.getElementById('equity'), {
      type: 'line',
      data: {
        labels: d.equity_curve.map(p => p.date),
        datasets: [{
          label: 'Portfolio', data: d.equity_curve.map(p => p.value),
          borderColor: '#58a6ff', backgroundColor: 'rgba(88,166,255,0.1)',
          fill: true, tension: 0.1,
        }],
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

  // Comparison table
  const cmpRows = [
    ['CAGR', b.cagr_pct, m.cagr_pct, '%'],
    ['Sharpe', b.sharpe, m.sharpe, ''],
    ['MaxDD', b.max_dd_pct, m.max_dd_pct, '%'],
    ['Win Rate', b.win_rate_pct, m.win_rate_pct, '%'],
    ['Trades (90d est)', Math.round(b.trades * 90 / b.holdout_days), m.trades, ''],
  ];
  document.querySelector('#compare tbody').innerHTML = cmpRows.map(([lab, bv, lv, suf]) => {
    const delta = lv - bv;
    return `<tr><td>${lab}</td><td class="num">${fmt(bv, suf)}</td><td class="num">${fmt(lv, suf)}</td><td class="num ${cls(delta)}">${delta >= 0 ? '+' : ''}${fmt(delta, suf)}</td></tr>`;
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


def main() -> int:
    journal = Journal.load()
    docs = Path(DOCS_DIR)
    docs.mkdir(parents=True, exist_ok=True)

    data = build_data_json(journal)
    (docs / "data.json").write_text(json.dumps(data, indent=2))
    (docs / "index.html").write_text(INDEX_HTML)
    print(f"Dashboard written to {docs}/")
    print(f"  Days elapsed:   {data['metrics']['days_elapsed']}/{data['test_days']}")
    print(f"  Portfolio:      ${data['metrics']['portfolio_value']:.2f}")
    print(f"  CAGR:           {data['metrics']['cagr_pct']:+.2f}%")
    print(f"  Trades:         {data['metrics']['trades']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
