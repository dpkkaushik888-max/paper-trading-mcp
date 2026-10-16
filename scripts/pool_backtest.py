"""Full-history backtest of pool lines through the real book.step + family code (S35 §7).

  python scripts/pool_backtest.py --line <id>|all [--venue com|us] [--start 2020-01-01] [--end YYYY-MM-DD]
  python scripts/pool_backtest.py --line connors_live_params --venue us --start 2026-04-20 --parity state/journal.json

Writes state/pool_baselines/<id>.json (aggregates only). Never run in the daily Action.
"""
from __future__ import annotations

import argparse
from datetime import timedelta
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from trading_engine.paper.config import STARTING_CAPITAL, UNIVERSE  # noqa: E402
from trading_engine.paper.pool import book as pool_book  # noqa: E402
from trading_engine.paper.pool.context import context_for_bar  # noqa: E402
from trading_engine.paper.pool.families import make_strategy  # noqa: E402
from trading_engine.paper.pool.gate import gate_result, years_beating  # noqa: E402
from trading_engine.paper.pool.registry import LINES, BookParams, Hypothesis, LineSpec  # noqa: E402

REF_ID = "connors_live_params"
REFERENCE = LineSpec(REF_ID, "connors", {}, book=BookParams(data_guard=False), registered_on="2026-10-16",
                     status="reference", hypothesis=Hypothesis("beats_live"), note="S18 live rules, no data guard")
BTC = "BTCUSDT"
DIMS = ("btc_trend", "btc_vol20_bucket", "basket_ret20_bucket", "dow")
COND = {"below200": ("below_both", "below200_above50")}


def load_bars(bar_dir: Path, venue: str) -> dict[str, pd.DataFrame]:
    syms = [s for s in UNIVERSE if not (venue == "com" and s == "MATICUSDT")]   # MATIC delisted on .com
    out = {}
    for s in syms:
        p = bar_dir / venue / f"{s}.csv"
        if p.exists():
            out[s] = pd.read_csv(p, index_col="date", parse_dates=True).sort_index()
    return out


def contexts(bars, days) -> dict:
    return {d: context_for_bar(bars, d, BTC) for d in days}


def simulate(spec, bars, days, ctxs) -> dict:
    strat = make_strategy(spec)
    rows = {s: strat.indicators(df).to_dict("index") for s, df in bars.items()}
    closes = {s: df["Close"].to_dict() for s, df in bars.items()}
    book = pool_book.Book(name=spec.id)
    eq, n_open = {}, {}
    for d in days:
        snap = {}
        for s, df in bars.items():
            if d in closes[s]:
                snap[s] = (float(closes[s][d]), rows[s][d])
            elif len(df) and df.index[0] < d:   # stale symbol: frozen last bar, as live sees it
                last = df.index[df.index.searchsorted(d) - 1]
                snap[s] = (float(closes[s][last]), rows[s][last])
        pool_book.step(book, spec, strat, snap, ctxs[d], d.date().isoformat())
        eq[d], n_open[d] = book.value, len(book.open_positions)
    return dict(equity=pd.Series(eq), n_open=pd.Series(n_open), trades=book.closed_trades)


def _perf(eq: pd.Series, base: float) -> dict:
    full = pd.concat([pd.Series([base]), eq.reset_index(drop=True)])
    r = full.pct_change().dropna()
    yrs = len(eq) / 365.25
    return dict(ret=float(eq.iloc[-1] / base - 1), cagr=float((eq.iloc[-1] / base) ** (1 / yrs) - 1),
                sharpe=float(r.mean() / r.std() * np.sqrt(365)) if r.std() > 0 else 0.0,
                maxdd=float(-(full / full.cummax() - 1).min()))


def _tstats(pnls: list[float]) -> dict:
    a = np.array(pnls, dtype=float)
    return dict(n=int(len(a)), wins=int((a > 0).sum()), mean=float(a.mean()) if len(a) else None,
                sd=float(a.std(ddof=1)) if len(a) > 1 else None)


def _cond_maxdd(rets: pd.Series, mask: pd.Series) -> float:
    curve = (1 + rets[mask.reindex(rets.index, fill_value=False)]).cumprod()
    return float(-(curve / curve.cummax() - 1).min()) if len(curve) else 0.0


def summarise(spec, res, ref, bars, ctxs, venue, start, end) -> dict:
    eq, tr = res["equity"], res["trades"]
    out = dict(line=spec.id, family=spec.family, params=dict(spec.params), venue=venue, period=[start, end],
               python=platform.python_version(),
               git_sha=subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip(),
               full=_perf(eq, STARTING_CAPITAL), trades=_tstats([t["pnl_pct"] for t in tr]))
    out["full"].update(trades=out["trades"]["n"], win_rate=out["trades"]["wins"] / max(1, out["trades"]["n"]),
                       mean_pnl_pct=out["trades"]["mean"])
    years, prev = {}, STARTING_CAPITAL
    for y in sorted({d.year for d in eq.index}):
        e = eq[eq.index.year == y]
        years[str(y)] = dict(**_perf(e, prev), trades=sum(t["exit_date"][:4] == str(y) for t in tr))
        prev = float(e.iloc[-1])
    out["years"] = years
    pre = eq[eq.index.year < 2023]
    oos = eq[eq.index.year >= 2023]
    out["oos_2023"] = _perf(oos, float(pre.iloc[-1]) if len(pre) else STARTING_CAPITAL) if len(oos) else None
    out["buckets"] = {dim: {str(k): _tstats([t["pnl_pct"] for t in tr if (t["entry_ctx"] or {}).get(dim) == k])
                            for k in sorted({str((t["entry_ctx"] or {}).get(dim)) for t in tr})} for dim in DIMS}
    if ref is not None:
        out["years_beating_s18"], out["n_years"] = years_beating(years, ref["years"])
        flat = {d.date().isoformat() for d, n in ref["n_open"].items() if n == 0}
        idle = [t["pnl_pct"] for t in tr if t["entry_date"] in flat]
        # bh_basket-equivalent: equal-weight, daily-rebalanced mean close-to-close return of the venue universe.
        closes = pd.DataFrame({s: df["Close"] for s, df in bars.items()}).reindex(eq.index)
        bench = closes.pct_change(fill_method=None).mean(axis=1).fillna(0.0)
        cond = (spec.hypothesis.cond if spec.hypothesis else None) or "below_both"
        mask = pd.Series({d: c.btc_trend in COND.get(cond, (cond,)) for d, c in ctxs.items()})
        line_rets = pd.concat([pd.Series([STARTING_CAPITAL]), eq.reset_index(drop=True)]).pct_change().dropna()
        line_rets.index = eq.index
        ev = dict(idle_n=len(idle), idle_mean=float(np.mean(idle)) if idle else None, cond=cond, cond_days=int(mask.sum()),
                  line_maxdd=_cond_maxdd(line_rets, mask), bench_maxdd=_cond_maxdd(bench, mask))
        out["hypothesis"] = dict(kind=spec.hypothesis.kind, cond=spec.hypothesis.cond) if spec.hypothesis else None
        out["evidence"] = ev
        out["gate"] = gate_result(spec.hypothesis.kind if spec.hypothesis else "", out["years_beating_s18"],
                                  out["n_years"], ev)
    return out


def parity(res, journal_path: Path) -> tuple[int, int]:
    """Backtest dates are bar dates; journal day D trades bar D-1, so shift +1 day."""
    sh = lambda s: (pd.Timestamp(s) + timedelta(days=1)).date().isoformat()  # noqa: E731
    mine = {(t["symbol"], sh(t["entry_date"]), sh(t["exit_date"])) for t in res["trades"]}
    live = [(t["symbol"], t["entry_date"], t["exit_date"]) for t in json.loads(journal_path.read_text())["closed_trades"]]
    return sum(t in mine for t in live), len(live)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--line", required=True, help="line id, connors_live_params, or all")
    ap.add_argument("--venue", default="com", choices=("com", "us"))
    ap.add_argument("--start", default="2020-01-01")
    ap.add_argument("--end")
    ap.add_argument("--bars", default="data/bars")
    ap.add_argument("--out", default="state/pool_baselines")
    ap.add_argument("--parity", help="journal.json: compare closed trades instead of writing baselines")
    a = ap.parse_args(argv)
    bars = load_bars(Path(a.bars), a.venue)
    last = max(df.index[-1] for df in bars.values())
    end = pd.Timestamp(a.end) if a.end else last
    days = [d for d in sorted(set().union(*[df.index for df in bars.values()])) if pd.Timestamp(a.start) <= d <= end]
    ctxs = contexts(bars, days)
    period = (days[0].date().isoformat(), days[-1].date().isoformat())
    ref = simulate(REFERENCE, bars, days, ctxs)
    if a.parity:
        m, n = parity(ref, Path(a.parity))
        print(f"parity {a.venue} {period}: {m}/{n} journal closed trades matched (symbol, entry, exit)")
        return 0
    specs = {l.id: l for l in LINES}
    ids = list(specs) if a.line == "all" else [] if a.line == REF_ID else [a.line]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ref_sum = summarise(REFERENCE, ref, None, bars, ctxs, a.venue, *period)
    ref_sum["years_beating_s18"], ref_sum["n_years"] = years_beating(ref_sum["years"], ref_sum["years"])
    (out / f"_s18_reference_{a.venue}.json").write_text(json.dumps(ref_sum, indent=1, sort_keys=True))
    ref_sum["n_open"] = ref["n_open"]
    print(f"{REF_ID}: CAGR {ref_sum['full']['cagr']:.4f} MaxDD {ref_sum['full']['maxdd']:.4f} "
          f"Sharpe {ref_sum['full']['sharpe']:.3f} n={ref_sum['trades']['n']} years "
          + " ".join(f"{y}:{v['ret']:+.4f}" for y, v in ref_sum["years"].items()))
    for lid in ids:
        s = summarise(specs[lid], simulate(specs[lid], bars, days, ctxs), ref_sum, bars, ctxs, a.venue, *period)
        name = f"{lid}.json" if a.venue == "com" else f"{lid}_{a.venue}.json"
        (out / name).write_text(json.dumps(s, indent=1, sort_keys=True))
        f = s["full"]
        print(f"{lid}: beats {s['years_beating_s18']}/{s['n_years']} CAGR {f['cagr']:.4f} MaxDD {f['maxdd']:.4f} "
              f"Sharpe {f['sharpe']:.3f} n={f['trades']} gate={s['gate']} ev={s['evidence']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
