"""Full-history backtest of pool lines through the real book.step + family code (S35 §7).

  python scripts/pool_backtest.py --line <id>|all [--venue com|us] [--start 2020-01-01] [--end YYYY-MM-DD]
  python scripts/pool_backtest.py --line connors_live_params --venue us --start 2026-04-20 --parity state/journal.json

Writes state/pool_baselines/<id>.json (aggregates only). Never run in the daily Action.
A --venue com run also adds a ``live_window_us`` block per line: the same line on .us bars from LIVE_US_START
to the last cached .us bar (design §7 W100_us/LIVE_us). ``generated_from`` records ``git rev-parse HEAD`` and
whether the tree was dirty; baselines regenerated just before a commit record that commit's parent.
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
from trading_engine.paper.config import COST_PCT, STARTING_CAPITAL, UNIVERSE  # noqa: E402
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
LIVE_US_START = "2026-04-20"   # first live journal day (S18 forward test on binance.us)
ROOT = Path(__file__).resolve().parent.parent


def _git(*args) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, cwd=ROOT).stdout.strip()


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


def _rets(eq: pd.Series) -> pd.Series:
    """Daily returns of a value path that starts at STARTING_CAPITAL (first day vs the starting capital)."""
    r = pd.concat([pd.Series([STARTING_CAPITAL]), eq.reset_index(drop=True)]).pct_change().dropna()
    r.index = eq.index
    return r


def bh_basket_path(bars, days) -> pd.Series:
    """Live ``bh_basket`` (benchmarks.init_bh_basket/mark_benchmark): STARTING_CAPITAL / len(UNIVERSE) per coin
    bought once at the first day's closes net of COST_PCT, never rebalanced; coins without a close that day get
    zero allocation (their share stays cash); a missing later close marks at the last seen close."""
    per_coin = STARTING_CAPITAL / len(UNIVERSE)
    d0 = days[0]
    shares = {s: per_coin * (1.0 - COST_PCT) / float(df["Close"][d0]) for s, df in bars.items() if d0 in df.index}
    cash = STARTING_CAPITAL - per_coin * len(shares)
    closes = pd.DataFrame({s: bars[s]["Close"] for s in shares}).reindex(days).ffill()
    return (closes * pd.Series(shares)).sum(axis=1) + cash


def _cond_maxdd(rets: pd.Series, mask: pd.Series) -> float:
    """MaxDD of the path stitched from the masked days' daily returns (same method for line and comparators)."""
    curve = (1 + rets[mask.reindex(rets.index, fill_value=False)]).cumprod()
    return float(-(curve / curve.cummax() - 1).min()) if len(curve) else 0.0


def _buckets(tr) -> dict:
    return {dim: {str(k): _tstats([t["pnl_pct"] for t in tr if (t["entry_ctx"] or {}).get(dim) == k])
                  for k in sorted({str((t["entry_ctx"] or {}).get(dim)) for t in tr})} for dim in DIMS}


def _full(eq, tr) -> dict:
    ts = _tstats([t["pnl_pct"] for t in tr])
    return dict(**_perf(eq, STARTING_CAPITAL), trades=ts["n"], win_rate=ts["wins"] / max(1, ts["n"]),
                mean_pnl_pct=ts["mean"])


def load_live_window(bar_dir: Path, start: str = LIVE_US_START) -> dict | None:
    bars = load_bars(bar_dir, "us")
    if not bars:
        return None
    last = max(df.index[-1] for df in bars.values())
    days = [d for d in sorted(set().union(*[df.index for df in bars.values()])) if pd.Timestamp(start) <= d <= last]
    if not days:
        return None
    ctxs = contexts(bars, days)
    flagged: dict[str, set] = {}
    for c in ctxs.values():
        for k, syms in c.data_flags.items():
            for sym in syms:
                flagged.setdefault(sym, set()).add(k)
    return dict(bars=bars, days=days, ctxs=ctxs, missing=sorted(set(UNIVERSE) - set(bars)),
                flagged={k: sorted(v) for k, v in sorted(flagged.items())})


def live_window(spec, lw) -> dict | None:
    """The line on .us bars over the live window; flagged symbols are handled by the line's own data guard."""
    if lw is None:
        return None
    res = simulate(spec, lw["bars"], lw["days"], lw["ctxs"])
    tr = res["trades"]
    guard = spec.book.data_guard
    return dict(venue="us", period=[lw["days"][0].date().isoformat(), lw["days"][-1].date().isoformat()],
                full=_full(res["equity"], tr), trades=_tstats([t["pnl_pct"] for t in tr]), buckets=_buckets(tr),
                data_guard=guard, missing_symbols=lw["missing"], flagged_symbols=lw["flagged"],
                skipped_symbols=sorted(set(lw["missing"]) | (set(lw["flagged"]) if guard else set())),
                note="skipped = no .us bars, or (data_guard on) stale/gap200 at some point in the window: entries "
                     "blocked as live; data_guard off trades flagged symbols at their frozen last bar as live")


def summarise(spec, res, ref, bars, ctxs, venue, start, end, generated_from=None) -> dict:
    eq, tr = res["equity"], res["trades"]
    out = dict(line=spec.id, family=spec.family, params=dict(spec.params), venue=venue, period=[start, end],
               python=platform.python_version(), git_sha=_git("rev-parse", "--short", "HEAD"),
               generated_from=generated_from, years_method="chained",
               full=_full(eq, tr), trades=_tstats([t["pnl_pct"] for t in tr]))
    years, prev = {}, STARTING_CAPITAL
    for y in sorted({d.year for d in eq.index}):
        e = eq[eq.index.year == y]
        years[str(y)] = dict(**_perf(e, prev), trades=sum(t["exit_date"][:4] == str(y) for t in tr))
        prev = float(e.iloc[-1])
    out["years"] = years
    pre = eq[eq.index.year < 2023]
    oos = eq[eq.index.year >= 2023]
    out["oos_2023"] = _perf(oos, float(pre.iloc[-1]) if len(pre) else STARTING_CAPITAL) if len(oos) else None
    out["buckets"] = _buckets(tr)
    if ref is not None:
        out["years_beating_s18"], out["n_years"] = years_beating(years, ref["years"])
        # Idle = the reference held nothing going into the bar (its n_open at the PREVIOUS day's close).
        prev_open = ref["n_open"].shift(1, fill_value=0)
        flat = {d.date().isoformat() for d, n in prev_open.items() if n == 0}
        idle = [t["pnl_pct"] for t in tr if t["entry_date"] in flat]
        cond = (spec.hypothesis.cond if spec.hypothesis else None) or "below200"
        mask = pd.Series({d: c.btc_trend in COND.get(cond, (cond,)) for d, c in ctxs.items()})
        # Conditional MaxDD vs the S18 reference run; the basket family is additionally held to live bh_basket.
        comps = {"s18_reference": _cond_maxdd(_rets(ref["equity"]), mask)}
        if spec.family == "basket_trend":
            comps["bh_basket"] = _cond_maxdd(_rets(ref["bh_basket"]), mask)
        ev = dict(idle_n=len(idle), idle_mean=float(np.mean(idle)) if idle else None, idle_def="reference n_open == 0 "
                  "at the previous bar", cond=cond, cond_days=int(mask.sum()), line_maxdd=_cond_maxdd(_rets(eq), mask),
                  comparators=comps, dd_method="max drawdown of the path compounded from the masked days' daily returns")
        out["hypothesis"] = dict(kind=spec.hypothesis.kind, cond=spec.hypothesis.cond) if spec.hypothesis else None
        out["evidence"] = ev
        out["gate"] = gate_result(spec.hypothesis.kind if spec.hypothesis else "", out["years_beating_s18"],
                                  out["n_years"], ev, waiver=spec.baseline_waiver)
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
    gen = dict(sha=_git("rev-parse", "HEAD"), dirty=bool(_git("status", "--porcelain")))
    lw = load_live_window(Path(a.bars)) if a.venue == "com" else None
    ref_sum = summarise(REFERENCE, ref, None, bars, ctxs, a.venue, *period, generated_from=gen)
    if a.venue == "com":
        ref_sum["live_window_us"] = live_window(REFERENCE, lw)
    ref_sum["years_beating_s18"], ref_sum["n_years"] = years_beating(ref_sum["years"], ref_sum["years"])
    (out / f"_s18_reference_{a.venue}.json").write_text(json.dumps(ref_sum, indent=1, sort_keys=True))
    ref_sum.update(n_open=ref["n_open"], equity=ref["equity"], bh_basket=bh_basket_path(bars, days))
    print(f"{REF_ID}: CAGR {ref_sum['full']['cagr']:.4f} MaxDD {ref_sum['full']['maxdd']:.4f} "
          f"Sharpe {ref_sum['full']['sharpe']:.3f} n={ref_sum['trades']['n']} years "
          + " ".join(f"{y}:{v['ret']:+.4f}" for y, v in ref_sum["years"].items()))
    for lid in ids:
        s = summarise(specs[lid], simulate(specs[lid], bars, days, ctxs), ref_sum, bars, ctxs, a.venue, *period,
                      generated_from=gen)
        if a.venue == "com":
            s["live_window_us"] = live_window(specs[lid], lw)
        name = f"{lid}.json" if a.venue == "com" else f"{lid}_{a.venue}.json"
        (out / name).write_text(json.dumps(s, indent=1, sort_keys=True))
        f = s["full"]
        print(f"{lid}: beats {s['years_beating_s18']}/{s['n_years']} CAGR {f['cagr']:.4f} MaxDD {f['maxdd']:.4f} "
              f"Sharpe {f['sharpe']:.3f} n={f['trades']} gate={s['gate']} ev={s['evidence']}")
        w = s.get("live_window_us")
        if w:
            print(f"  live_window_us {w['period']}: ret {w['full']['ret']:+.4f} MaxDD {w['full']['maxdd']:.4f} "
                  f"n={w['full']['trades']} skipped={w['skipped_symbols']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
