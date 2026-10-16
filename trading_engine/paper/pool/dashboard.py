"""Strategy-pool dashboard section (S35 §9, C4): server-rendered static HTML inserted into docs/index.html.

Read-only. Source = ``state/pool_report.json`` when present (malformed -> "pool report unavailable" box), else
``score_pool`` over the journal. Any failure renders the unavailable box; the live sections never depend on it.
Verdicts are recommendations; the owner rules. Live S18 is never changed by this section.
"""

from __future__ import annotations

import json
import math
from datetime import date
from html import escape
from pathlib import Path
from typing import Any, Optional

from .context import ret20_bucket
from .registry import LINES, LIVE_ID
from .scoring import DIMENSIONS, MIN_TRADES, load_context_history, score_pool

BEGIN, END = "<!-- S35 pool section: begin -->", "<!-- S35 pool section: end -->"
HEADER_NOTE = "Verdicts are recommendations; the owner rules. Live S18 is never changed by this table."
HEATMAP_CAPTION = "Exploratory; never drives verdicts (S35 §6). Cell = posterior mean win probability (Beta, k=10 backtest prior)."
WEEKS = 8
BADGE = {"KEEP": "#6e7681", "PROMOTE-cand": "#238636", "RETIRE-cand": "#d18616", "HARD-KILL": "#da3633",
         "DORMANT": "#1f6feb", "CONTROL": "#8957e5", "NOT-STEPPED": "#b1bac4", "INCONCLUSIVE-180": "#d4a72c",
         "LIVE": "#30363d"}
BUCKET_ORDER = {
    "btc_vol20_bucket": ["low", "mid", "high"],
    "basket_ret20_bucket": [ret20_bucket(x) for x in (-0.2, -0.05, 0.05, 0.2)],
    "dow": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
}
CSS = """<style>
  .pool-badge { display:inline-block; padding:0.1rem 0.45rem; border-radius:10px; font-size:0.75rem; font-weight:600; color:#fff; white-space:nowrap; }
  .pool-badge.light { color:#0d1117; }
  .pool-tag { display:inline-block; padding:0 0.35rem; border:1px solid #30363d; border-radius:8px; font-size:0.7rem; color:#7d8590; }
  .pool-wrap { overflow-x:auto; }
  .pool table { font-size:0.8rem; }
  .pool td, .pool th { padding:0.3rem 0.4rem; }
  .pool details summary { cursor:pointer; color:#7d8590; font-size:0.75rem; }
  .pool td.heat { text-align:center; font-size:0.72rem; white-space:nowrap; }
  .pool td.insuff { background:repeating-linear-gradient(45deg,#21262d,#21262d 4px,#30363d 4px,#30363d 8px); color:#7d8590; }
  .pool td.empty { color:#484f58; }
  .pool .unavail { border:1px dashed #d18616; padding:0.75rem; border-radius:6px; color:#d18616; }
</style>"""


# ── formatting (never emits "None" / "nan") ─────────────────────────────────
def _ok(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _pct(x, dp=1, sign=False) -> str:
    if not _ok(x):
        return "—"
    return f"{x * 100:+.{dp}f} %" if sign else f"{x * 100:.{dp}f} %"


def _num(x, fmt="{:.2f}") -> str:
    return fmt.format(x) if _ok(x) else "—"


def _e(x) -> str:
    return "—" if x is None or x == "" else escape(str(x))


def badge(label: str) -> str:
    colour = BADGE.get(label, "#6e7681")
    light = " light" if label in ("NOT-STEPPED", "INCONCLUSIVE-180") else ""
    return f'<span class="pool-badge{light}" style="background:{colour}">{escape(label)}</span>'


def unavailable_box(reason: str) -> str:
    return (f'{BEGIN}\n<div class="card pool" id="pool">\n  <h2>Strategy pool (S35)</h2>\n'
            f'  <div class="unavail">pool report unavailable: {escape(reason)[:300]}</div>\n</div>\n{END}\n')


# ── data ────────────────────────────────────────────────────────────────────
def load_report(report_path, journal: dict, baselines_dir, history_path, lines=None,
                B: Optional[int] = None) -> tuple[Optional[dict], Optional[str]]:
    """(report dict, None) or (None, reason). Never raises."""
    try:
        p = Path(report_path) if report_path else None
        if p is not None and p.exists():
            try:
                rep = json.loads(p.read_text())
            except ValueError as e:
                return None, f"malformed {p.name}: {e}"
            if not (isinstance(rep, dict) and isinstance(rep.get("lines"), list) and isinstance(rep.get("live"), dict)
                    and all(isinstance(l, dict) and "id" in l for l in rep["lines"])):
                return None, f"malformed {p.name}: missing live/lines"
            return rep, None
        kw = {} if B is None else {"B": B}
        if lines is not None:
            kw["lines"] = lines
        rep = score_pool(journal, baselines_dir, load_context_history(history_path), **kw)
        return rep.to_dict(), None
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {e}"


def _last_value(days: list[dict], lid: str):
    for d in reversed(days):
        if d.get("skipped"):
            continue
        if lid == LIVE_ID:
            return d.get("portfolio_value")
        v = ((d.get("lines") or {}).get(lid) or {}).get("value") or d.get(f"{lid}_value")
        if v:
            return v
    return None


def weekly_view(days: list[dict], line_ids: list[str], weeks: int = WEEKS) -> tuple[list[str], dict]:
    """Last ``weeks`` ISO weeks present in the journal: per line per week, summed LineDay counters."""
    agg: dict[str, dict[str, dict[str, int]]] = {}
    order: list[str] = []
    for d in days:
        try:
            y, w, _ = date.fromisoformat(d["date"]).isocalendar()
        except (KeyError, TypeError, ValueError):
            continue
        wk = f"{y}-W{w:02d}"
        if wk not in order:
            order.append(wk)
        for lid in line_ids:
            row = (d.get("lines") or {}).get(lid)
            if not isinstance(row, dict):
                continue
            c = agg.setdefault(lid, {}).setdefault(wk, dict(signals=0, entries=0, cap=0, gate=0, cash=0, data=0))
            c["signals"] += int(row.get("signals") or 0)
            c["entries"] += int(row.get("entries") or 0)
            for k, v in (row.get("blocked") or {}).items():
                if k in c:
                    c[k] += int(v or 0)
    return order[-weeks:], agg


def _bucket_cols(dim: str, cond: dict, ids: list[str]) -> list[str]:
    seen = sorted({b for i in ids for b in ((cond.get(i) or {}).get(dim) or {})})
    pref = BUCKET_ORDER.get(dim)
    if pref:
        key = lambda b: (pref.index(b) if b in pref else len(pref), b)
    else:
        key = lambda b: (b == "unknown", b)
    return sorted(seen, key=key)


def _heat_style(p) -> str:
    if not _ok(p):
        return ""
    t = max(-1.0, min(1.0, (p - 0.5) * 2.5))
    rgb = "63,185,80" if t >= 0 else "248,81,73"
    return f' style="background:rgba({rgb},{0.15 + 0.6 * abs(t):.2f})"'


# ── render ──────────────────────────────────────────────────────────────────
def _lines_table(rep: dict, days: list[dict]) -> str:
    specs = {l.id: l for l in LINES}
    head = ("Line", "Family", "Hypothesis", "Status", "Days", "Value", "Return", "vs live", "Trades / entry days",
            "Win % [90 % CI]", "P(win)", "PF", "Max DD", "Exposure", "Tier", "Holm p_adj", "Verdict")
    out = ['<div class="pool-wrap"><table id="pool-lines"><thead><tr>' +
           "".join(f"<th>{escape(h)}</th>" for h in head) + "</tr></thead><tbody>"]
    for s in [rep["live"]] + list(rep["lines"]):
        lid = s.get("id", "?")
        spec = specs.get(lid)
        hyp = s.get("hypothesis") or ("reference" if lid == LIVE_ID else None)
        if spec and spec.hypothesis and spec.hypothesis.cond:
            hyp = f"{hyp}({spec.hypothesis.cond})"
        tag = ' <span class="pool-tag">S34 shadow</span>' if lid.startswith("shadow_") else ""
        paired = s.get("paired") or {}
        wp = s.get("win_post") or {}
        win = (f"{_pct(s.get('win_rate'), 0)} [{_pct(wp.get('lo'), 0)}–{_pct(wp.get('hi'), 0)}]"
               if s.get("n_trades") else "—")
        pf = "no losses" if s.get("no_losses") and s.get("n_trades") else _num(s.get("profit_factor"))
        label = "INCONCLUSIVE-180" if s.get("rule") == "INCONCLUSIVE-180" else (s.get("verdict") or "—")
        extra = ""
        if label == "CONTROL":
            extra = f' <span class="muted">guard effect {_pct(s.get("guard_effect"), 2, sign=True)}</span>'
        sent = escape(s.get("auto_sentence") or "")
        reasons = "; ".join(escape(str(r)) for r in (s.get("reasons") or []))
        out.append(
            f"<tr><td>{escape(str(lid))}{tag}</td><td>{_e(s.get('family'))}</td><td>{_e(hyp)}</td>"
            f"<td>{_e(s.get('status'))}</td><td class=\"num\">{_num(s.get('days_active'), '{:d}')}</td>"
            f"<td class=\"num\">{_num(_last_value(days, lid), '${:,.2f}')}</td>"
            f"<td class=\"num\">{_pct(s.get('total_return'), 1, sign=True)}</td>"
            f"<td class=\"num\">{_num(paired["cum_ret_diff"] * 100 if _ok(paired.get("cum_ret_diff")) else None, '{:+.1f} pp')}</td>"
            f"<td class=\"num\">{_num(s.get('n_trades'), '{:d}')} / {_num(s.get('n_entry_days'), '{:d}')}</td>"
            f"<td class=\"num\">{win}</td><td class=\"num\">{_pct(wp.get('mean'), 0)}</td><td class=\"num\">{pf}</td>"
            f"<td class=\"num\">{_pct(s.get('max_dd'))}</td><td class=\"num\">{_pct(s.get('mean_exposure'), 0)}</td>"
            f"<td>{_e(s.get('tier'))}</td><td class=\"num\">{_num(s.get('holm_adjusted_p'))}</td>"
            f"<td title=\"{sent}\">{badge(label)}{extra}<details><summary>why</summary>{sent}"
            f"<div class=\"muted\">{reasons}</div></details></td></tr>")
    out.append("</tbody></table></div>")
    return "\n".join(out)


def _heatmaps(rep: dict) -> str:
    cond = rep.get("cond_table") or {}
    ids = [LIVE_ID] + [l["id"] for l in rep["lines"] if l["id"] in cond]
    out = [f'<h3>Condition × line</h3><div class="muted">{escape(HEATMAP_CAPTION)}</div>']
    for dim in DIMENSIONS:
        cols = _bucket_cols(dim, cond, ids)
        out.append(f'<div class="pool-wrap"><table class="pool-heat" id="pool-heat-{escape(dim)}"><thead><tr><th>{escape(dim)}</th>' +
                   "".join(f"<th>{escape(c)}</th>" for c in cols) + "</tr></thead><tbody>")
        if not cols:
            out.append(f'<tr><td class="muted">no closed trades with a {escape(dim)} tag yet</td></tr>')
        for lid in ids:
            if not cols:
                break
            cells = (cond.get(lid) or {}).get(dim) or {}
            tds = []
            for c in cols:
                cell = cells.get(c)
                if not cell:
                    tds.append('<td class="heat empty">—</td>')
                elif cell.get("tier") == "insufficient":
                    txt = f"n&lt;{MIN_TRADES}" if (cell.get("n") if _ok(cell.get("n")) else 0) < MIN_TRADES else "days&lt;5"
                    tds.append(f'<td class="heat insuff" title="n={_e(cell.get("n"))}, insufficient">{txt}</td>')
                else:
                    tds.append(f'<td class="heat"{_heat_style(cell.get("post_mean"))} title="P(win) '
                               f'{_pct(cell.get("post_mean"), 0)}, mean {_pct(cell.get("mean_pnl_pct"), 2, True)}/trade">'
                               f'n={_e(cell.get("n"))} w={_pct(cell.get("win_rate"), 0).replace(" ", "")}</td>')
            out.append(f"<tr><td>{escape(str(lid))}</td>{''.join(tds)}</tr>")
        out.append("</tbody></table></div>")
    return "\n".join(out)


def _weekly(rep: dict, days: list[dict]) -> str:
    ids = [l["id"] for l in rep["lines"]]
    weeks, agg = weekly_view(days, ids)
    out = ['<h3>Weekly triggering (last 8 ISO weeks)</h3><div class="muted">signals / entries / blocked; '
           'second line: blocked by cap · gate · cash · data (from LineDay rows)</div>',
           '<div class="pool-wrap"><table id="pool-weekly"><thead><tr><th>Line</th>' +
           "".join(f"<th>{escape(str(w))}</th>" for w in weeks) + "</tr></thead><tbody>"]
    for lid in ids:
        tds = []
        for w in weeks:
            c = (agg.get(lid) or {}).get(w)
            if not c:
                tds.append('<td class="num muted">—</td>')
                continue
            blk = c["cap"] + c["gate"] + c["cash"] + c["data"]
            tds.append(f'<td class="num">{escape(str(c["signals"]))} / {escape(str(c["entries"]))} / {escape(str(blk))}<div class="muted" '
                       f'style="font-size:0.7rem">{escape(str(c["cap"]))} · {escape(str(c["gate"]))} · {escape(str(c["cash"]))} · {escape(str(c["data"]))}</div></td>')
        out.append(f"<tr><td>{escape(str(lid))}</td>{''.join(tds)}</tr>")
    out.append("</tbody></table></div>")
    return "\n".join(out)


def _flags(rep: dict, days: list[dict]) -> str:
    last = days[-1] if days else {}
    errs = last.get("pool_errors") or {}
    stale = last.get("stale_symbols") or []
    df = ((last.get("context") or {}).get("data_flags") or {})
    ctrl = [l for l in rep["lines"] if l.get("hypothesis") == "control"]
    rows = [("Pool errors (today)", "; ".join(f"{escape(str(k))}: {escape(str(v))}" for k, v in errs.items()) or "none"),
            ("Stale symbols (today)", escape(", ".join(map(str, stale))) or "none"),
            ("Data flags: stale", escape(", ".join(map(str, df.get("stale") or []))) or "none"),
            ("Data flags: gap200", escape(", ".join(map(str, df.get("gap200") or []))) or "none")]
    for c in ctrl:
        rows.append((f"{escape(str(c['id']))} guard effect (control − live, common window)",
                     _pct(c.get("guard_effect"), 2, sign=True)))
    return ('<h3>Pool errors and data flags</h3><table id="pool-flags"><tbody>' +
            "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in rows) + "</tbody></table>")


def render_pool_section(rep: Optional[dict], reason: Optional[str], journal: dict) -> str:
    """Self-contained HTML block between BEGIN/END markers. Never raises."""
    if rep is None:
        return unavailable_box(reason or "no report")
    try:
        days = journal.get("days") or []
        notes = "".join(f"<li>{escape(str(n))}</li>" for n in rep.get("notes") or [])
        body = "\n".join([
            f'{BEGIN}\n{CSS}\n<div class="card pool" id="pool">',
            f'  <h2>Strategy pool (S35, paper-only)</h2>',
            f'  <div class="muted" id="pool-note"><strong>{escape(HEADER_NOTE)}</strong> As of {_e(rep.get("as_of"))}.</div>',
            _lines_table(rep, days), _heatmaps(rep), _weekly(rep, days), _flags(rep, days),
            f'  <details><summary>Notes and caveats</summary><ul class="muted">{notes}</ul></details>',
            f"</div>\n{END}\n"])
        return body
    except Exception as e:  # noqa: BLE001
        return unavailable_box(f"render failed: {type(e).__name__}: {e}")


def pool_summary(rep: Optional[dict], reason: Optional[str]) -> dict[str, Any]:
    """Aggregates for data.json key ``pool`` (no trade lists)."""
    if rep is None:
        return dict(available=False, reason=reason)
    return dict(available=True, as_of=rep.get("as_of"),
                lines=[dict(id=l.get("id"), verdict=l.get("verdict"), rule=l.get("rule"), tier=l.get("tier"),
                            holm_adjusted_p=l.get("holm_adjusted_p")) for l in rep["lines"]])
