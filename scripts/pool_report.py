"""Strategy-pool report (S35 C3): score every pool line vs live from the journal, write JSON, print a table.

    python scripts/pool_report.py [--journal state/journal.json] [--out state/pool_report.json] [--print] [--B 5000]
Read-only over the journal; recommendations only.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trading_engine.paper.pool.scoring import B_VERDICT, load_context_history, score_pool  # noqa: E402


def _f(x, scale=100.0, fmt="{:.1f}"):
    return "-" if x is None else fmt.format(x * scale)


def table(report) -> str:
    rows = [("id", "verdict", "n", "win %", "CI", "vs live", "p_adj", "tier")]
    for s in [report.live] + report.lines:
        ci = "-" if s.ci_lo is None else f"[{s.ci_lo * 100:+.1f},{s.ci_hi * 100:+.1f}]"
        vs = "-" if not s.paired else f"{s.paired['cum_ret_diff'] * 100:+.1f}pp"
        rows.append((s.id, s.verdict + ("*" if s.flags else ""), str(s.n_trades), _f(s.win_rate), ci, vs,
                     _f(s.holm_adjusted_p, 1.0, "{:.2f}"), s.tier))
    w = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    return "\n".join(" | ".join(c.ljust(w[i]) for i, c in enumerate(r)) for r in rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--journal", default="state/journal.json")
    ap.add_argument("--out", default="state/pool_report.json")
    ap.add_argument("--baselines", default=None, help="default: <journal dir>/pool_baselines")
    ap.add_argument("--context-history", default=None, help="default: <journal dir>/pool_context_history.json")
    ap.add_argument("--B", type=int, default=B_VERDICT)
    ap.add_argument("--print", action="store_true", dest="do_print")
    a = ap.parse_args(argv)
    jp = Path(a.journal)
    t0 = time.perf_counter()
    rep = score_pool(json.loads(jp.read_text()), a.baselines or jp.parent / "pool_baselines",
                     load_context_history(a.context_history or jp.parent / "pool_context_history.json"), B=a.B)
    dt = time.perf_counter() - t0
    if a.out:
        Path(a.out).write_text(rep.to_json())
    if a.do_print:
        print(f"as_of {rep.as_of} | B={a.B} | scored {1 + len(rep.lines)} lines in {dt:.2f}s")
        print(table(rep))
        for s in [rep.live] + rep.lines:
            print(" -", s.auto_sentence)
    return 0


if __name__ == "__main__":
    sys.exit(main())
