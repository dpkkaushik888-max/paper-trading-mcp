"""Write state/pool_context_history.json: the S35 day context for every existing journal day, from cached
.us bars (data/bars/us, see scripts/fetch_bars.py). Context dict only; journal rows are never rewritten.

  python scripts/pool_context_backfill.py [--journal state/journal.json] [--bars data/bars] [--out ...]
"""
from __future__ import annotations

import argparse
from datetime import timedelta
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from trading_engine.paper.pool.context import context_for_bar  # noqa: E402
from scripts.pool_backtest import BTC, load_bars  # noqa: E402

# B3: rows 2026-06-23..29 were backfilled with the pre-8a2acc9 `<=` as_of bug; flag them.
B3_BACKFILL_DAYS = {f"2026-06-{d}" for d in range(23, 30)}


def backfill(journal_path: Path, bars_dir: Path) -> dict:
    bars = load_bars(bars_dir, "us")
    out = {}
    for day in json.loads(journal_path.read_text())["days"]:
        d = day["date"]
        ctx = context_for_bar(bars, pd.Timestamp(d) - timedelta(days=1), BTC).to_dict()
        if d in B3_BACKFILL_DAYS:
            ctx["data_flags"]["b3_backfill"] = True
        out[d] = ctx
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--journal", default="state/journal.json")
    ap.add_argument("--bars", default="data/bars")
    ap.add_argument("--out", default="state/pool_context_history.json")
    a = ap.parse_args(argv)
    hist = backfill(Path(a.journal), Path(a.bars))
    Path(a.out).write_text(json.dumps(hist, indent=1, sort_keys=True))
    print(f"wrote {len(hist)} days to {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
