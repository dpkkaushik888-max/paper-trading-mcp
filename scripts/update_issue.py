"""Edit the pinned S18 progress issue with the latest daily summary.

Uses the `gh` CLI which is pre-installed on GitHub Actions runners and picks
up GITHUB_TOKEN automatically (from `env: GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}`).

Expected issue title: "S18 Paper-Forward — Progress"
If the issue doesn't exist yet, this script prints instructions and exits 0.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from trading_engine.paper.config import TEST_DAYS
from trading_engine.paper.journal import Journal

ISSUE_TITLE = "S18 Paper-Forward — Progress"


def find_issue_number() -> int | None:
    """Return the issue number for the pinned progress issue, or None."""
    try:
        out = subprocess.check_output(
            ["gh", "issue", "list", "--state", "open", "--json",
             "number,title", "--limit", "50"],
            text=True,
        )
        issues = json.loads(out)
        for iss in issues:
            if iss["title"].strip() == ISSUE_TITLE:
                return int(iss["number"])
    except (subprocess.CalledProcessError, FileNotFoundError, json.JSONDecodeError) as e:
        print(f"gh issue list failed: {e}", file=sys.stderr)
    return None


def build_body(j: Journal) -> str:
    n = j.days_elapsed()
    port = j.last_portfolio_value()
    start = j.starting_capital
    total_ret = (port / start - 1) * 100 if start > 0 else 0.0

    closed = j.closed_trades
    wins = [t for t in closed if t.pnl > 0]
    wr = (len(wins) / len(closed) * 100) if closed else 0.0

    total_win = sum(t.pnl for t in wins)
    total_loss = -sum(t.pnl for t in closed if t.pnl <= 0)
    pf = (total_win / total_loss) if total_loss > 0 else float("inf") if wins else 0.0
    pf_str = f"{pf:.2f}" if pf != float("inf") else "∞"

    last_day = j.days[-1] if j.days else None
    last_decisions = last_day["decisions"] if last_day else []
    enters = [d["symbol"] for d in last_decisions if d["action"] == "enter"]
    exits_ = [f"{d['symbol']} ({d['reason']})" for d in last_decisions if d["action"] == "exit"]

    open_str = ", ".join(p.symbol for p in j.open_positions) or "_none_"
    halt = _check_halt_msg(j)

    body = f"""## S18 Paper-Forward — Day {n} of {TEST_DAYS}

**Last update:** {datetime.now(timezone.utc).isoformat(timespec="seconds")}

### Summary
| Metric | Value |
|--------|-------|
| Portfolio | **${port:.2f}** ({total_ret:+.2f}% vs start) |
| Cash | ${j.cash:.2f} |
| Open positions | {len(j.open_positions)} / 6 — {open_str} |
| Closed trades | {len(closed)} |
| Win rate | {wr:.1f}% ({len(wins)}W / {len(closed) - len(wins)}L) |
| Profit factor | {pf_str} |
| Max drawdown | {j.max_drawdown() * 100:.2f}% |

### Today's decisions
- **Entered:** {", ".join(enters) if enters else "_none_"}
- **Exited:** {", ".join(exits_) if exits_ else "_none_"}

### Links
- 📊 [Full dashboard](https://dpkkaushik888-max.github.io/paper-trading-mcp/)
- 📓 [Journal source](../blob/master/state/journal.json)
- 📋 [S18 spec](../blob/master/specs/S18-paper-forward-validation.md)

{halt if halt else ""}

---
_This issue body is auto-edited by `.github/workflows/paper-forward.yml` on each daily run._
"""
    return body


def _check_halt_msg(j: Journal) -> str | None:
    from trading_engine.paper.config import (
        MAX_DRAWDOWN_PCT, MAX_CONSECUTIVE_LOSSES, MAX_NO_SIGNAL_DAYS,
    )
    if j.max_drawdown() > MAX_DRAWDOWN_PCT:
        return f"### ⚠️ HALT — Drawdown {j.max_drawdown() * 100:.2f}% exceeds {MAX_DRAWDOWN_PCT * 100:.0f}%"
    if j.consecutive_losses() >= MAX_CONSECUTIVE_LOSSES:
        return f"### ⚠️ HALT — {j.consecutive_losses()} consecutive losses"
    if j.days_since_last_trade() >= MAX_NO_SIGNAL_DAYS:
        return f"### ⚠️ HALT — {j.days_since_last_trade()} days without any trade"
    return None


def main() -> int:
    issue_num = find_issue_number()
    if issue_num is None:
        print(f"No open issue titled '{ISSUE_TITLE}'. Skipping update.")
        print("Create one manually with:")
        print(f"  gh issue create --title '{ISSUE_TITLE}' --body 'initialising'")
        return 0

    j = Journal.load()
    body = build_body(j)
    subprocess.run(
        ["gh", "issue", "edit", str(issue_num), "--body", body],
        check=True,
    )
    print(f"Updated issue #{issue_num}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
