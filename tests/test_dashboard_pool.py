"""S35 C4: strategy-pool dashboard section (lines table, heatmap, weekly view, flags) and the issue summary."""
import dataclasses
import importlib.util
import json
import re
from html.parser import HTMLParser
from pathlib import Path

import pytest

from trading_engine.paper.journal import Journal
from trading_engine.paper.pool import dashboard as pd_
from trading_engine.paper.pool import scoring as sc
from trading_engine.paper.pool.registry import LINES

ROOT = Path(__file__).parent.parent
FIXTURE = ROOT / "tests" / "fixtures" / "pool_journal_replay.json"   # trimmed 120-day replay, 2026-06-18..10-15


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


bd = _load("build_dashboard", "scripts/build_dashboard.py")
ui = _load("update_issue", "scripts/update_issue.py")


class _Cells(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tables, self.ids, self.cells, self._in = 0, [], [], 0

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            self.tables += 1
            self.ids.append(dict(attrs).get("id"))
        if tag in ("td", "th"):
            self._in += 1
            self.cells.append("")

    def handle_endtag(self, tag):
        if tag in ("td", "th"):
            self._in -= 1

    def handle_data(self, data):
        if self._in > 0:
            self.cells[-1] += data


def _parse(html):
    p = _Cells()
    p.feed(html)
    return p


def _strip_pool(html):
    return re.sub(re.escape(pd_.BEGIN) + r".*?" + re.escape(pd_.END) + r"\n", "", html, flags=re.S)


@pytest.fixture(scope="module")
def replay(tmp_path_factory):
    out = tmp_path_factory.mktemp("dash")
    fx = json.loads(FIXTURE.read_text())
    reg = fx["days"][0]["date"]   # the replay registered every line on its first day
    lines = tuple(dataclasses.replace(l, registered_on=reg) for l in LINES)
    rep = sc.score_pool(fx, ROOT / "state" / "pool_baselines", {}, B=500, lines=lines)
    (out / "pool_report.json").write_text(rep.to_json())
    assert bd.main(["--journal", str(FIXTURE), "--docs-dir", str(out / "docs"),
                    "--pool-report", str(out / "pool_report.json")]) == 0
    return out, rep


def test_fixture_is_small():
    assert FIXTURE.stat().st_size < 300_000


def test_replay_renders_all_sections(replay):
    out, rep = replay
    html = (out / "docs" / "index.html").read_text()
    p = _parse(html)
    assert len(html) < 1_000_000 and pd_.HEADER_NOTE in html and "Exploratory; never drives verdicts" in html
    assert {"pool-lines", "pool-weekly", "pool-flags"} <= set(p.ids)
    assert [i for i in p.ids if i and i.startswith("pool-heat-")] == [f"pool-heat-{d}" for d in sc.DIMENSIONS]
    for l in LINES:
        assert f"<td>{l.id}" in html
    for s in rep.lines:
        label = "INCONCLUSIVE-180" if s.rule == "INCONCLUSIVE-180" else s.verdict
        assert f">{label}</span>" in html
    assert 'class="pool-tag">S34 shadow' in html and "guard effect" in html
    weekly = html[html.index('id="pool-weekly"'):]
    assert len(re.findall(r"<th>\d{4}-W\d\d</th>", weekly[:weekly.index("</thead>")])) == 8
    assert 'class="heat insuff"' in html and "n&lt;8" in html
    bad = [c for c in p.cells if re.search(r"\b(None|nan|NaN|inf)\b", c)]
    assert bad == []
    data = json.loads((out / "docs" / "data.json").read_text())
    assert data["pool"]["available"] and len(data["pool"]["lines"]) == 9


def test_badge_colours():
    assert "#238636" in pd_.badge("PROMOTE-cand") and "#da3633" in pd_.badge("HARD-KILL")
    assert "#8957e5" in pd_.badge("CONTROL") and "#1f6feb" in pd_.badge("DORMANT")
    assert "#d18616" in pd_.badge("RETIRE-cand") and "#d4a72c" in pd_.badge("INCONCLUSIVE-180")


def test_live_sections_unchanged(replay, tmp_path):
    out, _ = replay
    html = (out / "docs" / "index.html").read_text()
    assert _strip_pool(html) == bd.INDEX_HTML.replace(bd.POOL_PLACEHOLDER, "")
    data = json.loads((out / "docs" / "data.json").read_text())
    ref = bd.build_data_json(Journal.load(FIXTURE))
    for k in ("generated_at", "pool"):
        data.pop(k, None)
        ref.pop(k, None)
    assert data == json.loads(json.dumps(ref))


def test_real_journal_no_report_computes_insufficient(tmp_path):
    jp = ROOT / "state" / "journal.json"
    assert bd.main(["--journal", str(jp), "--docs-dir", str(tmp_path), "--pool-report",
                    str(tmp_path / "missing.json")]) == 0
    html = (tmp_path / "index.html").read_text()
    assert "pool report unavailable" not in html and "insufficient" in html
    assert _strip_pool(html) == bd.INDEX_HTML.replace(bd.POOL_PLACEHOLDER, "")
    assert [c for c in _parse(html).cells if re.search(r"\b(None|nan|NaN)\b", c)] == []
    assert not (ROOT / "docs" / "pool_report.json").exists()


@pytest.mark.parametrize("payload", ["{not json", json.dumps({"lines": "x"}), json.dumps([1, 2])])
def test_malformed_report_renders_fallback(tmp_path, payload):
    (tmp_path / "pool_report.json").write_text(payload)
    assert bd.main(["--journal", str(FIXTURE), "--docs-dir", str(tmp_path / "d"),
                    "--pool-report", str(tmp_path / "pool_report.json")]) == 0
    html = (tmp_path / "d" / "index.html").read_text()
    assert "pool report unavailable" in html and 'id="pool-lines"' not in html
    assert _strip_pool(html) == bd.INDEX_HTML.replace(bd.POOL_PLACEHOLDER, "")
    assert json.loads((tmp_path / "d" / "data.json").read_text())["pool"]["available"] is False


def test_render_failure_is_contained():
    html = pd_.render_pool_section({"live": {}, "lines": [{"id": "x", "n_trades": "bad"}]}, None, {"days": []})
    assert pd_.BEGIN in html and pd_.END in html


def test_weekly_view_counts_and_window():
    days = [dict(date=f"2026-09-{d:02d}", lines={"a": dict(signals=2, entries=1, blocked=dict(cap=1, data=2))})
            for d in range(1, 30)] + [dict(date="2026-09-30", skipped=True)]
    weeks, agg = pd_.weekly_view(days, ["a"], weeks=3)
    assert weeks == ["2026-W38", "2026-W39", "2026-W40"]
    assert agg["a"]["2026-W39"] == dict(signals=14, entries=7, cap=7, gate=0, cash=0, data=14)


def test_issue_pool_lines(replay, tmp_path):
    out, rep = replay
    txt = ui.pool_lines(out / "pool_report.json")
    assert txt.count("\n- `") + txt.startswith("- `") == len(rep.lines) and "tier" in txt and "owner rules" in txt
    (tmp_path / "bad.json").write_text("{")
    assert ui.pool_lines(tmp_path / "bad.json") == "" and ui.pool_lines(tmp_path / "missing.json") == ""
    body = ui.build_body(Journal.load(ROOT / "state" / "journal.json"), out / "pool_report.json")
    assert "Strategy pool (S35" in body and body.index("Strategy pool") < body.index("### Links")
