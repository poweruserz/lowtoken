import json
from collections import Counter
import os

import pytest

from lowtoken import diet, oracle, prices, rent


def _call(mid, ctx_read, ctx_write, out, think=0, model="claude-opus-5-5", tools=()):
    content = [{"type": "tool_use", "id": t, "name": "Bash", "input": {"command": "npm test"}} for t in tools]
    return {"type": "assistant", "timestamp": "2026-11-26T10:00:00Z", "message": {
        "id": mid, "model": model, "content": content,
        "usage": {"input_tokens": 0, "cache_read_input_tokens": ctx_read, "cache_creation_input_tokens": ctx_write,
                  "cache_creation": {"ephemeral_1h_input_tokens": ctx_write, "ephemeral_5m_input_tokens": 0},
                  "output_tokens": out, "output_tokens_details": {"thinking_tokens": think}}}}


def _result(tid, text):
    return {"type": "user", "message": {"content": [{"type": "tool_result", "tool_use_id": tid, "content": text}]}}


@pytest.fixture
def transcript(tmp_path):
    lines = [
        _call("m1", 0, 10_000, 100, think=40, tools=["t1"]),
        _result("t1", "PASS test one\n" * 50),
        {"type": "attachment", "attachment": {"type": "hook_success", "hookEvent": "PostToolUse",
                                              "content": '{"session_id": "x", "hook_event_name": "PostToolUse"}'}},
        _call("m2", 10_000, 600, 50),
        _call("m3", 10_600, 60, 20),
        _call("m4", 3_000, 500, 10),          # compacted: context shrank
        _call("m5", 3_500, 20, 10),
    ]
    p = tmp_path / "s.jsonl"
    p.write_text("\n".join(json.dumps(x) for x in lines) + "\n", encoding="utf-8")
    return str(p)


def test_bill_is_exact(transcript):
    s = rent.parse(transcript)
    assert len(s.calls) == 5
    i, o, r = prices.price("claude-opus-5-5")
    want = sum(c["usage"]["cache_read_input_tokens"] * r + c["usage"]["cache_creation_input_tokens"] * i * 2
               + c["usage"]["output_tokens"] * o for c in s.calls)
    assert s.bill == pytest.approx(want)


def test_growth_is_split_and_rent_stops_at_compaction(transcript):
    s = rent.parse(transcript)
    kinds = {}
    for it in s.items:
        kinds.setdefault(it.kind, []).append(it)
    startup = kinds["startup"][0]
    assert startup.tokens == 10_000 and startup.reads == 3       # calls 1-3, then compaction
    assert kinds["startup"][1].label == "context after compaction" and kinds["startup"][1].reads == 2
    think, out = kinds["thinking"][0], kinds["output"][0]
    assert (think.tokens, out.tokens, think.gen, out.gen) == (40, 60, 40, 60)
    grow = sum(it.tokens for it in s.items if it.first == 1)
    assert grow == pytest.approx(600)                             # exactly the context growth before call 2
    tool = kinds["tool"][0]
    assert tool.label == "Bash: npm test" and tool.reads == 2


def test_echoing_hook_is_found(transcript):
    s = rent.parse(transcript)
    f = rent.findings([s])
    assert any(x["id"] == "echoing-hook" for x in f)


def test_interactive_cmd_hook_is_diagnosed(tmp_path):
    banner = ('Microsoft Windows [Version 10.0.26200.9974]\r\n\r\nC:\\p>{"session_id":"x",'
              '"hook_event_name":"UserPromptSubmit","prompt":"' + "long prompt " * 40 + '"}\r\n\r\nC:\\p>')
    lines = [_call("m1", 0, 10_000, 10),
             {"type": "attachment", "attachment": {"type": "hook_success", "hookEvent": "UserPromptSubmit",
                                                   "content": banner}},
             {"type": "user", "message": {"content": "next prompt"}},
             _call("m2", 10_000, 700, 10)]
    p = tmp_path / "s.jsonl"
    p.write_text("\n".join(json.dumps(x) for x in lines) + "\n", encoding="utf-8")
    f = rent.findings([rent.parse(str(p))])
    hit = next(x for x in f if x["id"] == "interactive-cmd-hook")
    assert "//c" in hit["fix"] and hit["usd"] > 0


CONTEXT_MD = """## Context Usage
### Estimated usage by category

| Category | Tokens | Percentage |
|----------|--------|------------|
| System prompt | 2.6k | 0.3% |
| Custom agents | 1.2k | 0.1% |
| Skills | 300 | 0.0% |

### MCP Tools

| Tool | Server | Tokens |
|------|--------|--------|
| mcp__big__do_thing | big | 900 |
| mcp__big__other_thing | big | 300 |

### Custom Agents

| Agent Type | Source | Tokens |
|------------|--------|--------|
| plug:reviewer | Plugin | 700 |
| plug:writer | Plugin | 500 |

### Memory Files

| Type | Path | Tokens |
|------|------|--------|
| Project | C:\\tmp\\x\\.claude\\rules\\s00000.md | 12 |
| Project | C:\\tmp\\x\\.claude\\rules\\s00001.md | 1.3k |
| Project | C:\\tmp\\x\\.claude\\rules\\z0000.md | 4 |

### Skills

| Skill | Source | Tokens |
|-------|--------|--------|
| tidy | User | < 20 |
| plug:deploy | Plugin | ~190 |
"""


def test_oracle_parse_skips_rounded_rows():
    rows = oracle._parse(CONTEXT_MD)
    assert rows == {"s00000.md": 12, "z0000.md": 4}


def test_diet_groups_unused_components():
    comps, cats = diet.components(CONTEXT_MD)
    assert cats["System prompt"] == 2600
    names = {(c.kind, c.name): c.tokens for c in comps}
    assert names[("agent", "plug:reviewer")] == 700
    assert names[("skill", "plug:deploy")] == 190
    assert ("mcp", "big") in names
    plan = diet.plan(comps, {"agent:plug:reviewer": 3}, calls=1000, sessions=10)
    plug = next(g for g in plan if g["group"] == "plug")
    assert plug["unused_tokens"] == 500 + 190 and plug["uses"] == 3
    assert any(g["group"] == "mcp: big" for g in plan)


def test_compact_payback_solves_the_break_even():
    from lowtoken.cli import SUMMARY, compact_payback
    i, o, r = prices.price("claude-opus-5-5")
    ctx, startup = 458_000, 45_000
    after = startup + SUMMARY
    n = compact_payback(ctx, startup, "claude-opus-5-5")
    assert n * (ctx - after) * r == pytest.approx(ctx * r + SUMMARY * o + after * i * 2, rel=0.08)
    assert compact_payback(80_000, 45_000, "claude-opus-5-5") is None  # not worth it


def test_write_multiplier_weights_mixed_ttls():
    u = {"cache_creation": {"ephemeral_5m_input_tokens": 300, "ephemeral_1h_input_tokens": 100}}
    assert prices.write_multiplier(u) == pytest.approx((300 * 1.25 + 100 * 2) / 400)


def test_partial_usage_lines_are_replaced_by_the_final_one(tmp_path):
    first = _call("m1", 0, 10_000, 2)
    final = _call("m1", 0, 10_000, 1093)
    p = tmp_path / "s.jsonl"
    p.write_text(json.dumps(first) + "\n" + json.dumps(final) + "\n", encoding="utf-8")
    s = rent.parse(str(p))
    assert len(s.calls) == 1 and s.calls[0]["usage"]["output_tokens"] == 1093


@pytest.fixture
def claude_home(tmp_path, monkeypatch):
    home = tmp_path / "claude"
    (home / "plugins").mkdir(parents=True)
    installed = {}
    for name, files in [("tidy", ["agents/a.md"]), ("busy", ["agents/a.md", "hooks/hooks.json"]),
                        ("used", ["skills/s/SKILL.md"])]:
        root = tmp_path / "cache" / name
        (root / ".claude-plugin").mkdir(parents=True)
        (root / ".claude-plugin" / "plugin.json").write_text(json.dumps({"name": name}))
        for f in files:
            (root / f).parent.mkdir(parents=True, exist_ok=True)
            (root / f).write_text("x")
        installed[f"{name}@m"] = [{"installPath": str(root)}]
    (home / "plugins" / "installed_plugins.json").write_text(json.dumps({"version": 2, "plugins": installed}))
    settings = {"enabledPlugins": {k: True for k in installed},
                "hooks": {"Stop": [{"hooks": [{"type": "command", "command": "cmd.exe /d /c set X=1&&echo"}]}]}}
    (home / "settings.json").write_text(json.dumps(settings, indent=2))
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(home))
    monkeypatch.setenv("LOWTOKEN_HOME", str(tmp_path / "state"))
    return home


def test_diet_switches_off_only_unused_passive_plugins_and_undoes(claude_home):
    groups = [{"group": g, "uses": u, "tokens": 100.0, "unused_tokens": 100.0 if not u else 0.0, "items": 2, "usd": 1.0}
              for g, u in [("tidy", 0), ("busy", 0), ("used", 3)]]
    off = diet.removable(groups)
    assert [k for k, _ in off] == ["tidy@m"]            # hooks keep "busy" on; "used" was used
    backup = diet.apply([k for k, _ in off])
    s = json.loads((claude_home / "settings.json").read_text())
    assert s["enabledPlugins"] == {"tidy@m": False, "busy@m": True, "used@m": True}
    assert os.path.exists(backup) and s["hooks"]          # nothing else in settings changed
    assert diet.undo() == ["tidy@m"]
    assert json.loads((claude_home / "settings.json").read_text())["enabledPlugins"]["tidy@m"] is True
    assert diet.undo() == []                              # undo only once


@pytest.mark.skipif(os.name != "nt", reason="the Git Bash /c rewrite only happens on Windows")
def test_diet_flags_cmd_hooks_that_git_bash_breaks(claude_home):
    assert diet.broken_hooks() == ["Stop"]


def test_statusline_cache_countdown_and_cold_cost(tmp_path):
    import calendar
    import time
    from lowtoken.cli import cache_state
    p = tmp_path / "t.jsonl"
    line = _call("m1", 400_000, 1_000, 10)
    line["timestamp"] = "2026-11-26T10:00:00Z"
    p.write_text(json.dumps(line) + "\n", encoding="utf-8")
    t0 = calendar.timegm(time.strptime("2026-11-26T10:00:00", "%Y-%m-%dT%H:%M:%S"))
    assert cache_state(str(p), "claude-opus-5-5", now=t0 + 23 * 60) == "cache warm 37m"
    i, _, _ = prices.price("claude-opus-5-5")
    assert cache_state(str(p), "claude-opus-5-5", now=t0 + 3601) == f"cache cold: next call re-writes ${401_000 * i * 2:.2f}"


def test_diet_counts_short_command_names_as_use():
    comps = [diet.Component("skill", "commit-commands:commit", "commit-commands", 30.0),
             diet.Component("skill", "frontend-design:frontend-design", "frontend-design", 40.0)]
    plan = diet.plan(comps, Counter({"skill:commit": 2, "skill:frontend-design": 1}), calls=100, sessions=2)
    assert plan == []   # used under their short names: nothing to switch off


def test_keepwarm_pays_only_where_the_cache_really_expired():
    def call(ts, ctx, wrote):
        u = {"cache_read_input_tokens": ctx - wrote, "cache_creation_input_tokens": wrote, "input_tokens": 0,
             "cache_creation": {"ephemeral_1h_input_tokens": wrote}}
        return {"ts": ts, "ctx": ctx, "usage": u, "model": "claude-opus-5-5", "miss": None, "bill": 0.0}
    s = rent.Session(path="x.jsonl")
    s.calls = [call("2026-11-26T10:00:00Z", 100_000, 100_000),
               call("2026-11-26T12:30:00Z", 101_000, 101_000)]     # back after 2.5 h: whole prefix re-written
    i, _, r = prices.price("claude-opus-5-5")
    kw = rent.keepwarm([s])
    penalty = 100_000 * (2 * i - r)
    assert kw["base"] == pytest.approx(penalty)
    table = dict(kw["table"])
    # one ping (at 59 min) keeps it to 1h59 only: still expired, and the session end wastes a ping too
    assert table[1] == pytest.approx(-(100_000 * r + 101_000 * r))
    # two pings bridge the 2.5 h gap; the session end costs two wasted pings
    assert table[2] == pytest.approx(penalty - 2 * 100_000 * r - 2 * 101_000 * r)
    assert kw["best"] == 2 and kw["expired"] == 1 and kw["ends"] == 1
    s.calls[1]["miss"] = {"type": "model_changed"}                  # a /model switch: pings cannot help
    assert rent.keepwarm([s]) is None
