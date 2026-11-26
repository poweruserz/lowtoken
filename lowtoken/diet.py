"""Startup-context diet: what you load on every call versus what you use.

Every agent description, skill listing line, MCP server and memory file sits in
the context of every API call of every session, and is paid for on each one.
`claude -p /context` (free, no model turn) gives the exact token cost of each
component; the transcripts tell how often each was actually used. A component
that costs tokens on every call and was never used is dead weight.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from collections import Counter
from dataclasses import dataclass

_ROW = re.compile(r"^\|\s*(?P<name>[^|]+?)\s*\|\s*(?P<src>[^|]*?)\s*\|\s*(?P<tok>[<~]?\s*[\d.]+k?)\s*\|\s*$", re.M)


def _tok(s: str) -> float:
    s = s.replace("~", "").strip()
    if s.startswith("<"):
        return float(s[1:]) / 2
    return float(s[:-1]) * 1000 if s.endswith("k") else float(s)


def _section(md: str, title: str) -> str:
    i = md.find(f"### {title}")
    if i < 0:
        return ""
    j = md.find("\n### ", i + 4)
    return md[i: j if j > 0 else len(md)]


def context_report(cwd: str | None = None, claude: str = "claude") -> str:
    """The markdown that /context prints for a fresh session in `cwd`."""
    p = subprocess.run([shutil.which(claude) or claude, "-p", "/context", "--no-session-persistence",
                        "--output-format", "json"], cwd=cwd, capture_output=True, text=True,
                       encoding="utf-8", timeout=300, env=dict(os.environ, MSYS_NO_PATHCONV="1"))
    return json.loads(p.stdout)["result"]


@dataclass
class Component:
    kind: str        # agent | skill | mcp | memory | category
    name: str
    group: str       # plugin or server it belongs to, for one-switch removal
    tokens: float
    uses: int = 0


def components(md: str) -> tuple[list[Component], dict[str, float]]:
    cats: dict[str, float] = {}
    for line in _section(md, "Estimated usage by category").splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) >= 2 and re.match(r"^[\d.]+k?$", cells[1]):
            cats[cells[0]] = _tok(cells[1])
    out: list[Component] = []
    for m in _ROW.finditer(_section(md, "Custom Agents")):
        n = m.group("name")
        if n not in ("Agent Type",) and not n.startswith("-"):
            out.append(Component("agent", n, n.split(":")[0] if ":" in n else "(user agents)", _tok(m.group("tok"))))
    for m in _ROW.finditer(_section(md, "Skills")):
        n = m.group("name")
        if n not in ("Skill",) and not n.startswith("-"):
            out.append(Component("skill", n, n.split(":")[0] if ":" in n else "(user skills)", _tok(m.group("tok"))))
    # Deferred MCP tools cost only their name in the "available tools" listing, but every
    # server lists all of its tools there on every session.
    from . import cost
    servers: dict[str, float] = {}
    for m in _ROW.finditer(_section(md, "MCP Tools")):
        n, srv = m.group("name"), m.group("src")
        if n.startswith("mcp__") and srv:
            servers[srv] = servers.get(srv, 0.0) + cost.estimate(n + "\n")
    for srv, tok in servers.items():
        out.append(Component("mcp", srv, f"mcp: {srv}", tok))
    return out, cats


def usage(paths: list[str]) -> Counter:
    """How often each agent, skill and MCP server was used in these transcripts."""
    used: Counter = Counter()
    cmd = re.compile(r"<command-name>/?([^<]+)</command-name>")
    for p in paths:
        with open(p, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if '"tool_use"' in line:
                    try:
                        d = json.loads(line)
                    except ValueError:
                        continue
                    for b in (d.get("message") or {}).get("content") or []:
                        if not isinstance(b, dict) or b.get("type") != "tool_use":
                            continue
                        name, inp = b.get("name", ""), b.get("input") or {}
                        if name == "Agent":
                            used["agent:" + str(inp.get("subagent_type") or "general-purpose")] += 1
                        elif name == "Skill":
                            used["skill:" + str(inp.get("skill") or "")] += 1
                        elif name.startswith("mcp__"):
                            used["mcp:" + name.split("__")[1]] += 1
                if "<command-name>" in line:
                    for c in cmd.findall(line):
                        used["skill:" + c.strip()] += 1
    return used


def plan(comps: list[Component], used: Counter, calls: int, sessions: int, model: str | None = None) -> list[dict]:
    """Groups (plugins) whose components were never used, with what carrying them cost."""
    from . import prices
    i, _, r = prices.price(model)
    per_token = calls * r + sessions * i * 2.0   # read on every call + one cache write per session
    groups: dict[str, dict] = {}
    for c in comps:
        c.uses = used.get(f"{c.kind}:{c.name}", 0)
        g = groups.setdefault(c.group, {"group": c.group, "tokens": 0.0, "uses": 0, "items": 0, "unused_tokens": 0.0})
        g["tokens"] += c.tokens
        g["uses"] += c.uses
        g["items"] += 1
        if not c.uses:
            g["unused_tokens"] += c.tokens
    out = []
    for g in groups.values():
        g["usd"] = g["unused_tokens"] * per_token
        if g["unused_tokens"] > 0:
            out.append(g)
    return sorted(out, key=lambda g: -g["usd"])
