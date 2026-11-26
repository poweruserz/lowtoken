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


# A plugin with any of these does work outside Claude's own choices (hooks, servers, styles...):
# switching it off could change behaviour even if Claude never invoked it, so diet never does.
_ACTIVE_KEYS = {"hooks", "mcpServers", "lspServers", "outputStyles", "statusLine", "monitors"}
_ACTIVE_FILES = ("hooks", ".mcp.json", ".lsp.json", "output-styles", "monitors")


def passive(install_path: str) -> bool:
    """True if the plugin only offers agents, skills and commands, which act only when invoked."""
    try:
        with open(os.path.join(install_path, ".claude-plugin", "plugin.json"), encoding="utf-8") as fh:
            manifest = json.load(fh)
    except (OSError, ValueError):
        return False  # unknown layout: not ours to switch off
    if _ACTIVE_KEYS & set(manifest):
        return False
    return not any(os.path.exists(os.path.join(install_path, f)) for f in _ACTIVE_FILES)


def _claude_home() -> str:
    return os.environ.get("CLAUDE_CONFIG_DIR") or os.path.join(os.path.expanduser("~"), ".claude")


def removable(groups: list[dict], home: str | None = None) -> list[tuple[str, dict]]:
    """(enabledPlugins key, plan group) for enabled, passive plugins none of whose parts were used."""
    home = home or _claude_home()
    try:
        with open(os.path.join(home, "settings.json"), encoding="utf-8") as fh:
            enabled = json.load(fh).get("enabledPlugins") or {}
        with open(os.path.join(home, "plugins", "installed_plugins.json"), encoding="utf-8") as fh:
            installed = json.load(fh).get("plugins") or {}
    except (OSError, ValueError):
        return []
    out = []
    for g in groups:
        if g["uses"] or g["unused_tokens"] < g["tokens"] or g["group"].startswith(("(", "mcp: ")):
            continue
        for key, on in enabled.items():
            if on and key.split("@")[0] == g["group"]:
                paths = [x.get("installPath") for x in installed.get(key) or [] if x.get("installPath")]
                if paths and all(passive(p) for p in paths):
                    out.append((key, g))
    return out


def _write_settings(path: str, settings: dict) -> None:
    tmp = path + ".lowtoken.tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(settings, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    os.replace(tmp, path)


def apply(keys: list[str], home: str | None = None) -> str:
    """Switch plugins off; returns the backup path. The list is kept so `undo` touches only these."""
    import shutil
    import time
    home = home or _claude_home()
    path = os.path.join(home, "settings.json")
    backup = f"{path}.bak-lowtoken-diet-{time.strftime('%Y%m%d-%H%M%S')}"
    shutil.copy2(path, backup)
    with open(path, encoding="utf-8") as fh:
        settings = json.load(fh)
    for k in keys:
        settings.setdefault("enabledPlugins", {})[k] = False
    _write_settings(path, settings)
    from .dedup import state_dir
    d = state_dir()
    if d:
        with open(os.path.join(d, "diet-applied.json"), "w", encoding="utf-8") as fh:
            json.dump({"keys": keys, "backup": backup}, fh)
    return backup


def undo(home: str | None = None) -> list[str]:
    """Switch back on exactly the plugins the last `apply` switched off."""
    from .dedup import state_dir
    home = home or _claude_home()
    d = state_dir()
    rec = os.path.join(d, "diet-applied.json") if d else ""
    try:
        with open(rec, encoding="utf-8") as fh:
            keys = json.load(fh)["keys"]
    except (OSError, ValueError, KeyError):
        return []
    path = os.path.join(home, "settings.json")
    with open(path, encoding="utf-8") as fh:
        settings = json.load(fh)
    for k in keys:
        settings.setdefault("enabledPlugins", {})[k] = True
    _write_settings(path, settings)
    os.remove(rec)
    return keys


def broken_hooks(home: str | None = None) -> list[str]:
    """Hook events whose command starts cmd.exe with single-slash switches.

    Claude Code runs hooks through Git Bash on Windows; there `/c` is rewritten into a path, so
    cmd.exe starts interactively and echoes the hook's input into the context."""
    if os.name != "nt":
        return []
    try:
        with open(os.path.join(home or _claude_home(), "settings.json"), encoding="utf-8") as fh:
            hooks = json.load(fh).get("hooks") or {}
    except (OSError, ValueError):
        return []
    return sorted({ev for ev, groups in hooks.items() for g in groups for h in g.get("hooks") or []
                   if re.match(r"^\s*cmd(\.exe)?\s+(/d\s+)?/c\s", h.get("command") or "", re.I)})


def plan(comps: list[Component], used: Counter, calls: int, sessions: int, model: str | None = None) -> list[dict]:
    """Groups (plugins) whose components were never used, with what carrying them cost."""
    from . import prices
    i, _, r = prices.price(model)
    per_token = calls * r + sessions * i * 2.0   # read on every call + one cache write per session
    groups: dict[str, dict] = {}
    for c in comps:
        c.uses = used.get(f"{c.kind}:{c.name}", 0)
        if ":" in c.name:  # also count "/commit" typed for "commit-commands:commit", or the plugin's own name
            plugin, short = c.name.split(":", 1)
            c.uses += used.get(f"{c.kind}:{short}", 0) + used.get(f"skill:{short}", 0) * (c.kind != "skill") \
                + used.get(f"skill:{plugin}", 0)
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
