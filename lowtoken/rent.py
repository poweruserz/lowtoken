"""Context rent: what each thing put into an agent's context really costs.

An LLM agent resends its whole context on every call, so a token added at
call k is paid once to write it to the prompt cache and again on every later
call that reads it. That repeated read is the rent. This module rebuilds, from
Claude Code transcripts, every call's exact usage and every item that entered
the context between calls, and charges each item its rent.

Exact: per-call usage and the bill (from the API's own `usage` blocks), model
output (output_tokens), and the size of everything appended between two calls
(the change in context size). Estimated: how that appended size splits
between tool results, prompts and injected text (cost.estimate, ~6-9% error
per item, the split is then rescaled to the exact total).
"""
from __future__ import annotations

import glob
import json
import math
import os
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from . import cost, prices

SHRINK = 0.85  # a call whose context is below 85% of the previous one dropped content


@dataclass
class Item:
    kind: str          # startup | thinking | output | tool | injected | user
    label: str         # e.g. "Bash: npm test", "Read: app.py", "hook: UserPromptSubmit"
    tokens: float
    first: int         # index of the first call that reads it
    reads: int = 0     # calls that read it, including the first (filled by _settle)
    usd: float = 0.0
    detail: str = ""
    gen: float = 0.0   # tokens generated (billed at the output price), for model output


@dataclass
class Session:
    path: str
    model: str | None = None
    calls: list[dict] = field(default_factory=list)   # {ctx, usage, model, ts, bill}
    items: list[Item] = field(default_factory=list)
    epoch_end: list[int] = field(default_factory=list)  # call index after which context was dropped
    rebuild_usd: float = 0.0
    rebuild_tokens: int = 0
    rebuild_idle: int = 0   # rebuilds after an idle gap longer than the cache TTL
    rebuild_reasons: dict = field(default_factory=dict)  # cause -> tokens re-written
    findings: list[dict] = field(default_factory=list)

    @property
    def bill(self) -> float:
        return sum(c["bill"] for c in self.calls)


def _text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(x.get("text", "") for x in content if isinstance(x, dict) and x.get("type") == "text")
    return ""


_CD = re.compile(r"^\s*(cd\s+[^;&|]+(&&|;)\s*|[A-Z_]+=\S*\s+)+")


def _label(name: str, inp: dict) -> tuple[str, str]:
    if name in ("Bash", "PowerShell"):
        cmd = _CD.sub("", (inp.get("command") or "").strip())
        words = cmd.split()
        fam = " ".join(words[:2]) if words and words[0] in (
            "git", "npm", "npx", "pnpm", "yarn", "bun", "cargo", "go", "gh", "docker", "uv", "pip",
            "python", "python3", "node", "dotnet", "kubectl", "make") else (words[0] if words else "?")
        return f"{name}: {fam}", cmd[:120]
    if name in ("Read", "Write", "Edit"):
        p = inp.get("file_path") or ""
        return f"{name}: {os.path.basename(p)}", p
    if name in ("Grep", "Glob"):
        return name, str(inp.get("pattern", ""))[:120]
    if name.startswith("mcp__"):
        parts = name.split("__")
        return f"mcp: {parts[1]}", parts[-1]
    if name == "Agent":
        return f"Agent: {inp.get('subagent_type') or 'general'}", str(inp.get("description", ""))[:120]
    return name, ""


def _att_visible(a: dict) -> tuple[str, str] | None:
    """(label, model-visible text) of a transcript attachment, or None."""
    t = a.get("type", "")
    if t.startswith("hook_"):
        c = a.get("content")
        c = c if isinstance(c, str) else json.dumps(c) if c else ""
        if not c.strip():
            return None
        ev = str(a.get("hookEvent") or a.get("hookName") or "").split(":")[0]
        return f"hook: {ev}", c
    if t in ("total_tokens_reminder", "environment", "permission_mode", "plan_mode"):
        return None
    body = a.get("content") if "content" in a else {k: v for k, v in a.items() if k != "type"}
    return t, body if isinstance(body, str) else json.dumps(body, ensure_ascii=False)


def parse(path: str) -> Session:
    s = Session(path)
    seen: dict[str, int] = {}   # message id -> call index
    last_out = 0
    last_think = 0
    pend_tool: list[Item] = []
    pend_user = 0.0
    pend_att: dict[str, list] = defaultdict(lambda: [0.0, ""])
    id2: dict[str, tuple[str, str, str]] = {}
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if '"isSidechain":true' in line.replace(" ", ""):
                continue
            try:
                d = json.loads(line)
            except ValueError:
                continue
            t = d.get("type")
            m = d.get("message") or {}
            if t == "system" and d.get("subtype") == "compact_boundary" and s.calls:
                s.epoch_end.append(len(s.calls) - 1)
            elif t == "assistant":
                for b in m.get("content") or []:
                    if isinstance(b, dict) and b.get("type") == "tool_use":
                        lab, det = _label(b.get("name", "?"), b.get("input") or {})
                        id2[b.get("id")] = (b.get("name", "?"), lab, det)
                u = m.get("usage")
                mid = m.get("id")
                if not u or m.get("model") == "<synthetic>":
                    continue
                if mid in seen:
                    # Claude Code logs one line per content block; early lines carry partial
                    # output_tokens (24% of output was missed by keeping the first), the last is final.
                    c = s.calls[seen[mid]]
                    c["usage"], c["bill"] = u, prices.bill(c["model"], u)
                    if seen[mid] == len(s.calls) - 1:
                        last_out = u.get("output_tokens") or 0
                        last_think = ((u.get("output_tokens_details") or {}).get("thinking_tokens") or 0)
                    continue
                seen[mid] = len(s.calls)
                ctx = (u.get("input_tokens") or 0) + (u.get("cache_read_input_tokens") or 0) \
                    + (u.get("cache_creation_input_tokens") or 0)
                model = m.get("model")
                s.model = s.model or model
                call = {"ctx": ctx, "usage": u, "model": model, "ts": d.get("timestamp"), "id": mid,
                        "bill": prices.bill(model, u), "miss": (m.get("diagnostics") or {}).get("cache_miss_reason")}
                k = len(s.calls)
                if k == 0 or ctx < s.calls[-1]["ctx"] * SHRINK:
                    if k:
                        s.epoch_end.append(k - 1)
                    s.items.append(Item("startup", "startup context" if k == 0 else "context after compaction",
                                        ctx, k))
                else:
                    delta = ctx - s.calls[-1]["ctx"]
                    _split(s, k, delta, last_out, last_think, pend_tool, pend_user, pend_att)
                s.calls.append(call)
                last_out = u.get("output_tokens") or 0
                last_think = ((u.get("output_tokens_details") or {}).get("thinking_tokens") or 0)
                pend_tool, pend_user, pend_att = [], 0.0, defaultdict(lambda: [0.0, ""])
            elif t == "user":
                c = m.get("content")
                if isinstance(c, list):
                    for b in c:
                        if not isinstance(b, dict):
                            continue
                        if b.get("type") == "tool_result":
                            name, lab, det = id2.get(b.get("tool_use_id"), ("?", "?", ""))
                            txt = _text(b.get("content"))
                            pend_tool.append(Item("tool", lab, cost.estimate(txt), 0, detail=det))
                        elif b.get("type") == "text":
                            pend_user += cost.estimate(b.get("text", ""))
                elif isinstance(c, str):
                    pend_user += cost.estimate(c)
            elif t == "attachment":
                v = _att_visible(d.get("attachment") or {})
                if v:
                    slot = pend_att[v[0]]
                    slot[0] += cost.estimate(v[1])
                    if len(slot[1]) < 1600:  # a sample of every injection under this label, for the detectors
                        slot[1] += v[1][:400] + "\n"
    _settle(s)
    _rebuilds(s)
    return s


def _split(s: Session, k: int, delta: int, out: int, think: int, tools: list[Item],
           user: float, att: dict) -> None:
    """Split the exact growth `delta` before call k into items."""
    visible = max(0, out - think)
    kept = min(out, max(0, delta))          # output the harness kept in context
    kept_visible = min(visible, kept)
    kept_think = kept - kept_visible
    rest = delta - kept
    est = sum(i.tokens for i in tools) + user
    scale = min(1.0, rest / est) if est > 0 else 1.0
    if think:
        s.items.append(Item("thinking", "thinking", kept_think, k, gen=think))
    if visible:
        s.items.append(Item("output", "model output", kept_visible, k, gen=visible))
    for i in tools:
        i.tokens *= scale
        i.first = k
        s.items.append(i)
    if user:
        s.items.append(Item("user", "user prompt", user * scale, k))
    injected = max(0.0, rest - est * scale)
    if injected > 0:
        w = sum(v[0] for v in att.values())
        if w > 0:
            for lab, (tok, sample) in att.items():
                s.items.append(Item("injected", lab, injected * tok / w, k, detail=sample))
        else:
            s.items.append(Item("injected", "harness (system reminders)", injected, k))


def _settle(s: Session) -> None:
    """Count reads per item and charge rent: one cache write, then cache reads."""
    n = len(s.calls)
    ends = sorted(set(s.epoch_end)) + [n - 1]
    for it in s.items:
        end = next(e for e in ends if e >= it.first) if it.first < n else n - 1
        it.reads = max(0, end - it.first + 1)
        if it.first >= n:
            continue
        model = s.calls[it.first]["model"]
        i, o, r = prices.price(model)
        it.usd = it.tokens * (i * prices.write_multiplier(s.calls[it.first]["usage"])
                              + r * max(0, it.reads - 1)) + it.gen * o


def _ts(x: str | None) -> float | None:
    if not x:
        return None
    try:
        return time.mktime(time.strptime(x[:19], "%Y-%m-%dT%H:%M:%S"))
    except ValueError:
        return None


def _rebuilds(s: Session) -> None:
    """Calls that rewrote more cache than was appended: the cached prefix was lost."""
    for k in range(1, len(s.calls)):
        c, p = s.calls[k], s.calls[k - 1]
        u = c["usage"]
        wrote = u.get("cache_creation_input_tokens") or 0
        grew = max(0, c["ctx"] - p["ctx"])
        lost = wrote - grew - 2000
        if lost > 0 and c["ctx"] >= p["ctx"] * SHRINK:
            i, _, r = prices.price(c["model"])
            s.rebuild_usd += lost * (i * prices.write_multiplier(u) - r)
            s.rebuild_tokens += lost
            # the API names the cause when it can; otherwise an idle gap past the previous write's lifetime
            reason = (c.get("miss") or {}).get("type")
            a, b = _ts(p["ts"]), _ts(c["ts"])
            ttl = 3600 if prices.write_multiplier(p["usage"]) > 1.6 else 300
            if not reason or reason in ("previous_message_not_found", "unavailable"):
                reason = "idle longer than the cache lifetime" if a and b and b - a > ttl else "unknown"
            if reason.startswith("idle"):
                s.rebuild_idle += 1
            s.rebuild_reasons[reason] = s.rebuild_reasons.get(reason, 0) + lost


# ------------------------------------------------------------------ corpus ---

def transcripts(root: str | None = None, days: float | None = 30, project: str | None = None,
                exclude: list[str] | None = None) -> list[str]:
    root = root or os.path.join(os.path.expanduser("~"), ".claude", "projects")
    cutoff = time.time() - days * 86400 if days else 0
    out = []
    for p in glob.glob(os.path.join(root, "**", "*.jsonl"), recursive=True):
        if project and project.lower() not in p.lower():
            continue
        if exclude and any(x.lower() in p.lower() for x in exclude):
            continue
        try:
            if os.path.getmtime(p) >= cutoff:
                out.append(p)
        except OSError:
            pass
    return out


def lindy(sessions: list[Session], ks=(1, 5, 10, 20, 50, 100, 200)) -> list[tuple[int, float, float]]:
    """(k, P(session reaches call k), mean calls still to come after call k)."""
    lens = [len(s.calls) for s in sessions if s.calls]
    rows = []
    for k in ks:
        rem = [n - k for n in lens if n > k]
        if rem:
            rows.append((k, len(rem) / len(lens), sum(rem) / len(rem)))
    return rows


_ECHO = re.compile(r'"(hook_event_name|session_id|transcript_path)"\s*:')


def keepwarm(sessions: list[Session], max_pings: int = 60) -> dict | None:
    """Would keeping the prompt cache warm through idle gaps pay, and for how long? (ski rental)

    A keeper re-sends the last request every TTL - 60 s, at most H times per gap; each re-send costs
    one cache read of the prefix. It pays only where the next call really re-wrote the expired prefix
    (the usage shows it and the API names no other cause); session ends and gaps after /clear,
    compaction or a model switch only cost pings. H is chosen on half of the gaps and scored on the
    other half, so the saving is not fitted to the same data it is reported on."""
    gaps = []   # (gap s, ttl, ping $, penalty $, really expired)
    for s in sessions:
        if os.sep + "subagents" + os.sep in s.path or "/subagents/" in s.path:
            continue   # a subagent's context is never resumed
        calls = [c for c in s.calls if c.get("ts")]
        one_h = any((c["usage"].get("cache_creation") or {}).get("ephemeral_1h_input_tokens") for c in calls)
        ttl, w = (3600, 2.0) if one_h else (300, 1.25)
        for k, c in enumerate(calls):
            i, _, r = prices.price(c["model"])
            g, expired = math.inf, False
            if k + 1 < len(calls):
                n = calls[k + 1]
                a, b = _ts(c["ts"]), _ts(n["ts"])
                if not (a and b):
                    continue
                g = b - a
                lost = (n["usage"].get("cache_creation_input_tokens") or 0) - max(0, n["ctx"] - c["ctx"])
                expired = lost > 0.5 * c["ctx"] and (n.get("miss") or {}).get("type") in (
                    None, "previous_message_not_found", "unavailable")
            if g > ttl - 60:
                gaps.append((g, ttl, c["ctx"] * r, c["ctx"] * (i * w - r), expired))
    if not any(x[4] for x in gaps):
        return None

    def cost(h: int, rows) -> float:
        total = 0.0
        for g, ttl, ping, penalty, expired in rows:
            t = ttl - 60
            sent = h if g == math.inf else min(h, int(g // t))
            warm = g != math.inf and (sent == int(g // t) or g <= sent * t + ttl)
            total += sent * ping + (penalty if expired and not warm else 0.0)
        return total

    hs = range(max_pings + 1)
    a_half, b_half = gaps[0::2], gaps[1::2]
    h_a = min(hs, key=lambda h: cost(h, a_half))
    h_b = min(hs, key=lambda h: cost(h, b_half))
    held_out = (cost(0, b_half) - cost(h_a, b_half)) + (cost(0, a_half) - cost(h_b, a_half))
    best = min(hs, key=lambda h: cost(h, gaps))
    base = cost(0, gaps)
    return {"base": base, "best": best, "best_saves": base - cost(best, gaps), "held_out_saves": held_out,
            "table": [(h, base - cost(h, gaps)) for h in (1, 2, 4, 8, 20, 40) if h <= max_pings],
            "ttl_1h_share": sum(x[1] == 3600 for x in gaps) / len(gaps),
            "expired": sum(x[4] for x in gaps), "ends": sum(x[0] == math.inf for x in gaps)}


def findings(sessions: list[Session]) -> list[dict]:
    """Concrete waste, each with the USD it cost in this period."""
    out = []
    echo = Counter()
    echo_n = Counter()
    shell = set()   # labels whose output is an interactive cmd.exe session
    big = []
    for s in sessions:
        for it in s.items:
            if it.kind == "injected" and it.label.startswith("hook:") and _ECHO.search(it.detail or ""):
                echo[it.label] += it.usd
                echo_n[it.label] += 1
                if "Microsoft Windows [Version" in (it.detail or ""):
                    shell.add(it.label)
            if it.kind == "tool" and it.tokens >= 8000:
                big.append(it)
    for lab, usd in echo.items():
        if lab in shell:
            # Reproduced: under Git Bash, `cmd.exe /d /c x` gets /d and /c rewritten into paths by MSYS,
            # so cmd starts interactively, reads the hook JSON from stdin as a command and echoes it.
            out.append({"id": "interactive-cmd-hook", "usd": usd,
                        "title": f"{lab} hook starts an interactive cmd.exe that echoes every prompt",
                        "detail": f"{echo_n[lab]} injections of the cmd banner plus the full hook JSON. Claude Code "
                                  f"runs hooks through Git Bash on Windows, which turns `/c` into a path, so the "
                                  f"command after it never runs either.",
                        "fix": "Hooks here run through Git Bash: write `cmd.exe //d //c ...`, or drop the cmd.exe wrapper."})
            continue
        out.append({"id": "echoing-hook", "usd": usd,
                    "title": f"{lab} hook prints its own JSON input back into the context",
                    "detail": f"{echo_n[lab]} injections. A hook that echoes its stdin (often a .cmd/.bat "
                              f"without '@echo off', or a script that prints the payload) adds the whole prompt "
                              f"again on every turn.",
                    "fix": "Silence the hook's stdout, or print only what Claude needs."})
    reb = sum(s.rebuild_usd for s in sessions)
    if reb > 0:
        why = Counter()
        for s in sessions:
            why.update(s.rebuild_reasons)
        causes = ", ".join(f"{k.replace('_', ' ')} {v / 1e6:.1f}M" for k, v in why.most_common(5))
        out.append({"id": "cache-rebuilds", "usd": reb,
                    "title": "Prompt cache rebuilt mid-session",
                    "detail": f"{sum(s.rebuild_tokens for s in sessions):,} tokens re-written to cache. "
                              f"Causes (from the API's own diagnostics where present): {causes}.",
                    "fix": "Pick the model before you start (a /model switch re-writes the whole context); "
                           "finish or /compact before long breaks; avoid changing CLAUDE.md, MCP servers or "
                           "settings mid-session."})
    if big:
        usd = sum(i.usd for i in big)
        out.append({"id": "huge-tool-results", "usd": usd,
                    "title": f"{len(big)} tool results over 8k tokens",
                    "detail": "Largest: " + "; ".join(f"{i.label} ({i.tokens:,.0f} tok, {i.reads} reads)"
                                                     for i in sorted(big, key=lambda x: -x.usd)[:3]),
                    "fix": "Filter at the source (head, grep, --quiet, jq) or let `lowtoken hook` fold them."})
    return sorted(out, key=lambda f: -f["usd"])


def summarize(sessions: list[Session]) -> dict:
    by_kind = Counter()
    by_label = Counter()
    tok_kind = Counter()
    bill = 0.0
    calls = 0
    counted: set = set()   # a resumed or forked session repeats earlier responses in a new file
    unknown = Counter()
    for s in sessions:
        for c in s.calls:
            if c.get("id") in counted:
                continue
            counted.add(c.get("id"))
            bill += c["bill"]
            calls += 1
            if not prices.known(c["model"]):
                unknown[c["model"]] += 1
        for it in s.items:
            by_kind[it.kind] += it.usd
            tok_kind[it.kind] += it.tokens
            if it.kind in ("tool", "injected"):
                by_label[it.label] += it.usd
        by_kind["cache rebuilds"] += s.rebuild_usd
    items = sorted((it for s in sessions for it in s.items if it.kind == "tool"), key=lambda i: -i.usd)[:10]
    return {"sessions": len(sessions), "calls": calls, "bill": bill, "by_kind": by_kind, "unpriced": unknown,
            "tokens_by_kind": tok_kind, "by_label": by_label, "top_items": items,
            "lindy": lindy(sessions), "keepwarm": keepwarm(sessions), "findings": findings(sessions)}
