"""lowtoken command line."""
from __future__ import annotations

import argparse
import json
import sys
import time

from . import __version__

KIND_NAMES = {
    "startup": "Startup context (system prompt, tools, CLAUDE.md, skills)",
    "thinking": "Model thinking (kept in context)",
    "output": "Model output (text, tool calls, file writes)",
    "tool": "Tool results",
    "injected": "Injected text (hooks, reminders, skill bodies)",
    "user": "Your prompts",
    "cache rebuilds": "Cache rebuilds (prefix lost, re-written)",
}


def _bar(frac: float, width: int = 24) -> str:
    n = round(frac * width)
    return "#" * n + "." * (width - n)


def _usd(x: float) -> str:
    return f"${x:,.2f}" if x < 1000 else f"${x:,.0f}"


def cmd_audit(a) -> int:
    from . import rent
    paths = rent.transcripts(a.root, a.days, a.project, a.exclude)
    if not paths:
        print("No Claude Code transcripts found for that period.")
        return 1
    t = time.time()
    sessions = []
    for i, p in enumerate(paths):
        try:
            s = rent.parse(p)
        except OSError:
            continue
        if s.calls:
            sessions.append(s)
        if sys.stderr.isatty() and i % 25 == 0:
            print(f"\rreading transcripts {i + 1}/{len(paths)}", end="", file=sys.stderr)
    if sys.stderr.isatty():
        print("\r" + " " * 40 + "\r", end="", file=sys.stderr)
    r = rent.summarize(sessions)
    if a.json:
        out = dict(r)
        out["top_items"] = [vars(i) for i in r["top_items"]]
        print(json.dumps(out, default=str, indent=1))
        return 0
    total = sum(r["by_kind"].values()) or 1
    period = f"last {a.days:g} days" if a.days else "all time"
    print(f"lowtoken audit · {period} · {r['sessions']} sessions · {r['calls']:,} API calls "
          f"· parsed in {time.time() - t:.0f}s")
    print(f"Billed at API list prices: {_usd(r['bill'])}  (attributed below: {_usd(total)})")
    if r["unpriced"]:
        from .prices import DEFAULT
        print(f"  priced as {DEFAULT} (unknown model): "
              + ", ".join(f"{m} x{n}" for m, n in r["unpriced"].most_common()))
    print()
    print("WHERE THE MONEY GOES  (generation + cache write + every later re-read)")
    for k, v in sorted(r["by_kind"].items(), key=lambda x: -x[1]):
        print(f"  {_bar(v / total)} {100 * v / total:5.1f}%  {_usd(v):>10}  {KIND_NAMES.get(k, k)}")
    print("\nTOP TOOL AND INJECTION SOURCES BY RENT")
    for lab, v in r["by_label"].most_common(a.top):
        print(f"  {100 * v / total:5.1f}%  {_usd(v):>10}  {lab}")
    print("\nMOST EXPENSIVE SINGLE TOOL RESULTS")
    for it in r["top_items"][: a.top]:
        print(f"  {_usd(it.usd):>9}  {it.tokens:>8,.0f} tok x {it.reads:>4} reads  {it.label}  {it.detail[:60]}")
    if r["lindy"]:
        print("\nLINDY EFFECT  (the older the session, the longer a new token stays)")
        print("  a token added at call k is read again on average ...")
        for k, p, rem in r["lindy"]:
            print(f"    k={k:<4} {rem:7.0f} more calls   ({100 * p:.0f}% of sessions get this far)")
    if r["findings"]:
        print("\nFINDINGS")
        for f in r["findings"]:
            print(f"  [{_usd(f['usd'])}] {f['title']}\n      {f['detail']}\n      fix: {f['fix']}")
    return 0


def cmd_diet(a) -> int:
    from . import diet, rent
    md = diet.context_report(a.cwd)
    comps, cats = diet.components(md)
    paths = rent.transcripts(a.root, a.days, None)
    used = diet.usage(paths)
    calls = sessions = 0
    model = None
    for p in paths:  # count calls cheaply: one usage line per API call id
        ids = set()
        with open(p, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if '"usage"' in line and '"assistant"' in line:
                    try:
                        m = json.loads(line).get("message") or {}
                    except ValueError:
                        continue
                    ids.add(m.get("id"))
                    model = model or m.get("model")
        calls += len(ids)
        sessions += bool(ids)
    plan = diet.plan(comps, used, calls, sessions, model)
    loaded = sum(v for k, v in cats.items() if "deferred" not in k and k not in ("Free space", "Autocompact buffer"))
    print(f"lowtoken diet · startup context {loaded:,.0f} tokens on every call · "
          f"{calls:,} calls in {sessions} sessions over {a.days:g} days\n")
    for k, v in sorted(cats.items(), key=lambda x: -x[1]):
        if k not in ("Free space", "Autocompact buffer"):
            print(f"  {v:>9,.0f}  {k}")
    total = sum(g["usd"] for g in plan)
    print(f"\nNEVER USED IN {a.days:g} DAYS, LOADED ANYWAY  (cost of carrying it: {_usd(total)})")
    for g in plan[: a.top]:
        print(f"  {_usd(g['usd']):>9}  {g['unused_tokens']:>7,.0f} tok  {g['group']}"
              f"  ({g['items']} items, {g['uses']} uses in total)")
    print("\nDisable a plugin with /plugin, a claude.ai connector in claude.ai settings, an MCP server with /mcp.")
    return 0


def _last_usage(transcript: str, first: bool = False) -> tuple[int, str | None]:
    """Context size and model of the last (or first) API call, from the transcript's tail (or head)."""
    try:
        with open(transcript, "rb") as fh:
            if not first:
                fh.seek(0, 2)
                fh.seek(max(0, fh.tell() - 400_000))
            lines = fh.read(400_000).decode("utf-8", "replace").splitlines()
    except OSError:
        return 0, None
    for line in (lines if first else reversed(lines)):
        if '"usage"' not in line:
            continue
        try:
            m = json.loads(line).get("message") or {}
        except ValueError:
            continue
        u = m.get("usage")
        if u:
            return ((u.get("input_tokens") or 0) + (u.get("cache_read_input_tokens") or 0)
                    + (u.get("cache_creation_input_tokens") or 0)), m.get("model")
    return 0, None


SUMMARY = 10_000  # shortcut: assumed size of a compaction summary; measure it from compact_boundary events


def compact_payback(ctx: int, startup: int, model: str | None) -> int | None:
    """Calls after which compacting now has paid for itself, or None if it never would.

    Compacting reads the context once (ctx*r), writes a summary (H*o) and re-caches the new, smaller
    context (C'*w); from then on each call reads C' instead of ctx. Break-even n solves
    n*(ctx - C')*r = ctx*r + H*o + C'*w. It ignores what the summary forgets; that is the user's call.
    """
    from . import prices
    i, o, r = prices.price(model)
    after = startup + SUMMARY
    if ctx < 2 * after or not startup:
        return None
    once = ctx * r + SUMMARY * o + after * i * 2.0
    return max(1, round(once / ((ctx - after) * r)))


def cmd_statusline(a) -> int:
    """One status line: context size, what each call re-reads it for, and what lowtoken saved."""
    from . import dedup, prices
    try:
        d = json.loads(sys.stdin.buffer.read().decode("utf-8", "replace") or "{}")
    except ValueError:
        d = {}
    ctx, model = _last_usage(d.get("transcript_path") or "")
    model = model or (d.get("model") or {}).get("id")
    _, _, r = prices.price(model)
    parts = []
    if ctx:
        parts.append(f"ctx {ctx / 1000:.1f}k")
        parts.append(f"rent ${ctx * r:.3f}/call")
        parts.append(f"next 100 calls ${ctx * r * 100:.2f}")
        n = compact_payback(ctx, _last_usage(d.get("transcript_path") or "", first=True)[0], model)
        if n is not None:
            parts.append(f"/compact pays back in {n} calls")
    saved = dedup.Store(d.get("session_id") or "default").saved if d.get("session_id") else 0
    if saved:
        parts.append(f"lowtoken -{saved / 1000:.1f}k tok")
    print(" · ".join(parts) or "lowtoken")
    return 0


def cmd_count(a) -> int:
    from . import cost
    text = _read(a.file)
    if a.exact:
        from . import oracle
        parts = _chunks(text, 1800)
        n = oracle.count(parts, model=a.model)
        if any(x is None for x in n):
            print("could not read some counts from /context", file=sys.stderr)
            return 1
        print(sum(n))
    else:
        print(cost.estimate(text))
    return 0


def _chunks(text: str, size: int) -> list[str]:
    out, buf = [], ""
    for line in text.splitlines(keepends=True):
        if len(buf) + len(line) > size and buf:
            out.append(buf)
            buf = ""
        while len(line) > size:
            out.append(line[:size])
            line = line[size:]
        buf += line
    if buf:
        out.append(buf)
    return out


def cmd_compress(a) -> int:
    from . import codec
    r = codec.encode(_read(a.file))
    sys.stdout.write(r.text)
    if not r.text.endswith("\n"):
        sys.stdout.write("\n")
    pct = 100 * r.saved / r.original_tokens if r.original_tokens else 0
    print(f"[lowtoken] {r.original_tokens} -> {r.tokens} tokens (-{pct:.1f}%) via {'+'.join(r.steps) or 'nothing'}",
          file=sys.stderr)
    return 0


def cmd_hook(a) -> int:
    from . import hook
    return hook.main(a.event)


def _read(path: str | None) -> str:
    if not path or path == "-":
        return sys.stdin.buffer.read().decode("utf-8", errors="replace")
    with open(path, encoding="utf-8", errors="replace") as fh:
        return fh.read()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="lowtoken", description="Pay less rent on your LLM context.")
    p.add_argument("--version", action="version", version=f"lowtoken {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("audit", help="where your Claude Code tokens and dollars really go")
    s.add_argument("--days", type=float, default=30, help="look back this many days (0 = all)")
    s.add_argument("--project", help="only transcripts whose path contains this text")
    s.add_argument("--exclude", action="append", help="skip transcripts whose path contains this text (repeatable)")
    s.add_argument("--root", help="transcript folder (default ~/.claude/projects)")
    s.add_argument("--top", type=int, default=10)
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_audit)

    s = sub.add_parser("diet", help="startup context you load on every call but never use")
    s.add_argument("--days", type=float, default=30)
    s.add_argument("--cwd", help="project folder to measure (default: current)")
    s.add_argument("--root", help="transcript folder (default ~/.claude/projects)")
    s.add_argument("--top", type=int, default=15)
    s.set_defaults(fn=cmd_diet)

    s = sub.add_parser("count", help="Claude token count of a file or stdin")
    s.add_argument("file", nargs="?")
    s.add_argument("--exact", action="store_true", help="exact count via your Claude Code (free, ~20s)")
    s.add_argument("--model", help="model whose tokenizer to use with --exact")
    s.set_defaults(fn=cmd_count)

    s = sub.add_parser("compress", help="verified-lossless compression of tool output")
    s.add_argument("file", nargs="?")
    s.set_defaults(fn=cmd_compress)

    s = sub.add_parser("statusline", help="status line: context size, rent per call, tokens saved")
    s.set_defaults(fn=cmd_statusline)

    s = sub.add_parser("hook", help="Claude Code hook entry point (reads the hook JSON on stdin)")
    s.add_argument("event", choices=["post", "session"])
    s.set_defaults(fn=cmd_hook)

    a = p.parse_args(argv)
    if a.cmd not in ("hook", "statusline"):
        from . import cost
        cost.precise()
    if getattr(a, "days", None) == 0:
        a.days = None
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
