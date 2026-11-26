"""Replay real Claude Code sessions through the lowtoken hook.

Each transcript is copied, line by line, into a fresh synthetic transcript.
For every tool result the hook runs first, exactly as Claude Code would call
it: the structured response Claude Code gave the hook (`toolUseResult`) and
the synthetic transcript so far. When the hook rewrites an output, the
rewritten text is what goes into the synthetic transcript, so later calls see
what Claude would have seen. Then it checks and measures:

  * every omitted run is identical to the source text its marker names
  * every line shown by a rewritten Read carries its true line number
  * tokens before/after, and the rent (cache write + later re-reads until the
    context was compacted) that the saving is worth at list prices

Prints aggregates only; no transcript content leaves the machine.

    python bench/replay.py [--days 30] [--limit N] [--exact-sample 300]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import tempfile
import time
from collections import Counter, defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from lowtoken import cost, dedup, hook, prices, rent  # noqa: E402

RAW: dict[tuple[str, int], list[str]] = {}
FAIL = Counter()
CHECKED = Counter()

_orig_add = dedup.Store.add
_orig_runs = dedup.Store.runs


def _add(self, label, lines):
    sid = _orig_add(self, label, lines)
    RAW[(self.path, sid)] = list(lines)
    return sid


def _strip(line: str) -> tuple[int, str]:
    n = dedup.key(line)[0]
    text = dedup._NUM.sub("", line, count=1) if n != -1 else line
    return n, text.rstrip("\r")


def _runs(self, lines):
    rs = _orig_runs(self, lines)
    for r in rs:
        src = RAW[(self.path, r.src)]
        for t in range(r.length):
            (n1, t1), (n2, t2) = _strip(lines[r.start + t]), _strip(src[r.src_start + t])
            CHECKED["omitted line"] += 1
            if t1 != t2 or (n1 != -1 and n1 != n2):
                FAIL["omitted line differs from its source"] += 1
    return rs


cost.precise()
dedup.Store.add = _add
dedup.Store.runs = _runs


def shown(name: str, resp) -> str:
    """The text Claude sees for a tool response (same rendering before and after)."""
    if name in ("Bash", "PowerShell") and isinstance(resp, dict):
        out = resp.get("stdout") or ""
        err = resp.get("stderr") or ""
        return out + ("\n" + err if err else "")
    if name == "Read" and isinstance(resp, dict) and isinstance(resp.get("file"), dict):
        f = resp["file"]
        s = int(f.get("startLine") or 1)
        return "\n".join(f"{s + i:>6}\t{x}" for i, x in enumerate((f.get("content") or "").split("\n")))
    if isinstance(resp, str):
        return resp
    if isinstance(resp, list):
        return "\n".join(b.get("text", "") for b in resp if isinstance(b, dict))
    return ""


def check_read(before: dict, after: dict) -> None:
    f0, f1 = before["file"], after["file"]
    orig = {int(f0.get("startLine") or 1) + i: x for i, x in enumerate(f0["content"].split("\n"))}
    s1 = int(f1["startLine"])
    for i, x in enumerate(f1["content"].split("\n")):
        if x.startswith("[lowtoken: "):
            continue
        CHECKED["read line number"] += 1
        if orig.get(s1 + i) != x:
            FAIL["read line shown under a wrong number"] += 1


def replay(path: str, root: str) -> list[dict]:
    sid = "replay-" + re.sub(r"\W", "_", path)[-80:]
    synth = os.path.join(root, sid + ".jsonl")
    store = dedup.Store(sid, None)
    store._blank()
    store.save()
    id2: dict[str, tuple[str, dict]] = {}
    calls, last_id, prev_ctx, epoch, model = 0, None, None, 0, None
    rows = []
    with open(path, encoding="utf-8", errors="replace") as fh, open(synth, "w", encoding="utf-8") as tw:
        def emit(obj):
            tw.write(json.dumps(obj) + "\n")
            tw.flush()

        for line in fh:
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if not isinstance(d, dict) or d.get("isSidechain"):
                continue
            t = d.get("type")
            m = d.get("message") or {}
            if t == "system" and d.get("subtype") == "compact_boundary":
                epoch += 1
            elif t == "assistant":
                for b in m.get("content") or []:
                    if isinstance(b, dict) and b.get("type") == "tool_use":
                        id2[b.get("id")] = (b.get("name", ""), b.get("input") or {})
                u = m.get("usage")
                if u and m.get("id") != last_id:
                    last_id = m.get("id")
                    model = model or m.get("model")
                    ctx = (u.get("input_tokens") or 0) + (u.get("cache_read_input_tokens") or 0) \
                        + (u.get("cache_creation_input_tokens") or 0)
                    if prev_ctx and ctx < prev_ctx * rent.SHRINK:
                        epoch += 1
                        emit({"type": "system", "subtype": "compact_boundary"})
                    prev_ctx = ctx
                    calls += 1
            if not (t == "user" and "toolUseResult" in d):
                emit(d)
                continue
            c = m.get("content")
            blk = next((b for b in c if isinstance(b, dict) and b.get("type") == "tool_result"), None) \
                if isinstance(c, list) else None
            name, inp = id2.get(blk.get("tool_use_id") if blk else None, ("", {}))
            resp = d["toolUseResult"]
            if blk is None or not (name in ("Bash", "PowerShell", "Read") or name.startswith("mcp__")):
                emit(d)
                continue
            t0 = time.perf_counter()
            out = hook.post({"session_id": sid, "tool_name": name, "tool_input": inp, "tool_response": resp,
                             "transcript_path": synth, "tool_use_id": blk.get("tool_use_id")})
            ms = 1000 * (time.perf_counter() - t0)
            new = resp
            if out:
                hs = out["hookSpecificOutput"]
                new = hs.get("updatedToolOutput", hs.get("updatedMCPToolOutput"))
                if name == "Read":
                    check_read(resp, new)
            a, b = shown(name, resp), shown(name, new)
            if out:
                blk["content"] = b   # what Claude would have seen
            emit(d)
            fam = "mcp" if name.startswith("mcp__") else name
            rows.append({"tool": fam, "before": cost.estimate(a), "after": cost.estimate(b), "ms": ms,
                         "call": calls, "epoch": epoch, "a": a if out else None, "b": b if out else None})
    os.remove(synth)
    ends = defaultdict(int)
    for r in rows:
        ends[r["epoch"]] = max(ends[r["epoch"]], r["call"])
    for r in rows:   # reads: calls from the next one to the end of the item's epoch
        r["reads"] = max(1, calls - r["call"]) if r["epoch"] == epoch else max(1, ends[r["epoch"]] - r["call"] + 1)
        r["model"] = model
    return rows


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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=float, default=30)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--exact-sample", type=int, default=0, help="re-measure N rewritten outputs with the exact oracle")
    ap.add_argument("--exclude", action="append", help="skip transcripts whose path contains this text")
    a = ap.parse_args()
    paths = rent.transcripts(days=a.days or None, exclude=a.exclude)
    random.Random(0).shuffle(paths)
    if a.limit:
        paths = paths[:a.limit]
    root = tempfile.mkdtemp(prefix="lowtoken-replay-")
    rows = []
    t = time.time()
    for i, p in enumerate(paths):
        rows += replay(p, root)
        print(f"\r{i + 1}/{len(paths)} transcripts, {len(rows)} tool results", end="", file=sys.stderr)
    print(file=sys.stderr)
    agg = defaultdict(Counter)
    for r in rows:
        i, _, rd = prices.price(r["model"])
        for k in (r["tool"], "ALL"):
            g = agg[k]
            g["n"] += 1
            g["changed"] += r["before"] != r["after"]
            g["before"] += r["before"]
            g["after"] += r["after"]
            g["rent_before"] += r["before"] * (2 * i + rd * (r["reads"] - 1))
            g["rent_after"] += r["after"] * (2 * i + rd * (r["reads"] - 1))
    print(f"replayed {len(paths)} transcripts, {len(rows)} tool results in {time.time() - t:.0f}s")
    print(f"{'tool':<12}{'results':>9}{'rewritten':>11}{'tokens before':>15}{'after':>12}{'saved':>8}{'rent saved':>12}")
    for k, g in sorted(agg.items(), key=lambda x: -x[1]["before"]):
        sv = 100 * (1 - g["after"] / g["before"]) if g["before"] else 0
        rs = g["rent_before"] - g["rent_after"]
        print(f"{k:<12}{g['n']:>9}{g['changed']:>11}{g['before']:>15,.0f}{g['after']:>12,.0f}{sv:>7.1f}%"
              f"{rs:>11,.2f}$")
    ms = sorted(r["ms"] for r in rows)
    if ms:
        print(f"hook time in-process: median {ms[len(ms) // 2]:.2f} ms, p99 {ms[int(len(ms) * .99)]:.1f} ms")
    print("checks:", dict(CHECKED), "failures:", dict(FAIL) or 0)
    if a.exact_sample:
        from lowtoken import oracle
        ch = [r for r in rows if r["a"] is not None]
        random.Random(1).shuffle(ch)
        ch = ch[:a.exact_sample]
        parts, owners = [], []
        for j, r in enumerate(ch):
            for side in ("a", "b"):
                for c in _chunks(r[side], 1800):
                    parts.append(c)
                    owners.append((j, side))
        counts = []
        for k in range(0, len(parts), 400):
            counts += oracle.count(parts[k:k + 400])
        tot = defaultdict(Counter)
        for (j, side), c in zip(owners, counts):
            tot[j][side] += c or 0
        A = sum(v["a"] for v in tot.values())
        B = sum(v["b"] for v in tot.values())
        ea = sum(r["before"] for r in ch)
        eb = sum(r["after"] for r in ch)
        print(f"exact Claude count on {len(ch)} rewritten outputs: {A:,} -> {B:,} tokens "
              f"(-{100 * (1 - B / A):.1f}%); estimator said -{100 * (1 - eb / ea):.1f}%")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
