"""Claude Code hook entry points.

post     PostToolUse: shrink Bash/PowerShell, Read and MCP tool output before
         Claude sees it (`updatedToolOutput` / `updatedMCPToolOutput`).
session  SessionStart: forget the session's dedup store after /clear or a
         compaction, when earlier output is no longer in the context.

Deduplication only references text the session transcript shows Claude
actually received; without a transcript it is off and only the verified-
lossless codec runs. A hook must never break a tool call: on any error it
prints nothing and Claude gets the original output.
"""
from __future__ import annotations

import json
import os
import sys

from . import codec, cost, dedup

MIN_SAVED = 30  # rewrite only when it saves at least this many tokens


def _log(entry: dict) -> None:
    path = os.environ.get("LOWTOKEN_LOG")
    if path:
        try:
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry) + "\n")
        except OSError:
            pass  # logging must never turn a good rewrite into a failed hook


def _fp(text: str) -> str:
    return dedup.blake2b(text.encode("utf-8", "replace"), digest_size=8).hexdigest()


def _shrink_text(store: dedup.Store | None, text: str, terminal: bool = True) -> tuple[str, int]:
    """render (terminal output only) -> dedup against what Claude has seen -> verified-lossless codec."""
    rendered = codec.render(text) if terminal else text
    lines = rendered.split("\n")
    body = rendered
    if store is not None:
        fp = _fp(rendered)
        if fp not in store.elided:  # Claude asking for the same output again gets it whole
            kept, _ = dedup.apply(store, lines)
            if len(kept) != len(lines):
                store.elided.append(fp)
                body = "\n".join(kept)
    total = len(lines) - (1 if lines and lines[-1] == "" else 0)  # what a fold header must state
    r = codec.encode(body, terminal=False, lines_total=total)
    return r.text, cost.estimate(text) - r.tokens


def _bash(store, resp: dict) -> dict | None:
    out = dict(resp)
    saved = 0
    for key in ("stdout", "stderr"):
        v = resp.get(key)
        if isinstance(v, str) and len(v) > 200:
            new, s = _shrink_text(store, v)
            if s > 0:
                out[key] = new
                saved += s
    return out if saved >= MIN_SAVED else None


def _read(store, resp: dict) -> dict | None:
    f = resp.get("file")
    if store is None or resp.get("type") != "text" or not isinstance(f, dict) \
            or not isinstance(f.get("content"), str):
        return None
    content = f["content"]
    start = int(f.get("startLine") or 1)
    lines = content.split("\n")
    fp = _fp(f"{f.get('filePath')}\0{start}\0{content}")
    if fp in store.elided:
        return None  # Claude read the same thing again after we trimmed it: give it whole
    new, new_start = dedup.read_markers(store, f.get("filePath") or "", start, lines)
    if new == lines:
        return None
    store.elided.append(fp)
    body = "\n".join(new)
    if cost.estimate(content) - cost.estimate(body) < MIN_SAVED:
        return None
    return dict(resp, file=dict(f, content=body, startLine=new_start, numLines=len(new)))


def _mcp(store, resp) -> object | None:
    if isinstance(resp, str):
        new, s = _shrink_text(store, resp, terminal=False)
        return new if s >= MIN_SAVED else None
    if isinstance(resp, list) and all(isinstance(b, dict) for b in resp):
        out, saved = [], 0
        for b in resp:
            if b.get("type") == "text" and isinstance(b.get("text"), str) and len(b["text"]) > 200:
                new, s = _shrink_text(store, b["text"], terminal=False)
                if s > 0:
                    out.append(dict(b, text=new))
                    saved += s
                    continue
            out.append(b)
        return out if saved >= MIN_SAVED else None
    return None


def _text(name: str, resp) -> str:
    """Roughly the text Claude reads for a tool response, for the savings counter."""
    if isinstance(resp, dict) and isinstance(resp.get("file"), dict):
        return resp["file"].get("content") or ""
    if isinstance(resp, dict):
        return (resp.get("stdout") or "") + (resp.get("stderr") or "")
    if isinstance(resp, list):
        return "\n".join(b.get("text", "") for b in resp if isinstance(b, dict))
    return str(resp or "")


def post(d: dict) -> dict | None:
    name = d.get("tool_name") or ""
    resp = d.get("tool_response")
    # A subagent reports its parent's session_id but has its own context: never dedup against
    # what the parent saw. Without the lock (a parallel call holds it) dedup is skipped too.
    store = dedup.Store(d.get("session_id") or "default", load=False)
    locked = not d.get("agent_id") and store.lock()
    try:
        if locked:
            store.load()
        s = store if locked and store.sync(d.get("transcript_path")) else None
        result = _rewrite(name, resp, s)
        if result and locked:
            new = next(iter(result.values()))
            store.saved += max(0, cost.estimate(_text(name, resp)) - cost.estimate(_text(name, new)))
        if locked:
            store.save()
    finally:
        store.unlock()
    if result:
        _log({"tool": name, "session": d.get("session_id")})
        return {"hookSpecificOutput": {"hookEventName": "PostToolUse", **result}}
    return None


def _rewrite(name: str, resp, s) -> dict | None:
    result = None
    if name in ("Bash", "PowerShell") and isinstance(resp, dict):
        new = _bash(s, resp)
        result = new and {"updatedToolOutput": new}
    elif name == "Read" and isinstance(resp, dict):
        new = _read(s, resp)
        result = new and {"updatedToolOutput": new}
    elif name == "Grep" and isinstance(resp, dict) and resp.get("mode") == "content" \
            and isinstance(resp.get("content"), str) and len(resp["content"]) > 200:
        r = codec.encode(resp["content"], json_=False, terminal=False)  # file text, grouped by path
        if r.saved >= MIN_SAVED:
            result = {"updatedToolOutput": dict(resp, content=r.text)}
    elif name.startswith("mcp__"):
        new = _mcp(s, resp)
        result = {"updatedMCPToolOutput": new} if new is not None else None
    return result


def session(d: dict) -> None:
    if d.get("source") in ("clear", "compact") and not d.get("agent_id"):
        store = dedup.Store(d.get("session_id") or "default", load=False)
        if store.lock(wait=5.0):
            try:
                store.load()
                store.sync(d.get("transcript_path"))  # move past what was dropped
                store.reset()
            finally:
                store.unlock()


def main(event: str) -> int:
    try:
        d = json.loads(sys.stdin.buffer.read().decode("utf-8", "replace"))
        if event == "post":
            out = post(d)
            if out:
                sys.stdout.write(json.dumps(out))
        elif event == "session":
            session(d)
    except Exception as e:  # noqa: BLE001 - a hook must never break the tool call
        _log({"error": repr(e)})
    return 0
