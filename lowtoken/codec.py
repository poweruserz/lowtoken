"""Verified-lossless compression of tool output for LLM context.

Every transform here has an exact decoder. `encode` runs each transform,
decodes the result and compares it with the input; a transform whose round
trip is not byte-identical is dropped. So the output is either provably
decodable back to the input or it is the input.

The one exception is `render`, which turns raw terminal bytes into what a
terminal would show (ANSI colours, carriage-return progress bars, backspaces).
That step is applied first and everything after it round-trips to its output.
"""
from __future__ import annotations

import json
import re
from typing import Callable

from . import cost as _cost

# ---------------------------------------------------------------- render ----

_ANSI = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07\x1b]*(?:\x07|\x1b\\)|[@-Z\\-_])")


def render(text: str) -> str:
    """What a terminal would display: strip ANSI, apply \\r overwrites and \\b."""
    if "\x1b" in text:
        text = _ANSI.sub("", text)
    if "\r" not in text and "\b" not in text:
        return text
    out = []
    for line in text.replace("\r\n", "\n").split("\n"):
        if "\r" in line or "\b" in line:
            buf: list[str] = []
            col = 0
            for ch in line:
                if ch == "\r":
                    col = 0
                elif ch == "\b":
                    col = max(0, col - 1)
                else:
                    if col < len(buf):
                        buf[col] = ch
                    else:
                        buf.append(ch)
                    col += 1
            line = "".join(buf).rstrip()
        out.append(line)
    return "\n".join(out)


# ------------------------------------------------------------- run folds ----
# A run of identical lines, or a block of lines repeated back to back, becomes
# one copy plus a marker line. The marker grammar must not occur in the input.

_RUN = "... (line above repeated {n} more times)"
_BLOCK = "... (previous {p} lines repeated {n} more times)"
_RUN_RE = re.compile(r"^\.\.\. \(line above repeated (\d+) more times\)$")
_BLOCK_RE = re.compile(r"^\.\.\. \(previous (\d+) lines repeated (\d+) more times\)$")
# The header states the true line count: measured on Haiku, without it the
# model sometimes counted the folded lines instead of the printed ones.
_HEAD = "[lowtoken: {n} lines; repeated lines folded]"
_HEAD_RE_RUNS = re.compile(r"^\[lowtoken: \d+ lines; repeated lines folded\]$")
MAX_PERIOD = 12


def fold_runs(lines: list[str], printed: int | None = None) -> list[str]:
    out: list[str] = []
    i, n = 0, len(lines)
    while i < n:
        best_p, best_k = 0, 0
        for p in range(1, min(MAX_PERIOD, (n - i) // 2) + 1):
            k = 0
            while i + (k + 2) * p <= n and lines[i + (k + 1) * p:i + (k + 2) * p] == lines[i:i + p]:
                k += 1
            # lines saved = k*p - 1 marker; prefer the period that saves most
            if k and k * p - 1 > best_k * best_p - 1:
                best_p, best_k = p, k
        if best_k and best_k * best_p > 1:
            out.extend(lines[i:i + best_p])
            out.append(_RUN.format(n=best_k) if best_p == 1 else _BLOCK.format(p=best_p, n=best_k))
            i += (best_k + 1) * best_p
        else:
            out.append(lines[i])
            i += 1
    if len(out) == n:
        return out
    if printed is None:
        printed = n - (1 if lines and lines[-1] == "" else 0)  # a final newline is not a line
    return [_HEAD.format(n=printed)] + out


def unfold_runs(lines: list[str]) -> list[str]:
    out: list[str] = []
    if lines and _HEAD_RE_RUNS.match(lines[0]):
        lines = lines[1:]
    for ln in lines:
        m = _RUN_RE.match(ln)
        if m and out:
            out.extend([out[-1]] * int(m.group(1)))
            continue
        m = _BLOCK_RE.match(ln)
        if m and 0 < int(m.group(1)) <= len(out):
            p, k = int(m.group(1)), int(m.group(2))
            out.extend(out[-p:] * k)
            continue
        out.append(ln)  # not a marker we could have written: keep it as text
    return out


# ------------------------------------------------------------ path groups ---
# grep -n / compiler / linter output: "path:12:text" lines sharing a path are
# grouped under the path, ripgrep --heading style.

_PATH = r"(?P<path>(?:[A-Za-z]:)?(?=[^:\s]*[/\\.])[^:\s]+)"  # no spaces; has / \ or .
_LOC = re.compile(r"^" + _PATH + r":(?P<rest>\d+[:\-].*)$")
_HEAD_RE = re.compile(r"^" + _PATH + r":$")
_ITEM_RE = re.compile(r"^  (?P<rest>\d+[:\-].*)$")
MIN_GROUP = 2


def group_paths(lines: list[str]) -> list[str]:
    out: list[str] = []
    i, n = 0, len(lines)
    while i < n:
        m = _LOC.match(lines[i])
        if m:
            j = i + 1
            while j < n:
                m2 = _LOC.match(lines[j])
                if not m2 or m2.group("path") != m.group("path"):
                    break
                j += 1
            if j - i >= MIN_GROUP:
                out.append(m.group("path") + ":")
                out.extend("  " + _LOC.match(lines[k]).group("rest") for k in range(i, j))
                i = j
                continue
        out.append(lines[i])
        i += 1
    return out


def ungroup_paths(lines: list[str]) -> list[str]:
    out: list[str] = []
    head = None
    for ln in lines:
        if head is not None:
            m = _ITEM_RE.match(ln)
            if m:
                out.append(f"{head}:{m.group('rest')}")
                continue
            head = None
        m = _HEAD_RE.match(ln)
        if m:
            head = m.group("path")
            continue
        out.append(ln)
    return out


# ------------------------------------------------------------------ json ----

def _minify_json(s: str) -> str:
    """Drop whitespace outside strings; keeps every lexeme (numbers, escapes)."""
    out = []
    in_str = esc = False
    for ch in s:
        if in_str:
            out.append(ch)
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
            out.append(ch)
        elif ch not in " \t\r\n":
            out.append(ch)
    return "".join(out)


class _Raw(str):
    """A JSON number kept as its original lexeme."""


def _loads(s: str):
    return json.loads(s, parse_float=_Raw, parse_int=_Raw)


def _scalar(v) -> bool:
    return v is None or isinstance(v, (bool, str))


_SAFE_CELL = re.compile(r"^[^,\n\"\\{}\[\]:]*$")
_NUMLIKE = re.compile(r"^-?\d+(\.\d+)?([eE][+-]?\d+)?$|^(true|false|null)$")


def _cell(v) -> str | None:
    if v is None:
        return "null"
    if v is True:
        return "true"
    if v is False:
        return "false"
    if isinstance(v, _Raw):
        return str(v)
    if isinstance(v, str):
        if v and _SAFE_CELL.match(v) and not _NUMLIKE.match(v) and v == v.strip():
            return v
        return json.dumps(v, ensure_ascii=False)
    return None


def _table(rows: list) -> str | None:
    """TOON-style table for an array of flat objects with identical keys."""
    if len(rows) < 2 or not all(isinstance(r, dict) for r in rows):
        return None
    keys = list(rows[0].keys())
    if not keys or any(list(r.keys()) != keys for r in rows):
        return None
    if any(not _SAFE_CELL.match(k) or not k for k in keys):
        return None
    body = []
    for r in rows:
        cells = []
        for k in keys:
            v = r[k]
            if not (_scalar(v) or isinstance(v, _Raw)):
                return None
            cells.append(_cell(v))
        body.append(",".join(cells))
    return f"[{len(rows)}]{{{','.join(keys)}}}:\n" + "\n".join(body)


def _untable(s: str):
    head, _, rest = s.partition("\n")
    m = re.match(r"^\[(\d+)\]\{(.*)\}:$", head)
    if not m:
        return None
    keys = m.group(2).split(",")
    rows = []
    for line in rest.split("\n"):
        cells = _split_cells(line)
        if len(cells) != len(keys):
            return None
        rows.append(dict(zip(keys, (_uncell(c) for c in cells))))
    return rows if len(rows) == int(m.group(1)) else None


def _split_cells(line: str) -> list[str]:
    cells, buf, in_str, esc = [], [], False, False
    for ch in line:
        if in_str:
            buf.append(ch)
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
            buf.append(ch)
        elif ch == ",":
            cells.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
    cells.append("".join(buf))
    return cells


def _uncell(c: str):
    if c == "null":
        return None
    if c == "true":
        return True
    if c == "false":
        return False
    if c.startswith('"'):
        return json.loads(c)
    if re.match(r"^-?\d+(\.\d+)?([eE][+-]?\d+)?$", c):
        return _Raw(c)
    return c


def _eq(a, b) -> bool:
    # numbers compare by lexeme, so 1.0 vs 1 is a difference
    if isinstance(a, _Raw) or isinstance(b, _Raw):
        return isinstance(a, _Raw) and isinstance(b, _Raw) and str(a) == str(b)
    if isinstance(a, dict) and isinstance(b, dict):
        return list(a) == list(b) and all(_eq(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_eq(x, y) for x, y in zip(a, b))
    return type(a) is type(b) and a == b


def encode_json(text: str) -> str | None:
    """Minified JSON, or a table when the document is a uniform array."""
    s = text.strip()
    if not s or s[0] not in "[{":
        return None
    try:
        data = _loads(s)
    except ValueError:
        return None
    if isinstance(data, list):
        t = _table(data)
        if t is not None:
            return t
    if isinstance(data, dict) and len(data) == 1:
        (k, v), = data.items()
        if isinstance(v, list):
            t = _table(v)
            if t is not None and _SAFE_CELL.match(k) and k:
                return f"{k}" + t
    return _minify_json(s)


def decode_json(enc: str):
    """Inverse of encode_json at the data level."""
    m = re.match(r"^([^\[\n]*)(\[\d+\]\{.*)$", enc, re.S)
    if m and "\n" in enc:
        rows = _untable(m.group(2))
        if rows is not None:
            return {m.group(1): rows} if m.group(1) else rows
    return _loads(enc)


# ------------------------------------------------------------- pipeline -----

class Result:
    __slots__ = ("text", "original_tokens", "tokens", "steps")

    def __init__(self, text: str, original_tokens: int, tokens: int, steps: list[str] | None = None):
        self.text, self.original_tokens, self.tokens, self.steps = text, original_tokens, tokens, steps or []

    @property
    def saved(self) -> int:
        return self.original_tokens - self.tokens


_LINE_STEPS: list[tuple[str, Callable, Callable]] = [
    ("runs", fold_runs, unfold_runs),
    ("paths", group_paths, ungroup_paths),
]


def encode(text: str, *, terminal: bool = True, json_: bool = True,
           count: Callable[[str], int] | None = None, min_gain: float = 0.02,
           lines_total: int | None = None) -> Result:
    """`lines_total`: line count to state in a fold header, when `text` was already shortened."""
    count = count or _cost.estimate
    base = render(text) if terminal else text
    original = count(text)
    steps = ["render"] if base != text else []
    best = base
    best_cost = count(base)

    if json_:
        cands = []
        for j in (encode_json(base), _minify_json(base.strip()) if base.strip()[:1] in "[{" else None):
            if j is None or j == base.strip() or j in cands:
                continue
            try:
                ok = _eq(decode_json(j), _loads(base.strip()))
            except ValueError:
                ok = False
            if ok:
                cands.append(j)
        if cands:  # a table is not always cheaper than minified JSON: price both
            j = min(cands, key=count)
            c = count(j)
            if c < best_cost * (1 - min_gain):
                return Result(j, original, c, steps + ["json"])

    # The full decoder must reproduce the input; if the input itself already
    # contains marker-like lines, nothing below can pass this check.
    if decode(best) == best:
        cur = best.split("\n")
        for name, fwd, _ in _LINE_STEPS:
            cand = fold_runs(cur, lines_total) if name == "runs" and lines_total else fwd(cur)
            if cand == cur:
                continue
            joined = "\n".join(cand)
            if decode(joined) != base:
                continue
            c = count(joined)
            if c < best_cost * (1 - min_gain):
                cur, best_cost = cand, c
                steps.append(name)
        best = "\n".join(cur)
    return Result(best, original, best_cost, steps)


def decode(text: str) -> str:
    """Inverse of the line transforms (JSON decodes with decode_json)."""
    lines = text.split("\n")
    for _, _, inv in reversed(_LINE_STEPS):
        lines = inv(lines)
    return "\n".join(lines)
