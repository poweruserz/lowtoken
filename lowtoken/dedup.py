"""Session-level, content-addressed deduplication against what Claude has seen.

Agents put the same text into their context again and again: a file read with
Read, later printed with cat or sed, grepped, re-read after a small edit, read
right after Claude wrote it. Measured on real Claude Code sessions, 9-12% of
tool-result tokens repeat runs of lines that are still in the context.

Ground truth is the session transcript, not the tool output: another hook may
have replaced an output, Claude Code may have sent a "file unchanged" stub, a
compaction may have dropped everything. So the store is fed only from the
transcript (incrementally, by byte offset), and a marker may only point at
text that the transcript shows Claude actually received or wrote.

Two kinds of reference:
  * Read of a file: lines Claude already has at the same line numbers (from an
    earlier Read or its own Write), at the start or end of the new read,
    become one marker each. Claude Code numbers Read lines itself, so the
    marker takes the number of the last omitted line and every shown line
    keeps its true number.
  * Anything else (Bash, MCP): greedy LZ77 over lines against earlier outputs.
    A run long enough to pay for its marker becomes
    "[lowtoken: N lines omitted, identical to lines a-b of <source>]".

A line's identity is its exact text; a `cat -n` style number prefix is split
off and must match too, so omitted numbers are always recoverable.
"""
from __future__ import annotations

import json
import os
import re

from . import cost

try:  # _blake2 avoids importing OpenSSL through hashlib (~12 ms per hook call)
    from _blake2 import blake2b
except ImportError:  # pragma: no cover
    from hashlib import blake2b

WINDOW = 300         # sources kept for line matching; a memory bound, not a correctness one
MIN_LINES = 3
MAX_SEGMENTS = 4     # earlier reads one Read marker may cite
# `cd "<project>" && ` and `FOO=bar ` in front of a command say nothing about what it printed
_CD = re.compile(r"^(?:cd\s+(?:\"[^\"]*\"|'[^']*'|\S+)\s*(?:&&|;)\s*|[A-Z_][A-Z0-9_]*=\S*\s+)+")
MIN_SAVED = 40       # tokens a replacement must save after paying for its marker
_NUM = re.compile(r"^ *(\d+)\t")
_OURS = re.compile(r"^\[lowtoken: ")
_SHIFT = re.compile(r"^\[lowtoken: lines (\d+)-(\d+) unchanged since .+? \(they were lines (\d+)-(\d+) then\)\]$")


def _hash(text: str) -> int:
    return int.from_bytes(blake2b(text.encode("utf-8", "replace"), digest_size=8).digest(), "big")


def text_hash(line: str) -> int:
    """Hash of a raw file line (Read content has no number prefix)."""
    return _hash(line[:-1] if line.endswith("\r") else line)


def key(line: str) -> tuple[int, int]:
    """(number or -1, hash of the text) of a shown line."""
    if line.endswith("\r"):
        line = line[:-1]
    m = _NUM.match(line)
    if m:
        return int(m.group(1)), _hash(line[m.end():])
    return -1, _hash(line)


def _solid(line: str) -> bool:
    return len(line.strip()) >= 8


def _visible(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text")
    return ""


def _norm_path(p: str) -> str:
    return os.path.normcase(os.path.normpath(p)) if p else ""


def _display(path: str, cwd: str) -> str:
    """The path as a marker names it: relative to the project when inside it, else in full.

    A bare file name would be ambiguous between two app.py files in one session."""
    if not path:
        return "?"
    try:
        rel = os.path.relpath(path, cwd) if cwd else path
    except ValueError:  # another drive on Windows
        rel = path
    return path if rel.startswith("..") else rel.replace("\\", "/")


class Run:
    """A run of `length` new lines from index `start`, equal to source `src` from `src_start`."""
    __slots__ = ("start", "length", "src", "src_start", "numbered")

    def __init__(self, start: int, length: int, src: int, src_start: int, numbered: bool):
        self.start, self.length, self.src, self.src_start, self.numbered = start, length, src, src_start, numbered


def state_dir() -> str | None:
    """A private per-user directory for session stores, or None when it is not safe to use.

    Never a shared temp dir: another local user could pre-create it to feed the hook false
    "Claude has seen this" state, or plant symlinks at predictable file names."""
    d = os.environ.get("LOWTOKEN_HOME")
    if not d:
        if os.name == "nt":
            d = os.path.join(os.environ.get("LOCALAPPDATA") or os.path.expanduser("~"), "lowtoken")
        else:
            d = os.path.join(os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache"), "lowtoken")
    try:
        os.makedirs(d, mode=0o700, exist_ok=True)
        if os.name != "nt":  # on Windows the per-user profile's ACLs already keep others out
            import stat
            st = os.lstat(d)
            if stat.S_ISLNK(st.st_mode) or not stat.S_ISDIR(st.st_mode) or st.st_uid != os.getuid():
                return None
            if st.st_mode & 0o077:
                os.chmod(d, 0o700)
    except OSError:
        return None
    return d


def _create(path: str):
    """Open a new file exclusively, never through a symlink."""
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0)
    return os.open(path, flags, 0o600)


class Store:
    def __init__(self, session_id: str, root: str | None = None, load: bool = True):
        root = root or state_dir()
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", session_id or "default")
        self.path = os.path.join(root, f"{safe}.json") if root else ""  # "" = no safe place: dedup stays off
        self._blank()
        if load:
            self.load()

    def load(self) -> None:
        if not self.path:
            return
        try:
            with open(self.path, encoding="utf-8") as fh:
                d = json.load(fh)
            for k in ("n", "sources", "elided", "offset", "pending", "seq", "files"):
                setattr(self, k, d[k])
            self.saved = d.get("saved", 0)
            self.ctx = d.get("ctx", 0)
        except (OSError, ValueError, KeyError):
            pass

    def _blank(self) -> None:
        self.n = 0
        self.sources: list[dict] = []
        self.elided: list[str] = []   # fingerprints of outputs we shortened
        self.offset = 0               # transcript bytes already read
        self.pending: dict[str, list] = {}   # tool_use_id -> [name, path or command]
        self.seq: dict[str, int] = {}        # label -> how many times seen
        self.files: dict[str, dict[str, list]] = {}  # path -> {line number: [hash, label, delta, source id]}
        self.saved = getattr(self, "saved", 0)  # tokens saved this session (kept across resets)
        self.ctx = 0                  # context size of the last API call seen in the transcript

    def lock(self, wait: float = 2.0) -> bool:
        """Claude Code runs hooks of parallel tool calls at the same time; one writer at a time.

        A lock older than 30 s is from a hook that was killed and is taken over."""
        import time
        if not self.path:
            return False
        lk = self.path + ".lock"
        end = time.time() + wait
        while True:
            try:
                os.close(_create(lk))
                self._locked = lk
                return True
            except FileExistsError:
                try:
                    if time.time() - os.path.getmtime(lk) > 30:
                        os.remove(lk)
                        continue
                except OSError:
                    pass
                if time.time() > end:
                    return False
                time.sleep(0.02)
            except OSError:
                return False

    def unlock(self) -> None:
        lk = getattr(self, "_locked", None)
        if lk:
            try:
                os.remove(lk)
            except OSError:
                pass
            self._locked = None

    def reset(self) -> None:
        offset = self.offset
        self._blank()
        self.offset = offset
        self.save()

    def save(self) -> None:
        # keep the store bounded in long sessions: drop what markers may no longer cite
        for path in list(self.files):
            live = {k: v for k, v in self.files[path].items() if self.live(v[3])}
            if live:
                self.files[path] = live
            else:
                del self.files[path]
        if len(self.pending) > 200:
            self.pending = dict(list(self.pending.items())[-200:])
        if not self.path:
            return
        tmp = f"{self.path}.{os.urandom(6).hex()}.tmp"
        with os.fdopen(_create(tmp), "w", encoding="utf-8") as fh:
            json.dump({"n": self.n, "sources": self.sources, "elided": self.elided[-200:], "offset": self.offset,
                       "pending": self.pending, "seq": self.seq, "files": self.files, "saved": self.saved,
                       "ctx": self.ctx}, fh)
        os.replace(tmp, self.path)

    # ---------------------------------------------------------------- sync --
    def sync(self, transcript: str | None) -> bool:
        """Read new transcript lines; False when there is no transcript to trust."""
        if not transcript or not os.path.isfile(transcript):
            return False
        size = os.path.getsize(transcript)
        if size < self.offset:  # a different or rewritten transcript
            self._blank()
        with open(transcript, "rb") as fh:
            fh.seek(self.offset)
            data = fh.read()
        end = data.rfind(b"\n")
        if end < 0:
            return True
        self.offset += end + 1
        for raw in data[:end].split(b"\n"):
            try:
                d = json.loads(raw)
            except ValueError:
                continue
            if isinstance(d, dict) and not d.get("isSidechain"):
                self._event(d)
        return True

    def _event(self, d: dict) -> None:
        t = d.get("type")
        u = (d.get("message") or {}).get("usage") if t == "assistant" else None
        if t == "system" and d.get("subtype") == "compact_boundary" or u and self._shrank(u):
            # Text is in the context for as long as the context never got smaller. Measured on 40,706
            # consecutive calls: it only shrinks on compaction (no silent clearing), so any drop resets.
            offset, saved = self.offset, self.saved
            self._blank()          # saved once, with everything after the boundary, by the caller
            self.offset, self.saved = offset, saved
            if not u:
                return
        if u:
            self.ctx = sum(u.get(k) or 0 for k in ("input_tokens", "cache_read_input_tokens",
                                                   "cache_creation_input_tokens"))
        content = (d.get("message") or {}).get("content")
        if not isinstance(content, list):
            return
        cwd = d.get("cwd") or ""
        for b in content:
            if not isinstance(b, dict):
                continue
            if t == "assistant" and b.get("type") == "tool_use":
                name, inp = b.get("name", ""), b.get("input") or {}
                path = inp.get("file_path")
                path = path if isinstance(path, str) else ""
                shown = _display(path, cwd)
                # Labels are numbered when Claude makes the call, in call order, failed calls included:
                # that is how Claude counts its own Reads. Results can arrive in another order.
                if name == "Read":
                    label = self._ordinal(f"Read of {shown}", numbered=True)
                elif name in ("Write", "Edit"):
                    label = ""
                else:
                    cmd = _CD.sub("", " ".join(str(inp.get("command") or "").split()))
                    label = self._ordinal(f"the output of `{cmd[:60]}{'...' if len(cmd) > 60 else ''}`" if cmd
                                          else f"the {name.split('__')[-1]} result")
                self.pending[b.get("id", "")] = [name, path, label]
                if name == "Write" and isinstance(inp.get("content"), str):
                    self._wrote(path, self._ordinal(f"your Write of {shown}"), inp["content"].split("\n"))
                elif name == "Edit" and isinstance(inp.get("new_string"), str):
                    self.add(self._ordinal(f"your Edit of {shown}"), inp["new_string"].split("\n"))
            elif t == "user" and b.get("type") == "tool_result":
                name, path, label = (list(self.pending.pop(b.get("tool_use_id", ""), None) or []) + ["", "", ""])[:3]
                text = _visible(b.get("content"))
                if not label or not text.strip() or b.get("is_error"):
                    continue
                if name == "Read":
                    self._seen_read(path, label, text.split("\n"))
                else:
                    self.add(label, text.split("\n"))

    def _ordinal(self, base: str, numbered: bool = False) -> str:
        k = self.seq.get(base, 0) + 1
        self.seq[base] = k
        if numbered:  # "Read of a.py" -> "Read #2 of a.py"
            head, _, rest = base.partition(" ")
            return f"{head} #{k} {rest}"
        return base if k == 1 else f"{base} (#{k})"

    def _wrote(self, path: str, label: str, lines: list[str]) -> None:
        sid = self.add(label, lines)
        self.files[_norm_path(path)] = {str(i + 1): [text_hash(x), label, 0, sid] for i, x in enumerate(lines)}

    def _seen_read(self, path: str, label: str, lines: list[str]) -> None:
        if not any(_NUM.match(ln) for ln in lines):
            return  # a stub (file unchanged, replaced by another hook...): nothing new was shown
        sid = self.add(label, lines)
        fmap = self.files.setdefault(_norm_path(path), {})
        for ln in lines:
            m = _NUM.match(ln)
            if not m:
                continue
            num, text = int(m.group(1)), ln[m.end():]
            sm = _SHIFT.match(text)
            if sm:  # our shifted-suffix marker: the old lines now live at new numbers
                a, b, pa = int(sm.group(1)), int(sm.group(2)), int(sm.group(3))
                old = dict(fmap)
                for t in range(b - a + 1):
                    v = old.get(str(pa + t))
                    if v:
                        fmap[str(a + t)] = [v[0], v[1], v[2] + (pa - a), v[3]]
                continue
            if _OURS.match(text):
                continue  # lines behind our own marker keep what Claude saw before
            fmap[str(num)] = [text_hash(text), label, 0, sid]

    # -------------------------------------------------------------- sources --
    def add(self, label: str, lines: list[str]) -> int:
        self.n += 1
        ks = [key(x) for x in lines]
        self.sources.append({"id": self.n, "label": label, "h": [k[1] for k in ks], "num": [k[0] for k in ks]})
        self.sources = [s for s in self.sources if s["id"] > self.n - WINDOW]
        return self.n

    def _shrank(self, u: dict) -> bool:
        ctx = sum(u.get(k) or 0 for k in ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"))
        return bool(self.ctx) and ctx < self.ctx

    def live(self, sid: int) -> bool:
        return True  # file maps stay valid until the context shrinks (then the store is reset)

    def source(self, sid: int) -> dict | None:
        return next((s for s in self.sources if s["id"] == sid), None)

    def runs(self, lines: list[str]) -> list[Run]:
        ks = [key(x) for x in lines]
        index: dict[int, list[tuple[int, int]]] = {}
        for s in self.sources:
            for j, h in enumerate(s["h"]):
                index.setdefault(h, []).append((s["id"], j))
        by_id = {s["id"]: s for s in self.sources}

        def same(i: int, src: dict, j: int) -> bool:
            n, h = ks[i]
            return h == src["h"][j] and (n == -1 or n == src["num"][j]) and not _OURS.match(lines[i])

        out: list[Run] = []
        i = 0
        while i < len(lines):
            best = None
            if _solid(lines[i]) and ks[i][1] in index:
                for sid, j in index[ks[i][1]][-64:]:
                    src = by_id[sid]
                    k = 0
                    base = src["num"][j]
                    # a marker names one contiguous range, so numbered sources must not skip numbers
                    while i + k < len(ks) and j + k < len(src["h"]) and same(i + k, src, j + k) \
                            and (base == -1 or src["num"][j + k] == base + k):
                        k += 1
                    if k and (best is None or k > best.length or (k == best.length and sid > best.src)):
                        best = Run(i, k, sid, j, ks[i][0] != -1)
            if best and best.length >= MIN_LINES and sum(map(_solid, lines[i:i + best.length])) >= 2:
                out.append(best)
                i += best.length
            else:
                i += 1
        return out


def marker(run: Run, src: dict) -> str:
    if run.numbered:
        a = src["num"][run.src_start]
        return f"[lowtoken: lines {a}-{a + run.length - 1} omitted, identical to {src['label']} shown earlier]"
    a = src["num"][run.src_start] if src["num"][run.src_start] != -1 else run.src_start + 1
    return f"[lowtoken: {run.length} lines omitted, identical to lines {a}-{a + run.length - 1} of {src['label']}]"


def apply(store: Store, lines: list[str]) -> tuple[list[str], int]:
    """Replace repeated runs with markers; returns (new lines, tokens saved)."""
    out: list[str] = []
    saved = 0
    pos = 0
    for r in store.runs(lines):
        src = store.source(r.src)
        if src is None:
            continue
        m = marker(r, src)
        gain = cost.estimate("\n".join(lines[r.start:r.start + r.length])) - cost.estimate(m)
        if gain < MIN_SAVED:
            continue
        out.extend(lines[pos:r.start])
        out.append(m)
        pos = r.start + r.length
        saved += gain
    out.extend(lines[pos:])
    return out, saved


def read_markers(store: Store, path: str, start: int, lines: list[str]) -> tuple[list[str], int]:
    """Trim the start and end of a Read whose lines Claude already has.

    Returns (content lines, startLine). Lines in the middle are never removed:
    Claude Code numbers the content itself, so cutting only at the ends is the
    one way to keep every shown number true.
    """
    fmap = {int(n): v for n, v in store.files.get(_norm_path(path), {}).items() if store.live(v[3])}
    if not fmap:
        return lines, start
    hs = [text_hash(x) for x in lines]

    def known(i: int, shift: int = 0):
        v = fmap.get(start + i + shift)
        return v if v and v[0] == hs[i] else None

    def tag(v) -> tuple:
        return (v[1], v[2])

    # Prefix: same text at the same number. It may span several earlier reads (a file read in
    # pieces); the marker then lists each piece, up to MAX_SEGMENTS of them.
    p = 0
    segs: list[list] = []  # [first index, last index, entry]
    while p < len(lines):
        v = known(p)
        if not v:
            break
        if not segs or tag(v) != tag(segs[-1][2]):
            if len(segs) == MAX_SEGMENTS:
                break
            segs.append([p, p, v])
        segs[-1][1] = p
        p += 1
    # Suffix: the run must reach the very last line, since a line left after the
    # marker would be shifted to a wrong number. The shift (lines inserted or
    # deleted above) is found from the last non-blank line.
    last = len(lines) - 1
    solid = last
    while solid > p and not lines[solid].strip():
        solid -= 1
    q, qshift, qref = 0, 0, None
    if solid > p:
        cands = {n - (start + solid) for n, v in fmap.items() if v[0] == hs[solid]}
        for shift in sorted(cands, key=abs)[:8]:
            ref = known(last, shift)
            if not ref:
                continue
            k = 0
            while last - k >= p:
                v = known(last - k, shift)
                if not v or tag(v) != tag(ref):
                    break
                k += 1
            if k > q:
                q, qshift, qref = k, shift, ref
    out = list(lines)
    new_start = start
    if q >= MIN_LINES and qref:
        a, b = start + len(lines) - q, start + last
        v = qref
        delta = qshift + v[2]
        was = f" (they were lines {a + delta}-{b + delta} then)" if delta else ""
        m = f"[lowtoken: lines {a}-{b} unchanged since {v[1]}{was}]"
        if cost.estimate("\n".join(lines[len(lines) - q:])) - cost.estimate(m) >= MIN_SAVED:
            out = out[:len(lines) - q] + [m]
    if p >= MIN_LINES and segs:
        if len(segs) == 1:
            v = segs[0][2]
            delta = v[2]
            was = f" (they were lines {start + delta}-{start + p - 1 + delta} then)" if delta else ""
            m = f"[lowtoken: lines {start}-{start + p - 1} unchanged since {v[1]}{was}]"
        else:
            parts = []
            for i0, i1, v in segs:
                a, b = start + i0, start + i1
                was = (f" as line {a + v[2]}" if a == b else f" as lines {a + v[2]}-{b + v[2]}") if v[2] else ""
                parts.append(f"{a if a == b else f'{a}-{b}'}: {v[1]}{was}")
            m = f"[lowtoken: lines {start}-{start + p - 1} unchanged since you saw them ({'; '.join(parts)})]"
        if cost.estimate("\n".join(lines[:p])) - cost.estimate(m) >= MIN_SAVED:
            # the marker takes the number of the last omitted line, so every shown line keeps its number
            out = [m] + out[p:]
            new_start = start + p - 1
    return out, new_start
