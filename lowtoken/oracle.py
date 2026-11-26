"""Exact Claude token counts for free, through the user's own Claude Code.

`claude -p /context` runs no model turn (cost $0) and lists every loaded memory
file with its exact token count from the count_tokens endpoint. We write each
sample into its own `.claude/rules/` file inside a throwaway project, run one
`/context`, and subtract the count of a file holding one token. No API key.

This counts the user's own text, which is what count_tokens is for. It does
not, and must not, be used to enumerate the tokenizer's vocabulary.

Limits: /context prints counts >= 1000 as "1.2k", so samples are kept under
~900 tokens each (callers split longer text and sum the parts).
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile

FENCE_MIN = 5
_ROW = re.compile(r"^\|\s*Project\s*\|\s*(?P<path>[^|]+?)\s*\|\s*(?P<tok>[\d.]+k?)\s*\|\s*$", re.M)


def _fence(sample: str) -> str:
    # A code fence keeps @imports, HTML comments and frontmatter in the sample inert.
    longest = max((len(m) for m in re.findall(r"`+", sample)), default=0)
    return "`" * max(FENCE_MIN, longest + 1)


def _wrap(sample: str, fence: str) -> str:
    return f"{fence}\n{sample}\n{fence}"


def _parse(result: str) -> dict[str, int]:
    out = {}
    for m in _ROW.finditer(result):
        tok = m.group("tok")
        if tok.endswith("k"):
            continue  # rounded, unusable
        out[os.path.basename(m.group("path").strip())] = int(tok)
    return out


def count(samples: list[str], model: str | None = None, claude: str = "claude",
          timeout: int = 600) -> list[int | None]:
    """Exact Claude token count of each sample (None when it could not be read)."""
    if not samples:
        return []
    fence = "`" * max(len(_fence(s)) for s in samples)
    root = tempfile.mkdtemp(prefix="lowtoken-oracle-")
    try:
        rules = os.path.join(root, ".claude", "rules")
        os.makedirs(rules)
        names = [f"s{i:05d}.md" for i in range(len(samples))]
        base = "zbase.md"  # same length as sNNNNN.md so the framing costs the same
        for name, s in zip(names, samples):
            with open(os.path.join(rules, name), "w", encoding="utf-8", newline="") as fh:
                fh.write(_wrap(s, fence))
        with open(os.path.join(rules, base.replace("zbase", "z0000")), "w", encoding="utf-8", newline="") as fh:
            fh.write(f"{fence}\na\n{fence}")  # "a" is one token; an empty body would merge the two newlines
        empty = os.path.join(root, "mcp.json")
        with open(empty, "w") as fh:
            json.dump({"mcpServers": {}}, fh)
        cmd = [shutil.which(claude) or claude, "-p", "/context", "--system-prompt", "x", "--tools", "",
               "--strict-mcp-config", "--mcp-config", empty, "--no-session-persistence",
               "--output-format", "json"]
        if model:
            cmd += ["--model", model]
        env = dict(os.environ, MSYS_NO_PATHCONV="1")
        proc = subprocess.run(cmd, cwd=root, capture_output=True, text=True, encoding="utf-8",
                              timeout=timeout, env=env)
        try:
            result = json.loads(proc.stdout)["result"]
        except (ValueError, KeyError) as e:
            raise RuntimeError(f"claude /context failed: {proc.stderr[-500:] or proc.stdout[-500:]}") from e
        rows = _parse(result)
        zero = rows.get("z0000.md")
        if zero is None:
            raise RuntimeError("baseline row missing from /context output")
        return [rows[n] - zero + 1 if n in rows else None for n in names]
    finally:
        shutil.rmtree(root, ignore_errors=True)
