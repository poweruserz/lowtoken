"""A/B on real bug-fix tasks: does Claude still finish the job with lowtoken, and for how much?

Each task is a small repo with a bug and a failing test. Claude (claude -p, isolated from your own
plugins and hooks) is asked to fix it. Success is decided afterwards by running the tests ourselves,
not by what Claude says. Tokens and cost come from the API's own usage.

    python bench/tasks.py [--model haiku] [--trials 3] [--only name]

Costs real usage on your Claude plan (a few cents per run on Haiku).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WORDS = ["amber", "basil", "cedar", "delta", "ember", "flint", "garnet", "harbor", "iris", "juniper"]


def _write(d: str, path: str, text: str) -> None:
    full = os.path.join(d, path)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)


def big_module(d: str) -> str:
    """A 400-line module; the bug sits in the middle, the fix needs two edits far apart."""
    funcs = []
    for i in range(1, 80):
        funcs.append(f"def metric_{i}(values):\n    \"\"\"Weighted metric {i}.\"\"\"\n"
                     f"    return sum(v * {i % 7 + 1} for v in values) / max(len(values), 1)\n")
    funcs[39] = ("def metric_40(values):\n    \"\"\"Weighted metric 40.\"\"\"\n"
                 "    return sum(v * 6 for v in values) / max(len(values) - 1, 1)\n")
    funcs[70] = ("def metric_71(values):\n    \"\"\"Weighted metric 71.\"\"\"\n"
                 "    return sum(v * 2 for v in values) / max(len(values) + 1, 1)\n")
    _write(d, "stats/metrics.py", "\n\n".join(funcs))
    _write(d, "tests/test_metrics.py",
           "from stats.metrics import metric_40, metric_71, metric_3\n\n"
           "def test_metric_40():\n    assert metric_40([1, 2, 3]) == 12\n\n"
           "def test_metric_71():\n    assert metric_71([1, 2, 3]) == 4\n\n"
           "def test_metric_3():\n    assert metric_3([2, 2]) == 8\n")
    _write(d, "stats/__init__.py", "")
    return "The tests in tests/ fail. Find the bugs in stats/metrics.py and fix them; every metric_N divides by len(values)."


def noisy_tests(d: str) -> str:
    """The test run prints hundreds of repeated deprecation warnings around one real failure."""
    _write(d, "app/parse.py",
           "import warnings\n\n"
           "def parse_line(line):\n"
           "    warnings.warn('parse_line: legacy separator handling is deprecated', DeprecationWarning)\n"
           "    key, _, value = line.partition('=')\n"
           "    return key.strip(), value\n")
    _write(d, "app/__init__.py", "")
    cases = "\n".join(f"    ('k{i} = v{i}', ('k{i}', 'v{i}'))," for i in range(120))
    _write(d, "tests/test_parse.py",
           "import pytest\nfrom app.parse import parse_line\n\n"
           f"CASES = [\n{cases}\n]\n\n"
           "@pytest.mark.parametrize('line,want', CASES)\n"
           "def test_parse(line, want):\n    print('checking', line)\n    assert parse_line(line) == want\n")
    _write(d, "pytest.ini", "[pytest]\nfilterwarnings = default\naddopts = -rA -s\n")
    return "Run the tests with `python -m pytest`, find why they fail and fix app/parse.py."


def json_config(d: str) -> str:
    """A big uniform JSON file the code reads; one field name in the code is wrong."""
    hosts = [{"name": f"{WORDS[i % 10]}-{i}", "region": ["eu-west", "us-east", "ap-south"][i % 3],
              "port": 8000 + i, "enabled": i % 4 != 0, "weight": i % 5 + 1} for i in range(60)]
    _write(d, "config/hosts.json", json.dumps(hosts, indent=2))
    _write(d, "app/routing.py",
           "import json, os\n\n"
           "def load():\n"
           "    with open(os.path.join(os.path.dirname(__file__), '..', 'config', 'hosts.json')) as fh:\n"
           "        return json.load(fh)\n\n"
           "def total_weight(region):\n"
           "    return sum(h['wieght'] for h in load() if h['region'] == region and h['enabled'])\n")
    _write(d, "app/__init__.py", "")
    want = sum(h["weight"] for h in hosts if h["region"] == "eu-west" and h["enabled"])
    _write(d, "tests/test_routing.py",
           f"from app.routing import total_weight\n\ndef test_eu():\n    assert total_weight('eu-west') == {want}\n")
    return "tests/test_routing.py fails. Look at config/hosts.json and app/routing.py and fix the code."


def rename_everywhere(d: str) -> str:
    """A function was renamed in one file; callers in several files still use the old name."""
    _write(d, "lib/core.py", "def compute_total(items):\n    return sum(i['price'] * i['qty'] for i in items)\n")
    _write(d, "lib/__init__.py", "")
    for k, name in enumerate(["orders", "invoices", "reports", "carts"]):
        body = [f"from lib.core import calc_total\n\n"]
        for j in range(25):
            body.append(f"def {name}_{j}(items):\n    # {name} helper {j}\n    return calc_total(items) + {j}\n\n")
        _write(d, f"lib/{name}.py", "".join(body))
    _write(d, "tests/test_all.py",
           "from lib import orders, invoices, reports, carts\n\nITEMS = [{'price': 2, 'qty': 3}]\n\n"
           "def test_all():\n    assert orders.orders_1(ITEMS) == 7 and invoices.invoices_2(ITEMS) == 8\n"
           "    assert reports.reports_3(ITEMS) == 9 and carts.carts_4(ITEMS) == 10\n")
    return "The tests fail with an import error. The function in lib/core.py is called compute_total; fix the callers."


def log_bug(d: str) -> str:
    """A script whose output repeats a lot; the bug is in how it counts."""
    lines = []
    for k in range(1, 6):
        lines += [f"WARN retrying connection to db-{k}"] * (10 + k * 7) + ["INFO heartbeat ok"] * k
    lines.append("ERROR giving up")
    _write(d, "run.log", "\n".join(lines) + "\n")
    _write(d, "tools/summarize.py",
           "import sys\n\n"
           "def count_retries(path, host):\n"
           "    n = 0\n"
           "    for line in open(path):\n"
           "        if line.startswith('WARN') and line.strip().endswith(host[:-1]):\n"
           "            n += 1\n"
           "    return n\n\n"
           "if __name__ == '__main__':\n"
           "    print(open(sys.argv[1]).read())\n")
    _write(d, "tools/__init__.py", "")
    _write(d, "tests/test_summarize.py",
           "from tools.summarize import count_retries\n\n"
           "def test_db3():\n    assert count_retries('run.log', 'db-3') == 31\n\n"
           "def test_db5():\n    assert count_retries('run.log', 'db-5') == 45\n")
    return "Run `python tools/summarize.py run.log` to see the log, then fix count_retries so the tests pass."


TASKS = {f.__name__: f for f in (big_module, noisy_tests, json_config, rename_everywhere, log_bug)}


def _tests_digest(d: str) -> str:
    import hashlib
    h = hashlib.sha256()
    for base, _, files in sorted(os.walk(os.path.join(d, "tests"))):
        for f in sorted(files):
            if f.endswith(".py"):
                with open(os.path.join(base, f), "rb") as fh:
                    h.update(f.encode() + fh.read())
    return h.hexdigest()


def run(name: str, on: bool, model: str, trial: int) -> dict:
    d = tempfile.mkdtemp(prefix=f"lt-task-{name}-")
    subprocess.run(["git", "init", "-q"], cwd=d)
    goal = TASKS[name](d)
    with open(os.path.join(d, "mcp.json"), "w") as fh:
        json.dump({"mcpServers": {}}, fh)
    prompt = f"{goal} Use python -m pytest to check your fix. Do not edit the tests."
    cmd = [shutil.which("claude") or "claude", "-p", prompt, "--model", model, "--setting-sources", "project",
           "--strict-mcp-config", "--mcp-config", "mcp.json", "--allowedTools", "Read", "Edit", "Write", "Bash",
           "Grep", "Glob", "--output-format", "json", "--max-turns", "40"]
    if on:
        cmd[3:3] = ["--plugin-dir", ROOT]
    before = _tests_digest(d)
    t = time.time()
    p = subprocess.run(cmd, cwd=d, capture_output=True, text=True, encoding="utf-8", timeout=900,
                       env=dict(os.environ, MSYS_NO_PATHCONV="1"))
    secs = time.time() - t
    tests_untouched = _tests_digest(d) == before  # a "fix" that edits the tests does not count
    check = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"], cwd=d,
                           capture_output=True, text=True)
    try:
        out = json.loads(p.stdout)
    except ValueError:
        out = {}
    mu = next(iter((out.get("modelUsage") or {}).values()), {})
    res = {"name": name, "on": on, "trial": trial, "solved": check.returncode == 0 and tests_untouched,
           "context": (mu.get("inputTokens") or 0) + (mu.get("cacheReadInputTokens") or 0)
           + (mu.get("cacheCreationInputTokens") or 0),
           "output": mu.get("outputTokens") or 0, "usd": out.get("total_cost_usd") or 0,
           "turns": out.get("num_turns") or 0, "secs": secs}
    shutil.rmtree(d, ignore_errors=True)
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="haiku")
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--only")
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    names = [a.only] if a.only else list(TASKS)
    jobs = [(n, on, t) for n in names for t in range(a.trials) for on in (False, True)]
    with ThreadPoolExecutor(a.jobs) as ex:
        res = list(ex.map(lambda j: run(j[0], j[1], a.model, j[2]), jobs))
    print(f"model {a.model}, {a.trials} trial(s) per arm\n")
    print(f"{'task':<18}{'arm':<10}{'solved':>8}{'context tok':>13}{'output tok':>12}{'turns':>7}{'usd':>8}")
    tot = {False: [0, 0, 0, 0, 0.0, 0], True: [0, 0, 0, 0, 0.0, 0]}
    for n in names:
        for on in (False, True):
            rs = [r for r in res if r["name"] == n and r["on"] == on]
            ok = sum(r["solved"] for r in rs)
            ctx = sum(r["context"] for r in rs)
            outp = sum(r["output"] for r in rs)
            usd = sum(r["usd"] for r in rs)
            turns = sum(r["turns"] for r in rs)
            t = tot[on]
            t[0] += ok; t[1] += len(rs); t[2] += ctx; t[3] += outp; t[4] += usd; t[5] += turns
            print(f"{n:<18}{'lowtoken' if on else 'plain':<10}{ok:>4}/{len(rs):<3}{ctx:>13,}{outp:>12,}"
                  f"{turns:>7}{usd:>8.3f}")
    print()
    for on in (False, True):
        ok, k, ctx, outp, usd, turns = tot[on]
        print(f"{'lowtoken' if on else 'plain':<10} solved {ok}/{k}  context {ctx:,}  output {outp:,}  "
              f"turns {turns}  ${usd:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
