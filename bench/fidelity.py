"""Does Claude still get the right answer when lowtoken shrinks its tool output?

Runs real Claude Code sessions (`claude -p`, isolated from your own plugins and
hooks) on deterministic scenarios, once with the lowtoken plugin and once
without, asks questions whose answers sit in the parts lowtoken folds or
omits, and grades them exactly. Token counts come from the API's own usage.

    python bench/fidelity.py [--model haiku] [--trials 2] [--only edit_reread]

Costs real usage on your Claude plan (a few cents per run on Haiku).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WORDS = ["amber", "basil", "cedar", "delta", "ember", "flint", "garnet", "harbor", "iris", "juniper", "kelp",
         "lumen", "maple", "nectar", "onyx", "pepper", "quartz", "raven", "sage", "tundra"]


def _code_line(i: int) -> str:
    return f"item_{i} = register(weight={i * 7 % 13}, tag='{WORDS[i % 20]}-{i * 31 % 97}')"


def edit_reread(d: str) -> tuple[str, dict]:
    lines = [_code_line(i) for i in range(1, 151)]
    open(os.path.join(d, "items.py"), "w", newline="\n").write("\n".join(lines) + "\n")
    new80 = "item_80 = register(weight=99, tag='edited-line')"
    steps = (f"1) Read items.py (whole file). 2) Edit items.py: replace the line `{lines[79]}` with `{new80}`. "
             "3) Read items.py again (whole file).")
    truth = {"line_40": lines[39], "line_80": new80, "line_81": lines[80], "line_149": lines[148],
             "tag_of_item_120": lines[119].split("tag='")[1].rstrip("')")}
    return steps, truth


def pieces_reread(d: str) -> tuple[str, dict]:
    """The file is read in two pieces, edited near the end, then read whole: one marker cites both pieces."""
    lines = [_code_line(i) for i in range(1, 241)]
    open(os.path.join(d, "big.py"), "w", newline="\n").write("\n".join(lines) + "\n")
    new = "item_230 = register(weight=0, tag='edited-near-end')"
    steps = ("1) Read big.py with offset 1 and limit 100. 2) Read big.py with offset 101 and limit 100. "
             f"3) Edit big.py: replace the line `{lines[229]}` with `{new}`. 4) Read big.py again (whole file).")
    truth = {"line_37": lines[36], "line_163": lines[162], "line_230": new, "line_231": lines[230],
             "tag_of_item_199": lines[198].split("tag='")[1].rstrip("')")}
    return steps, truth


def cat_after_read(d: str) -> tuple[str, dict]:
    lines = [_code_line(i) for i in range(1, 121)]
    open(os.path.join(d, "conf.py"), "w", newline="\n").write("\n".join(lines) + "\n")
    steps = "1) Read conf.py (whole file). 2) Run with Bash: sed -n '30,90p' conf.py"
    truth = {"line_35": lines[34], "line_77": lines[76], "weight_of_item_88": str(88 * 7 % 13),
             "tag_of_item_31": lines[30].split("tag='")[1].rstrip("')")}
    return steps, truth


def json_table(d: str) -> tuple[str, dict]:
    users = [{"id": i, "name": f"{WORDS[i % 20].title()} {WORDS[(i * 3) % 20].title()}",
              "email": f"user{i}@{WORDS[(i * 7) % 20]}.dev", "active": i % 3 != 0,
              "city": ["Izmir", "Ankara", "Bursa", "Konya"][i % 4], "score": round(i * 1.75, 2)} for i in range(1, 41)]
    open(os.path.join(d, "users.json"), "w", newline="\n").write(json.dumps(users, indent=2))
    steps = "1) Run with Bash: cat users.json"
    truth = {"email_of_id_17": users[16]["email"], "city_of_id_23": users[22]["city"],
             "active_count": str(sum(u["active"] for u in users)), "score_of_id_33": str(users[32]["score"])}
    return steps, truth


def log_repeats(d: str) -> tuple[str, dict]:
    log = ["INFO starting worker pool size=8"]
    for k in range(1, 6):
        log += [f"WARN retrying connection to db-{k}"] * (10 + k * 7)
        log += ["INFO heartbeat ok", "DEBUG queue depth=0"] * k
    log += ["ERROR giving up after 5 hosts"]
    open(os.path.join(d, "gen.py"), "w", newline="\n").write(
        "import sys\nsys.stdout.write(open('run.log').read())\n")
    open(os.path.join(d, "run.log"), "w", newline="\n").write("\n".join(log) + "\n")
    steps = "1) Run with Bash: python gen.py (it prints a log)"
    truth = {"retries_db_3": str(10 + 3 * 7), "heartbeats_total": str(sum(range(1, 6))),
             "lines_printed_by_gen_py": str(len(log)), "last_line_printed_by_gen_py": log[-1]}
    return steps, truth


def grep_group(d: str) -> tuple[str, dict]:
    os.makedirs(os.path.join(d, "src"), exist_ok=True)
    truth_lines = {}
    for f, n in [("alpha.py", 60), ("beta.py", 45), ("gamma.py", 70)]:
        body, hits = [], []
        for i in range(1, n + 1):
            if (i * 5 + len(f)) % 9 == 0:
                body.append(f"    # TODO({WORDS[i % 20]}): revisit threshold {i}")
                hits.append(i)
            else:
                body.append(f"    value_{i} = compute_metric({i}, mode='{WORDS[(i + 3) % 20]}')")
        open(os.path.join(d, "src", f), "w", newline="\n").write("\n".join(body) + "\n")
        truth_lines[f] = hits
    steps = "1) Run with Bash: grep -rn TODO src/ | sort -t: -k1,1 -k2,2n"
    truth = {"beta_todo_line_numbers_csv": ",".join(map(str, truth_lines["beta.py"])),
             "todo_count_total": str(sum(len(v) for v in truth_lines.values())),
             "owner_of_gamma_line_" + str(truth_lines["gamma.py"][2]): WORDS[truth_lines["gamma.py"][2] % 20]}
    return steps, truth


def _forget_session(d: str) -> None:
    """Remove the transcript folder Claude Code created for this throwaway project."""
    slug = re.sub(r"[^A-Za-z0-9]", "-", os.path.realpath(d))
    proj = os.path.join(os.path.expanduser("~"), ".claude", "projects")
    for name in os.listdir(proj) if os.path.isdir(proj) else []:
        if name.lower() == slug.lower() and "lt-fid-" in name:
            shutil.rmtree(os.path.join(proj, name), ignore_errors=True)


CLEANUP = False
SCENARIOS = {f.__name__: f for f in (edit_reread, pieces_reread, cat_after_read, json_table, log_repeats, grep_group)}


def run(name: str, with_lowtoken: bool, model: str, trial: int) -> dict:
    d = tempfile.mkdtemp(prefix=f"lt-fid-{name}-")
    subprocess.run(["git", "init", "-q"], cwd=d)
    steps, truth = SCENARIOS[name](d)
    keys = ", ".join(f'"{k}"' for k in truth)
    prompt = (f"Use your tools for these steps, in order, with no commentary between them: {steps}\n"
              f"Then answer from the tool results you received, without running anything else. "
              f"Copy text exactly (no quotes or backticks around lines); give lists as comma-separated values "
              f"without spaces. End your reply with one line: "
              f"ANSWERS: {{json object with keys {keys}, all values strings}}")
    with open(os.path.join(d, "mcp.json"), "w") as fh:
        json.dump({"mcpServers": {}}, fh)
    cmd = [shutil.which("claude") or "claude", "-p", prompt, "--model", model, "--setting-sources", "project",
           "--strict-mcp-config", "--mcp-config", "mcp.json", "--allowedTools", "Read", "Edit", "Bash",
           "--output-format", "json"]  # sessions are persisted: deduplication needs the transcript
    if with_lowtoken:
        cmd[3:3] = ["--plugin-dir", ROOT]
    env = dict(os.environ, MSYS_NO_PATHCONV="1")
    t = time.time()
    p = subprocess.run(cmd, cwd=d, capture_output=True, text=True, encoding="utf-8", timeout=600, env=env)
    try:
        out = json.loads(p.stdout)
    except ValueError:
        return {"name": name, "on": with_lowtoken, "trial": trial, "error": (p.stderr or p.stdout)[-300:]}
    text = out.get("result") or ""
    m = re.findall(r"ANSWERS:\s*(\{.*\})", text, re.S)
    try:
        ans = json.loads(m[-1]) if m else {}
    except ValueError:
        ans = {}
    norm = lambda v: re.sub(r"\s+", " ", str(v).strip().strip("`").strip())
    score = {k: norm(ans.get(k, "")) == norm(v) for k, v in truth.items()}
    mu = next(iter((out.get("modelUsage") or {}).values()), {})
    shutil.rmtree(d, ignore_errors=True)
    if CLEANUP:
        _forget_session(d)
    return {"name": name, "on": with_lowtoken, "trial": trial, "score": score, "answers": ans, "truth": truth,
            "input": (mu.get("inputTokens") or 0) + (mu.get("cacheReadInputTokens") or 0)
            + (mu.get("cacheCreationInputTokens") or 0),
            "usd": out.get("total_cost_usd") or 0, "secs": time.time() - t}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="haiku")
    ap.add_argument("--trials", type=int, default=2)
    ap.add_argument("--only")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--cleanup", action="store_true", help="delete the throwaway sessions from ~/.claude/projects")
    a = ap.parse_args()
    global CLEANUP
    CLEANUP = a.cleanup
    names = [a.only] if a.only else list(SCENARIOS)
    jobs = [(n, on, t) for n in names for t in range(a.trials) for on in (False, True)]
    with ThreadPoolExecutor(a.jobs) as ex:
        res = list(ex.map(lambda j: run(j[0], j[1], a.model, j[2]), jobs))
    print(f"model {a.model}, {a.trials} trial(s) per arm\n")
    print(f"{'scenario':<16}{'arm':<10}{'correct':>9}{'context tok':>13}{'usd':>8}")
    tot = {False: [0, 0, 0, 0.0], True: [0, 0, 0, 0.0]}
    for n in names:
        for on in (False, True):
            rs = [r for r in res if r["name"] == n and r["on"] == on]
            errs = [r for r in rs if "error" in r]
            ok = sum(sum(r["score"].values()) for r in rs if "score" in r)
            q = sum(len(r["score"]) for r in rs if "score" in r)
            tok = sum(r.get("input", 0) for r in rs)
            usd = sum(r.get("usd", 0) for r in rs)
            t = tot[on]
            t[0] += ok; t[1] += q; t[2] += tok; t[3] += usd
            arm = "lowtoken" if on else "plain"
            print(f"{n:<16}{arm:<10}{ok:>5}/{q:<3}{tok:>13,}{usd:>8.3f}" + (f"  errors: {len(errs)}" if errs else ""))
            for r in rs:
                for k, good in (r.get("score") or {}).items():
                    if not good:
                        print(f"      miss {k}: got {str(r['answers'].get(k))[:70]!r} want {r['truth'][k][:70]!r}")
    print()
    for on in (False, True):
        ok, q, tok, usd = tot[on]
        print(f"{'lowtoken' if on else 'plain':<10} accuracy {ok}/{q} ({100 * ok / max(q, 1):.1f}%)  "
              f"context tokens {tok:,}  ${usd:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
