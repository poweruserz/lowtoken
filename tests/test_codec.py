import json
import random

from lowtoken import codec, cost


def test_render_strips_ansi_and_progress():
    raw = "\x1b[32mPASS\x1b[0m a\nDownloading 10%\rDownloading 55%\rDownloading 100%\nab\bc"
    assert codec.render(raw) == "PASS a\nDownloading 100%\nac"


def test_render_keeps_crlf_text():
    assert codec.render("a\r\nb\r\n") == "a\nb\n"


def test_runs_roundtrip_and_shrink():
    lines = ["start"] + ["warning: x"] * 40 + ["a", "b", "c"] * 10 + ["end"]
    folded = codec.fold_runs(lines)
    assert len(folded) < 10
    assert codec.unfold_runs(folded) == lines


def test_paths_roundtrip_windows_and_unix():
    lines = [r"C:\src\app.py:12:import os", r"C:\src\app.py:40:x = 1", "src/b.ts:3:let a",
             "src/b.ts:9:let b", "src/b.ts:11-context line", "plain line"]
    g = codec.group_paths(lines)
    assert g[0] == r"C:\src\app.py:" and g[1] == "  12:import os"
    assert codec.ungroup_paths(g) == lines


def test_timestamps_and_blame_are_not_paths():
    # regression: "2026-11-26 10:04:15 INFO" was grouped under a made-up path "2026-11-26 10"
    log = "\n".join(f"2026-11-26 10:{m:02d}:15 INFO worker {m} finished batch" for m in range(30))
    blame = "\n".join(f"a1b2c3d4 (Ayse Yilmaz 2026-11-26 10:0{m % 10}:00 +0300 {m + 1}) x = f({m})"
                      for m in range(20))
    for text in (log, blame):
        assert "paths" not in codec.encode(text).steps
        assert codec.group_paths(text.split("\n")) == text.split("\n")
    win = "\n".join(rf"C:\src\app\mod.py:{m}:x = {m}" for m in range(1, 8))
    assert codec.group_paths(win.split("\n"))[0] == r"C:\src\app\mod.py:"


def test_header_states_the_count_it_is_given():
    lines = ["a"] + ["x"] * 40
    assert codec.fold_runs(lines, 300)[0] == "[lowtoken: 300 lines; repeated lines folded]"
    r = codec.encode("\n".join(lines), lines_total=300)
    assert r.text.startswith("[lowtoken: 300 lines;") and codec.decode(r.text) == "\n".join(lines)


def test_json_picks_the_cheaper_of_table_and_minified():
    rows = [{"a": i} for i in range(3)]  # a tiny table can cost more than compact JSON
    r = codec.encode(json.dumps(rows, indent=4))
    assert "json" in r.steps
    assert r.tokens == min(cost.estimate(codec._minify_json(json.dumps(rows))),
                           cost.estimate(codec.encode_json(json.dumps(rows))))


def test_json_minify_keeps_lexemes():
    src = '{\n  "a": 1.0,\n  "b": "x y",\n  "c": [1e5, -0.50]\n}'
    enc = codec.encode_json(src)
    assert enc == '{"a":1.0,"b":"x y","c":[1e5,-0.50]}'
    assert codec._eq(codec.decode_json(enc), codec._loads(src))


def test_json_table_roundtrip():
    rows = [{"id": i, "name": f"item {i}", "ok": i % 2 == 0, "note": None if i else "a,b"} for i in range(6)]
    src = json.dumps(rows, indent=2)
    enc = codec.encode_json(src)
    assert enc.startswith("[6]{id,name,ok,note}:")
    assert codec._eq(codec.decode_json(enc), codec._loads(src))


def test_json_table_quotes_ambiguous_cells():
    rows = [{"v": "true"}, {"v": "12"}, {"v": " pad"}, {"v": "x"}]
    enc = codec.encode_json(json.dumps(rows))
    assert codec._eq(codec.decode_json(enc), codec._loads(json.dumps(rows)))


def _check(text):
    r = codec.encode(text)
    base = codec.render(text)
    if "json" in r.steps:
        assert codec._eq(codec.decode_json(r.text), codec._loads(base.strip()))
    elif set(r.steps) - {"render"}:
        assert codec.decode(r.text) == base, (text, r.text)
    else:
        assert r.text == base
    assert r.tokens <= cost.estimate(base)


def test_marker_lines_in_input_are_never_misdecoded():
    tricky = [
        "a\n... (line above repeated 3 more times)\nb",
        "x\ny\n... (previous 2 lines repeated 1 more times)",
        "Note:\n  12: something\nother:\n  3-ctx",
        "p.py:1:a\np.py:2:b\n... (line above repeated 2 more times)",
    ]
    for t in tricky:
        _check(t)
        _check(t + "\n" + "\n".join(["dup line here"] * 30))


def test_fuzz_roundtrip():
    rng = random.Random(1234)
    atoms = ["ok", "warning: deprecated", "src/a.py:12:x = 1", "src/a.py:13:y = 2", "C:\\w\\b.ts:4:z",
             "", "  12: indented", "Note:", "... (line above repeated 2 more times)", "\x1b[31mred\x1b[0m",
             "progress 10%\rprogress 90%", "{\"a\": 1}", "--", "x" * 50]
    for _ in range(3000):
        n = rng.randint(1, 60)
        lines = []
        for _ in range(n):
            a = rng.choice(atoms)
            lines.extend([a] * (rng.randint(2, 6) if rng.random() < 0.2 else 1))
        _check("\n".join(lines))
