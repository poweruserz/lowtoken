import json
import os
import re
import subprocess
import sys
import uuid

import pytest

from lowtoken import codec, dedup, hook

MARK = re.compile(r"^\[lowtoken: ")


class Session:
    """A fake Claude Code session: writes the transcript the way Claude Code does."""

    def __init__(self, tmp_path):
        self.sid = "test-" + uuid.uuid4().hex
        self.transcript = str(tmp_path / f"{self.sid}.jsonl")
        open(self.transcript, "w").close()

    def _append(self, d):
        d.setdefault("cwd", "C:/p")  # Claude Code writes the working directory on every line
        with open(self.transcript, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(d) + "\n")

    def call(self, name, inp, resp, shown_override=None):
        tid = "toolu_" + uuid.uuid4().hex[:12]
        self._append({"type": "assistant", "message": {"content": [
            {"type": "tool_use", "id": tid, "name": name, "input": inp}]}})
        out = hook.post({"session_id": self.sid, "transcript_path": self.transcript, "tool_use_id": tid,
                         "tool_name": name, "tool_input": inp, "tool_response": resp})
        new = resp
        if out:
            hs = out["hookSpecificOutput"]
            new = hs.get("updatedToolOutput", hs.get("updatedMCPToolOutput"))
        self._append({"type": "user", "message": {"content": [
            {"type": "tool_result", "tool_use_id": tid, "content": shown_override or visible(name, new)}]}})
        return out, new

    def compact(self):
        self._append({"type": "system", "subtype": "compact_boundary"})


def visible(name, resp):
    if name == "Read":
        f = resp["file"]
        return "\n".join(f"{f['startLine'] + i:>6}\t{x}" for i, x in enumerate(f["content"].split("\n")))
    if name in ("Bash", "PowerShell"):
        return resp.get("stdout", "") + (("\n" + resp["stderr"]) if resp.get("stderr") else "")
    if isinstance(resp, list):
        return "\n".join(b.get("text", "") for b in resp)
    return str(resp)


@pytest.fixture
def s(tmp_path, monkeypatch):
    monkeypatch.setenv("LOWTOKEN_HOME", str(tmp_path))
    return Session(tmp_path)


def _file(n=200):
    return [f"def function_number_{i}(argument): return compute_value({i}) * 2" for i in range(1, n + 1)]


def read_resp(lines, start=1, total=None, path="C:/p/app.py"):
    return {"type": "text", "file": {"filePath": path, "content": "\n".join(lines), "numLines": len(lines),
                                     "startLine": start, "totalLines": total or start + len(lines) - 1}}


def assert_true_numbers(new, truth):
    f = new["file"]
    for i, line in enumerate(f["content"].split("\n")):
        if not MARK.match(line):
            assert truth[f["startLine"] + i - 1] == line


def test_read_after_edit_keeps_every_line_number(s):
    v1 = _file()
    out, _ = s.call("Read", {"file_path": "C:/p/app.py"}, read_resp(v1))
    assert out is None
    v2 = v1[:120] + ["    # a new comment line inserted by an edit"] * 3 + v1[120:]
    s.call("Edit", {"file_path": "C:/p/app.py", "old_string": v1[119], "new_string": v2[119] + "\n" + v2[120]},
           "The file has been updated.")
    out, new = s.call("Read", {"file_path": "C:/p/app.py"}, read_resp(v2))
    shown = new["file"]["content"].split("\n")
    assert len(shown) < 10
    assert_true_numbers(new, v2)
    assert "unchanged since Read #1 of app.py" in shown[0]
    assert any("they were lines" in x for x in shown)


def test_nothing_is_left_after_a_suffix_marker(s):
    # regression: trailing blank lines kept after the marker were shown under shifted numbers
    v1 = [x if i % 7 else "" for i, x in enumerate(_file(300))]
    s.call("Read", {"file_path": "C:/p/app.py"}, read_resp(v1))
    v2 = v1[:150] + ["inserted line that shifts the rest"] * 2 + v1[150:]
    for start, limit in [(100, 80), (140, 61), (1, 302), (200, 49)]:
        part = v2[start - 1:start - 1 + limit]
        out, new = s.call("Read", {"file_path": "C:/p/app.py"}, read_resp(part, start, 302))
        if out:
            assert_true_numbers(new, v2)
            assert not new["file"]["content"].split("\n")[-1] == ""


def test_third_read_follows_markers_back_to_the_text(s):
    v1 = _file()
    s.call("Read", {"file_path": "C:/p/app.py"}, read_resp(v1))
    v2 = v1[:100] + ["changed line one hundred and one"] + v1[101:]
    s.call("Read", {"file_path": "C:/p/app.py"}, read_resp(v2))
    v3 = v2[:150] + ["changed line one hundred and fifty one"] + v2[151:]
    out, new = s.call("Read", {"file_path": "C:/p/app.py"}, read_resp(v3))
    assert_true_numbers(new, v3)
    shown = new["file"]["content"].split("\n")
    # line 101 was shown by read #2, the rest of the prefix by read #1: one marker cites both
    assert shown[0] == ("[lowtoken: lines 1-150 unchanged since you saw them "
                        "(1-100: Read #1 of app.py; 101: Read #2 of app.py; 102-150: Read #1 of app.py)]")


def test_read_after_write_is_one_marker(s):
    body = _file(80)
    s.call("Write", {"file_path": "C:/p/new.py", "content": "\n".join(body)}, "File created.")
    out, new = s.call("Read", {"file_path": "C:/p/new.py"}, read_resp(body, path="C:/p/new.py"))
    shown = new["file"]["content"].split("\n")
    assert len(shown) == 1 and "unchanged since your Write of new.py" in shown[0]
    assert new["file"]["startLine"] == 80  # the marker carries the last omitted number


def test_output_replaced_by_another_hook_is_never_referenced(s):
    v1 = _file()
    s.call("Read", {"file_path": "C:/p/app.py"}, read_resp(v1), shown_override="[indexed by another plugin]")
    out, _ = s.call("Read", {"file_path": "C:/p/app.py"}, read_resp(v1))
    assert out is None
    out, new = s.call("Bash", {"command": "cat app.py"}, {"stdout": "\n".join(v1[:60]), "stderr": ""})
    assert "Read #1" not in new["stdout"]       # read #1 only showed the other plugin's stub
    assert "of Read #2 of app.py" in new["stdout"]  # read #2 really showed these lines


def test_bash_repeat_is_elided_then_shown_whole(s):
    body = "\n".join(_file(60))
    s.call("Read", {"file_path": "C:/p/app.py"}, read_resp(body.split("\n"), 1, 60))
    out, new = s.call("Bash", {"command": "cat app.py"}, {"stdout": body, "stderr": ""})
    assert "60 lines omitted, identical to lines 1-60 of Read #1 of app.py" in new["stdout"]
    out, new = s.call("Bash", {"command": "cat app.py"}, {"stdout": body, "stderr": ""})
    assert new["stdout"] == body  # asked again: Claude gets it whole


def test_compaction_resets_store(s):
    body = "\n".join(_file(60))
    s.call("Bash", {"command": "cat a"}, {"stdout": body, "stderr": ""})
    s.compact()
    out, _ = s.call("Bash", {"command": "cat b"}, {"stdout": body, "stderr": ""})
    assert out is None


def test_no_transcript_means_no_dedup(tmp_path, monkeypatch):
    monkeypatch.setenv("LOWTOKEN_HOME", str(tmp_path))
    body = "\n".join(_file(60))
    for _ in range(2):
        out = hook.post({"session_id": "x", "tool_name": "Bash", "tool_input": {"command": "cat a"},
                         "tool_response": {"stdout": body, "stderr": ""}})
        assert out is None


def test_mcp_json_table(s):
    rows = [{"id": i, "title": f"Issue number {i}", "state": "open", "comments": i * 3} for i in range(40)]
    out, new = s.call("mcp__github__list_issues", {}, [{"type": "text", "text": json.dumps(rows, indent=2)}])
    assert new[0]["text"].startswith("[40]{id,title,state,comments}:")


def _usage(s, ctx):
    s._append({"type": "assistant", "message": {"id": "m" + uuid.uuid4().hex[:8], "content": [], "usage": {
        "input_tokens": 0, "cache_read_input_tokens": ctx, "cache_creation_input_tokens": 0, "output_tokens": 1}}})


def test_prefix_marker_can_cite_a_file_read_in_pieces(s):
    v1 = _file(300)
    s.call("Read", {"file_path": "C:/p/app.py"}, read_resp(v1[:100], 1, 300))
    s.call("Read", {"file_path": "C:/p/app.py"}, read_resp(v1[100:220], 101, 300))
    v2 = v1[:250] + ["a line changed near the end"] + v1[251:]
    out, new = s.call("Read", {"file_path": "C:/p/app.py"}, read_resp(v2))
    shown = new["file"]["content"].split("\n")
    assert shown[0] == ("[lowtoken: lines 1-220 unchanged since you saw them "
                        "(1-100: Read #1 of app.py; 101-220: Read #2 of app.py)]")
    assert_true_numbers(new, v2)


def test_command_labels_drop_the_cd_prefix(s):
    body = "\n".join(_file(60))
    s.call("Bash", {"command": 'cd "C:/p/some dir" && PYTHONPATH=. python gen.py'}, {"stdout": body, "stderr": ""})
    out, new = s.call("Bash", {"command": "cat app.py"}, {"stdout": body, "stderr": ""})
    assert "of the output of `python gen.py`" in new["stdout"]


def test_a_shrinking_context_forgets_everything(s):
    v1 = _file()
    _usage(s, 100_000)
    s.call("Read", {"file_path": "C:/p/app.py"}, read_resp(v1))
    _usage(s, 120_000)
    out, _ = s.call("Read", {"file_path": "C:/p/app.py"}, read_resp(v1[:150], 1, 200))
    assert out is not None                      # still in context: trimmed
    _usage(s, 60_000)                           # something was dropped from the context
    out, _ = s.call("Read", {"file_path": "C:/p/app.py"}, read_resp(v1[:120], 1, 200))
    assert out is None


def test_rereading_a_trimmed_read_gives_it_whole(s):
    v1 = _file()
    s.call("Read", {"file_path": "C:/p/app.py"}, read_resp(v1))
    v2 = v1[:100] + ["changed"] + v1[101:]
    out, _ = s.call("Read", {"file_path": "C:/p/app.py"}, read_resp(v2))
    assert out is not None
    out, _ = s.call("Read", {"file_path": "C:/p/app.py"}, read_resp(v2))
    assert out is None                          # same read again: Claude could not find it, so no markers


def test_subagent_never_sees_markers_for_the_parents_reads(s):
    v1 = _file()
    s.call("Read", {"file_path": "C:/p/app.py"}, read_resp(v1))
    out = hook.post({"session_id": s.sid, "agent_id": "a1", "transcript_path": s.transcript,
                     "tool_name": "Read", "tool_input": {"file_path": "C:/p/app.py"}, "tool_response": read_resp(v1)})
    assert out is None
    out = hook.post({"session_id": s.sid, "agent_id": "a1", "transcript_path": s.transcript, "tool_name": "Bash",
                     "tool_input": {"command": "cat app.py"}, "tool_response": {"stdout": "\n".join(v1), "stderr": ""}})
    assert out is None or "lowtoken: " not in out["hookSpecificOutput"]["updatedToolOutput"]["stdout"]


def test_read_labels_count_failed_and_reordered_calls(s):
    v1 = _file()
    s.call("Read", {"file_path": "C:/p/app.py"}, read_resp(v1), shown_override="File content exceeds limit")
    s._append({"type": "user", "message": {"content": [{"type": "tool_result", "tool_use_id": "x", "is_error": True,
                                                         "content": "error"}]}})
    s.call("Read", {"file_path": "C:/p/app.py"}, read_resp(v1))          # Claude's 2nd Read of the file
    out, new = s.call("Bash", {"command": "cat app.py"}, {"stdout": "\n".join(v1[:50]), "stderr": ""})
    assert "Read #2 of app.py" in new["stdout"]


def test_numbered_lines_that_skip_are_not_called_a_range(s):
    numbered = "\n".join(f"{n:>6}\tdef function_{n}(self, request, response): return None" for n in range(12, 400, 17))
    s.call("Bash", {"command": "cat -n a.py | grep def"}, {"stdout": numbered, "stderr": ""})
    out, new = s.call("Bash", {"command": "cat -n a.py | grep def"}, {"stdout": numbered + "\n", "stderr": ""})
    assert "lines 12-" not in new["stdout"]


def test_grep_text_is_not_rendered_as_a_terminal(s):
    content = "\n".join(f"src/legacy/data.txt:{n}:header value {n}" for n in range(1, 40)) + "\rX"
    out, new = s.call("Grep", {"pattern": "header"}, {"mode": "content", "content": content, "numLines": 39})
    assert codec.decode(new["content"]) == content


def test_fold_header_counts_the_lines_before_dedup(s):
    body = _file(200)
    s.call("Read", {"file_path": "C:/p/app.py"}, read_resp(body))
    out, new = s.call("Bash", {"command": "tool"}, {"stdout": "\n".join(body + ["retry"] * 100), "stderr": ""})
    assert new["stdout"].startswith("[lowtoken: 300 lines; repeated lines folded]")


def test_held_lock_means_codec_only_and_no_store_write(s):
    body = "\n".join(_file(60))
    s.call("Read", {"file_path": "C:/p/app.py"}, read_resp(body.split("\n"), 1, 60))
    st = dedup.Store(s.sid, load=False)
    assert st.lock()
    try:
        out, new = s.call("Bash", {"command": "cat app.py"}, {"stdout": body, "stderr": ""})
        assert new["stdout"] == body  # no dedup without the lock
    finally:
        st.unlock()


def test_unwritable_log_never_breaks_the_hook(tmp_path):
    env = dict(os.environ, LOWTOKEN_LOG=str(tmp_path))  # a directory: open() fails
    p = subprocess.run([sys.executable, "-m", "lowtoken", "hook", "post"], input=b"not json", capture_output=True,
                       env=env)
    assert p.returncode == 0 and p.stdout == b""


def test_bad_input_never_breaks_the_tool_call():
    for payload in [b"not json", b"{}", b'{"tool_name":"Read","tool_response":{"type":"text","file":5}}',
                    b'{"tool_name":"Bash","transcript_path":"/nonexistent","tool_response":{"stdout":7}}']:
        p = subprocess.run([sys.executable, "-m", "lowtoken", "hook", "post"], input=payload, capture_output=True)
        assert p.returncode == 0 and p.stdout == b""


@pytest.mark.skipif(os.name == "nt", reason="POSIX ownership and symlink checks")
def test_state_dir_refuses_a_symlink_and_tightens_permissions(tmp_path, monkeypatch):
    real = tmp_path / "real"
    real.mkdir(mode=0o755)
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    monkeypatch.setenv("LOWTOKEN_HOME", str(link))
    assert dedup.state_dir() is None                    # a planted symlink is never followed
    monkeypatch.setenv("LOWTOKEN_HOME", str(real))
    assert dedup.state_dir() == str(real)
    assert (os.stat(real).st_mode & 0o777) == 0o700    # others lose access


def test_no_safe_state_dir_means_codec_only(monkeypatch):
    monkeypatch.setattr(dedup, "state_dir", lambda: None)
    body = "\n".join(["warn: retrying connection"] * 50)
    out = hook.post({"session_id": "x", "tool_name": "Bash", "tool_input": {"command": "x"},
                     "tool_response": {"stdout": body, "stderr": ""}})
    assert out["hookSpecificOutput"]["updatedToolOutput"]["stdout"].startswith("[lowtoken: 50 lines;")
