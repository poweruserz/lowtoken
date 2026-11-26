# lowtoken

**A coding agent does not pay for a token once. It pays rent on it, on every call that follows.**

lowtoken measures that rent exactly, with Claude's own tokenizer, for free, from the transcripts
Claude Code already writes. Then it cuts the rent without changing what Claude knows: every
rewrite it makes either decodes back to the original byte for byte, or points at text the
transcript proves Claude already has.

```
lowtoken audit · last 30 days · 170 sessions · 40,670 API calls

WHERE THE MONEY GOES  (generation + cache write + every later re-read)
  ######..................  25.7%  Model output (text, tool calls, file writes)
  ######..................  23.7%  Startup context (system prompt, tools, CLAUDE.md, skills)
  #####...................  21.1%  Model thinking (kept in context)
  ####....................  15.2%  Tool results
  ##......................   7.5%  Injected text (hooks, reminders, skill bodies)
  #.......................   5.6%  Cache rebuilds (prefix lost, re-written)
  ........................   1.3%  Your prompts
```

<sub>Real output from one heavy Claude Code user's 30 days (the tool also prints dollar amounts at
list prices; they are left out here). Everything below was measured on the same data unless it
says otherwise.</sub>

## The problem in four numbers

1. **80% of the bill is the context, not the work.** Each call resends the whole context: re-reading
   it from the prompt cache is 57% of the bill and writing it to the cache 24%. Generating output is 20%.
2. **A tool-output token is re-read 129 times on average** (weighted by size) before the session
   ends or compacts.
3. **The Lindy effect.** Session lengths are heavy-tailed, so the longer a session has run, the longer
   it is expected to keep running. A token added at call 1 is re-read about 270 more times; one added
   at call 100, about 960 times. The same `cat` costs several times more late in a session.
4. **Everyone counts tokens wrong for Claude.** On 2,398 real tool outputs, prompts and replies,
   counted exactly with Claude's tokenizer: `chars / 4` (what RTK reports) undercounts by **48%**,
   OpenAI's o200k (what TOON and repomix report) by **34%**.

## What lowtoken does

| Command | What it answers |
|---|---|
| `lowtoken audit` | Where did my money go? Exact bill from the API's own `usage` blocks, then every item that entered the context (each tool result, hook injection, the thinking, the startup context) charged its rent: one cache write plus every later re-read, until a compaction drops it. Ranks the most expensive single tool results and flags waste. |
| `lowtoken diet` | What do I load on every call and never use? Exact per-agent, per-skill, per-MCP-server token cost (free, via `/context`) crossed with 30 days of real usage. `--apply` switches off the plugins that are safe to drop (backed up, `--undo` restores). |
| hook (plugin) | Shrinks Bash, PowerShell, Read, Grep and MCP output before Claude reads it, losslessly (below). |
| `lowtoken count --exact` | Exact Claude token count of any text, no API key, $0. |
| `lowtoken statusline` | `ctx 300.0k · rent $0.060/call · next 100 calls $6.00 · /compact pays back in 14 calls · cache warm 37m · lowtoken -12.3k tok` (example, Opus 5.5 prices) |

`cache warm 37m` counts down the prompt cache's lifetime (1 hour on a subscription, 5 minutes on 5m
writes). Once it is cold, the next call re-writes the whole context at twice the input price instead of
reading it at the cache rate, 40 times more on Opus 5.5, and the status line says what that will cost.
Idle gaps caused most of the cache rebuilds measured above.

The `/compact` payback is the break-even of the rent model: compacting reads the context once, writes
a summary and re-caches a smaller context, after which every call reads less. It solves
`n (C - C') r = C r + H o + C' w` for `n`. Whether the summary keeps what you need stays your call.

### `audit` findings it surfaced on real data

* A third-party app's `UserPromptSubmit` hook written as `cmd.exe /d /c ...`. Claude Code runs hooks
  through Git Bash on Windows, whose path conversion turns `/c` into a file path, so cmd.exe started
  *interactively*, read the hook's JSON from stdin as a command, and echoed the cmd banner plus the
  whole prompt back into the context on every turn. The command after `/c` never ran at all. The same
  wrapper sat in a `Stop` hook. Cost: **1.8% of the bill** (3,164 + 2,402 injections), for hooks that
  did nothing. Fix under Git Bash: `cmd.exe //d //c ...` (in PowerShell or cmd the single-slash form is the right one).
* **Prompt-cache rebuilds: 5.7% of the bill**, with causes from the API's own `cache_miss_reason` where
  it gives one: 28.2M tokens re-written after idle gaps longer than the cache lifetime, 4.9M after the
  tool set changed mid-session, 3.0M after a `/model` switch.
* 177 single tool results over 8k tokens; the worst was read 222 times.

### `diet` on the same machine

Startup context was 44,500 tokens on every call. Components never used in 30 days, loaded anyway
(names replaced):

| component | tokens on every call | uses in 30 days | share of the bill |
|---|---:|---:|---:|
| plugin A (68 agents and skills) | 4,960 | 20, all in a few of its items | 1.1% |
| connector B (MCP server) | 3,097 | 0 | 0.7% |
| plugin C (7 agents) | 2,110 | 0 | 0.5% |
| plugin D (11 agents and skills) | 1,329 | 0 | 0.3% |
| **all unused components** | | | **4.2%** |

`diet --apply` switches a plugin off only when every one of its agents, skills and commands went
unused (short command names such as `/commit` count as use) and it offers nothing that acts on its
own: no hooks, MCP or LSP servers, output styles or status lines. On this machine that was 12 plugins
and 3,946 tokens on every call. It backs up `settings.json` first, and `--undo` switches back on
exactly those plugins and nothing else.

## The hook: lossless by construction

Three transforms, each checked on every call:

1. **Terminal rendering.** What a terminal would show: ANSI colours removed, `\r` progress bars
   collapsed to their final state. Colour codes alone made a typical test log 80% longer in Claude tokens.
2. **Verified folding.** Repeated lines and repeated blocks fold into one copy plus
   `... (line above repeated 39 more times)`, with a header that states the true line count;
   `grep -n` style output is grouped under each path; JSON is minified, and an array of same-shaped
   objects becomes a table. Each fold has an exact decoder. The hook encodes, decodes, compares,
   and keeps a fold only if the round trip is byte-identical (JSON: identical data, numbers compared
   as written). Fuzzed with 3,000 adversarial inputs, including inputs that already contain the markers.
3. **Deduplication against what Claude has already seen.** Agents re-show the same text constantly:
   a file read, then `cat`-ed, then `sed -n`-ed, then re-read after a one-line edit. A run of lines
   Claude already received becomes
   `[lowtoken: 59 lines omitted, identical to lines 1-59 of Read #1 of app.py]`.
   * **The transcript is the only source of truth.** Another plugin may have replaced an output, Claude
     Code may have sent a "file unchanged" stub, a compaction may have dropped everything. lowtoken reads
     the session transcript incrementally and only ever points at text it shows Claude received.
     (This was a real bug caught in an end-to-end test: another installed plugin was replacing Read output.)
   * **Read line numbers stay true.** Claude Code numbers Read output itself, so removing lines from the
     middle would renumber everything after. lowtoken only trims the start and end of a read; the
     marker takes the number of the last omitted line. Every shown line keeps its real number. A file
     read in pieces is covered by one marker that names each piece:
     `[lowtoken: lines 1-200 unchanged since you saw them (1-100: Read #1 of big.py; 101-200: Read #2 of big.py)]`.
     Reads are numbered as Claude made them, failed calls included.
   * **Text counts as "in context" until the context shrinks.** Every API call's context size is in
     the transcript. On 40,706 consecutive calls it shrank only on compaction, never by silent
     clearing; any drop at all resets what lowtoken may point at.
   * **Subagents are left alone.** They report their parent's session id but have their own context.
   * If Claude asks for the same output again, it gets it whole: no loop where a model keeps
     re-requesting text it cannot find.

### Does Claude still get the right answer?

Real `claude -p` sessions, isolated from the user's own plugins, run each scenario with and without
lowtoken. The questions target exactly the parts lowtoken folds or omits. Graded by exact match.

Six scenarios: a re-read after an edit, a file read in two pieces and then whole (one marker cites
both pieces), `sed` after Read, a big JSON file, a log full of repeated lines, `grep -rn` across files.

| Model | Without lowtoken | With lowtoken | Cost with lowtoken |
|---|---|---|---|
| Claude Haiku 5.5, 10 trials, 250 questions per arm | 229/250 (91.6%) | **250/250 (100%)** | **-10%** |
| Claude Sonnet 5.5, 4 trials, 100 questions per arm | 99/100 | **100/100** | **-12%** |

On Haiku the gap is not luck: 21 errors against 0, Fisher's exact test p = 3e-7. With 0 errors in 250,
the 95% upper bound on lowtoken's error rate there is 1.2%. On Sonnet both arms are near perfect and
the difference is not significant (p = 0.5): lowtoken costs less and loses nothing. Earlier, smaller
runs agree (Haiku 91/100 vs 100/100, 90/100 vs 100/100, 54/60 vs 60/60; Sonnet 50/50 vs 50/50, 39/40
vs 40/40). No answer was lost in any scenario on any model in any run. Where the plain arm failed
(counting repeated log lines), the folded form is not just shorter but easier: the model no longer
counts 31 identical lines by eye. Without lowtoken, Haiku answered 32 or 33 retries and 183 to 212
lines, and Sonnet once said 190; with lowtoken, 31 and 187, every time.

That came from a measured fix. An earlier version folded repeats without saying how many lines there
were, and Haiku sometimes reported the folded count (26) instead of the printed one (187). The header
`[lowtoken: 187 lines; repeated lines folded]` was added for that, and is part of the verified grammar.

### Does Claude still finish real tasks?

`bench/tasks.py` gives Claude five small repos, each with a bug and failing tests (a fix that needs two
edits far apart in a 400-line module, a test run buried in repeated warnings, a wrong key against a big
JSON config, a rename across five files, a log-counting bug). Success is decided afterwards by running
the tests ourselves; edited tests count as a failure.

| Model | Solved without | Solved with | Context tokens | Cost |
|---|---|---|---|---|
| Haiku 5.5, 25 runs per arm | 25/25 | **25/25** | +6% | same |
| Sonnet 5.5, 10 runs per arm | 10/10 | **10/10** | -9% | same |

Nothing broke. The token difference is noise from how the model chose to work: these tasks take
5-7 turns, so the startup context dominates and there is no time for rent to build up (lowtoken
rewrote an output in only 19 of the runs, and in no run did Claude repeat a call after a rewrite).
The savings live in long sessions, which is where the Lindy effect says the money is.

### Replay on real sessions

`bench/replay.py` feeds every tool result of 30 days of real transcripts through the hook, in order,
writing a synthetic transcript of what Claude would have seen, and checks every claim the hook makes:

524 transcripts (benchmark sessions excluded), 36,965 tool results, 2,969 of them rewritten:

* every omitted line is identical to the source line its marker names: **35,702 checked, 0 failures**
* every line a rewritten Read shows sits under its true line number: **8,959 checked, 0 failures**
* counted exactly with Claude's tokenizer, 300 rewritten outputs went from 306,648 to 218,165 tokens:
  **-28.9% per rewritten output** (the fast estimator predicted -31.3%, so it runs a little optimistic)
* hook time: 24 ms median, 95 ms p99 (store lock and transcript sync included), plus Python's start-up

| tool | results | tokens saved |
|---|---:|---:|
| Bash | 16,958 | 3.1% |
| MCP | 7,640 | 7.2% |
| PowerShell | 6,008 | 2.7% |
| Read | 6,359 | 2.1% |
| all | 36,965 | 3.5% |

Honest scale: most tool output is unique text, so lossless rewriting saves a few percent overall,
and about a quarter of each output it does touch. For most users the larger wins are in `audit` and `diet`.

Is there more to take from Read without losing anything? Measured at the hook, on 3,835 real Reads:
trimming leaves only 0.09% of Read tokens on the table (runs too short to pay for a marker). The rest
of a Read is text Claude has not seen at those line numbers, which only a lossy tool would drop.

Installed from the marketplace into a real Claude Code setup with ten other plugins (one of which
also rewrites tool output), the hook folded a test log and tabled a JSON file in a live session, and
Claude answered every question about them correctly, noting on its own that a layer had reformatted
the output.

## How it compares

Read from each project's source code (November 2026). Most tools in this space trade some output
away for savings; lowtoken only folds what decodes back or points at what the transcript proves
Claude already has.

| Tool | Approach | Counts tokens with | Lossless |
|---|---|---|---|
| RTK | ~160 per-command filters, PreToolUse command rewrite | chars / 4 | no (full output can be saved to a file) |
| snip, ztk | declarative / compiled per-command filters | chars / 4, bytes | no |
| kompressor | API proxy, lossless / recoverable / aggressive modes | estimate fitted to tiktoken | recoverable by a retrieve tool |
| context-mode | outputs over 4 KB replaced by matching chunks plus a search tool | bytes | no (recoverable by search) |
| headroom | API proxy, JSON crushing, compress-cache-retrieve | OpenAI tokenizers | mixed |
| token-optimizer | diff-only re-reads, archived outputs | chars / 4 | partly |
| lean-ctx | context gateway; a stub replaces re-reads of files unchanged on disk | tiktoken o200k | for that stub, yes |
| TOON | compact data format | o200k | yes |
| **lowtoken** | rent accounting + verified folding + transcript-checked dedup | **Claude's tokenizer, exact** | **yes, checked on every call** |

Re-read deduplication as such is not new: Claude Code itself, lean-ctx and token-optimizer all shorten
the re-read of a file that did not change. What lowtoken adds, as far as a code search of GitHub and a
source read of about 30 tools could find, is: references checked against the transcript (what Claude
actually received, not what the tool produced); re-reads of *changed* files trimmed so every shown
line keeps its true number, across several earlier reads; line-level references across tools (a
`cat` or `sed` of text an earlier Read showed); folds that are decoded and compared before use; per-item
rent accounting with the Lindy effect; and exact Claude token counts through `/context`.

kompressor's own 89-task A/B on Claude Code is worth knowing: its recoverable mode solved the same
tasks but cost 9% more per trial, because fetching dropped text back takes extra turns, and cache
re-reads were already cheap. That is the case for not dropping anything.

## How it works

### Rent

An item of `s` tokens added before call `k` of a session that ends (or compacts) at call `N` costs

```
rent = s * (w * p_in  +  (N - k) * p_read)        w = 2 for the 1-hour cache, 1.25 for 5 minutes
```

plus `s * p_out` if the model generated it. The audit computes this per item, exactly per call, from
the API `usage` blocks Claude Code logs. The growth of the context between two calls is known exactly
(the difference of two usage totals); only its split between tool results, prompts and injected text
is estimated, and that split is rescaled to the exact total. The attributed total lands within 1.3%
of the bill.

Accounting traps it avoids, most found by reading other trackers' code and bug reports: Claude Code
logs one usage line per content block and the early ones carry partial `output_tokens` (keeping the
first line misses 24% of output; lowtoken itself had this bug until a research pass caught it);
1-hour and 5-minute cache writes cost differently and can mix in one call; resumed and forked sessions
repeat earlier responses in new files; fast mode doubles the rates; web searches are billed per search.
Cache rebuilds are explained with the API's own `cache_miss_reason` when the transcript carries it.

`N - k` is unknown while a session runs. Its expectation, `E[N - k | N > k]`, is the mean residual life
of the session-length distribution. For light-tailed distributions it shrinks as `k` grows. For
heavy-tailed ones it grows (for a Pareto tail it grows linearly in `k`). Measured:

| a token added at call k | is re-read on average | sessions that get this far |
|---|---|---|
| 1 | 274 more calls | 87% |
| 10 | 612 | 38% |
| 100 | 962 | 22% |
| 200 | 1,113 | 17% |

### Exact Claude token counts for free

Claude's tokenizer is not public. But `claude -p /context` runs no model turn (cost $0) and prints the
exact token count of every loaded memory file, computed by the count_tokens endpoint. So lowtoken
writes each sample into its own `.claude/rules/*.md` file in a throwaway project, wrapped in a code
fence so imports and comments stay inert, runs one `/context`, and subtracts a one-token baseline.
600 samples take 24 seconds. lowtoken uses this only to count your own text. It does not and must not
be used to enumerate the tokenizer.

Some things it shows about the Claude 5 tokenizer (Opus 5.5, exact counts vs o200k):

| text | Claude | o200k |
|---|---:|---:|
| `"a" * 400` | 400 | 50 |
| English sentence (163 chars) | 46 | 31 |
| Turkish sentence (142 chars) | 73 | 37 |
| 10 Windows paths | 339 | 239 |
| 10 UUIDs | 275 | 239 |
| JSON, pretty / compact / as CSV | 198 / 122 / 62 | 172 / 97 / 51 |
| 4, 8 or 16 spaces of indent | same cost | same cost |
| `cat -n` line-number prefix | 3 tokens per line | |

So re-indenting code saves nothing, while structure (repeated prefixes, repeated lines, JSON
punctuation) and presentation noise are where lossless savings live. That is what the hook targets.

### The fast estimator

The hook cannot call `/context` on every tool call, so `lowtoken/cost.py` is a linear model over
cheap pre-token features (word-length buckets, digits, punctuation, whitespace runs, non-ASCII),
fitted on 2,398 exact counts: **9.1% error per item** (cross-validated), 5.8% with `tiktoken`
installed, under 0.1% bias in totals. It loads in milliseconds; the whole hook runs in about 30 ms
plus Python's start-up.

## Install

As a Claude Code plugin (the hook):

```
/plugin marketplace add poweruserz/lowtoken
/plugin install lowtoken@lowtoken
```

Python 3.9+ must be on `PATH` as `python3` or `python`. No packages needed.

The CLI (`audit`, `diet`, `count`, `statusline`):

```
pip install git+https://github.com/poweruserz/lowtoken
lowtoken audit
lowtoken diet
```

Status line, in `~/.claude/settings.json`:

```json
{ "statusLine": { "type": "command", "command": "lowtoken statusline" } }
```

## Limits

* `audit` counts what the transcripts record. Claude Code's own cost counter also includes side
  requests (titles, summaries) that are not in the transcript; for about two thirds of sessions the
  two agree within 1%.
* Prices are list prices cached on 2026-11-03 (`lowtoken/prices.py`). On a subscription, read them as
  a share of your plan's limits rather than dollars.
* The exact oracle depends on the format of `/context` output, which Claude Code may change.
* The estimator was fitted on Opus 5.5 counts. Refit it if the tokenizer changes.
* Deduplication assumes, as measured, that Claude Code only drops context by compacting. If a future
  version clears old tool output without shrinking the context, a marker could point at text that is
  gone; it would still say exactly where the text was and how to get it again.
* On short tasks (5-7 turns) the hook saves little: the startup context dominates. It pays off in the
  long sessions where most of the money goes.

## Privacy

Everything runs locally. Transcripts are read, never uploaded. The hook keeps its per-session state in
`~/.cache/lowtoken` (Windows: `%LOCALAPPDATA%\lowtoken`), a directory only your user can open; it
refuses to use a directory that is a symlink or owned by someone else, and then runs without
deduplication rather than trust state another account could have written. The exact counter sends your text to
Anthropic through your own Claude Code, exactly as `/context` does. The benchmarks print aggregates only.

## Reproduce

```
python -m pytest                                      # unit, property, adversarial and regression tests
python bench/replay.py --days 30 --exact-sample 300   # replay your own sessions through the hook
python bench/fidelity.py --model haiku --trials 3     # Q&A A/B with real Claude sessions (a few cents)
python bench/tasks.py --model haiku --trials 5        # bug-fix A/B: tests decide success, not Claude
```

The test suite runs on Python 3.9 through 3.15; CI covers Linux, macOS and Windows. Every fix that
came out of review has a regression test, and each of those tests was checked to fail when its fix
is reverted.

MIT licence.
