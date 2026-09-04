<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) 2026 John William Creighton (s243a) -->

# Local-agent prompt — Gemini 4.8 supervised pipeline for the C/C++ WAM targets

> For a Claude Code session ON THE USER'S PC (not the cloud coordinator).
> That machine has: `~/Projects/agent-automation-admin/` (incl.
> `skills/sanitized-agy-review`), the `gemini` CLI with the user's Google
> credentials, and a clone of `s243a/UnifyWeaver`. The cloud coordinator
> (this branch's ledger-keeper) merges whatever branches this pipeline
> produces via the usual clean-extend + local-gate discipline.

---

## Copy from here ↓↓↓

You are Claude Code running locally on my PC. Your job: build and pilot a
SUPERVISED agent pipeline in which **Gemini 4.8** (via the `gemini` CLI,
which has no auto mode) does implementation work on my UnifyWeaver clone,
with **Claude Haiku** as the interactive supervisor answering the CLI's
prompts, and produce branches the UnifyWeaver cloud coordinator can merge.

### Step 0 — read my conventions FIRST
Read `~/Projects/agent-automation-admin/skills/sanitized-agy-review` in
full. It is the house pattern for supervised agent runs — including the
**gamma-distribution human-answer-delay method**: use ITS method and
parameters, do not invent your own. Summarize back to me (a) the skill's
supervision loop, (b) its delay method, (c) anything in it that conflicts
with the requirements below, before building anything.

### Step 1 — the harness (in `~/Projects/agent-automation-admin/`)
Build a supervisor that:
1. **Drives `gemini` through a pty** (pexpect or equivalent), detecting
   its interactive prompts and submitting answers.
2. **Haiku as the supervisor model**: each prompt from the CLI (plus a
   rolling transcript window) goes to Claude Haiku, which decides the
   answer (approve / deny / steer) per a written policy file. Haiku is
   for high-frequency, low-stakes decisions; it must ESCALATE rather
   than answer when: the agent proposes a destructive command (rm -rf,
   force-push, chmod on system paths, anything touching credentials or
   `~/.ssh`), edits outside the packet's ownership list, a push to any
   branch not named in the packet, network access beyond git/google, or
   the same prompt recurring >3 times (loop). Escalation = pause and ask
   ME (or, if I've enabled it, a single Sonnet-4.5-class review call at
   LOW reasoning for judgment calls that are above Haiku but below
   human — log which tier answered every prompt).
3. **Gamma-distributed answer delays** per the skill, applied to every
   submitted answer. Note in the harness README: delays are for TUI
   robustness and polite pacing; the Gemini access itself must run under
   an auth arrangement that permits automated use (API-key tier) — the
   harness never pretends to be anything it isn't.
4. **Environment hygiene (hard requirement)**: the `gemini` subprocess
   gets a WHITELISTED environment only — `PATH`, `HOME`, `TERM`, locale,
   and Google's own variables. No `ANTHROPIC_*`, no `CLAUDE_*`, no
   `*_API_KEY` other than Google's, no session tokens. Whitelist, not
   blacklist; assert it in a startup self-test that dumps the child env
   to the log and greps it clean.
5. **Full transcript logging** (pty stream + supervisor decisions +
   delays + escalations) per run, in the skill's log layout if it has
   one.
6. **Work-packet interface**: a packet is a markdown file giving the
   Gemini agent its task, ownership list, frozen list, gates, and target
   branch. The supervisor seeds the Gemini session with the packet and
   keeps it on-task (steer answer: "stay within the packet").

### Step 2 — the first work packets (UnifyWeaver C/C++ campaign)
My UnifyWeaver clone should be on
`claude/peerhailer-exploratory-docs-aodas5`, freshly pulled. Packets in
order — pilot with Packet 1 ONLY, me watching:

**Packet 1 — wam_cpp frameless-Y verdict (small, bounded — the pilot).**
The A2 "frameless-Y ITE barrier" hazard was fixed on Rust/Python/Haskell/
Go and verdicted on R/F#/Lua/LLVM; `cpp` is one of two flag-passers never
re-audited. Task: read ledger rows D50/D52/D53 in
`docs/proposals/JS_TARGETS_PARITY_PUNCHLIST.md`, the fleet-gaps A2 rows,
and `tests/test_wam_python_frameless_ite_level.pl` (probe style).
Reproduce-or-rule-out on wam_cpp (the shape: `get_level` in an
`Allocate`-less ITE clause under `ite_use_y_level(true)`, caller with
live Y1); verdict = broken-fixed (CP-levels model, per the Rust/Python/Go
precedent) / safe-by-design / unreachable — probe-pinned either way as
`tests/test_wam_cpp_frameless_ite_level.pl`; update `docs/WAM_CPP_STATUS.md`
+ the fleet-gaps cpp cell. Owns: `wam_cpp_target.pl`,
`wam_cpp_lowered_emitter.pl`, cpp templates/runtime files,
`tests/test_wam_cpp_*.pl` (additive), the two docs. Frozen: everything
else. Gates: existing cpp suites green (pre-existing failures verified
pristine-identical); new probe green (red-on-pristine if a fix landed).
Branch: `gemini/wam-cpp-frameless-y`.

**Packet 2 — §9 cut-semantics probe corpora for cpp, then c.** Port the
35-probe corpus (`tests/test_wam_javascript_cut_semantics.pl` is the
source; Go/Rust ports show the pattern) to `tests/test_wam_cpp_cut_semantics.pl`
and `tests/test_wam_c_cut_semantics.pl`; SWI-oracled; probes whose shapes
a target cannot compile refuse LOUDLY. Fix what the probes catch (the
Rust port found 11 defect classes; expect finds). Branches:
`gemini/wam-cpp-cut-semantics`, `gemini/wam-c-cut-semantics`.

**Packet 3 — wam_c register bounds.** The C runtime has fixed 256-slot
register arrays with no bounds check (fleet-gaps c row); a >256-X fact
indexes out of bounds. Add checked access or sized allocation + a probe
that would have crashed. Branch: `gemini/wam-c-register-bounds`.

**Packet 4 (later, after 1–3 merge) — resolver P3 ports for c/cpp**, per
the D61 Go playbook (builtin inventory first; corpus 51/51; differential
2600/0).

### Rules that bind every packet
- Run SWI under `LC_ALL=C.UTF-8`. Never edit `wam_target.pl`,
  `wam_text_parser.pl`, `examples/` outside a packet's ownership,
  other targets' files, or the shared harness.
- Wrong output is worse than refusal; existing suites are the gate.
- No AI-model identifiers in code, comments, or commit messages (repo
  rule). Commit messages: conventional, descriptive; I'll tell you what
  attribution trailer to use for Gemini-authored commits.
- Push each packet's branch to origin when its gates pass; the cloud
  coordinator re-verifies everything independently before merging, so
  a green report here is necessary, not sufficient.

### Step 3 — pilot
Run Packet 1 end-to-end with me present: harness self-test (env dump
clean), seed the packet, supervise to completion or clean failure, show
me the transcript + the branch diff + the gate results. Do not start
Packet 2 until I approve the pilot.

### Report back (I'll relay it to the cloud coordinator)
Per packet: branch name + head commit, verdict/results, gate outputs,
supervisor stats (prompts answered by Haiku / escalated / to whom,
delay distribution used), and anything the supervision policy got wrong.

## ↑↑↑ Copy to here
