<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) 2026 John William Creighton (s243a) -->

# Local-agent prompt — supervised Gemini pipeline for the C/C++ WAM targets

> For a Claude Code session ON THE USER'S PC. Aligned with the house
> pattern in `agent-automation-admin/skills/sanitized-agy-review` (read by
> the coordinator from a clone): tool-capable runs go through
> `mcp-acp-bridge` + `agy-dual-gated` with a spawn supervisor wrapper,
> `env -i` + dedicated Agy home, and the established gamma pacing profile.
> The delta this round builds: a Haiku-backed operator wrapper and an
> IMPLEMENTATION policy (packet-scoped writes) extending the skill's
> review-scoped default-deny. The cloud coordinator merges resulting
> branches via clean-extend + independent re-verification.

---

## Copy from here ↓↓↓

You are Claude Code running locally on my PC. Goal: run **Gemini 4.8**
("Agy") as a supervised implementation agent on my UnifyWeaver clone,
following my house pattern, with **Claude Haiku** as the operator wrapper.

### Step 0 — conventions first
Read, in order:
1. `~/Projects/agent-automation-admin/skills/sanitized-agy-review/SKILL.md`
   and its `references/supervision.md` — the binding pattern.
2. The docs it points to: `s243a/mcp-acp-bridge` `docs/agents.md` +
   `docs/supervisor.md`; `s243a/peerhailer` `docs/composer.md`;
   `s243a/t3code` `docs/fork/agy-broker-pty.md`.
Summarize back to me the supervision contract (wrapper protocol,
fail-closed rules, the pacing profile) and flag anything below that
conflicts with it BEFORE building.

### Step 1 — the Haiku operator wrapper + implementation policy
Per supervision.md, the bridge owns the permission gate and the spawn
supervisor wrapper receives `{tool,args}` and outputs
`approve | reject | pass`. Build:

1. **`haiku-operator` wrapper**: forwards each `{tool,args}` plus the
   active PACKET POLICY (below) and a rolling decision log to Claude
   Haiku; Haiku returns approve/reject/pass with a one-line reason
   (logged). HARD RULES enforced in the wrapper itself, not delegated to
   the model: timeout, crash, malformed output, and silence NEVER
   approve (skill rule); `pass` (defer to my UI) for anything the policy
   marks human-tier; independently scrubbed credentials — the wrapper's
   Anthropic key lives in ITS environment only, never the Agy child's
   (`env -i` + allowlist per the skill; supervisor and reviewer
   environments stay separate).
2. **Implementation packet policy** (new, extends the skill's
   review-mode default-deny): writes permitted ONLY at workspace-
   confined, symlink-aware paths matching the packet's ownership list;
   shell permitted ONLY for the packet's enumerated exact command forms
   (each packet lists its build/test commands verbatim — e.g.
   `swipl -q -g run_tests -t halt tests/test_wam_cpp_*.pl`, the target's
   `build.sh`, `gcc`/`g++` on generated files); git commit/push
   permitted only to the packet's named branch. Everything else: reject
   (in-policy denials) or pass (novel/ambiguous → me). Escalate-to-me
   always: destructive commands, anything touching credentials/`~/.ssh`,
   edits outside ownership, pushes to other branches, the same request
   repeating >3 times.
3. **Verify the model slug** with `agy models` (never silently
   substitute — skill rule); record the exact Gemini 4.8 slug + effort
   in every run report. Run under the auth arrangement that permits
   automated use.
4. **Pacing**: the established profile verbatim —
   `--supervisor-timing '{"min":2000,"max":30000,"dist":"gamma","shape":2}'`
   (latency only; no authorization effect — keep supervision.md's
   framing in the README).
5. **Optional middle tier**: if a decision is above Haiku but below me,
   allow one Sonnet low-reasoning consult; log which tier answered every
   request. Reporting per the skill: model+effort, capability mode,
   inherited env NAMES (no values), tool/permission counts, and
   independently verified results.

### Step 2 — the work packets (UnifyWeaver C/C++ campaign)
UnifyWeaver clone on `claude/peerhailer-exploratory-docs-aodas5`,
freshly pulled; SWI runs under `LC_ALL=C.UTF-8`. Each packet is one
markdown file (task, ownership, frozen list, enumerated commands, gates,
branch) seeded as Agy's prompt. Pilot with Packet 1 ONLY, me watching.

**Packet 1 — wam_cpp frameless-Y verdict (the pilot; small, bounded).**
The A2 frameless-Y ITE-barrier hazard was fixed on Rust/Python/Haskell/Go
and verdicted on R/F#/Lua/LLVM; `cpp` is one of two flag-passers never
re-audited. Read ledger rows D50/D52/D53
(`docs/proposals/JS_TARGETS_PARITY_PUNCHLIST.md`), the fleet-gaps A2
rows, and `tests/test_wam_python_frameless_ite_level.pl` (probe style).
Reproduce-or-rule-out on wam_cpp (`get_level` in an Allocate-less ITE
clause under `ite_use_y_level(true)`, caller with live Y1); verdict =
broken-fixed (CP-levels model per the Rust/Python/Go precedent) /
safe-by-design / unreachable — probe-pinned either way as
`tests/test_wam_cpp_frameless_ite_level.pl`; update
`docs/WAM_CPP_STATUS.md` + the fleet-gaps cpp cell.
Owns: `wam_cpp_target.pl`, `wam_cpp_lowered_emitter.pl`, cpp
templates/runtime files, `tests/test_wam_cpp_*.pl` (additive), those two
docs. Frozen: everything else. Gates: existing cpp suites green
(pre-existing failures verified pristine-identical); new probe green,
red-on-pristine if a fix landed. Branch: `gemini/wam-cpp-frameless-y`.

**Packet 2 — §9 cut-semantics corpora for cpp, then c.** Port the
35-probe corpus (`tests/test_wam_javascript_cut_semantics.pl` source;
Go/Rust ports show the pattern) to `tests/test_wam_cpp_cut_semantics.pl`
and `tests/test_wam_c_cut_semantics.pl`; SWI-oracled; uncompilable
shapes refuse LOUDLY; fix what the probes catch (Rust's port surfaced 11
defect classes — expect finds). Branches: `gemini/wam-cpp-cut-semantics`,
`gemini/wam-c-cut-semantics`.

**Packet 3 — wam_c register bounds.** Fixed 256-slot register arrays,
no bounds check (fleet-gaps c row): add checked access or sized
allocation + a probe that would have crashed.
Branch: `gemini/wam-c-register-bounds`.

**Packet 4 (after 1–3 merge) — resolver P3 ports for c/cpp** per the
D61 Go playbook (builtin inventory first; corpus 51/51; differential
2600/0).

### Rules binding every packet
- Never edit `wam_target.pl`, `wam_text_parser.pl`, shared harness,
  other targets, or `examples/` outside a packet's ownership.
- Wrong output is worse than refusal; existing suites are the gate.
- No AI-model identifiers in code, comments, or commit messages (repo
  rule); I'll specify the attribution trailer for these commits.
- Push the packet branch when its gates pass; the cloud coordinator
  re-verifies independently before merging — a green local report is
  necessary, not sufficient.

### Step 3 — pilot
Run Packet 1 end-to-end with me present: env self-test (child env dump →
names logged, no `ANTHROPIC_*`/`CLAUDE_*`/foreign keys), slug check,
seed, supervise to completion or clean failure; show me the decision log,
branch diff, and gate outputs. No Packet 2 until I approve.

### Report back (relayed to the cloud coordinator)
Per packet: branch + head commit, verdict/results, gate outputs, the
skill-format run report (model/effort, mode, env names, request counts,
tier-per-decision stats, pacing profile), and anything the policy got
wrong.

## ↑↑↑ Copy to here
