<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) 2026 John William Creighton (s243a) -->

# Grok prompt — Tier-2 lowered emitter for wam_javascript

> The biggest remaining WAM parity gap: 14 of 18 WAM targets have a per-predicate
> lowered emitter (Tier-2); `wam_javascript` is one of only 4 without. Branch from
> `grok/wamjs-indexing` (the latest JS-WAM state: interpreter + ISO bagof/setof +
> first-arg indexing) and EXTEND — do not rebuild.

---

## Copy from here ↓↓↓

You are contributing to **UnifyWeaver** (`github.com/s243a/unifyweaver`), a compiler
that lowers Prolog to a shared WAM bytecode and runs it on per-language WAM
interpreters. Branch from **`grok/wamjs-indexing`** (it has the full JS WAM:
interpreter tier, ISO bagof/setof, and first-argument indexing) as
`grok/wamjs-lowered-emitter`. Prolog dialect is SWI-Prolog (`swipl`); the JS runtime
targets Node (`node`, v18+).

### The gap
`wam_javascript` is **interpreter-tier only**: every predicate runs through the WAM
instruction loop in `runtime.js.mustache`. Mature targets add a **Tier-2 lowered
emitter** (`wam_<lang>_lowered_emitter.pl`) that compiles suitable predicates to
**direct host functions** — no per-goal interpreter dispatch — with the interpreter as
fallback. 14 of 18 WAM targets have this; JS is one of only 4 without
(`docs/proposals/JS_TARGETS_PARITY_PUNCHLIST.md` G-W1, census denominators).

### Goal
Add a lowered-emitter tier to `wam_javascript`:
1. An emit-mode resolver `javascript_wam_resolve_emit_mode/2` supporting
   `interpreter | functions | mixed(List)` (JS currently supports `interpreter` only —
   `wam_javascript_target.pl` throws on other modes).
2. A new `src/unifyweaver/targets/wam_javascript_lowered_emitter.pl` that emits a
   **direct JS function** for a lowered predicate (deterministic/simple clauses first),
   falling back to the interpreter for anything it can't lower.
3. `functions` mode = lower all eligible predicates; `mixed(List)` = lower only the
   named predicates, interpret the rest. `interpreter` mode stays exactly as today.

### Reference to port FROM (closest peer)
`src/unifyweaver/targets/wam_lua_lowered_emitter.pl` + `wam_lua_target.pl`'s
`lua_wam_resolve_emit_mode/2` — Lua is dynamically typed like JS, so its lowering
strategy (clause-1 fast path + interpreter fallback, how it decides eligibility, how it
stitches lowered functions alongside the interpreter runtime) is the model. Cross-check
`wam_haskell`/`wam_rust` lowered emitters for the `mixed(List)` shape. Read
`docs/WAM_TARGET_ROADMAP.md` for the tier definitions.

### Read first (grounding)
- `templates/targets/javascript_wam/runtime.js.mustache` and
  `templates/targets/javascript_wam/program.js.mustache` — the current VM + how
  predicates are emitted/dispatched, so the lowered functions integrate cleanly.
- `src/unifyweaver/targets/wam_javascript_target.pl` — `write_wam_javascript_project/3`,
  the (interpreter-only) emit-mode handling to extend, and how indexing emits switch tables.
- `src/unifyweaver/targets/wam_lua_lowered_emitter.pl` + `wam_lua_target.pl` (the model).
- `docs/WAM_JAVASCRIPT_STATUS.md`.

### Deliverables (SPDX headers preserved)
- `src/unifyweaver/targets/wam_javascript_lowered_emitter.pl` (new).
- Extend `wam_javascript_target.pl`: `javascript_wam_resolve_emit_mode/2`
  (interpreter|functions|mixed) + wire the lowered path into
  `write_wam_javascript_project/3` (choose interpreter vs lowered per predicate).
- Template changes in `templates/targets/javascript_wam/` if the lowered functions need
  a shared prelude/runtime hook.
- Extend `tests/test_wam_javascript_builtins.pl` (or add
  `tests/test_wam_javascript_lowered.pl`) with: a predicate compiled in `functions`
  mode runs under `node` and matches SWI; a `mixed([p/n])` build lowers only `p` and
  interprets the rest; and a determinism check that a lowered deterministic predicate
  leaves no choice point.
- Update `docs/WAM_JAVASCRIPT_STATUS.md`: lowered emitter / emit modes → implemented,
  with residual notes (which clause shapes still fall back to the interpreter).
- `INTEGRATION_PATCH.md` ONLY if a shared file must change (e.g. if you want a
  conformance arm in `functions` mode). Do NOT edit `core/target_registry.pl`,
  `docs/BINDING_MATRIX.md`, `tests/test_advanced.pl`, `glue/js_glue.pl`, or the
  cross-target conformance harness directly.

### Constraints
- **`interpreter` mode must remain byte-for-byte behaviorally identical** — the default
  path and the 48-query conformance must not change.
- Lowered functions must produce identical solutions/order to the interpreter for the
  predicates they handle; fall back to the interpreter rather than emit wrong code.
- No new runtime dependency (stock Node). Keep the six WAM conventions intact.

### Acceptance (must pass before handoff)
1. `swipl -q -g run_tests -t halt tests/test_wam_javascript_builtins.pl` → green
   (interpreter-mode probes + 48 classics unchanged) plus the new lowered/mixed tests.
2. `CONFORMANCE_TARGETS=javascript swipl -q -g run_tests -t halt tests/test_wam_cross_target_conformance.pl`
   → still 48/48 green (this arm runs interpreter mode — must be unaffected).
3. Show a predicate compiled in `functions` mode: the generated JS is a direct function
   (not an interpreter-vector entry), runs under `node`, and matches SWI; and a
   `mixed([...])` build lowers only the named predicates.

### Handoff format
Return: the new lowered emitter, the extended target/templates/tests, the updated status
doc, `INTEGRATION_PATCH.md` (only if needed), and a note stating you branched from
`grok/wamjs-indexing` and EXTENDED the files (list net additions), which clause shapes
you lower vs fall back on, and how `functions`/`mixed` are wired.

## ↑↑↑ Copy to here
