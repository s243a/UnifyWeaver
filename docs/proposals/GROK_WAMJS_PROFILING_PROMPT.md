<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) 2026 John William Creighton (s243a) -->

# Grok prompt — opt-in runtime profiling for wam_javascript (GP-PROF)

> Branch from `grok/wamjs-string-literal` (latest JS-WAM: interpreter + full string
> support incl. compiled literals) and EXTEND — do not rebuild. Profiling is opt-in
> instrumentation: ZERO behavior or output change when off (the default).

---

## Copy from here ↓↓↓

You are contributing to **UnifyWeaver** (`github.com/s243a/unifyweaver`), a compiler
that lowers Prolog to a shared WAM bytecode and runs it on per-language WAM interpreters.
Branch from **`grok/wamjs-string-literal`** (the latest JS-WAM state) as
`grok/wamjs-profiling`. Prolog dialect is SWI-Prolog (`swipl`); the JS runtime targets
Node (`node`, v18+).

### The gap (GP-PROF)
The JS WAM runtime has no way to see where a program spends its time or work. Before
we invest in performance items (indexing depth, parallelism), we need a baseline:
**opt-in profiling instrumentation** in the interpreter.

### Goal
1. **Activation** — profiling is OFF by default. Turn it on via environment variable
   `UW_PROFILE=1` on the emitted program (also accept `UW_PROFILE=json`), and expose a
   runtime toggle (e.g. `Runtime.profile(true)`) for programmatic use.
2. **Collected per run** (keep the hot-path cost near zero when off — a single flag
   check, no allocation):
   - per-predicate: call count, total instructions executed under it, wall-clock ns
     (aggregate, `process.hrtime.bigint()`), choice points created, max choice-point
     depth seen;
   - global: total instructions, unify calls, trail pushes, heap cells allocated,
     backtracks, GC-ish resets if any, total wall time.
3. **Report** — on process exit (or `Runtime.profileReport()`), print a compact table
   to **stderr** (never stdout — stdout is program output and the conformance harness
   reads it): predicates sorted by wall time desc, with count / instr / CPs / time,
   then the global totals line. With `UW_PROFILE=json`, emit one JSON object to stderr
   instead (stable key names — document them in the status doc).
4. **Lowered tier**: the lowered emitter's generated functions should at minimum count
   calls per lowered predicate when profiling is on (instruction-level detail is only
   for the interpreter tier — say so in the report output).

### Reference
- `templates/targets/javascript_wam/runtime.js.mustache` — the interpreter main loop
  (instruction dispatch), call/execute handling, choice-point creation, trail — the
  spots to hook. Keep each hook behind ONE `if (this._prof)`-style guard.
- Grep the mature WAM runtimes (rust/haskell) and `swipl`'s own `profile/1` output
  format for inspiration on what a useful per-predicate table looks like; the exact
  format is yours to design, but keep it one-screen compact.
- `src/unifyweaver/targets/wam_javascript_target.pl` / lowered emitter — only if a
  small emit-time thread-through is needed (e.g. predicate-name tables for pretty
  reporting).

### The six WAM conventions (do not regress)
1. Cons: `put_list` AND `put_structure [|]/2` intern the same functor; `[]` is the atom.
2. Functor `name/arity` where name may contain `/` (`///2`); parse arity as trailing `/<digits>`.
3. Nested terms outer-first via placeholders — `put_*` must bind+trail the placeholder.
4. `deref` before every type test.  5. `is/2` yields an integer for integral results.
6. Unhandled instruction ⇒ a real one-slot NoOp.

### Deliverables (SPDX headers)
- Extend `templates/targets/javascript_wam/runtime.js.mustache` with the instrumented
  hooks + report writer (stderr only).
- `wam_javascript_target.pl` / lowered emitter only if needed for name tables / call
  counters in lowered functions.
- Update `docs/WAM_JAVASCRIPT_STATUS.md`: profiling section — activation, metrics,
  the JSON schema of `UW_PROFILE=json`, tier caveat (lowered = call counts only).
- Extend `tests/test_wam_javascript_builtins.pl` with probes:
  (a) run a program (e.g. fib or member) with `UW_PROFILE=1` — stdout is BYTE-IDENTICAL
  to the unprofiled run and stderr contains the table with a plausible call count;
  (b) `UW_PROFILE=json` — stderr parses as JSON and the profiled predicate appears with
  count ≥ 1; (c) default run — stderr has NO profiling output.
- `INTEGRATION_PATCH.md` ONLY if a shared file must change. Do NOT edit
  `core/target_registry.pl`, `docs/BINDING_MATRIX.md`, `tests/test_advanced.pl`,
  `glue/js_glue.pl`, `wam_target.pl`, `wam_text_parser.pl`, or the cross-target
  conformance harness.

### Constraints — CRITICAL
- OFF by default; when off, stdout AND stderr byte-identical to today, and the hot
  loop pays only a flag check. When ON, stdout still byte-identical (report → stderr).
- Do NOT break the 48/48 conformance, or the builtin/parser/fact-source/lowered/
  term-meta/string suites — re-run all. Interpreter tier; no new runtime dependency.

### Acceptance (must pass before handoff)
1. `swipl -q -g run_tests -t halt tests/test_wam_javascript_builtins.pl` → green incl. the 3 profiling probes.
2. `swipl -q -g run_tests -t halt tests/test_wam_javascript_lowered.pl` and
   `swipl -q -g run_tests -t halt tests/test_wam_javascript_fact_sources.pl` → green.
3. `CONFORMANCE_TARGETS=javascript swipl -q -g run_tests -t halt tests/test_wam_cross_target_conformance.pl` → still 48/48 (profiling off).
4. Show a `UW_PROFILE=1` run of a recursive program (fib(20) or similar) with its
   stderr table, next to the unprofiled run's identical stdout.

### Handoff format
Return: the changed runtime/target files, the extended test file, the updated status
doc, `INTEGRATION_PATCH.md` (only if needed), and a note stating you branched from
`grok/wamjs-string-literal` and EXTENDED it (net additions), the metric set + JSON
schema, measured overhead when off (rough), and any residual.

## ↑↑↑ Copy to here
