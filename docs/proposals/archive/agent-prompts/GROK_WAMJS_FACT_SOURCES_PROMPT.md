<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) 2026 John William Creighton (s243a) -->

# Grok prompt — external fact sources for wam_javascript (G-W4)

> Branch from `grok/wamjs-parser` (latest JS-WAM: interpreter + bagof/setof + indexing +
> lowered emitter + builtin breadth + term parser) and EXTEND — do not rebuild.

---

## Copy from here ↓↓↓

You are contributing to **UnifyWeaver** (`github.com/s243a/unifyweaver`), a compiler
that lowers Prolog to a shared WAM bytecode and runs it on per-language WAM interpreters.
Branch from **`grok/wamjs-parser`** (the latest JS-WAM state) as `grok/wamjs-fact-sources`.
Prolog dialect is SWI-Prolog (`swipl`); the JS runtime targets Node (`node`, v18+).

### The gap (G-W4)
`wam_javascript` has no external fact-source / data tier — a predicate's facts must be
compiled inline. 14 of 18 WAM targets consume external fact sources; JS has only an empty
`fact_sources: {}` scaffold in its templates. The **lightweight reference peer is
`wam_lua`** (dynamically typed, interpreter-tier like JS), which supports a
`lua_fact_sources(Sources)` option — the full LMDB/CSR machinery of rust/haskell is
overkill; mirror Lua's lightweight file-backed approach.

### Goal
Add a lightweight external fact-source option to `wam_javascript` so a predicate's facts
can be loaded at runtime from a file (start with **TSV/CSV** and/or **JSON-lines**, the
formats Node can read with zero deps via `fs`/`readline`), instead of being interned
inline. Model the option shape and wiring on `wam_lua`.

### Reference to port FROM (study first)
- `src/unifyweaver/targets/wam_lua_target.pl`:
  - `lua_fact_source_spec/4` (~:545) — reads the `lua_fact_sources(Sources)` option and
    matches a predicate indicator (`fact_source_pi_match/3` ~:551).
  - `lua_fact_source_entry/3` (~:556) — turns a `file(Path)` spec into a runtime entry.
  - the emit site (~:475-483) that assembles fact-source entries, and the
    `'fact_sources'=FactSourcesCode` template binding (~:784).
  - `templates/targets/lua_wam/*` — how the Lua runtime consumes `fact_sources` at run time.
- `templates/targets/javascript_wam/runtime.js.mustache` — the JS runtime; note the
  existing empty `fact_sources: {}` scaffold hook to fill in.
- `src/unifyweaver/targets/wam_javascript_target.pl` — `write_wam_javascript_project/3`
  and where facts are currently emitted; add the option handling here mirroring Lua.
- The oracle is SWI-Prolog with the facts loaded from the same file.

### Deliverables (SPDX headers)
- Add a `javascript_wam_fact_sources(Sources)` (or reuse a target-neutral name if one
  exists) option, parsed like Lua's, that emits — instead of inline facts — a runtime
  loader in the emitted Node project that reads the file (TSV/CSV and/or JSONL) and
  answers the predicate from it. Fill the `fact_sources` template hook.
- Extend `templates/targets/javascript_wam/runtime.js.mustache` (and `program.js.mustache`
  if needed) with the file-backed fact reader (Node `fs`/`readline`, no npm deps).
- Update `docs/WAM_JAVASCRIPT_STATUS.md`: external fact sources → implemented (file/TSV/CSV/
  JSONL), note LMDB/CSR as out of scope (matches the Lua peer).
- Extend `tests/test_wam_javascript_builtins.pl` (or a new `tests/test_wam_javascript_fact_sources.pl`)
  with a predicate backed by a fixture file, compiled and run under `node`, answers
  matching SWI over the same data.
- `INTEGRATION_PATCH.md` ONLY if a shared file must change. Do NOT edit
  `core/target_registry.pl`, `docs/BINDING_MATRIX.md`, `tests/test_advanced.pl`,
  `glue/js_glue.pl`, or the cross-target conformance harness.

### The six WAM conventions (do not regress)
1. Cons: `put_list` AND `put_structure [|]/2` intern the same functor; `[]` is the atom.
2. Functor `name/arity` where name may contain `/` (`///2`); parse arity as trailing `/<digits>`.
3. Nested terms outer-first via placeholders — `put_*` must bind+trail the placeholder.
4. `deref` before every type test.  5. `is/2` yields an integer for integral results.
6. Unhandled instruction ⇒ a real one-slot NoOp.

### Constraints
- Do NOT break the 48/48 conformance, the builtin probes, the lowered-emitter tests, or the
  parser probes — re-run all after changes. Interpreter tier; no new runtime dependency
  (Node built-ins only). Inline-facts behavior (no fact-source option) must be unchanged.

### Acceptance (must pass before handoff)
1. `swipl -q -g run_tests -t halt tests/test_wam_javascript_builtins.pl` (+ any new
   fact-source test) → green.
2. `swipl -q -g run_tests -t halt tests/test_wam_javascript_lowered.pl` → green.
3. `CONFORMANCE_TARGETS=javascript swipl -q -g run_tests -t halt tests/test_wam_cross_target_conformance.pl` → still 48/48, no xfail/skip.
4. Show a predicate backed by a fixture file (TSV/CSV or JSONL) compiled + run under `node`,
   its answers matching SWI over the same data.

### Handoff format
Return: the changed target/runtime/template/test files, the updated status doc,
`INTEGRATION_PATCH.md` (only if needed), and a note stating you branched from
`grok/wamjs-parser` and EXTENDED it (net additions), which source formats are supported
(TSV/CSV/JSONL), and what's left (LMDB/CSR = out of scope, matching the Lua peer).

## ↑↑↑ Copy to here
