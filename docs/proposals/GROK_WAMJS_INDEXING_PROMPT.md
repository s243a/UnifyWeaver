<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) 2026 John William Creighton (s243a) -->

# Grok prompt — first-argument indexing for wam_javascript

> Use AFTER the current `main` (which now has the full `wam_javascript` target
> incl. bagof/setof) — branch from `main` and EXTEND the existing files.

---

## Copy from here ↓↓↓

You are contributing to **UnifyWeaver** (`github.com/s243a/unifyweaver`), a compiler
that lowers Prolog to a shared WAM bytecode and runs it on per-language WAM
interpreters. Branch from the current `main` as `grok/wamjs-indexing`. Prolog dialect
is SWI-Prolog (`swipl`); the JS runtime targets Node (`node`, v22).

### Starting point — these ALREADY EXIST on main; extend them, do not recreate
- `templates/targets/javascript_wam/runtime.js.mustache` — the Node WAM VM. It already
  implements deref/unify/trail, choice points (try/retry/trust chains), Y-register
  frames, `evalArith`, and the builtins `findall/functor/arg/=../copy_term/\+/call/
  aggregate_all/bagof/setof`.
- `src/unifyweaver/targets/wam_javascript_target.pl` — the emitter.
- `templates/targets/javascript_wam/program.js.mustache` — instruction vector + labels + CLI.
- `docs/WAM_JAVASCRIPT_STATUS.md` — status; the `switch_on_*` limitation is noted there.
- `tests/test_wam_javascript_builtins.pl` — plunit probes + the 48 classic queries.

Read the current runtime first, especially how it (a) dispatches a call to a
predicate's clause chain and (b) handles the `switch_on_constant`,
`switch_on_structure`, and `switch_on_term` instructions today.

### The gap to close
Right now the `switch_on_constant`, `switch_on_structure`, and `switch_on_term`
instructions are emitted as **one-slot NoOps** (per WAM convention 6 — "unhandled ⇒
NoOp, never drop") and correctness comes entirely from trying every clause via the
try/retry/trust chain. That is correct but leaves **first-argument indexing**
unimplemented: a call like `p(foo, X)` against a predicate whose clauses have distinct
ground first arguments still walks all clauses and creates spurious choice points.

Implement real first-argument indexing so that:
- On a call whose first argument dereferences to a **ground** atom/number/compound, the
  VM jumps directly to the matching clause group (via the switch tables), skipping
  non-matching clauses and NOT leaving a choice point when only one clause can match.
- On a call whose first argument is **unbound** (a variable), dispatch falls back to the
  full try/retry/trust chain over all clauses (unchanged behavior) — indexing must
  never lose solutions.
- The existing determinism/backtracking semantics are preserved exactly: same solutions,
  same order; the only observable differences are fewer residual choice points and
  faster deterministic dispatch.

### Reference to port FROM
The mature WAM backends implement this. Study, in order of closeness:
- `src/unifyweaver/targets/wam_lua_target.pl` + `templates/targets/lua_wam/*` (dynamically
  typed, closest to JS) — how it builds and consults the switch-on tables.
- `src/unifyweaver/targets/wam_python_target.pl` and `wam_rust_target.pl` for a second view.
- `docs/WAM_SWITCH_INDEXING_CROSS_TARGET.md` if present — cross-target notes on switch
  indexing.
- `docs/WAM_BACKEND_CONVENTIONS.md` — you are now IMPLEMENTING conventions previously
  NoOp'd; keep all six intact (esp. deref-before-type-test, and label-PC alignment: an
  implemented switch must consume exactly the instruction slots the emitter laid out).

The oracle is SWI-Prolog: for every test the Node WAM answers (and success/failure)
must match `swipl`.

### Deliverables (SPDX headers preserved)
- Extend `runtime.js.mustache` to implement `switch_on_constant/structure/term` as real
  indexed dispatch, with variable-first-arg fallback to the clause chain.
- Update `wam_javascript_target.pl` only if the emitter must emit switch tables (check
  whether `wam_target.pl` already emits the switch instructions and the emitter just
  needs to lower them, vs. needing to build the index).
- Extend `tests/test_wam_javascript_builtins.pl` with indexing tests:
  a multi-clause predicate with distinct ground first args (e.g. `color(red,1). color(green,2).
  color(blue,3).`) — assert `color(green,X)` yields `X=2` and, ideally, that it is
  deterministic (no leftover choice point → a second solution request fails). Include a
  case proving variable-first-arg still enumerates all clauses
  (`findall(C-N, color(C,N), L)` → all three).
- Update `docs/WAM_JAVASCRIPT_STATUS.md`: move first-argument indexing from "not
  implemented / NoOp" to implemented; note any residual (e.g. deep/second-argument
  indexing still absent, which is fine).
- `INTEGRATION_PATCH.md` ONLY if a shared file needs changing (likely none). Do NOT edit
  `core/target_registry.pl`, `docs/BINDING_MATRIX.md`, `tests/test_advanced.pl`,
  `glue/js_glue.pl`, or the cross-target conformance harness/fixtures directly.

### Constraints
- Interpreter tier only; no new runtime dependency (stock Node).
- Do NOT break the existing 48/48 conformance or the builtin probes — re-run both after
  every change. Indexing is an optimization + choice-point cleanup; it must be
  behavior-preserving.

### Acceptance (must pass before handoff)
1. `swipl -q -g run_tests -t halt tests/test_wam_javascript_builtins.pl` → green,
   including the new indexing tests.
2. `CONFORMANCE_TARGETS=javascript swipl -q -g run_tests -t halt tests/test_wam_cross_target_conformance.pl`
   → javascript test passes, no xfail/skip (all 48 queries).
3. Show that an indexed deterministic call leaves no spurious choice point (e.g. the
   second-solution request fails) while a variable first arg still enumerates all
   clauses, and that both match SWI.

### Handoff format
Return: the changed runtime/emitter files, the extended test file, the updated status
doc, `INTEGRATION_PATCH.md` (only if needed), and a note stating you branched from
`main` and EXTENDED the files (list net additions), plus how you handled the
variable-first-arg fallback and any residual indexing not covered (e.g. second-arg /
deep indexing).

## ↑↑↑ Copy to here
