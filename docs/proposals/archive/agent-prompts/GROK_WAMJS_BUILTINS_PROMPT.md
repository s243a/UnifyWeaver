<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) 2026 John William Creighton (s243a) -->

# Grok prompt — expand the wam_javascript builtin surface

> Self-contained brief to paste into Grok. Closes the biggest gap between the
> new `wam_javascript` WAM target and the mature `wam_lua`/`wam_python`/`wam_rust`
> targets: the JS interpreter currently covers only the conformance set +
> arithmetic. Port the missing builtins from the Lua reference.

---

## Copy from here ↓↓↓

You are contributing to **UnifyWeaver** (`github.com/s243a/unifyweaver`), a compiler
that lowers Prolog to a shared WAM bytecode and runs it on per-language WAM
interpreters. Work in a clean checkout on a branch named `grok/wamjs-builtins`.
Prolog dialect is SWI-Prolog (`swipl`); the JS runtime targets Node (present as
`node`, v22).

### Context
A `wam_javascript` hybrid target already exists and is GREEN on the cross-target
conformance suite (48/48 queries: member, append, reverse, fib, ack, builtins,
wide, nested, buildnest, repeatvar, emptylist). Its files:
- `src/unifyweaver/targets/wam_javascript_target.pl` — the emitter (consumes the
  shared `wam_target.pl` bytecode; emit mode `interpreter`).
- `templates/targets/javascript_wam/runtime.js.mustache` — the Node WAM virtual
  machine (deref, unify, trail, choice points, arithmetic via `evalArith`,
  per-clause Y-register frames, reserve-slot write model).
- `templates/targets/javascript_wam/program.js.mustache` — instruction vector +
  label table + intern seed + CLI shim.
- `src/unifyweaver/bindings/javascript_wam_bindings.pl` — builtin catalogue +
  reserved lowered-tier map.
- `docs/WAM_JAVASCRIPT_STATUS.md` — status + the six WAM conventions checklist.

### Your task
Port the **missing builtins** that the mature `wam_lua` target already has, into the
JS WAM runtime. In priority order:
1. `findall/3` (and if tractable from the same mechanism: `bagof/3`, `setof/3`,
   `aggregate_all/3` at least for `count`, `sum`, `bag`, `set`).
2. `functor/3`, `arg/3`, `=../2` (univ) — term construction/inspection.
3. `copy_term/2`.
4. `\+/1` (negation-as-failure metacall) and `call/1`.
These are exactly the gaps named in `docs/WAM_JAVASCRIPT_STATUS.md` under "remaining".

### The reference to port FROM
`src/unifyweaver/targets/wam_lua_target.pl` and its runtime template under
`templates/targets/lua_wam/` implement all of the above. JS is dynamically typed
like Lua, so the Lua runtime is the closest model — study how it:
- re-enters the solver to collect `findall` solutions (sub-goal execution +
  solution copying onto the heap),
- builds/decomposes terms for `functor`/`arg`/`=..`,
- deep-copies with fresh variables for `copy_term`,
- implements `\+`/`call` as metacalls over the same instruction loop.
Mirror the mechanism; write idiomatic JS in `runtime.js.mustache`.

### Read first (grounding)
- `templates/targets/javascript_wam/runtime.js.mustache` (the VM you're extending —
  understand its term representation, `deref`, `unify`, `bind`/trail, choice-point
  stack, and how builtins are dispatched today).
- `src/unifyweaver/targets/wam_lua_target.pl` + `templates/targets/lua_wam/*` (the
  reference implementation of every builtin above).
- `docs/WAM_BACKEND_CONVENTIONS.md` — the six conventions you MUST NOT violate.
- `docs/WAM_JAVASCRIPT_STATUS.md` — current status + gap list.
- `src/unifyweaver/bindings/javascript_wam_bindings.pl` — register new builtins here.

### The six conventions (do not regress any)
1. Cons cells: `put_list` AND `put_structure [|]/2` intern the same functor; `[]` is the interned atom.
2. Functor `name/arity` where name may contain `/` (`///2`); parse arity as trailing `/<digits>`.
3. Nested terms built outer-first via placeholder vars — `put_*` must bind+trail the placeholder.
4. `deref` before every type test.
5. `is/2` yields an integer for integral results.
6. Unhandled instruction ⇒ a real one-slot NoOp (never drop/throw); keeps label PCs aligned.

### Deliverables (SPDX headers preserved)
- Extend `templates/targets/javascript_wam/runtime.js.mustache` with the new builtins.
- Register them in `src/unifyweaver/bindings/javascript_wam_bindings.pl` and, if the
  emitter needs to recognize new builtin opcodes/handlers, update
  `src/unifyweaver/targets/wam_javascript_target.pl`.
- Update `docs/WAM_JAVASCRIPT_STATUS.md`: move the ported builtins from "remaining"
  to "implemented"; keep the six-conventions checklist accurate.
- A test artifact: a small Prolog program per builtin (e.g. `findall(X, member(X,[1,2,3]), L)`,
  `functor(foo(a,b), N, A)`, `arg(2, foo(a,b), X)`, `foo(a,b) =.. L`, `copy_term(f(X,X), C)`,
  `\+ member(9,[1,2,3])`) plus a runner that compiles → `node` → checks the result. Put
  it under `tests/` following the repo's plunit style, or extend the conformance fixtures.
- `INTEGRATION_PATCH.md` (do NOT edit these yourself): any additions needed to
  `tests/test_wam_cross_target_conformance.pl` or `tests/wam_conformance_fixtures.pl`
  to add new builtin programs to the shared spec — list the exact clauses for a
  coordinator to apply.

### Constraints
- Do NOT break the existing 48/48 conformance. Re-run it after every change.
- No new runtime dependency; the generated JS must run on stock Node.
- Keep the lowered/FFI tiers out of scope — interpreter tier only.
- Do NOT edit `core/target_registry.pl`, `docs/BINDING_MATRIX.md`,
  `tests/test_advanced.pl`, `glue/js_glue.pl`, or the conformance harness/fixtures
  directly — put those in `INTEGRATION_PATCH.md`.
- SPDX header on every new file.

### Acceptance (must pass before handoff)
1. Conformance still green:
   `CONFORMANCE_TARGETS=javascript swipl -q -g run_tests -t halt tests/test_wam_cross_target_conformance.pl`
   → the `javascript` test passes, no xfail/skip.
2. Each new builtin: compile a Prolog probe, run under `node`, result matches
   SWI-Prolog's answer. Show the outputs (e.g. `findall` → `[1,2,3]`,
   `functor(foo(a,b),N,A)` → `N=foo, A=2`, `=..` → `[foo,a,b]`, `\+ member(9,...)` → true).

### Handoff format
Return: the changed runtime/emitter/bindings files, the new test artifact,
`INTEGRATION_PATCH.md`, the updated status doc, and a note on which builtins are
fully working vs. partial (e.g. if `setof`'s ordering or `bagof`'s free-variable
grouping is only partially handled, say so precisely).

## ↑↑↑ Copy to here
