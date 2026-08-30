<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) 2026 John William Creighton (s243a) -->

# Grok prompt — full ISO bagof/3 and setof/3 for wam_javascript

> Use this AFTER the `claude/peerhailer-exploratory-docs-aodas5` branch is merged
> to `main`. The `wam_javascript` target and its builtins will then exist on
> `main`, so this task EXTENDS them — it does not rebuild the target.

---

## Copy from here ↓↓↓

You are contributing to **UnifyWeaver** (`github.com/s243a/unifyweaver`), a compiler
that lowers Prolog to a shared WAM bytecode and runs it on per-language WAM
interpreters. Branch from the current `main` as `grok/wamjs-bagof-setof`. Prolog
dialect is SWI-Prolog (`swipl`); the JS runtime targets Node (`node`, v22).

### Starting point — these files ALREADY EXIST on main; extend them, do not recreate
- `src/unifyweaver/targets/wam_javascript_target.pl` — the emitter.
- `templates/targets/javascript_wam/runtime.js.mustache` — the Node WAM VM (deref,
  unify, trail, choice points, Y-frames, evalArith, and the builtins already ported:
  `findall/3`, `functor/3`, `arg/3`, `=../2`, `copy_term/2`, `\+/1`, `call/1`,
  `aggregate_all/3`, plus a **partial** `bagof/3`/`setof/3`).
- `src/unifyweaver/bindings/javascript_wam_bindings.pl` — builtin catalogue.
- `tests/test_wam_javascript_builtins.pl` — plunit probes + the 48 classic queries.
- `docs/WAM_JAVASCRIPT_STATUS.md` — status; `bagof/3` and `setof/3` are listed as
  PARTIAL there.

Read all of the above first, especially the current `bagof`/`setof` implementation in
`runtime.js.mustache`, so you extend it rather than duplicate it.

### The gap to close
The current `bagof/3` and `setof/3` collect solutions (and setof uniques + sorts) but
do NOT implement ISO semantics:
1. **Free-variable grouping.** In `bagof(Template, Goal, Bag)`, any variable that
   appears in `Goal` but not in `Template` and is not existentially quantified is a
   *witness* (free) variable. bagof must BACKTRACK over each distinct grouping of
   witness bindings, yielding one `Bag` per witness group (not one flat bag).
2. **`^/2` existential quantification.** `Var^Goal` (and nested `V1^V2^Goal`) removes
   those variables from the witness set, so they do not create groups.
3. **Empty-goal failure.** Unlike `findall/3` (which yields `[]`), `bagof/3` and
   `setof/3` **fail** when `Goal` has no solutions.
4. **setof ordering.** `setof/3` = `bagof/3` then sort each bag into the **standard
   order of terms** with duplicates removed. Standard order:
   `Var < Number < Atom < String < Compound`; compounds ordered by arity, then
   functor name, then arguments left-to-right. The current setof uses an approximate
   order — make it match SWI for mixed-type element lists.

### Reference
- Check `src/unifyweaver/targets/wam_lua_target.pl` + `templates/targets/lua_wam/*` —
  if the Lua WAM implements full ISO bagof/setof, port its mechanism (JS and Lua are
  both dynamically typed). If Lua's is also partial, implement per ISO / SWI semantics
  directly.
- The oracle is SWI-Prolog itself: for every test, the Node WAM answer must match what
  `swipl` produces for the same goal.

### Deliverables (SPDX headers preserved)
- Extend `templates/targets/javascript_wam/runtime.js.mustache` with full ISO
  `bagof/3` and `setof/3` (witness grouping, `^` handling, empty-goal failure, standard
  order). Reuse the existing `findall` collection + `copy_term` + term-compare
  machinery already in the runtime; add a standard-order comparator if one isn't
  already present.
- Update `src/unifyweaver/bindings/javascript_wam_bindings.pl` / `wam_javascript_target.pl`
  only if new dispatch is needed.
- Extend `tests/test_wam_javascript_builtins.pl` with a battery covering:
  free-var grouping (e.g. `bagof(C, age(_,C), Cs)` over facts with repeated ages),
  `^` quantification (`bagof(C, N^age(N,C), Cs)`), empty-goal failure
  (`\+ bagof(x, fail, _)`), and standard-order sorting over a mixed-type list
  (numbers, atoms, compounds). Each asserts equality with the SWI answer.
- Update `docs/WAM_JAVASCRIPT_STATUS.md`: move `bagof/3`/`setof/3` from PARTIAL to
  implemented, noting any residual limitation precisely.
- `INTEGRATION_PATCH.md` ONLY IF a shared file needs changing (likely none — you are
  extending existing files). Do NOT edit `core/target_registry.pl`,
  `docs/BINDING_MATRIX.md`, `tests/test_advanced.pl`, `glue/js_glue.pl`, or the
  cross-target conformance harness/fixtures directly.

### The six WAM conventions (do not regress)
1. Cons: `put_list` AND `put_structure [|]/2` intern the same functor; `[]` is the interned atom.
2. Functor `name/arity` where name may contain `/` (`///2`); parse arity as trailing `/<digits>`.
3. Nested terms built outer-first via placeholder vars — `put_*` must bind+trail the placeholder.
4. `deref` before every type test.
5. `is/2` yields an integer for integral results.
6. Unhandled instruction ⇒ a real one-slot NoOp (never drop/throw).

### Constraints
- Interpreter tier only; no new runtime dependency (stock Node).
- Do NOT break the existing 48/48 conformance or the current builtin probes — re-run
  both after every change.

### Acceptance (must pass before handoff)
1. `swipl -q -g run_tests -t halt tests/test_wam_javascript_builtins.pl` → green,
   including the new bagof/setof battery.
2. Existing conformance unaffected:
   `CONFORMANCE_TARGETS=javascript swipl -q -g run_tests -t halt tests/test_wam_cross_target_conformance.pl`
   → javascript test passes, no xfail/skip.
3. For each new case, show the Node WAM output next to the SWI answer and confirm they
   match (free-var groups, `^` quantification, empty-goal failure, mixed-type setof order).

### Handoff format
Return: the changed runtime/bindings/emitter files, the extended test file, the updated
status doc, `INTEGRATION_PATCH.md` (only if needed), and a short note stating that you
branched from `main` and EXTENDED the existing files (list the net additions), plus any
residual ISO corner you did not cover.

## ↑↑↑ Copy to here
