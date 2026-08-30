<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) 2026 John William Creighton (s243a) -->

# Grok prompt — term-meta builtins for wam_javascript (G-W3 follow-up)

> Branch from `grok/wamjs-fact-sources` (latest JS-WAM: interpreter + bagof/setof +
> indexing + lowered emitter + builtin breadth + term parser + fact sources) and
> EXTEND — do not rebuild.

---

## Copy from here ↓↓↓

You are contributing to **UnifyWeaver** (`github.com/s243a/unifyweaver`), a compiler
that lowers Prolog to a shared WAM bytecode and runs it on per-language WAM interpreters.
Branch from **`grok/wamjs-fact-sources`** (the latest JS-WAM state) as
`grok/wamjs-term-meta`. Prolog dialect is SWI-Prolog (`swipl`); the JS runtime targets
Node (`node`, v18+).

### The gap (G-W3 follow-up)
The JS WAM interpreter has broad builtins now but is missing the **term-meta** family that
G-W3 explicitly left for follow-up: `term_variables/2`, `numbervars/3`, and `=@=/2`
(variant equality) + its negation `\=@=/2`. Add them.

### Semantics (match SWI-Prolog exactly)
- **`term_variables(+Term, -List)`** — `List` is the list of distinct unbound variables in
  `Term`, in first-occurrence (left-to-right, depth-first) order.
- **`numbervars(+Term, +Start, -End)`** — binds each distinct unbound variable in `Term` to
  a term `'$VAR'(N)` with N counting from `Start`; `End` is `Start + count`. (Write/print of
  `'$VAR'(N)` as `A`,`B`,… is a printing concern — implement the binding semantics; note in
  the status doc whether `write` renders `'$VAR'` specially.)
- **`=@=/2`** — structural equivalence up to variable renaming (variant). `\=@=/2` is its
  negation. Ground terms compare as `==`; variables match variables consistently (a
  consistent bijection between the two terms' variables).

### Reference
- A mature WAM runtime that implements these (grep the rust/haskell/lua WAM runtimes and
  bindings for `term_variables`, `numbervars`, `=@=`/`variant`), ported to idiomatic JS
  over the existing term representation.
- `templates/targets/javascript_wam/runtime.js.mustache` — the VM term rep (deref, structs,
  var cells, trail) + how existing builtins like `copy_term` walk terms (copy_term already
  does a variable-respecting term walk — reuse that machinery).
- `src/unifyweaver/bindings/javascript_wam_bindings.pl` — register the new builtins.
- Oracle = SWI-Prolog.

### The six WAM conventions (do not regress)
1. Cons: `put_list` AND `put_structure [|]/2` intern the same functor; `[]` is the atom.
2. Functor `name/arity` where name may contain `/` (`///2`); parse arity as trailing `/<digits>`.
3. Nested terms outer-first via placeholders — `put_*` must bind+trail the placeholder.
4. `deref` before every type test.  5. `is/2` yields an integer for integral results.
6. Unhandled instruction ⇒ a real one-slot NoOp.

### Deliverables (SPDX headers)
- Extend `templates/targets/javascript_wam/runtime.js.mustache` with the three builtins
  (reuse the `copy_term` term-walk + the existing var/struct model; `numbervars` binds+trails).
- Register them in `src/unifyweaver/bindings/javascript_wam_bindings.pl` (+
  `wam_javascript_target.pl` only if new dispatch needed).
- Update `docs/WAM_JAVASCRIPT_STATUS.md`: move the three from follow-up to implemented; note
  any residual (e.g. `numbervars` write-rendering of `'$VAR'`).
- Extend `tests/test_wam_javascript_builtins.pl` with probes: `term_variables(f(X,Y,X),L)`
  → `[X,Y]`; `numbervars(f(X,Y),0,E)` → E=2 with the vars bound; `f(X,Y) =@= f(A,B)` true,
  `f(X,X) =@= f(A,B)` false — each run under `node`, matching SWI.
- `INTEGRATION_PATCH.md` ONLY if a shared file must change. Do NOT edit
  `core/target_registry.pl`, `docs/BINDING_MATRIX.md`, `tests/test_advanced.pl`,
  `glue/js_glue.pl`, or the cross-target conformance harness.

### Constraints
- Do NOT break the 48/48 conformance, the builtin/parser/fact-source probes, or the
  lowered-emitter tests — re-run all. Interpreter tier; no new runtime dependency.

### Acceptance (must pass before handoff)
1. `swipl -q -g run_tests -t halt tests/test_wam_javascript_builtins.pl` → green incl. the new probes.
2. `swipl -q -g run_tests -t halt tests/test_wam_javascript_lowered.pl` and
   `swipl -q -g run_tests -t halt tests/test_wam_javascript_fact_sources.pl` → green.
3. `CONFORMANCE_TARGETS=javascript swipl -q -g run_tests -t halt tests/test_wam_cross_target_conformance.pl` → still 48/48, no xfail/skip.
4. For each of the three builtins, show a probe's Node output next to SWI's and confirm they match.

### Handoff format
Return: the changed runtime/bindings/target files, the extended test file, the updated
status doc, `INTEGRATION_PATCH.md` (only if needed), and a note stating you branched from
`grok/wamjs-fact-sources` and EXTENDED it (net additions), plus any residual (e.g.
`numbervars` `'$VAR'` write-rendering, cyclic-term handling).

## ↑↑↑ Copy to here
