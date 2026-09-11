<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) 2026 John William Creighton (s243a) -->

# Grok prompt — real string type tag for wam_javascript (G-W3 residual)

> Branch from `grok/wamjs-op3` (latest JS-WAM: interpreter + bagof/setof + indexing +
> lowered emitter + builtin breadth + term parser + fact sources + term-meta + op/3)
> and EXTEND — do not rebuild. This is a runtime term-representation change; proceed
> carefully and keep the 48/48 conformance green throughout.

---

## Copy from here ↓↓↓

You are contributing to **UnifyWeaver** (`github.com/s243a/unifyweaver`), a compiler
that lowers Prolog to a shared WAM bytecode and runs it on per-language WAM interpreters.
Branch from **`grok/wamjs-op3`** (the latest JS-WAM state) as `grok/wamjs-string-tag`.
Prolog dialect is SWI-Prolog (`swipl`); the JS runtime targets Node (`node`, v18+).

### The gap (G-W3 residual)
The JS WAM runtime has **no distinct string type** — SWI strings intern as atoms
(documented residual in `docs/WAM_JAVASCRIPT_STATUS.md`). This affects fidelity:
`atom_string/2`, `split_string/4`, `string_concat/3`, `string_chars/2`,
`number_string/2` all currently yield atoms, and the standard order of terms treats
strings as atoms (SWI orders `Var < Number < Atom < String < Compound`).

### Goal
Introduce a distinct **string** term type in the runtime term representation and make
the string builtins produce/consume it, with correct standard ordering. Concretely:
1. Add a string tag to the term model (alongside atom/int/float/var/struct). `deref`,
   `unify` (a string unifies only with an equal string), `copy_term`, `==`/`\==`,
   and `compare/3` / standard order must handle it (String sorts **after** Atom,
   **before** Compound).
2. Make `atom_string/2`, `string_concat/3`, `string_chars/2`, `string_to_atom/2`,
   `number_string/2`, and `split_string/4` produce **strings** (not atoms) where SWI
   does; keep atom-producing builtins producing atoms. `string/1` type-check true only
   for the string type; `atom/1` false for strings.
3. `write/print` renders a string as its text (SWI `write` prints the text without
   quotes; `writeq`/`print` quote it with `"`) — match SWI's `write` at least.

### Reference
- SWI-Prolog semantics for strings, `string/1`, and standard order are the oracle.
- A mature WAM/runtime that carries a distinct string type (grep the rust/haskell WAM
  runtimes for a `String`/`Str` term variant and how `compare`/`write` treat it), ported
  to the JS term model.
- `templates/targets/javascript_wam/runtime.js.mustache` — the term rep (tags, deref,
  unify, compare/standard-order, write, and the existing string builtins that currently
  return atoms — this is what you're upgrading).
- `src/unifyweaver/bindings/javascript_wam_bindings.pl` — builtin catalogue.

### The six WAM conventions (do not regress)
1. Cons: `put_list` AND `put_structure [|]/2` intern the same functor; `[]` is the atom.
2. Functor `name/arity` where name may contain `/` (`///2`); parse arity as trailing `/<digits>`.
3. Nested terms outer-first via placeholders — `put_*` must bind+trail the placeholder.
4. `deref` before every type test.  5. `is/2` yields an integer for integral results.
6. Unhandled instruction ⇒ a real one-slot NoOp.

### Deliverables (SPDX headers)
- Extend `templates/targets/javascript_wam/runtime.js.mustache`: string tag + unify/compare/
  copy_term/write handling + upgrade the string builtins to produce/consume strings +
  `string/1` type check. Ensure `atom/1` is false for strings and standard order is
  `… < Atom < String < Compound`.
- Update `src/unifyweaver/bindings/javascript_wam_bindings.pl` if dispatch/catalogue needs it.
- Update `docs/WAM_JAVASCRIPT_STATUS.md`: string tag implemented; adjust the earlier
  "strings intern as atoms" residual notes (atom_string/split_string/string_concat etc.).
- Extend `tests/test_wam_javascript_builtins.pl`: probes for `atom_string(a,S), string(S)`,
  `split_string("a,b,c",",","",L)` (list of strings), `string_concat("x","y",Z)`,
  standard-order `sort([foo, "foo", 1, bar], L)` placing the string after atoms — each run
  under `node`, matching SWI.
- `INTEGRATION_PATCH.md` ONLY if a shared file must change. Do NOT edit
  `core/target_registry.pl`, `docs/BINDING_MATRIX.md`, `tests/test_advanced.pl`,
  `glue/js_glue.pl`, or the cross-target conformance harness.

### Constraints — CRITICAL
- The introduction of a string tag must **not** regress the 48/48 conformance (the
  classics use atoms/ints/lists, not strings — but `compare`/`sort`/unify changes could
  ripple). Re-run the full conformance arm AND the builtin/parser/fact-source/lowered/
  term-meta suites after every change.
- Existing atom behavior is unchanged; only SWI-string-producing builtins switch to the
  new tag. Interpreter tier; no new runtime dependency.

### Acceptance (must pass before handoff)
1. `swipl -q -g run_tests -t halt tests/test_wam_javascript_builtins.pl` → green incl. new string probes.
2. `swipl -q -g run_tests -t halt tests/test_wam_javascript_lowered.pl` and
   `swipl -q -g run_tests -t halt tests/test_wam_javascript_fact_sources.pl` → green.
3. `CONFORMANCE_TARGETS=javascript swipl -q -g run_tests -t halt tests/test_wam_cross_target_conformance.pl` → still 48/48, no xfail/skip.
4. Show `string/1`, `atom_string`, `split_string`, `string_concat`, and standard-order
   probes' Node output next to SWI's and confirm they match.

### Handoff format
Return: the changed runtime/bindings files, the extended test file, the updated status doc,
`INTEGRATION_PATCH.md` (only if needed), and a note stating you branched from
`grok/wamjs-op3` and EXTENDED it (net additions), which builtins now produce strings, how
standard order/`compare` handle strings, and any residual (`writeq` quoting, string↔codes
edge cases, cyclic).

## ↑↑↑ Copy to here
