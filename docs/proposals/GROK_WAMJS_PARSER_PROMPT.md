<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) 2026 John William Creighton (s243a) -->

# Grok prompt — richer runtime term parser for wam_javascript (G-W2)

> Branch from `grok/wamjs-builtin-breadth` (latest JS-WAM: interpreter + bagof/setof +
> indexing + lowered emitter + builtin breadth) and EXTEND — do not rebuild.

---

## Copy from here ↓↓↓

You are contributing to **UnifyWeaver** (`github.com/s243a/unifyweaver`), a compiler
that lowers Prolog to a shared WAM bytecode and runs it on per-language WAM interpreters.
Branch from **`grok/wamjs-builtin-breadth`** (the latest JS-WAM state) as
`grok/wamjs-parser`. Prolog dialect is SWI-Prolog (`swipl`); the JS runtime targets Node
(`node`, v18+).

### The gap (G-W2)
`wam_javascript`'s CLI/runtime term parser handles only **integers and atoms**
(`templates/targets/javascript_wam/runtime.js.mustache`, the `parse_cli_atom_or_int`
region) — no floats, lists, or compound terms — and `wam_javascript` is NOT registered
in `src/unifyweaver/targets/wam_runtime_parser_capability.pl` (7 targets are:
wam_r/cpp/python/fsharp/haskell/rust/go). This blocks any query with structured
arguments from the CLI, and blocks `read_term`-style builtins.

### Goal
1. Extend the runtime term parser so CLI/`read_term` input accepts **floats**, **lists**
   (`[a,b,c]`, `[H|T]`), and **compound terms** (`foo(a, bar(b), 3)`), in addition to the
   current ints/atoms — interning them into the same term representation the VM uses
   (cons cells via the interned `[|]/2` + `[]` atom; compounds via `put_structure`-style
   structs; floats as a numeric term). Respect the six WAM conventions.
2. Register `wam_javascript` in `wam_runtime_parser_capability.pl` at the appropriate
   capability level — study how go/haskell/rust register (`compiled`) and c++/r register
   (`native(parse_term)`), and pick whichever matches your implementation (a JS
   hand-written recursive-descent term reader is effectively the `compiled`/native
   in-runtime parser). Follow their registration shape exactly.

### Reference
- `src/unifyweaver/targets/wam_runtime_parser_capability.pl` (registration entries ~:164-172).
- `docs/WAM_RUNTIME_PARSER_STATUS.md` (what each level means).
- A mature runtime's parser for the term grammar (e.g. the rust/haskell/go WAM runtime
  templates' `parse_term`), ported to idiomatic JS.
- `templates/targets/javascript_wam/runtime.js.mustache` — the existing
  `parse_cli_atom_or_int` + the VM term representation (deref/unify/intern) to build into.
- The oracle is SWI-Prolog: a CLI query with structured args must bind/run identically.

### The six WAM conventions (do not regress)
1. Cons: `put_list` AND `put_structure [|]/2` intern the same functor; `[]` is the atom.
2. Functor `name/arity` where name may contain `/` (`///2`); parse arity as trailing `/<digits>`.
3. Nested terms built outer-first via placeholder vars — `put_*` must bind+trail the placeholder.
4. `deref` before every type test.
5. `is/2` yields an integer for integral results.
6. Unhandled instruction ⇒ a real one-slot NoOp.

### Deliverables (SPDX headers)
- Extend `templates/targets/javascript_wam/runtime.js.mustache` with the richer parser
  (recursive-descent: number [int/float], atom [incl. quoted], variable, list, compound).
- Update `wam_javascript_target.pl` only if emit-side changes are needed.
- `INTEGRATION_PATCH.md` for the `wam_runtime_parser_capability.pl` registration line
  (that file is shared — do NOT edit it directly; put the exact clause in the patch for
  the coordinator). Do NOT edit `core/target_registry.pl`, `docs/BINDING_MATRIX.md`,
  `tests/test_advanced.pl`, `glue/js_glue.pl`, or the cross-target conformance harness.
- Update `docs/WAM_JAVASCRIPT_STATUS.md`: parser scope int+atom → full term reader;
  runtime-parser capability registered.
- Extend `tests/test_wam_javascript_builtins.pl` with CLI/parse probes: a query with a
  list arg, a compound arg, and a float, run under `node`, matching SWI.

### Constraints
- Do NOT break the 48/48 conformance, the builtin probes, or the lowered-emitter tests —
  re-run all after changes. Interpreter tier; no new runtime dependency.

### Acceptance (must pass before handoff)
1. `swipl -q -g run_tests -t halt tests/test_wam_javascript_builtins.pl` → green, incl.
   the new parser probes.
2. `swipl -q -g run_tests -t halt tests/test_wam_javascript_lowered.pl` → green.
3. `CONFORMANCE_TARGETS=javascript swipl -q -g run_tests -t halt tests/test_wam_cross_target_conformance.pl` → still 48/48, no xfail/skip.
4. Show a CLI query with a list arg, a compound arg, and a float parsing + running under
   `node`, matching SWI.

### Handoff format
Return: the changed runtime/target files, the extended test file, the updated status doc,
`INTEGRATION_PATCH.md` (the parser-capability registration line), and a note stating you
branched from `grok/wamjs-builtin-breadth` and EXTENDED it (net additions), which term
shapes parse fully vs partially (quoted atoms, negative numbers, operators), and the
capability level you registered.

## ↑↑↑ Copy to here
