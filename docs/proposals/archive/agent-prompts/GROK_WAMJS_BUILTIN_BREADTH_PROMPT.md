<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) 2026 John William Creighton (s243a) -->

# Grok prompt — ISO/library builtin breadth for wam_javascript (G-W3)

> Branch from `grok/wamjs-lowered-emitter` (latest JS-WAM state: interpreter +
> bagof/setof + indexing + lowered emitter) and EXTEND — do not rebuild.

---

## Copy from here ↓↓↓

You are contributing to **UnifyWeaver** (`github.com/s243a/unifyweaver`), a compiler
that lowers Prolog to a shared WAM bytecode and runs it on per-language WAM
interpreters. Branch from **`grok/wamjs-lowered-emitter`** (the latest JS-WAM state:
interpreter tier, ISO bagof/setof, first-arg indexing, Tier-2 lowered emitter) as
`grok/wamjs-builtin-breadth`. Prolog dialect is SWI-Prolog (`swipl`); the JS runtime
targets Node (`node`, v18+).

### The gap (G-W3)
`wam_javascript` has the aggregate/all-solutions family and term-inspection builtins,
but lacks the ISO/library breadth that the mature WAM targets carry. Add the missing
builtins to the JS WAM runtime, grouped by family (do as many as land cleanly; prioritize
top-to-bottom):
1. **Sort family:** `sort/2`, `sort/4`, `msort/2`, `keysort/2`, `predsort/3`.
2. **Structural list library:** `append/3` (as a builtin, if not already), `reverse/2`,
   `nth0/3`, `nth1/3`, `last/2`, `sum_list/2` (`sumlist/2`), `max_list/2`, `min_list/2`,
   `list_to_set/2`, `select/3`, `exclude/3`/`include/3` if tractable.
3. **Atom/string ops:** `atom_concat/3`, `atom_length/2`, `atom_chars/2`, `atom_codes/2`,
   `char_code/2`, `sub_atom/5` (at least the common modes), `atom_string/2`,
   `number_codes/2`, `number_string/2`, `split_string/4`, `string_concat/3`,
   `string_chars/2`, `upcase_atom/2`, `downcase_atom/2`.
4. **format/IO:** `format/2`, `format/3` (the common directives: `~w ~a ~d ~p ~n ~q`),
   `tab/1`, `writeln/1` (if not present).
5. **assoc library (optional):** `empty_assoc/1`, `list_to_assoc/2`, `get_assoc/3`,
   `put_assoc/4`, `assoc_to_list/2`, `assoc_to_keys/2`.

### Reference
- `src/unifyweaver/bindings/rust_wam_bindings.pl` — declares `append, atom_concat,
  atom_string, sub_atom, split_string, number_string, nth, format, assoc ops`; the best
  breadth reference for what to cover and the naming.
- `templates/targets/lua_wam/*` + `wam_lua_target.pl` — dynamically typed peer; port the
  mechanism where similar.
- `templates/targets/javascript_wam/runtime.js.mustache` — where the existing builtins
  live and how dispatch works; extend it.
- `src/unifyweaver/bindings/javascript_wam_bindings.pl` — the JS WAM builtin catalogue;
  register each new builtin here.
- The oracle is SWI-Prolog: every new builtin's Node WAM answer must match `swipl`.

### The six WAM conventions (do not regress)
1. Cons: `put_list` AND `put_structure [|]/2` intern the same functor; `[]` is the atom.
2. Functor `name/arity` where name may contain `/` (`///2`); parse arity as trailing `/<digits>`.
3. Nested terms built outer-first via placeholder vars — `put_*` must bind+trail the placeholder.
4. `deref` before every type test.
5. `is/2` yields an integer for integral results.
6. Unhandled instruction ⇒ a real one-slot NoOp (never drop/throw).

### Deliverables (SPDX headers)
- Extend `templates/targets/javascript_wam/runtime.js.mustache` with the new builtins.
- Register them in `src/unifyweaver/bindings/javascript_wam_bindings.pl` (+ update
  `wam_javascript_target.pl` only if new dispatch is needed).
- Update `docs/WAM_JAVASCRIPT_STATUS.md`: move the added builtins to "implemented",
  note families still missing.
- Extend `tests/test_wam_javascript_builtins.pl` with a probe per builtin family
  (sort, list lib, atom/string, format) compiled → run under `node` → matching SWI.
- `INTEGRATION_PATCH.md` ONLY if a shared file must change. Do NOT edit
  `core/target_registry.pl`, `docs/BINDING_MATRIX.md`, `tests/test_advanced.pl`,
  `glue/js_glue.pl`, or the cross-target conformance harness/fixtures directly.

### Constraints
- Do NOT break the existing 48/48 conformance, the builtin probes, or the lowered-emitter
  tests — re-run all after changes. Interpreter tier only; no new runtime dependency.

### Acceptance (must pass before handoff)
1. `swipl -q -g run_tests -t halt tests/test_wam_javascript_builtins.pl` → green, incl.
   the new builtin-family probes.
2. `swipl -q -g run_tests -t halt tests/test_wam_javascript_lowered.pl` → green.
3. `CONFORMANCE_TARGETS=javascript swipl -q -g run_tests -t halt tests/test_wam_cross_target_conformance.pl` → still 48/48, no xfail/skip.
4. For each new family, show a probe's Node output next to SWI's and confirm they match.

### Handoff format
Return: the changed runtime/bindings/target files, the extended test file, the updated
status doc, `INTEGRATION_PATCH.md` (only if needed), and a note stating you branched from
`grok/wamjs-lowered-emitter` and EXTENDED it (list net additions), which builtins are full
vs partial (e.g. `sub_atom` modes, `format` directives covered), and which families you
left for a follow-up.

## ↑↑↑ Copy to here
