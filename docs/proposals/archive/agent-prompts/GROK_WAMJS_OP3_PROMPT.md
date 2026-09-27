<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) 2026 John William Creighton (s243a) -->

# Grok prompt — user-defined operators (op/3) for the wam_javascript parser (G-W2 follow-up)

> Branch from `grok/wamjs-term-meta` (latest JS-WAM) and EXTEND — do not rebuild.
> NOTE: this is the last substantive WAM parity item; the JS WAM lane is otherwise
> at practical parity with the mature fleet.

---

## Copy from here ↓↓↓

You are contributing to **UnifyWeaver** (`github.com/s243a/unifyweaver`), a compiler
that lowers Prolog to a shared WAM bytecode and runs it on per-language WAM interpreters.
Branch from **`grok/wamjs-term-meta`** (the latest JS-WAM state) as `grok/wamjs-op3`.
Prolog dialect is SWI-Prolog (`swipl`); the JS runtime targets Node (`node`, v18+).

### The gap (G-W2 follow-up)
The JS WAM term parser (`Runtime.parse_term` in `templates/targets/javascript_wam/runtime.js.mustache`)
uses a fixed ISO default operator table (so `1+2` reads as `+(1,2)`), but does not support
**user-defined operators** declared via `op/3`, nor **postfix** operators. Add them.

### Goal
1. Support the `op/3` directive so a program can declare operators
   (`:- op(Priority, Type, Name)` — Type ∈ `xfx xfy yfx` (infix), `fy fx` (prefix),
   `yf xf` (postfix)) and the parser then reads terms using the combined default + declared
   table. At minimum: dynamic infix + prefix; postfix if it lands cleanly.
2. Make the Pratt reader consult this dynamic table (declared ops override/extend the ISO
   defaults at their given priority/associativity).

### Reference
- The existing ISO default table + Pratt reader in
  `templates/targets/javascript_wam/runtime.js.mustache` (extend it — don't rewrite).
- How a mature WAM/runtime handles `op/3` (grep rust/haskell/swipl-style op tables) for the
  priority/associativity rules; SWI's `op/3` semantics are the oracle.
- `wam_javascript_target.pl` — where directives/program setup are emitted, to thread declared
  ops into the runtime's op table at startup.

### The six WAM conventions (do not regress)
1. Cons: `put_list` AND `put_structure [|]/2` intern the same functor; `[]` is the atom.
2. Functor `name/arity` where name may contain `/` (`///2`); parse arity as trailing `/<digits>`.
3. Nested terms outer-first via placeholders — `put_*` must bind+trail the placeholder.
4. `deref` before every type test.  5. `is/2` yields an integer for integral results.
6. Unhandled instruction ⇒ a real one-slot NoOp.

### Deliverables (SPDX headers)
- Extend `templates/targets/javascript_wam/runtime.js.mustache` (dynamic op table + Pratt
  consultation) and, if needed, `wam_javascript_target.pl` (emit declared ops into the
  runtime's initial op table from `:- op/3` directives in the source program).
- Update `docs/WAM_JAVASCRIPT_STATUS.md`: op/3 (infix/prefix[/postfix]) supported; note residuals.
- Extend `tests/test_wam_javascript_builtins.pl` with an op/3 probe: declare a custom infix
  op, parse/run a term using it, matching SWI.
- `INTEGRATION_PATCH.md` ONLY if a shared file must change. Do NOT edit
  `core/target_registry.pl`, `docs/BINDING_MATRIX.md`, `tests/test_advanced.pl`,
  `glue/js_glue.pl`, or the cross-target conformance harness.

### Constraints
- Do NOT break the 48/48 conformance, the builtin/parser/fact-source/term-meta probes, or
  the lowered-emitter tests — re-run all. Interpreter tier; no new runtime dependency.
- If postfix proves fiddly, ship infix+prefix `op/3` and note postfix as follow-up.

### Acceptance (must pass before handoff)
1. `swipl -q -g run_tests -t halt tests/test_wam_javascript_builtins.pl` → green incl. the op/3 probe.
2. `swipl -q -g run_tests -t halt tests/test_wam_javascript_lowered.pl` and
   `swipl -q -g run_tests -t halt tests/test_wam_javascript_fact_sources.pl` → green.
3. `CONFORMANCE_TARGETS=javascript swipl -q -g run_tests -t halt tests/test_wam_cross_target_conformance.pl` → still 48/48, no xfail/skip.
4. Show a custom-operator term parsed + run under `node`, matching SWI.

### Handoff format
Return: the changed runtime/target/test files, the updated status doc, `INTEGRATION_PATCH.md`
(only if needed), and a note stating you branched from `grok/wamjs-term-meta` and EXTENDED it
(net additions), which op types landed (infix/prefix/postfix), and any residual.

## ↑↑↑ Copy to here
