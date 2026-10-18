<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) 2026 John William Creighton (s243a) -->

# CLJSFIX Integration Patch

Shared-file notes for the **CLJSFIX (clean numeric recursion codegen, drop
PAR-1 xfail)** work. Per the delegation plan's shared-file rule, this worktree
does **not** touch `core/target_registry.pl`, `docs/BINDING_MATRIX.md`,
`src/unifyweaver/core/advanced/test_advanced.pl`, or `glue/js_glue.pl`. Nothing
in this card actually needs any of those four hot files; the one item below is
optional and recorded here for INT-0 only.

## Root cause

The ClojureScript numeric programs (`fib`, `factorial`, `sum`, `listsum`) were
marked `ja_xfail` in the PAR-1 harness because the generated CLJS did not run.
The gap was in the **shared Clojure base** (`clojure_target.pl`) native
clause-lowering path, not in the JVM→JS interop rewrite:

- Recursive/predicate-call body goals (e.g. `cfib(N1,R1)`) were **dropped**
  entirely. `clojure_output_goal/4` only handled `is/2` and `=/2`, so a call
  goal fell through to a catch-all that produced no binding and no code. Its
  output var then leaked into the return expression as a raw Prolog var name
  (`_62080`), and the surrounding `(let [...]` forms were emitted **unclosed**
  (open `(let [v ...]` lines with no matching `)`), yielding unbalanced,
  non-runnable code.
- `listsum` never reached the native path at all — a cons head `[H|T]` was not
  destructured, so the predicate fell through to the "Hello from …" stub.

Because the bug is in the shared base and is host-neutral (it breaks the JVM
Clojure output identically — the JVM Clojure tests just never exercised a
multi-goal recursive clause), the fix belongs in `clojure_target.pl`, guarded by
re-running the JVM Clojure suite. See "Files changed in-worktree" below.

## Files changed in-worktree (NOT hot shared files — applied here directly)

- `src/unifyweaver/targets/clojure_target.pl` — the fix. Added a straight-line
  recursion/computation lowering path to `native_clojure_clause/5`:
  - `clojure_input_head_analysis/6` — input-head guards + destructuring:
    plain var (nothing), `[]` (`(empty? argN)`), `[H|T]` (`(seq argN)` +
    `h (first argN)` / `t (rest argN)` binds), other literal (`(= argN lit)`).
  - `clojure_straightline_goals/1` — recognises a body that is only guards and
    single-output value goals (`is` / `=` / predicate-call with a var last arg),
    no control flow; control-flow bodies fall through to the existing
    `classify_goal_sequence` path unchanged.
  - `clojure_lower_outputs/5` + `clojure_output_rhs/3` — lower value goals to
    `Name-Expr` let-binding pairs; a predicate call `p(In…,Out)` becomes
    `(p cIn…)`. The last goal's value is the return expression.
  - `clojure_wrap_let/3` — emits a **single, properly closed**
    `(let [n1 e1 n2 e2 …] ret)` (or just `ret` when there are no bindings).
  - `clojure_head_conditions/4` — now emits `(empty? argN)` for `[]` and a
    non-empty guard for a cons in the generic path (was `(= argN "[…]")`).
  - `clojure_native_cli_entry/4` + `clojure_pred_list_input/2` — the standalone
    CLI entry parses the single argv as a comma-separated integer vector when the
    predicate's first argument is a list, else as a single integer.
    `Integer/parseInt` is emitted so the bb (JVM) path stays valid; the CLJS
    interop rewrite maps it to `js/parseInt` for nbb/Scittle.
- `src/unifyweaver/core/recursive_compiler.pl` — the CLJS transitive-closure
  path threads `TcOptions` into `clojurescript_from_clojure/3` (this is the
  CLJS-1 additive change; not a hot file).
- `tests/test_js_pattern_cross_target_conformance.pl` — removed the four
  `ja_xfail(clojurescript, {fib,factorial,sum,listsum})` entries; the CLJS
  numeric arm now asserts correctness.
- `tests/core/test_clojurescript_runtime_smoke.pl` — added nbb (and bb, gated)
  numeric-recursion arms: fib, factorial, sum, listsum (incl. empty list).

## 1. `core/target_registry.pl` — optional, INT-0 only (no CLJSFIX dependency)

CLJSFIX does not require any registry change: `compile_predicate_to_clojurescript/3`
is called directly and the `clojurescript` target is already registered on
`main`. The `nbb`/`babashka` capability additions belong to the **CLJS-1** patch
(`CLJS-1_INTEGRATION_PATCH.md`) and are not duplicated here.

## Acceptance (nbb present)

- Generated CLJS for fib/factorial/sum/listsum runs under `nbb` and matches the
  Prolog oracle (fib 10→55, factorial 6→720, sum 10→55, listsum [1,2,3,4]→10,
  listsum []→0).
- `test_clojurescript_target`, `test_clojurescript_recursive`,
  `test_clojurescript_runtime_smoke` (nbb arms run, not skipped), and
  `test_clojure_native_lowering` (JVM base, no regression) all green.
- PAR-1 clojurescript numeric arm passes with the xfail removed
  (`JS_CONFORMANCE_TARGETS=clojurescript`), and the full PAR-1 run is green.
