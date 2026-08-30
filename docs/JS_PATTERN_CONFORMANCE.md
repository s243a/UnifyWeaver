<!--
SPDX-License-Identifier: MIT OR Apache-2.0
Copyright (c) 2026 John William Creighton (@s243a)
-->
# JS Pattern Cross-Target Conformance Harness

The **JS pattern conformance harness** is the JavaScript-pattern analogue of
the WAM cross-target conformance harness
([`docs/WAM_CROSS_TARGET_CONFORMANCE.md`](WAM_CROSS_TARGET_CONFORMANCE.md)). It
compiles one **shared** fixture set with each JavaScript *pattern* target,
runs the generated code under that target's runtime, and diffs the result
against a single shared Prolog-oracle spec.

Before this harness, only the WAM backends had a cross-target parity net.
Each JS pattern target had its own per-target tests that re-declared their own
expectations, so a pattern backend could silently diverge from the shared
semantics with nothing noticing. This harness closes that gap.

## Files

| File | Role |
|------|------|
| `tests/js_pattern_conformance_fixtures.pl` | Shared spec: the programs, their query vectors, and the Prolog oracle. One source of truth. |
| `tests/test_js_pattern_cross_target_conformance.pl` | The harness: per-arm compile → run → diff, plus an always-run oracle self-check. |

## Running it

```sh
swipl -q -g run_tests -t halt tests/test_js_pattern_cross_target_conformance.pl
```

The suite is **green whenever the available arms agree with the oracle** (or
have no runnable programs). Arms whose runtime or target is absent are
reported *skipped*, never failed.

## Arms

| Arm | Compile path | Runtime | Notes |
|-----|--------------|---------|-------|
| `typescript` | `compile_recursion/3` (linear/tail/list-fold canned templates) and `compile_module/3` (factorial) | `tsc` (or `npx tsc`) then `node` | The robust, runnable pattern-target path today. |
| `annotated_js` | registry `compile_predicate/3` | `node` | Fuller JS pattern target being built in parallel. Skips cleanly when not registered in this worktree. |
| `vanilla_js` | registry `compile_predicate/3` | `node` | Same as `annotated_js`. |
| `clojurescript` | `compile_predicate_to_clojurescript/3` | `nbb` (node-babashka) | Numeric programs are currently `xfail` (the CLJS pattern recursion path is not yet clean); with `nbb` absent the whole arm skips. |

### The value oracle (contract difference from the WAM harness)

The WAM harness uses a **boolean** oracle — a ground query holds or it does
not, and every backend runs a 0-arity success/failure wrapper. The JS pattern
targets are **functional**: their generated code is a function that *returns a
value* (`fib(10)` → `55`). So the numeric fixtures here use a **value**
oracle:

```prolog
js_conformance_query(Program, Inputs, Expected).
%   Inputs   - the function's input args (Prolog terms)
%   Expected - the value the generated function must return
```

Expected values are hand-specified from standard Prolog semantics and
cross-checked at test time against a live Prolog oracle (`js_oracle/3`). The
`js_oracle_self_check` test runs every program's clauses and asserts the
hand-specified `Expected` matches actual Prolog evaluation — so the suite has
a meaningful, always-green test even with no JS toolchain installed.

### Program families

The pattern JS targets are **not** full WAM compilers. Programs are tagged by
family:

- **`numeric`** — a recursive numeric / list-fold function that the
  recursion-pattern codegen can emit and `node` can run (`fib`, `factorial`,
  `sum`, `listsum`). This is the family the pattern targets exercise today.
- **`structural`** — a classic list / head-shape predicate (`member`,
  `append`, `reverse`, and a nested-head `headshape` case). No pattern JS
  target compiles these today; they are carried in the spec so it honours the
  WAM-fixture philosophy (classics + a head-shape case) and so the fuller
  `annotated_js` / `vanilla_js` targets have a ready target to satisfy. The
  pattern-target arms **skip** the structural family cleanly.

## Skip behavior (skip, never fail)

The harness is defensive at every level:

1. **Missing runtime** (`node`, `tsc`, `nbb`) → the arm's plunit
   `condition/1` fails → the arm is reported **skipped**.
2. **Not-yet-existing target** (`annotated_js` / `vanilla_js` not registered
   in this worktree) → `ja_target_present/1` fails → the arm is **skipped**.
   These targets are being built in parallel; the harness runs green without
   them.
3. **Unsupported family** for a present arm (e.g. `structural` on
   `typescript`) → the program is **skipped and logged**, not failed.
4. **Genuine mismatch** on a *supported* program → a **real failure** (this
   is the safety net), unless the `(arm, program)` pair is registered in
   `ja_xfail/2` (tolerated and logged, like the WAM harness's `ct_xfail`).

## Environment knobs

Analogous to the WAM harness's `CONFORMANCE_TARGETS` / `CONFORMANCE_PROGRAMS`:

| Variable | Effect |
|----------|--------|
| `JS_CONFORMANCE_TARGETS` | Comma-separated arms to run (e.g. `typescript,clojurescript`). Default: all arms. |
| `JS_CONFORMANCE_PROGRAMS` | Comma-separated programs to run (e.g. `fib,sum`). Default: all programs. |

## Adding a target

Implement `ja_build/3` + `ja_run/4` (+ `ja_teardown/2`) for the new arm,
register it in `ja_arm/1` / `ja_default_arm/1`, declare which families it
supports with `ja_supported_family/2`, and add a `ja_runtime_present/1`
clause probing its toolchain. Missing toolchains skip rather than fail.
Known-divergent `(arm, program)` pairs go in `ja_xfail/2` so the suite stays
green while the underlying gap is tracked.
