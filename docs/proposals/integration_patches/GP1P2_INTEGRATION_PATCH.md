<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) 2026 John William Creighton (s243a) -->

# G-P1 / G-P2 Integration Patch (for INT-0)

**Task:** G-P1 (canned Fibonacci recursion hooks) + G-P2 (no structural
recursion) for the TypeScript pattern target.
**Worktree:** `agent-a01eb0b47eda96f05`
**Shared-file rule:** this agent did **NOT** edit `core/target_registry.pl`,
`docs/BINDING_MATRIX.md`, `tests/test_advanced.pl`, or `glue/js_glue.pl`, and
did **NOT** touch `runtime.js.mustache`, `wam_javascript_target.pl`, or any
`wam_*` file.

## Files changed (all inside the allowed set — no central wiring required)

| File | Change |
|------|--------|
| `src/unifyweaver/targets/typescript_target.pl` | The tree / multicall / direct-multicall recursion hooks now DERIVE the base cases, recursive-call offsets and aggregation from the actual clause body (G-P1); a new native structural-list lowering path (`native_ts_structural/3`) compiles member/append/list-length/reverse-accumulator shapes to real recursive TS over arrays (G-P2); the arity-2 tail hook no longer stubs `return items.length`. |
| `tests/core/test_typescript_target.pl` | Added non-fib tree/multicall/direct derivation tests, a fib-shape control, a no-clause fallback test, and structural member/append/length tests — including node-run cross-checks against the SWI oracle (gated on node ≥ 22). |

**No shared-file change is needed for this task.** Both edited files are in the
allowed set. The fix propagates to `annotated_js` / `vanilla_js` automatically
(they delegate each `compile_*_pattern` clause to the `typescript` clause and
then type-strip); both inheritor suites stay green.

## Backward compatibility

- The four numeric hooks fall back to the canned memoized Fibonacci body only
  when there is **no recursive clause to derive from** (e.g. the bare
  dispatch-shape calls in `test_vanilla_js_target.pl` that pass a pred name with
  no `user:` clauses and `RecClauses = []`). When real clauses are present the
  body is derived. fib/factorial/sum keep producing the same result.
- The structural path is tried before the numeric native path in
  `compile_predicate_to_typescript/3` and only fires for genuine list-recursive
  predicates (a cons pattern in a recursive-clause head), so facts and numeric
  predicates are unaffected.

## Optional follow-up (NOT done here — noted so it isn't re-raised)

`tests/js_pattern_conformance_fixtures.pl` already carries the structural
programs (member/append/reverse/headshape). The `typescript` arm of
`tests/test_js_pattern_cross_target_conformance.pl` still declares only
`ja_supported_family(typescript, numeric)`, so it SKIPs the structural family
and the harness stays green.

TypeScript can now compile **member/2, append/3 and list-length/2** end-to-end
(verified under node in the core suite). A future harness enhancement could add:

```prolog
ja_supported_family(typescript, structural).
% + a ja_build(typescript, structural, ...) adapter that calls
%   typescript_target:compile_predicate/3 and appends a per-shape driver
% + ja_xfail(typescript, reverse).    % crev/2 -> crev_acc/3 is cross-predicate
% + ja_xfail(typescript, headshape).  % compound-term heads (cnum/csym/cwd) unsupported
```

This was left out because `reverse` (a two-predicate wrapper `crev/2 → crev_acc/3`)
and `headshape` (compound terms as tagged head shapes) are outside the scope of
the single-predicate structural lowering, so flipping the family on wholesale
would turn those two programs from clean SKIPs into failures without the xfail
rows above. `crev_acc/3` on its own does compile and run correctly.
