<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) 2026 John William Creighton (s243a) -->

# G-P7 Integration Patch (for INT-0)

**Task:** G-P7 (minimal slice) — extend guard-goal codegen in the TypeScript and
Clojure pattern targets so the two guard families the shared classifier already
routes to the guard renderer — **negation** (`\+`/`not`) and **type-check
predicates** (`integer/1`, `atom/1`, `is_list/1`, ...) — actually render instead
of failing. Regex `match/2,3` guards and `constraint_analyzer.pl` dedup are the
explicitly-deferred follow-ups and are NOT in this slice.

**Worktree:** `agent-a2f35778cf87f66d0`

**Shared-file rule:** this agent did **NOT** edit
`src/unifyweaver/core/clause_body_analysis.pl` (the shared classifier is correct
— it already classifies `\+`/`not` and `type_check_pred/1` goals as guards), nor
any `wam_*` file, `core/target_registry.pl`, `docs/BINDING_MATRIX.md`,
`tests/test_advanced.pl`, or `glue/js_glue.pl`.

## Files changed (all inside the allowed set — no central wiring required)

| File | Change |
|------|--------|
| `src/unifyweaver/targets/typescript_target.pl` | Added clauses to `ts_guard_condition/3` for `\+(Inner)`, `not(Inner)`, `member(X,List)`, and unary type-check predicates, plus helpers `ts_member_list/3`, `ts_member_elem/3`, `ts_type_check/4`. Reuses the existing `ts_expr/3`. Inherited unchanged by annotated_js and vanilla_js (they delegate to TS then type-strip). |
| `src/unifyweaver/targets/clojure_target.pl` | Symmetric clauses added to `clojure_guard_condition/3` (`\+`/`not` → `(not ...)`, `member` → `(some #(= % x) list)`, type-checks), plus helpers `clojure_member_list/3`, `clojure_member_elem/3`, `clojure_type_check/4`. Reuses the existing `clojure_expr/3`. Inherited unchanged by ClojureScript (CLJS rewrites host calls then falls through to clojure clause compilation). |
| `tests/core/test_typescript_target.pl` | Added a `G-P7` section: structural tests (batch + pipeline) and three `condition(node_available)` node-execution tests vs the SWI oracle. Reuses the existing `node_available/0` and `ts_write_run_stdin/3` helpers. |
| `tests/core/test_clojurescript_target.pl` | Added a `G-P7` section: structural tests plus two `condition(nbb_available)` nbb-execution tests vs the SWI oracle. Added `nbb_available/0`, `cljs_write_run/3`, `cljs_nbb_exec/3` helpers and `library(process)`/`library(lists)` imports. |

**No central wiring is required.** No new public exports (the helpers are
module-internal; dispatch is through the already-exported guard renderers and the
`clause_body_analysis:render_guard_condition/4` multifile hooks, which are
unchanged).

## Renderer mapping (both targets)

Atoms are represented as strings in both targets, unbound vars as
`undefined`/`nil`, lists/compounds as arrays/collections — the type-check
mapping follows those conventions.

| Prolog guard | TypeScript | Clojure |
|---|---|---|
| `\+ G` / `not(G)` | `!(<render G>)` | `(not <render G>)` |
| `member(X, L)` | `L.includes(x)` (array literal or bound name) | `(some #(= % x) L)` (vector literal or bound name) |
| `integer(X)` | `Number.isInteger(x)` | `(integer? x)` |
| `float(X)` | `(typeof x === "number" && !Number.isInteger(x))` | `(float? x)` |
| `number(X)` | `typeof x === "number"` | `(number? x)` |
| `atom(X)` | `typeof x === "string"` | `(string? x)` |
| `is_list(X)` | `Array.isArray(x)` | `(sequential? x)` |
| `compound(X)` | `(typeof x === "object" && x !== null)` | `(coll? x)` |
| `var(X)` | `(x === undefined)` | `(nil? x)` |
| `nonvar(X)` / `ground(X)` | `(x !== undefined)` | `(some? x)` |

## Negation of a non-guard goal

`\+ Inner` recurses into the same guard renderer for `Inner`. `member/2` is
handled specially (positive `member` is not classified as a guard upstream, so
it is only reached via negation). For any **other** non-guard inner goal the
recursive render call **fails**, so `ts_guard_condition/3` /
`clojure_guard_condition/3` fail cleanly — no code is emitted rather than wrong
code. In the batch native path this makes the enclosing goal-sequence fall back
to `clause_guard_output_split/4`; no invalid output is ever produced.

## Notes / forward-compat (no action required for this slice)

- **`atomic/1`** is not currently in `clause_body_analysis:type_check_pred/1`, so
  a bare `atomic(X)` is not routed to the guard renderer. Both renderers already
  carry an `atomic` clause (TS: not-an-object; Clojure: `(not (coll? x))`) so if
  the shared classifier later adds `atomic` to `type_check_pred/1`, no target
  change is needed.
- **Pre-existing (out of scope):** a *guard-only* predicate compiled through the
  **batch** native path returns the `"error"`/`nil` sentinel (and the arity-1
  batch function signature omits its parameter). This predates G-P7. The clean,
  value-returning routes exercised by the acceptance tests are TS
  streaming filter/transform (`mode(pipeline)`/`mode(generator)`) and the
  Clojure/CLJS if-then-else lowering; flagged here only so INT-0 is aware.
