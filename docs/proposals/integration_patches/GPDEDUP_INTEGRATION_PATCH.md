<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) 2026 John William Creighton (s243a) -->

# G-P-dedup Integration Patch (for INT-0)

**Task:** G-P-dedup — make the **TypeScript** pattern target (and, by
inheritance, `annotated_js` / `vanilla_js`) and the **Clojure** target (and, by
inheritance, ClojureScript → CLJS) honor the `constraint_analyzer`
`unique`/`unordered` constraints, the way the mature **rust**/**go** targets do:
a predicate declared `unique(true)` (or the default per `constraint_analyzer`)
emits a **deduplicated** result collection; `unordered(true)` additionally
permits **sort-based** dedup; `unique(false)` leaves output untouched.
**Worktree:** `agent-a373c29a127ede097`
**Shared-file rule:** this agent did **NOT** edit `constraint_analyzer.pl`
(consumed only), any `wam_*` file, the `annotated_js`/`vanilla_js` targets,
`core/target_registry.pl`, `docs/BINDING_MATRIX.md`, `tests/test_advanced.pl`,
or `glue/js_glue.pl`.

## No central wiring is required

All edits are inside the allowed set. `annotated_js` and `vanilla_js` inherit
the change automatically because both delegate `compile_facts/3` to
`typescript_target`; ClojureScript inherits it because
`clojurescript_target:compile_facts_to_clojurescript/3` delegates to the shared
`clojure_target:compile_facts_to_clojure/3`. **INT-0 has nothing to wire.**

## How rust/go apply constraints (the model mirrored)

- `go_target.pl:4029` calls `constraint_analyzer:get_constraints(Pred/Arity, C)`
  then `merge_go_options/3` (runtime options override declared constraints), and
  threads `option(unique(U), …, true)` into each emitter. `unique(true)` emits
  an order-preserving hash dedup (`seen := map[string]bool` / `map` keys for
  facts); `unique(false)` emits none. rust mirrors this.
- `constraint_analyzer` defaults are `unique(true)`, `unordered(true)`; the
  `sort -u` vs hash-dedup split (per `docs/CONSTRAINT_SYSTEM.md`) is what bash
  does. The JS/Clojure output shapes have no runtime option-merge layer at this
  site, so we call `get_constraints/2` directly (no runtime-override merge — a
  follow-up if per-call `unique(...)` options are ever wanted for these targets).

## Files changed (all inside the allowed set)

| File | Change |
|------|--------|
| `src/unifyweaver/targets/typescript_target.pl` | Added `:- use_module('../core/constraint_analyzer', [get_constraints/2])`. In `compile_facts/3`, call `get_constraints(Pred/Arity, C)` and build the fact-array initialiser via the new helper `ts_facts_rhs/3`. The `export const <pred>Facts: <T>Fact[] = ` prefix is unchanged (so the AJS/VJS rewriters keep handling it); only the RHS expression changes. `queryX`/`isX` read that array, so the whole facts surface inherits the dedup. |
| `src/unifyweaver/targets/clojure_target.pl` | Added the same `use_module`. In `compile_facts_to_clojure/3`, call `get_constraints/2` and build the `(def facts …)` value via the new helper `clojure_facts_expr/3`. `get-all`/`contains?`/`-main` read `facts`, so all inherit it. Uses only portable `clojure.core` (`distinct`/`sort`/`vec`), so CLJS needs no interop rewrite. |
| `tests/core/test_typescript_target.pl` | New `G-P-dedup` block: structural checks (default → `new Set`+`.sort()`+`JSON.parse`; `unique(false)` → raw array; `unique(true),ordered` → order-preserving, no sort) + node executions vs SWI `setof`. |
| `tests/core/test_clojurescript_target.pl` | New `G-P-dedup` block: structural checks (default → `(vec (sort (distinct …)))`; `unique(false)` → raw vector; ordered → `(vec (distinct …))`, no sort) + nbb executions vs SWI `setof`. |

## Emitted dedup (semantics)

For gathered facts `[…]`, keyed by `JSON.stringify` (TS) / vector identity (CLJ):

| unique | unordered | TypeScript RHS | Clojure `(def facts …)` |
|--------|-----------|----------------|--------------------------|
| true   | true (default) | `[...new Set([…].map(f=>JSON.stringify(f)))].sort().map(s=>JSON.parse(s))` | `(vec (sort (distinct […])))` |
| true   | false (ordered) | `[…].map(f=>JSON.stringify(f)).filter((s,i,a)=>a.indexOf(s)===i).map(s=>JSON.parse(s))` | `(vec (distinct […]))` |
| false  | *         | `[…]` (raw, unchanged) | `[…]` (raw, unchanged) |

The RHS is plain JavaScript / plain `clojure.core` — no TS-only type syntax and
no JVM interop — so it survives the `vanilla_js` type-strip, the `annotated_js`
JSDoc rewrite, and the CLJS interop rewrite untouched (all verified under
node / nbb).

## Output shapes covered vs deferred

- **Covered:** the **facts / query-helper** output shape (TS `compile_facts/3`;
  Clojure `compile_facts_to_clojure/3`), which is the analogue of go's
  `map[string]bool` facts and its `seen`-set rule dedup — the one clear
  single "result collection" site in these targets. AJS/VJS/CLJS inherit it.
- **Deferred (documented follow-up):** the recursion / native-clause / aggregate
  / streaming shapes emit a **single computed value or a generator/reducer**,
  not a materialised result multiset, so there is no clean single dedup site
  there (unlike rust/go's uniform query-plan/stream model). Dedup for those is a
  separate, larger task if ever needed and is **not** required for datalog-style
  fact/query parity. Also deferred: a **runtime-option override** merge
  (per-call `unique(...)`/`unordered(...)` in Options) at these sites, matching
  go's `merge_go_options/3` — currently only the declared/default constraints
  are consulted.

## Acceptance

All five suites green (0 failures), and node/nbb executions confirm duplicates
removed under `unique`/default and retained under `unique(false)`, cross-checked
against SWI `setof`. See the task report for verbatim output.
