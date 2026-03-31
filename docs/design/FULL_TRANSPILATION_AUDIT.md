# Full Transpilation Pipeline Audit

## Context

The effective distance benchmark currently uses `generate_pipeline.py`
to produce self-contained programs per target. This audit tests whether
UnifyWeaver can compile the **full** `effective_distance.pl` natively
to each target, eliminating the generator script.

## Predicates in the Pipeline

```
effective_distance/3
  ├── setof/3 (collect unique articles)
  ├── member/2 (iterate articles)
  ├── root_category/1 (fact lookup)
  ├── aggregate_all/3 (sum over paths)
  │   └── path_to_root/3
  │       ├── article_category/2 (fact lookup)
  │       ├── root_category/1 (fact lookup)
  │       └── category_ancestor/3 (recursive transitive closure)
  │           └── category_parent/2 (fact lookup)
  └── is/2 (arithmetic: Hops^(-5), WeightSum^(-0.2))
```

## Audit Results

### category_ancestor/3 (recursive transitive closure + arithmetic)

| Target | Status | Notes |
|--------|--------|-------|
| C# Query | ✅ | With mode declaration + body-constant workaround |
| Go | ✅ | Via compile_general_recursive_to_go |
| Python | ✅ | Via generate_ternary_worker |
| AWK | ✅ | Via DFS fixpoint in BEGIN block |
| Rust | ✅ | Via compile_general_recursive_to_rust |

All targets compile this predicate. This was the focus of the deepening
work in PRs #1054-#1056.

### path_to_root/3 (multi-clause, calls recursive + fact predicates)

| Target | Status | Gap |
|--------|--------|-----|
| C# Query | ❌ | "variable not bound" — multi-clause with different body structures |
| Go | ❌ | "multiple rules without match constraints" — same as category_ancestor was before deepening |
| Python | ⚠️ | Compiles but output may not be functional (schema warnings) |
| AWK | ✅ | Compiles (generates OR pattern) |
| Rust | ❌ | Silent failure — no output |

**Root cause**: `path_to_root/3` has two clauses with different body
structures:
- Clause 1: `article_category(A, Cat), root_category(Cat), Root = Cat`
- Clause 2: `article_category(A, Cat), category_ancestor(Cat, Anc, H), root_category(Anc), Root = Anc, Hops is H + 1`

The second clause calls `category_ancestor/3` (a recursive predicate)
while the first doesn't. This **multi-clause heterogeneous body** pattern
isn't handled by most targets' compilation paths.

### effective_distance/3 (aggregation + built-ins)

| Target | Status | Gap |
|--------|--------|-----|
| C# Query | ❌ | Can't compile path_to_root dependency |
| Go | ❌ | Infinite debug loop on setof/member/aggregate_all |
| Python | ⚠️ | "Compiles" but setof/member/aggregate_all not functional |
| AWK | ⚠️ | "Compiles" but setof/member/aggregate_all not functional |
| Rust | ⚠️ | "Compiles" but built-ins not implemented |

**Root cause**: `effective_distance/3` uses built-in predicates that
no target compiles to functional code:
- `setof/3` — collect unique solutions
- `member/2` — iterate over a list
- `aggregate_all/3` — aggregate over all solutions of a goal

The C# Query Engine supports `aggregate_all/3` natively (AggregateNode),
but the other targets would need target-specific implementations of
these higher-order predicates.

## Gap Summary

| Gap | Targets Affected | Effort |
|-----|-----------------|--------|
| Multi-clause heterogeneous bodies | C#, Go, Rust | Medium — extend multi-rule compilation to handle different body predicate types per clause |
| `setof/3` compilation | All except C# (partial) | High — requires collecting solutions, deduplicating |
| `member/2` compilation | All | Medium — iterate over in-memory list |
| `aggregate_all/3` compilation | Go, Python, AWK, Rust | High — requires collecting all solutions then aggregating |
| Predicate-calls-predicate | All | Medium — compile multiple predicates and wire calls between them |

## What Works End-to-End Today

Only the **C# Query Engine** can compile the combination of
`category_ancestor/3` + `aggregate_all/3` in a single query plan.
But it can't compile `path_to_root/3` or `effective_distance/3`
because of the multi-clause and `setof/member` issues.

**No target** can compile the full `effective_distance.pl` natively today.

## Recommendations

### Short-term: Keep the generator script

`generate_pipeline.py` remains the correct approach for the benchmark.
Each target implements the same algorithm (DFS + aggregation) in native
code, which is what we want for fair comparison.

### Medium-term: Fix multi-clause compilation

The most impactful fix is supporting **multi-clause predicates with
heterogeneous bodies** (like `path_to_root/3`). This would let targets
compile predicates that dispatch between fact lookup and recursive calls.
This is the same "multiple rules" issue that was already fixed for
`category_ancestor` in Go/Rust — it needs extending to handle clauses
that call different predicates.

### Long-term: Built-in predicate compilation

`setof/3`, `member/2`, and `aggregate_all/3` need target-specific
implementations. The C# Query Engine is closest (has AggregateNode).
Other targets would need:
- A solution-collection framework (equivalent of `findall`)
- Aggregation operators (sum, count, etc.)
- List operations (member, append, etc.)

This is the work described in the original benchmark spec
(`docs/proposals/CROSS_TARGET_EFFECTIVE_DISTANCE_SPEC.md`).
