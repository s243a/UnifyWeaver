# Mode-Directed Tabling Runtime Support for the C# Query Engine

This document supersedes the earlier handoff-style note for this work.

Previous version:
- `git show d0a1ee91:docs/proposals/CODEX_HANDOFF_MODE_DIRECTED_TABLING.md`

## Purpose

Add mode-directed tabling semantics to the C# parameterized query engine
for path-aware recursive closure.

The immediate target is `min`/`max`-style pruning for shortest-path-like
queries. The broader goal is to let recursive query lowering distinguish
between:

- `All` paths must be preserved
- only the best path per key matters
- other aggregation-like accumulator modes may be added later

## Current State

### Phase 1: Compiler-side support is complete

The compiler now parses `:- table pred(_, _, min).` directives and
threads the resulting mode information into C# query plans.

Files already changed:

1. `src/unifyweaver/core/advanced/pattern_matchers.pl`
   - `declared_table_modes/3`
   - `parse_table_spec/2`
   - `has_directed_table_mode/4`
   - `table_enables_pruning/2`

2. `src/unifyweaver/targets/csharp_target.pl`
   - `maybe_path_aware_transitive_closure_plan` now records `table_modes`
   - `emit_plan_expression` for `path_aware_transitive_closure` emits a
     `TableMode.*` argument
   - `table_mode_csharp_enum/2` maps Prolog table modes to C# enum names

3. `tests/core/test_csharp_query_target.pl`
   - path-aware transitive-closure plan assertions were updated to check
     the threaded `table_modes`

Current compiler output shape:

```csharp
new PathAwareTransitiveClosureNode(
    new PredicateId("category_parent", 2),
    new PredicateId("category_ancestor", 3),
    1, 1, 10, TableMode.Min)
```

The missing work is runtime support for `TableMode`.

## Recent Semantic Correction

This design must be read in light of the later path-multiplicity fix.

The C# query runtime previously collapsed distinct simple paths when
they happened to produce the same `(target, depth)` or
`(target, accumulator)` tuple. That was corrected in:

- `6420efdf` `fix(csharp-query): preserve path multiplicity`

That means:

- `All` mode must preserve full per-path semantics
- `All` mode must not reintroduce tuple-level deduplication
- only `Min`/`Max`/similar directed-table modes should use
  best-known-value pruning

This is the most important correction to the earlier handoff note.

## Runtime Design

### Step 2.1: Add `TableMode` enum

```csharp
public enum TableMode
{
    All,
    Min,
    Max,
    First,
    Sum,
    Count
}
```

Initial required behavior:

- `All`:
  - preserve current per-path visited semantics
  - do not prune by best-known accumulator
  - do not dedup by `(target, depth)`

- `Min`:
  - keep the best minimum accumulator per key
  - prune worse-or-equal branches

- `Max`:
  - symmetric to `Min`

The remaining modes can be stubbed or left unimplemented if needed, but
the enum should reflect the compiler-side vocabulary already parsed.

### Step 2.2: Add `TableMode` to `PathAwareTransitiveClosureNode`

```csharp
public sealed record PathAwareTransitiveClosureNode(
    PredicateId EdgeRelation,
    PredicateId Predicate,
    int BaseDepth,
    int DepthIncrement,
    int MaxDepth = 0,
    TableMode AccumulatorMode = TableMode.All
) : PlanNode;
```

This should remain backward compatible for plans without an explicit
table directive.

### Step 2.3: Implement pruning in `AppendPathAwareRowsForSeed`

The runtime currently enumerates path-aware rows for `All` mode.

For directed-table modes:

- `Min`
  - maintain `bestKnown` per `(seed, target)` or per target within a
    given seeded expansion
  - prune when `nextDepth >= bestKnown[target]`
  - update the best bound when a strictly better depth is found

- `Max`
  - prune when `nextDepth <= bestKnown[target]`

Conceptually:

```csharp
var bestKnown = new Dictionary<object?, int>();

if (accumulatorMode == TableMode.Min)
{
    if (bestKnown.TryGetValue(next, out var best) && nextDepth >= best)
        continue;
    bestKnown[next] = nextDepth;
}
else if (accumulatorMode == TableMode.Max)
{
    if (bestKnown.TryGetValue(next, out var best) && nextDepth <= best)
        continue;
    bestKnown[next] = nextDepth;
}
```

### Important semantic note

The earlier handoff said:

- "`All` mode should keep current behavior with `emitted` HashSet"

That is no longer correct.

After the path-multiplicity fix, `All` mode should keep:

- per-path visited cycle prevention
- full path multiplicity

It should **not**:

- dedup by `(target, depth)`
- dedup by `(target, accumulator)`

So:

- `Min`/`Max` use best-known-value pruning
- `All` remains multiplicity-preserving

### Step 2.4: Update plan/debug formatting

Plan explanation strings should include the directed-table mode:

```csharp
PathAwareTransitiveClosureNode closure =>
    $"PathAwareTransitiveClosure edge={closure.EdgeRelation} maxDepth={closure.MaxDepth} mode={closure.AccumulatorMode}"
```

That makes benchmark traces and runtime debugging much easier.

## Correctness Model

### `All` mode

Semantics:

- enumerate all simple paths up to `MaxDepth`
- avoid only repeated nodes within a single path
- preserve path multiplicity

Suitable for:

- effective distance
- category influence propagation
- semantic-distance-style weighted accumulations

### `Min` mode

Semantics:

- return only the shortest path accumulator per key
- prune branches once they cannot improve the best-known result

Suitable for:

- shortest path to root
- plain nearest-ancestor or minimal-hop queries

### Relationship to per-path visited

Mode-directed tabling is not a replacement for per-path visited state.

They solve different problems:

- per-path visited prevents infinite cyclic derivations
- directed tabling decides which accumulated results are worth keeping

Both are needed.

## Testing Plan

### Existing tests

The current `test_pathaware_reach/3` shape should remain `TableMode.All`
unless explicitly table-directed.

That protects backward compatibility and the all-path semantics we
recently fixed.

### New tests for `Min`

Add a dedicated fixture such as:

```prolog
:- dynamic user:table/1.
assertz(user:table(test_pathaware_min_reach(_, _, min))).
```

with a graph containing:

- at least two different simple paths to the same target
- one path strictly shorter than the other

Expected behavior:

- `All` mode returns both path-derived rows when multiplicity matters
- `Min` mode returns only the shortest-path result per `(source, target)`

### Runtime verification

Use both:

- codegen-only suite coverage
- direct `dotnet` harness execution where available

The harness portability work in:

- `4d08dfbe` `fix(test): make csharp harness env fallback portable`

means these targeted runtime checks can now run on Unix-like systems as
well as Windows.

## Benchmark Workloads

### Primary evaluation workload

- `examples/benchmark/shortest_path_to_root.pl`

This is the clearest benchmark for `Min` mode because the query contract
only needs the best path, not all paths.

### Contrast workloads

- `examples/benchmark/effective_distance.pl`
  - should stay `All`
  - no pruning benefit from `Min`
  - useful as a semantic counterexample

- `examples/benchmark/category_influence.pl`
  - likely also an `All`-paths semantic workload

## Current Benchmark Context

Post path-multiplicity fix, effective-distance runtime looks like:

| Scale | C# Query | C# DFS | Rust DFS | Query vs C# DFS |
|-------|----------|--------|----------|-----------------|
| 300 | 0.598s | 0.435s | — | match |
| 1k | 0.407s | 1.249s | 1.361s | match |
| 5k | 1.247s | 5.362s | 7.327s | match |
| 10k | 3.125s | 9.961s | 13.676s | match |

This matters because:

- `All` mode is now semantically correct
- `Min` mode is the next optimization layer, not a substitute for that fix

Expected impact for shortest-path workloads:

- tuple counts should drop significantly relative to `All`
- pruning should restore a stronger speedup curve for shortest-path
  queries

## Recommended Implementation Order

1. Add `TableMode` to the C# runtime
2. Extend `PathAwareTransitiveClosureNode` to carry it
3. Implement `Min` branch-and-bound pruning
4. Add `Max` if trivial from the same structure
5. Add shortest-path regression tests
6. Benchmark `shortest_path_to_root.pl`
7. Only then consider extending directed-table semantics to
   `PathAwareAccumulationNode`

That last step should be separate, because weighted accumulations may
need stronger monotonicity assumptions before pruning is correct.

## Commits and Documents to Review

Relevant commits:

- `0097b5be` — table directive parsing + C# plan builder wiring
- `b6cf89f5` — shortest-path benchmark
- `d0a1ee91` — earlier handoff note version of this document
- `6420efdf` — path multiplicity fix
- `4d08dfbe` — portable harness env fallback

Related docs:

- `docs/proposals/MODE_DIRECTED_TABLING_PROPOSAL.md`
- `docs/proposals/COMPUTED_RECURSIVE_INCREMENT_PHILOSOPHY.md`
- `docs/proposals/COMPUTED_RECURSIVE_INCREMENT_SPEC.md`
- `docs/proposals/COMPUTED_RECURSIVE_INCREMENT_PLAN.md`

## Open Questions

1. Should `First` mode be implemented now or deferred until `Min`/`Max`
   are stable?
2. Should `bestKnown` be keyed only by target within a seeded expansion,
   or by a wider tuple if future grouped closures reuse the machinery?
3. Should pruning support be extended only to counted closure first, with
   weighted accumulation deferred until monotonicity is explicit?

## Changelog

- Converted this file from a handoff note into a corrected design doc.
- Updated the semantics to reflect the later path-multiplicity fix.
- Removed the outdated claim that `All` mode should keep tuple-level
  deduplication.
- Replaced stale benchmark context with current post-fix numbers.
- Added explicit linkage to the prior document revision in git.
- Edited by `gpt-5.4 (medium)`.
