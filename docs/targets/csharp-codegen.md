# C# Code Generation Target (`target(csharp_codegen)`)

`csharp_codegen` (and its native implementation `csharp_native`) emits idiomatic C# source files that mirror the Bash streaming templates. It builds on the existing `csharp_native_target.pl` module and is intended for scenarios where developers want tangible .NET source artefacts that can be compiled, debugged, and extended manually.

## Current Scope
- Recursive predicates: Supported via semi-naive iteration with HashSet deduplication. Two code styles available:
  - **Inline (default)**: Explicit `while (delta.Count > 0)` loops in generated code
  - **LINQ style**: `TransitiveClosure()` extension method via `linq_recursive(true)` option
- Non-recursive predicates: Facts, single-rule bodies, and multi-clause unions translate to LINQ pipelines that operate on in-memory arrays.
- Dedup semantics: Follows the Bash model—`Distinct()` for `unique(true)`, ordered variants are pending.
- Generated structure: Each predicate becomes a static class in the `UnifyWeaver.Generated` namespace with:
  - Fact arrays (`(string,string)[]` or similar).
  - Helper methods (`PredicateFactStream`, `PredicateStream`, `PredicateReverseStream`).
  - An optional `Main` for command-line execution.
  - Inline XML summary and usage comment ahead of the primary stream method for quick reference.
- Configuration: `compile_predicate_to_csharp/3` accepts options such as dedup strategy and will soon recognise `target(csharp_codegen)` explicitly.

## Tuple Representation & Arity Limits
- Rows are emitted as C# `ValueTuple` instances. This keeps generated code idiomatic and avoids custom record types while still enabling deconstruction.
- The current backend supports predicate arities 1–3 end-to-end (fact arrays, helpers, and joins). The design scales to 7 columns, which is the practical limit for `ValueTuple` before nesting or custom structs become necessary.
- Predicates that exceed this limit—or that require sparse/cached representations—should fall back to the C# query runtime target (or another backend) where we can swap in alternative storage strategies.

## Pipeline Semantics
- Clause compilation tracks each logical variable through the join plan. After every join we immediately project down to the minimal set of variables required by remaining joins and by the clause head. This prevents the `(X,Y,Y,Z)`-style blow-up that naïve joins can create.
- Because the generated code uses LINQ iterators, evaluation stays lazy: projections run before the next element is requested, so intermediate tuples are never materialised as large buffers.
- Variable reuse in heads (e.g. `p(X,Y,X)`) is handled by a final projection that reorders and duplicates fields as needed before results are returned or printed.

## Performance Characteristics
- **Memory:** O(1) for streaming execution. LINQ iterators only keep the current tuple from each source Enumerable.
- **Time:** Each join is currently a nested-loop join (`Join` over the in-memory arrays). This is adequate for modest data sets and mirrors the Bash target; larger workloads should migrate to the query runtime for hash/semi-naive evaluation.
- **Optimisations:** Because joins drop unused variables immediately, the tuple width stays bounded by the clause head and upcoming join keys. No sparse representation is required in this layer; more advanced layouts can be introduced in the query runtime if needed.

## Outer Join Support

The C# codegen target supports LEFT, RIGHT, and FULL OUTER joins through automatic pattern detection.

### Syntax

```prolog
% LEFT JOIN - all left records, matched right or null
left_join(X, Z) :-
    left_table(X, Y),
    (right_table(Y, Z) ; Z = null).

% RIGHT JOIN - all right records, matched left or null
right_join(X, Z) :-
    (left_table(X, Y) ; X = null),
    right_table(Y, Z).

% FULL OUTER JOIN - all records from both sides
full_outer(X, Z) :-
    (left_table(X, Y) ; X = null),
    (right_table(Y, Z) ; Z = null).
```

### Implementation

The compiler generates LINQ code using:
- `GroupJoin` + `SelectMany` + `DefaultIfEmpty` for LEFT JOIN
- Swapped tables with LEFT JOIN pattern for RIGHT JOIN
- LEFT JOIN concatenated with unmatched right records for FULL OUTER

Example generated code for LEFT JOIN:
```csharp
LeftTableStream()
    .GroupJoin(RightTableStream(),
               left => left.Y,
               right => right.Y,
               (left, rightGroup) => new { left, rightGroup })
    .SelectMany(
        x => x.rightGroup.DefaultIfEmpty(),
        (x, right) => (x.left.X, right?.Z))
```

## Strengths
- Developer tooling: Outputs standard C# source compatible with IDEs, analyzers, and debuggers.
- Performance: Compiled IL can outperform shell scripts for heavy workloads, especially with in-memory joins.
- Embedding: Easy to ship as part of a larger .NET application, enabling direct function calls without spawning processes.
- Full outer join support: LEFT, RIGHT, and FULL OUTER joins with LINQ patterns.

## Limitations (as of current design)
- Code size: Each predicate generates a substantial block of C#; larger rule sets may benefit more from the Query Runtime’s shared engine.
- Flexibility: Regenerating code is required for any change (e.g., new dedup strategies), whereas the query runtime can evolve independently.

## LINQ Recursive Style

The `linq_recursive(true)` option generates cleaner code using the `UnifyWeaver.Native` runtime library:

```prolog
% Inline style (default)
compile_predicate_to_csharp(ancestor/2, [], Code).

% LINQ style - uses TransitiveClosure extension
compile_predicate_to_csharp(ancestor/2, [linq_recursive(true)], Code).
```

**LINQ style generates:**
```csharp
using UnifyWeaver.Native;

public static IEnumerable<(string, string)> AncestorStream()
{
    if (_cache != null) return _cache;
    _cache = ParentStream().TransitiveClosure(
        (d, baseRel) => baseRel
            .Where(b => b.Item2 == d.Item1)
            .Select(b => (b.Item1, d.Item2))
    ).ToList();
    return _cache;
}
```

The `UnifyWeaver.Native` library (`src/unifyweaver/targets/csharp_native_runtime/`) provides:
- `TransitiveClosure<T>()` - Transitive closure with semi-naive iteration
- `SafeRecursiveJoin<T>()` - Safe join for self-referential queries
- `Fixpoint<T>()` - General fixpoint computation

## Roadmap
1. Formalise target selection: Wire `target(csharp_codegen)` through the recursive compiler so non-recursive predicates delegate here when requested.
2. Improve dedup/control flow: Add support for ordered dedup, constant folding, and guard blocks matching Bash templates.
3. ~~Recursive extensions: Investigate emitting loops or iterator blocks for specific recursion classes (tail recursion, transitive closure).~~ ✅ Implemented via semi-naive iteration and LINQ TransitiveClosure
4. Packaging: Provide project scaffolding or shared libraries so generated code can be consumed with minimal boilerplate.

## Relationship to Query Runtime
Both targets share the same clause analysis pipeline. The codegen path materialises clauses as standalone C# source, while the query runtime path creates an IR consumed by a shared engine. Future releases may let `target(csharp)` pick whichever backend satisfies the requested features, using codegen for simple cases and the query runtime when recursion or advanced optimisation is required.
