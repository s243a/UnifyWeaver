# C# Streaming Target (LINQ)

## Overview

The C# streaming target mirrors the Bash stream compiler but emits a fully self-contained C# program. The generated code materializes predicate facts into in-memory arrays and exposes LINQ-based join pipelines that yield tuple streams (`IEnumerable<(string,string)>`). A simple `Main` function prints joined tuples as `a:b` lines, making the output compatible with existing downstream consumers.

**Current scope**
- Only handles predicates compiled as single non-recursive rules composed exclusively of binary relations.
- Cross-predicate joins are expressed with LINQ `.Join` operations chained in pipeline order.
- Optional deduplication is represented with `.Distinct()` when the merged options include `unique(true)`.

## Compilation Flow

1. **Clause introspection** – `compile_predicate_to_csharp/3` locates all clauses for `Pred/Arity` and classifies the shape (facts, single rule, multi-rule). We currently only accept the single-rule case.
2. **Term collection** – `collect_predicate_terms/2` extracts each relational goal used in the body while filtering out constraints and variables.
3. **Fact gathering** – For every unique predicate signature, all fact clauses are collected and stored as `(string,string)` tuples.
4. **Helper emission** – The generator creates a static array plus a `PredicateStream()` helper method for each relation. These helpers supply the left/right inputs to the LINQ pipeline.
5. **Pipeline assembly** – The ordered predicates from the rule body drive the LINQ pipeline. The first predicate seeds the stream, downstream predicates contribute `.Join` clauses that stitch tuples on the shared key.
6. **Deduplication** – If the merged options request `unique(true)`, the pipeline appends `.Distinct()` to match the Bash-level dedup behaviour.
7. **Program synthesis** – `compose_csharp_program/6` wraps everything in a `UnifyWeaver.Generated` namespace with a `ModuleNameModule` class and companion `TargetNameStream()` function.

## Generated Code Structure

The generated file contains:

- A metadata header noting UnifyWeaver version, target, and timestamp.
- `using` statements for `System`, `System.Collections.Generic`, and `System.Linq`.
- Static arrays of `(string,string)` tuples for each relation used in the rule.
- Helper methods `RelationStream()` that expose the tuples as `IEnumerable<(string,string)>`.
- A `TargetStream()` method that chains the helpers via `.Join` to produce the final stream.
- A `Main` entry point that iterates the stream and writes colon-delimited output to stdout.

## Example (grandparent/2)

Compiling the rule `grandparent(X,Z) :- parent(X,Y), parent(Y,Z).` produces a pipeline equivalent to:

```csharp
ParentStream()
    .Join(ParentStream(),
          left => left.Item2,
          right => right.Item1,
          (left, right) => (left.Item1, right.Item2))
    .Distinct();
```

The first `ParentStream()` provides `(Parent,Child)` facts. The join aligns the second predicate on the shared `Y` variable and emits `(X,Z)` tuples. `.Distinct()` reflects `unique(true)` when the compiler deduces or receives it via options.

## Limitations (current state)

- Fact-only predicates (`Body == true`) are rejected because the C# target does not yet emit lookup/stream-only modules without joins.
- Multi-clause (OR) predicates and recursion are not supported.
- Non-binary relations (arity != 2) raise errors during compilation.
- Deduplication beyond simple `.Distinct()` (e.g. stable ordering, keyed grouping) is not implemented.
- The generator emits a single source file with embedded data; separate data files or reusable libraries are not yet produced.

## Next Steps

- Add fact-only support by emitting array-backed lookup methods without join pipelines.
- Introduce multi-clause orchestration, mirroring the Bash pipeline approach (e.g. helper functions per clause and union-like merges).
- Extend to non-binary joins by translating higher-arity relations into record types or tuples with more fields.
- Allow external data loading instead of compiling facts directly into the assembly.
- Define configuration for project/namespace layout so generated components can integrate with larger C# solutions.
- Coordinate streaming contracts with other targets (Bash, PowerShell) through well-defined serialization (e.g. JSON) or proxy services.
