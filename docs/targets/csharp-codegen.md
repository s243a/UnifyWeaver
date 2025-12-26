# C# Code Generation Target (`target(csharp_codegen)`)

`csharp_codegen` (and its native implementation `csharp_native`) emits idiomatic C# source files that mirror the Bash streaming templates. It builds on the existing `csharp_native_target.pl` module and is intended for scenarios where developers want tangible .NET source artefacts that can be compiled, debugged, and extended manually.

## Current Scope
- Recursive predicates: Supported via "Procedural" mode using method-to-method calls and `yield return`. Memoization is used to ensure termination on cyclic data.
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

## Strengths
- Developer tooling: Outputs standard C# source compatible with IDEs, analyzers, and debuggers.
- Performance: Compiled IL can outperform shell scripts for heavy workloads, especially with in-memory joins.
- Embedding: Easy to ship as part of a larger .NET application, enabling direct function calls without spawning processes.

## Limitations (as of current design)
- Code size: Each predicate generates a substantial block of C#; larger rule sets may benefit more from the Query Runtime’s shared engine.
- Flexibility: Regenerating code is required for any change (e.g., new dedup strategies), whereas the query runtime can evolve independently.

## Roadmap
1. Formalise target selection: Wire `target(csharp_codegen)` through the recursive compiler so non-recursive predicates delegate here when requested.
2. Improve dedup/control flow: Add support for ordered dedup, constant folding, and guard blocks matching Bash templates.
3. Recursive extensions: Investigate emitting loops or iterator blocks for specific recursion classes (tail recursion, transitive closure).
4. Packaging: Provide project scaffolding or shared libraries so generated code can be consumed with minimal boilerplate.

## Relationship to Query Runtime
Both targets share the same clause analysis pipeline. The codegen path materialises clauses as standalone C# source, while the query runtime path creates an IR consumed by a shared engine. Future releases may let `target(csharp)` pick whichever backend satisfies the requested features, using codegen for simple cases and the query runtime when recursion or advanced optimisation is required.
