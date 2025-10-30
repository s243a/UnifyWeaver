# C# Code Generation Target (`target(csharp_codegen)`)

`csharp_codegen` emits idiomatic C# source files that mirror the Bash streaming templates. It builds on the existing `csharp_stream_target.pl` module and is intended for scenarios where developers want tangible .NET source artefacts that can be compiled, debugged, and extended manually.

## Current Scope
- Non-recursive predicates: Facts, single-rule bodies, and multi-clause unions translate to LINQ pipelines that operate on in-memory arrays.
- Dedup semantics: Follows the Bash model—`Distinct()` for `unique(true)`, ordered variants are pending.
- Generated structure: Each predicate becomes a static class in the `UnifyWeaver.Generated` namespace with:
  - Fact arrays (`(string,string)[]` or similar).
  - Helper methods (`PredicateFactStream`, `PredicateStream`, `PredicateReverseStream`).
  - An optional `Main` for command-line execution.
- Configuration: `compile_predicate_to_csharp/3` accepts options such as dedup strategy and will soon recognise `target(csharp_codegen)` explicitly.

## Strengths
- Developer tooling: Outputs standard C# source compatible with IDEs, analyzers, and debuggers.
- Performance: Compiled IL can outperform shell scripts for heavy workloads, especially with in-memory joins.
- Embedding: Easy to ship as part of a larger .NET application, enabling direct function calls without spawning processes.

## Limitations (as of current design)
- Recursion: Not yet supported beyond the streaming case. Recursive predicates fall back to Bash or the query runtime.
- Code size: Each predicate generates a substantial block of C#; larger rule sets may benefit more from the Query Runtime’s shared engine.
- Flexibility: Regenerating code is required for any change (e.g., new dedup strategies), whereas the query runtime can evolve independently.

## Roadmap
1. Formalise target selection: Wire `target(csharp_codegen)` through the recursive compiler so non-recursive predicates delegate here when requested.
2. Improve dedup/control flow: Add support for ordered dedup, constant folding, and guard blocks matching Bash templates.
3. Recursive extensions: Investigate emitting loops or iterator blocks for specific recursion classes (tail recursion, transitive closure).
4. Packaging: Provide project scaffolding or shared libraries so generated code can be consumed with minimal boilerplate.

## Relationship to Query Runtime
Both targets share the same clause analysis pipeline. The codegen path materialises clauses as standalone C# source, while the query runtime path creates an IR consumed by a shared engine. Future releases may let `target(csharp)` pick whichever backend satisfies the requested features, using codegen for simple cases and the query runtime when recursion or advanced optimisation is required.
