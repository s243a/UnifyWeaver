# C# Query Runtime Target (`target(csharp_query)`)

This document specifies the proposed Query IR approach for the managed C# backend. The goal is to reuse UnifyWeaver’s clause analysis while executing via a reusable runtime that interprets relational plans using LINQ.

## Objectives
- Declarative IR: Represent clause bodies as structured query plans instead of hard-coded C# statements.
- Engine reuse: Centralise recursion, deduplication, and constraint handling inside a shared runtime library.
- Extensibility: Enable future features (memoised recursion, distributed execution, streaming objects) without regenerating target-specific source for every predicate.

## Pipeline Overview
1. Clause classification (existing): facts, single-rule bodies, multi-rule alternation, and constraint extraction.
2. IR construction (new): translate each clause to a plan made of relational operators (enumeration, selection, projection, join, distinct, ordering).
3. Plan packaging: emit a C# artefact containing:
   - The IR (likely as serialisable data or C# expression trees).
   - Metadata (predicate name, arity, unique/unordered flags, security hints).
4. Runtime execution: load the IR into the Query Engine library, which:
   - Resolves referenced predicates to other plans or fact stores.
   - Executes base clauses eagerly.
   - Runs a semi-naive fixpoint loop for recursive clauses, using delta sets and `HashSet<T>` for deduplication.
5. Output adapters: expose enumerable streams (`IEnumerable<Tuple<...>>`), synchronous materialisation, or streaming writers.

## Intermediate Representation
The IR is target-language-agnostic data. Initial design candidates:

| Component          | Description                                                  | Example                                    |
|--------------------|--------------------------------------------------------------|--------------------------------------------|
| `RelationRef`      | Symbolic link to a fact set or another compiled predicate    | `Ref("parent", arity:2)`                   |
| `Selection`        | Predicate on tuple elements                                  | `Arg0 == Arg1`, inequality constraints     |
| `Projection`       | Tuple shaping, renaming                                      | Select `child` and `ancestor` columns      |
| `Join`             | N-ary join with key selectors                                | Theta-joins via `SelectMany` / nested loops|
| `Union`            | Merge outputs from multiple clauses                          | Clause alternation                         |
| `Distinct`         | Deduplication strategy (global unique vs. per-iteration)     | `Distinct`, `HashSet` increments           |
| `Order`/`Limit`    | Optional ordering hints                                      | Forward compatibility for sorted targets   |

Two serialization options are under consideration:
- Expression Trees: compile plans into `System.Linq.Expressions.Expression<Func<...>>`. Pros: integrates with existing LINQ providers. Cons: less portable outside the CLR.
- Custom DTOs: simple discriminated unions recorded as JSON or binary. Pros: easier to inspect; runtime converts them to delegates.

Initial milestone will likely use DTOs for clarity and emit static C# builders that assemble the plan.

## Runtime Responsibilities
- Registry: map predicate identifiers to compiled plans or factual data.
- Constraint handling: apply `unique/1`, `unordered/1`, and other dedup strategies via `HashSet<T>` or sorted containers.
- Fixpoint Driver:
  1. Seed `current` with base facts (non-recursive clauses).
  2. Initialise `delta` with the same base results.
  3. While `delta` is non-empty:
     - Evaluate recursive plans referencing `delta` appropriately.
     - Remove tuples already seen (`HashSet.Contains`).
     - Emit new results, update `current`, and compute the next `delta`.
  4. Expose `current` as the final stream.
- Diagnostics: log iterations, show clause contributions, and surface firewall policy violations.

## Configuration
- New preference atom: `target(csharp_query)`.
- Optional runtime hints:
  - `fixpoint(strategy(semi_naive|naive))`
  - `distinct(strategy(hash|ordered|none))`
  - `materialize(full|lazy)` to control when results are generated.
- The generic `target(csharp)` option will initially alias `csharp_query` for recursion-heavy workloads while allowing smart fallback (see comparison doc).

## Security & Isolation
- The runtime runs inside the .NET sandbox; firewall modules must confirm that the target is allowed.
- Plans should include provenance metadata so execution logs can trace which Prolog clause emitted each operator.

## Roadmap
1. MVP: non-recursive support using IR + runtime, parity with current streaming target.
2. Recursion: implement semi-naive iteration, guard unsupported patterns with explicit errors.
3. Advanced patterns: extend runtime to handle memoization, tail recursion optimisation, and transitive closures.
4. Distribution hooks: allow plans to reference remote relations, enabling pipeline execution across nodes.

By funnelling all complex evaluation through this runtime, we keep the Prolog-side compiler small and declarative while unlocking richer execution strategies in managed environments.
