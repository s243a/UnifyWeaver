# C# Query Runtime Target (`target(csharp_query)`)

This document describes the Query IR approach for the managed C# backend. The goal is to reuse UnifyWeaver’s clause analysis while executing via a reusable runtime that interprets relational plans using LINQ.

Roadmap: `docs/targets/csharp-query-runtime-roadmap.md`

## Status (v0.1)
- **Non-recursive clauses** – fact scans, joins, selections, projections, and unions translate to query nodes executed via LINQ.
- **Arithmetic & comparisons** – `is/2`, inequality operators, and `dif/2` become arithmetic or selection nodes with runtime evaluation.
- **Recursive predicates** – semi-naive fixpoint driver supports single-predicate recursion; canonical reachability patterns compile to `TransitiveClosureNode` for faster closure evaluation.
- **Mutual recursion** – strongly connected predicate groups emit `mutual_fixpoint` plans composed of `cross_ref` nodes, enabling even/odd style dependencies.
- **Negation (safe/stratified)** – `\+/1` compiles to `NegationNode` and supports stratified negation over derived predicates via program definition materialisation.
- **Deduplication** – per-predicate `HashSet<object[]>` mirrors Bash distinct semantics.
- **Diagnostics** – `QueryPlanExplainer.Explain(plan)` and `QueryExecutionTrace` provide plan inspection and basic per-node execution stats.
- **Cancellation + async adapter** – `CancellationToken` support in `Execute(...)` and `ExecuteAsync(...)` returning `IAsyncEnumerable<object[]>`.

## Objectives
- Declarative IR: Represent clause bodies as structured query plans instead of hard-coded C# statements.
- Engine reuse: Centralise recursion, deduplication, and constraint handling inside a shared runtime library.
- Extensibility: Enable future features (memoised recursion, distributed execution, streaming objects) without regenerating target-specific source for every predicate.

## Pipeline Overview
1. Clause classification (existing): facts, single-rule bodies, multi-rule alternation, and constraint extraction.
2. IR construction (new): translate each clause to a plan made of relational operators (enumeration, selection, projection, join, distinct, ordering).
3. Plan packaging: emit a C# artefact that builds a `QueryPlan` by instantiating node records (`RelationScanNode`, `JoinNode`, `MutualFixpointNode`, etc.) and seeding an `InMemoryRelationProvider`.
4. Runtime execution: load the plan into the Query Engine library, which:
   - Resolves referenced predicates to other plans or fact stores.
   - Executes base clauses eagerly.
   - Runs semi-naive fixpoint loops for recursive clauses (single predicate or mutual groups) using delta sets and `HashSet<T>` for deduplication.
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
| `Distinct`         | Deduplication strategy (global unique vs. per-iteration)     | `HashSet<object[]>` per predicate          |
| `Order`/`Limit`    | Optional ordering hints                                      | Forward compatibility for sorted targets   |

The current implementation emits static C# builders that assemble the plan via node constructors. Future iterations may introduce serialisation so plans can be stored as JSON/binary DTOs or expression trees for dynamic loading.

## Runtime Responsibilities
- Registry: map predicate identifiers to compiled plans or factual data.
- Constraint handling: apply `unique/1`, `unordered/1`, and other dedup strategies via `HashSet<T>` or sorted containers.
- Fixpoint driver (single predicate):
  1. Seed `current` with base facts (non-recursive clauses).
  2. Initialise `delta` with the same base results.
  3. While `delta` is non-empty:
     - Evaluate recursive plans referencing `delta` appropriately.
     - Remove tuples already seen (`HashSet.Contains`).
     - Emit new results, update `current`, and compute the next `delta`.
  4. Expose `current` as the final stream.
- Mutual fixpoint driver:
  - Maintain per-predicate totals/deltas inside an evaluation context so strongly connected components iterate together.
  - Iterate until every predicate’s delta is empty, updating totals and dedup sets for each member.
- Diagnostics: log iterations, show clause contributions, and surface firewall policy violations.

## Negation and Stratification
- Safe negation only: every variable referenced inside a negated predicate must be bound earlier in the clause body (Datalog-style safety).
- Non-stratified programs are rejected: a predicate cannot (directly or indirectly) depend on itself through a negated edge.
- Stratified derived predicates are materialised before use: the query planner emits a `ProgramNode` that evaluates lower-strata `define_relation` / `define_mutual_fixpoint` definitions first, then evaluates the final query body (so `\+` can consult derived results).

## Ordering, Purity, and Effects
- UnifyWeaver may reorder joins/goals when a predicate is declared `unordered(true)` (the default). Use `ordered` / `unordered(false)` to disable goal reordering when evaluation order is semantically meaningful (e.g. effectful predicates).
- The query runtime’s default output order is unspecified (hash-set semantics). For deterministic ordering, use `order_by/...` (and `distinct(strategy(ordered))` when relevant) before `limit/offset`.
- Cache/index reuse (e.g. `new QueryExecutorOptions(ReuseCaches: true)`) assumes deterministic/pure relations. Disable caches (or clear them via `executor.ClearCaches()`) if underlying facts/providers can change or have side effects.
- Seeded transitive-closure caches treat the seed list as a set (deduped + order-insensitive), so calls with the same seeds in different orders share a cache entry.
- Seeded transitive-closure caches are bounded via `QueryExecutorOptions(SeededCacheMaxEntries: ...)` (default `4096` per cache key); set `0` to disable seeded transitive cache reuse while keeping other caches enabled.
- Seeded transitive-closure cache admission can be tuned via `QueryExecutorOptions(SeededCacheAdmissionMinRows: ...)` (default `0` = always admit; values `> 0` skip storing tiny result sets to reduce LRU churn).
- Single concrete transitive pair probes (`source,target` both bound) now cache exact probe results (`TransitiveClosurePairsSingleProbe` and `GroupedTransitiveClosurePairsSingleProbe`) to avoid repeating one-off BFS checks across repeated calls.
- Pair-probe caches are bounded via `QueryExecutorOptions(PairProbeCacheMaxEntries: ...)` (default `4096` per cache key); set `0` to disable pair-probe caching while keeping other cache reuse enabled.
- Pair-probe cache admission can be tuned via `QueryExecutorOptions(PairProbeCacheAdmissionMinCost: ...)` (default `0` = always admit; values `> 0` require a minimum directional probe-cost estimate before storing).
- Bounded seeded/pair probe caches use LRU eviction (recent cache hits refresh recency).

## Current Limitations
- Tail-recursive optimisation and memoised aggregates still fall back to iterative evaluation without specialised nodes.
- Ordering/paging are opt-in via query plan modifiers (`order_by/1`, `order_by/2`, `limit/1`, `offset/1`); default results follow hash-set semantics.
- Plans currently materialise relation facts inside the generated module; external fact providers will arrive in later releases.
- Negation does not support existential variables (unbound variables inside a negated literal); use explicit joins/aggregation instead.
- Runtime assumes in-process execution (`dotnet run`); distributed execution and persistence hooks remain future work.

## Optional Integrations (No Hard Dependencies)
The core query runtime is intended to stay dependency-free (no LiteDB, ONNX, etc.). Optional integrations live in separate C# files/projects so consumers only take on extra dependencies when they opt in.

- Core library project: `src/unifyweaver/targets/csharp_query_runtime/UnifyWeaver.QueryRuntime.Core.csproj`
  - Includes: `QueryRuntime.cs`, `IEmbeddingProvider.cs`
  - External deps: none
- Pearltrees + LiteDB project: `src/unifyweaver/targets/csharp_query_runtime/UnifyWeaver.QueryRuntime.Pearltrees.csproj`
  - Includes: `Pt*.cs` + `PtHarness.cs`
  - External deps: LiteDB (uses `lib/LiteDB.dll` when present; otherwise falls back to NuGet `LiteDB` package)
- ONNX embeddings: `src/unifyweaver/targets/csharp_query_runtime/OnnxEmbeddingProvider.cs`
  - Project: `src/unifyweaver/targets/csharp_query_runtime/UnifyWeaver.QueryRuntime.Onnx.csproj`
  - External deps: `Microsoft.ML.OnnxRuntime`

## Smoke Testing (Runtime Execution)
The Prolog test suite can generate per-plan C# console projects in codegen-only mode, and the PowerShell runner can then build/run them with dotnet and verify outputs.

- Runner (recommended): `pwsh -NoProfile -File scripts/testing/run_csharp_query_runtime_smoke.ps1`
  - Options: `-KeepArtifacts`, `-OutputDir tmp/csharp_query_smoke`, `-SkipCodegen`
- Environment variables (used by the test harness):
  - `SKIP_CSHARP_EXECUTION=1` (generate C# projects but do not execute via Prolog)
  - `CSHARP_QUERY_OUTPUT_DIR=...` (where generated projects are written)
  - `CSHARP_QUERY_KEEP_ARTIFACTS=1` (keep generated projects instead of auto-deleting)

## Diagnostics
- Plan inspection: `Console.WriteLine(QueryPlanExplainer.Explain(plan));`
- Execution stats (rows/time per node, fixpoint iteration counts, plus cache/index and join-strategy summaries):
  - `var trace = new QueryExecutionTrace();`
  - `foreach (var row in executor.Execute(plan, parameters, trace)) { ... }`
  - `Console.WriteLine(trace.ToString());`
  - Cache traces include eviction and admission counters (`evictions`, `admissions`, `admission_skips`) for bounded caches.
- Cancellation (long-running queries/fixpoints):
  - `using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(10));`
  - `foreach (var row in executor.Execute(plan, parameters, trace, cts.Token)) { ... }`
- Async consumption adapter (wraps the synchronous engine):
  - `await foreach (var row in executor.ExecuteAsync(plan, parameters, trace, cts.Token)) { ... }`
- Prepared-style cache reuse (useful for repeated parameterized calls):
  - `var executor = new QueryExecutor(provider, new QueryExecutorOptions(ReuseCaches: true));`
  - `var executor = new QueryExecutor(provider, new QueryExecutorOptions(ReuseCaches: true, SeededCacheMaxEntries: 1024));`
  - `var executor = new QueryExecutor(provider, new QueryExecutorOptions(ReuseCaches: true, SeededCacheMaxEntries: 1024, SeededCacheAdmissionMinRows: 2));`
  - `var executor = new QueryExecutor(provider, new QueryExecutorOptions(ReuseCaches: true, PairProbeCacheMaxEntries: 1024));`
  - `var executor = new QueryExecutor(provider, new QueryExecutorOptions(ReuseCaches: true, PairProbeCacheMaxEntries: 1024, PairProbeCacheAdmissionMinCost: 2));`
  - `executor.ClearCaches();` (if underlying facts change)
  - See “Ordering, Purity, and Effects” below for ordering/side-effect assumptions.

## Configuration
- New preference atom: `target(csharp_query)`.
- Optional runtime hints:
  - `fixpoint(strategy(semi_naive|naive))`
  - `materialize(full|lazy)` to control when results are generated.
- Query modifiers:
  - `distinct(strategy(hash|ordered|none))`
  - `order_by(Index)`, `order_by(Index, asc|desc)`, `order_by([(Index, asc|desc), ...])` (0-based output column indices)
  - `limit(N)`, `offset(N)`
- Aggregates:
  - `count`, `sum(Var)`, `avg(Var)`, `min(Var)`, `max(Var)`, `set(Var)`, `bag(Var)` via `aggregate_all/3,4` and `aggregate/4`
- The generic `target(csharp)` option will initially alias `csharp_query` for recursion-heavy workloads while allowing smart fallback (see comparison doc).

## Security & Isolation
- The runtime runs inside the .NET sandbox; firewall modules must confirm that the target is allowed.
- Plans should include provenance metadata so execution logs can trace which Prolog clause emitted each operator.

## Roadmap
1. Memoisation & advanced patterns – extend the runtime with tail-recursive optimisations, cached aggregates, and transitive-closure helpers.
2. Ordered evaluation – ordered deduplication + limit/offset nodes (done); future work includes more deterministic output strategies.
3. Streaming adapters – `IAsyncEnumerable` adapter + cancellation tokens (done); future work includes true async sources and non-blocking execution.
4. Distribution hooks – allow plans to reference remote relations, enabling pipeline execution across nodes.

By funnelling all complex evaluation through this runtime, we keep the Prolog-side compiler small and declarative while unlocking richer execution strategies in managed environments.
