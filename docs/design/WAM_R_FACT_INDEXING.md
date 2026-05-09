# WAM-R Fact Table Indexing

## Status

**Implemented.** Per-arg hash indexes landed on the `claude/r-hybrid-wam-target-cMqaZ`
branch (commit `907d979`). Generalises the first-arg-only hash index that
shipped with PR #1921. Memory: O(N · F) for F facts, arity N. No composite-key
indexes.

This document records the design discussion: what we built, what alternatives we
considered and ruled out, the theoretical context this slots into, and how it
relates to the parameterized C# query runtime
(`src/unifyweaver/targets/csharp_query_runtime/`), which has the most
sophisticated query-engine-style code in the repo.

> **Why a design doc for one PR?** The decision space here -- per-attribute
> hashing, composite keys, successive hashing, bitmap intersection, smallest-
> bucket selection -- is the same decision space mature query engines work in.
> WAM-R is the second target (after the C# parameterized runtime) where the
> engine has to make these choices, and it would be nice to keep the reasoning
> in one place rather than re-deriving it the next time another target hits it.

## TL;DR

For each fact-table predicate `p/N`, the codegen now emits **N independent hash
indexes**, one per arg position. Each index maps `(value -> integer-vector of
matching tuple indices)`. At dispatch time, the runtime walks the bound atom/int
args, looks each one up in its own index, **picks the smallest matching bucket**
(the most selective bound arg), and iterates just that bucket. Per-tuple
unification then filters on all the other args.

The position of an arg is encoded **structurally** by which env you look in, not
as part of the key. So `"aa"` in `arg1_env` and `"aa"` in `arg2_env` mean two
totally different things, and there are no hash collisions between positions.

Memory grows linearly in arity (`N × F` entries vs. `F` for first-arg only). It
does **not** grow exponentially, because we build indexes for the N singleton
arg-position subsets only -- not for combinations of args. (Note: the
"exponential" cost of composite indexing comes from building all `2^N - 1`
subsets, not from the key-space size of any single composite index. Each
individual hash table only stores entries for keys that appear in the data.
So a workload-driven *small* set of composite indexes would also be linear.
We just don't have any today.)

## What we built

### Storage

For a fact table with arity N and F ground tuples:

- `<pred>_facts <- list( list(arg1, arg2, ..., argN), ... )` -- master tuple
  list (unchanged).
- `<pred>_index_arg<K> <- new.env(...)` for K = 1..N -- one R env per arg
  position. Keys: `"a<atom-id>"` for atom-tag args, `"i<int-value>"` for
  integer-tag args. Value: an R integer vector of 1-based tuple indices into
  `<pred>_facts`.
- `<pred>_indexes <- list(<pred>_index_arg1, ..., <pred>_index_argN)` -- the
  bundle passed to `WamRuntime$fact_table_dispatch`.

For runtime-loaded fact sources (CSV today, future LMDB / TSV), the equivalent
runtime helper is `WamRuntime$build_fact_indexes(facts, arity)` which returns
the same list-of-envs shape.

Total entries across all indexes: at most `N × F`. Each fact contributes at
most one entry per arg position (skipped at positions whose value isn't atom or
int, e.g. floats, structs, unbounds at build time). Memory cost vs. the
first-arg-only baseline: `N×` more entries.

### Dispatch (`WamRuntime$fact_table_dispatch`)

Pseudocode:

```
function dispatch(args, indexes, tuples):
  best_bucket = NULL; best_size = +inf
  for j in 1..arity:
    if indexes[j] is NULL: continue
    v = deref(args[j])
    if v is unbound or struct: continue
    bk = key_for(v)                # "a<id>" or "i<val>"; NULL for floats
    if bk is NULL: continue
    if not exists(bk in indexes[j]): return FALSE   # short-circuit
    bucket = indexes[j][bk]
    if size(bucket) < best_size:
      best_bucket = bucket; best_size = size(bucket)
  if best_bucket is not NULL:
    return iter_subset(tuples, best_bucket, args)   # per-tuple unify
  return iter_full(tuples, args)                    # no usable bound arg
```

Two important properties:

1. **Short-circuit on miss.** If any bound atom/int arg's bucket is missing
   from its index, no fact can possibly match the conjunction, so we return
   FALSE without iterating. This is sound because: (a) the index contains
   *every* fact whose arg has that tag at that position; (b) a missing bucket
   means no fact has the queried value at that position; (c) the conjunction
   "all bound args must match" is therefore unsatisfiable.

2. **Smallest-bucket pick.** Any matching tuple must appear in *every* bound
   arg's bucket (each bucket contains exactly the tuples matching that one
   arg). So picking *any* bound arg's bucket is sound -- we just iterate fewer
   tuples by picking the smallest one. This is the runtime equivalent of the
   classic "build hash on smaller relation" hash-join heuristic.

### Soundness sketch

Let `B_j(v) = { i | tuples[i][j] = v }` be the set of tuple indices in arg `j`'s
bucket for value `v`. For a query that binds args `j1, ..., jk` to values
`v1, ..., vk`, the set of matching tuples is `B_{j1}(v1) ∩ ... ∩ B_{jk}(vk)`.

Key observation: the matching set is a **subset** of every `B_{ji}(vi)`. So
iterating any single `B_{ji}(vi)` and applying per-tuple unify (which checks
all args) yields exactly the matching set. Picking the smallest `B` minimizes
work. Q.E.D.

## Alternatives considered

### A. Composite-key indexes (exhaustive)

Keep a separate index for each subset of bound positions: `(arg1)`, `(arg2)`,
`(arg1, arg2)`, `(arg1, arg3)`, ..., `(arg1, ..., argN)`. Lookup becomes O(1)
hash for any binding pattern.

**Why not:** building all `2^N - 1` subsets means each fact contributes one
entry per subset, so total storage is `O(F × 2^N)` -- exponential in arity,
even though *each individual index* is just O(F) and the hash buckets only
ever materialize for keys that actually appear in the data. Build time is
exponential too. Even at N = 6 you have 63 indexes per table. This is the
"n-tuple problem" -- and it's only a problem when the index choice is
exhaustive.

> **Note on what's actually exponential.** The *key space* for composite
> indexing is exponential in arity (`2^N - 1` binding patterns, each with a
> potentially large value-space). Hash tables only store entries for keys that
> actually appear in the data, so a *single* composite index is O(F) regardless
> of key-space size. The exponential storage cost only appears when you build
> indexes for *every* subset of bound positions; the alternative -- workload-
> driven composite indexing on a chosen few subsets -- is O(F × k) for a small
> constant k. See option B below and the future-directions section.

### B. Composite-key indexes (workload-driven, opt-in)

Same as A, but the user declares which composite indexes to build per
workload, e.g. `r_fact_composite_index(p/3, [1, 2])` to build only the
`(arg1, arg2)` index. Storage is O(F × k) for k declared indexes -- linear,
not exponential.

**Why not (yet):** no infrastructure for declaring or maintaining these in
WAM-R today. Worth adding when a real workload shows that per-arg buckets
aren't selective enough on a specific multi-arg binding pattern. Listed in
future directions below. The parameterized C# query runtime already supports
this in its manifest format (`DelimitedRelationArtifactIndexManifest.Columns`
is `List<int>`) but currently emits only single-column indexes.

### C. Successive / rolling hash chain

Hash incrementally over the args of a fact: `h0 = init; h1 = mix(h0, arg1);
h2 = mix(h1, arg2); ...; hN`. Store `hN -> tuple_index` for exact-tuple
lookup.

**Why not:** breaks under partial bindings. A query with `arg2` unbound can't
compute `h2 = mix(h1, arg2)`. You'd have to either (a) skip arg2 in the chain
(producing a different hash sequence than facts that included it -- so they
don't match), or (b) precompute every possible `h2`, which collapses back to
the exponential case. Successive hashing works for primary-key style lookups
where every arg is always bound, but doesn't generalize to "any subset bound."

### D. Single env with composite (position, value) keys

One R env keyed by strings like `"1:a"`, `"2:b"`, where the position is part
of the key string instead of being implicit in env identity.

**Why not chosen:** functionally equivalent to per-arg envs, but with longer
keys (one extra string concat per lookup, slightly worse hash distribution
because keys share a position-prefix). Per-arg envs avoid both. Same memory.

### E. Bitmap intersection

Maintain per-(position, value) bitmaps over tuple indices; for each query,
AND-together the bitmaps for all bound args, then iterate the result.

**Why not:** at F facts, each bitmap is ~F bits, so each AND is O(F)
regardless of how selective the args are. The smallest-bucket pick is O(smallest
bucket), which is strictly better when at least one arg is selective. Bitmaps
win only when *all* bound args have very high cardinality buckets that need to
be intersected -- an uncommon case for fact tables.

### F. Sorted per-arg indexes (range-aware)

Same as our approach but with sorted vectors per bucket, supporting range
queries (`X > 5`, `X between A B`). Strict superset of what we did.

**Why not yet:** range queries aren't a current pattern in WAM-R workloads,
and the per-tuple-scan path handles them correctly (just slowly). Listed as
a future direction.

## Theoretical context

### RDBMS indexing tradition

What we did fits squarely in the **secondary-index** family from RDBMS theory:
- One index per column (~ B-tree on a single column in PostgreSQL/SQLite).
- Bucket-per-key shape (hash index, not sorted).
- Pre-filter via index, post-filter via per-row predicate evaluation.

Mature systems extend this with:
- **Range / sorted indexes** (B+ tree) for `>`, `<`, `BETWEEN`. (Our future #8.)
- **Composite / covering indexes** (`CREATE INDEX ON t(a, b)`) for high-
  selectivity multi-column queries. Not exponential in practice because users
  declare them per workload, not exhaustively.
- **Cost-based index selection** during query planning. Picks the right index
  before execution, using statistics. Our smallest-bucket pick is a runtime
  shortcut: we can't pre-plan because the WAM emits a single dispatch
  instruction per call without knowing arg cardinalities.
- **Index intersection** for queries that can use multiple indexes (related to
  bitmap intersection above; PostgreSQL's "BitmapAnd" plan node).

The RDBMS tradition tells us: per-column indexes are the well-trodden default,
composite indexes are workload-driven extras, and intersection/cost-based
planning are the more elaborate add-ons. Our design lands at the well-trodden
default.

### Datalog evaluation tradition

Datalog engines (Soufflé, LogicBlox, Coral, DLV, ...) typically:
- Index every relation on every column where it appears as a join key
  (roughly: per-attribute, like us).
- Use **semi-naive evaluation** for recursive predicates, so each round only
  scans newly-derived facts. Not applicable here -- WAM-R's recursive
  predicates go through the kernel detector (BFS / Dijkstra / A*) which
  bypasses the WAM stepper entirely; non-kernel recursion uses the WAM
  interpreter loop with no semi-naive optimization.
- Apply **magic-set transformation** to push selections through recursion,
  effectively doing demand-driven evaluation. Not applicable here for the
  same reason -- we don't have a Datalog query planner; WAM-R is interpreting
  WAM bytecode that was emitted from Prolog clauses with their evaluation
  order baked in.
- Maintain **delta relations** for incremental view maintenance. Not
  applicable -- our facts are static once loaded.

The piece of Datalog tradition we directly use: per-attribute hash indexes
to make the join probing fast. WAM-R doesn't have a join planner, but the
WAM's choice-point machinery effectively does nested-loop joins, and per-arg
indexes accelerate the inner loop.

### Where this design sits

On a spectrum from "no indexing" to "full cost-based query optimizer":

```
no indexing                                              Soufflé / Postgres
| full table scan                                        | per-attribute index +
                                                         | composite index +
                                                         | range index +
                                                         | cost-based join planning +
                                                         | semi-naive eval +
                                                         | magic-set transformation
|                                                        |
|              [WAM-R, after this PR]                    |
|              per-attribute hash index                  |
|              + smallest-bucket runtime pick            |
|              + per-tuple unify post-filter             |
|                                                        |
v                                                        v
0%                                                       100%
```

WAM-R is at "minimum viable per-attribute indexing" + a runtime selectivity
heuristic. Each step further (range, composite, cardinality estimation,
semi-naive, magic sets) is a future direction listed below.

## Connection to the C# query runtimes

The repo has three layers of C# code targeting different abstraction levels:

| Layer | Path | What it does |
|---|---|---|
| Parameterized query runtime (current) | `src/unifyweaver/targets/csharp_query_runtime/` | Plan-execution runtime: takes IR plans (param_seed, materialize, KeyJoinNode, NegationNode, AggregateNode, ...), executes them over relation providers. Has indexing, cardinality, and join-planning hooks. |
| Older non-parameterized native runtime | `src/unifyweaver/targets/csharp_native_runtime/` (`LinqRecursive.cs`) | Recursive-pattern helpers using LINQ. Predates the plan-based runtime. |
| Generator-side bridge | `src/unifyweaver/targets/csharp_runtime/` (`custom_csharp.pl`) | Prolog-side custom-handler integration. Predates the structured query target. |

The **parameterized query runtime** is the relevant comparison for WAM-R fact
indexing -- it's the one place in the repo with a real query-engine-style
indexing model.

### Direct alignment

The parameterized C# runtime implements essentially the same per-column
indexing design we just landed:

- **`IIndexedRelationProvider.TryLookupFacts(predicate, columnIndex, keys, ...)`**
  (`QueryRuntime.cs:351`) -- single-column lookup, the C# analogue of
  `WamRuntime$fact_table_dispatch`'s per-arg env lookup.
- **`IIndexedRelationBucketProvider.TryReadIndexedBuckets(predicate, columnIndex, ...)`**
  (`QueryRuntime.cs:362`) -- streams whole buckets for one column, the C#
  analogue of "iterate this arg's bucket."
- **Per-predicate per-column fact-index cache** (per the parameterized-queries
  status doc): "uses cached per-predicate/per-column fact indices to pick a
  selective bound pattern slot." That is *exactly* our smallest-bucket-pick
  policy, in a runtime that also has joins and aggregates.

### Where the C# runtime goes further

Things the C# parameterized runtime does that we do not (yet) do in WAM-R:

- **Manifest-level support for composite indexes.**
  `DelimitedRelationArtifactIndexManifest.Columns` is `List<int>`, so the
  artifact format can describe `(col1, col2)` indexes. The current writers
  only emit single-column indexes, but the format is ready. (`QueryRuntime.cs:294`.)
- **Two physical index kinds per column:** `offset_directory` (key -> file
  offsets, on-disk) and `covering_bucket` (key -> the rows themselves,
  amortizes I/O). WAM-R has only one in-memory shape (key -> integer vector
  of indices into a master list).
- **Cardinality-aware join build-side selection.**
  `KeyJoinNode` (`QueryRuntime.cs:11748-11754`) chooses build-vs-probe side
  based on `IRelationCardinalityProvider.TryGetRelationCardinality`. WAM-R
  has no joins (the WAM does nested-loop joins via choice points) and no
  cardinality estimation -- our runtime smallest-bucket pick is a poor man's
  cardinality oracle that uses the actual bucket size as a proxy.
- **Mode declarations** drive which indexes are needed. With `mode(p(+, -))`,
  only arg1 is ever used as a query input, so only arg1's index is needed.
  WAM-R doesn't yet do mode analysis, so we eagerly build indexes for all
  args. Mode analysis is follow-up #4 in the WAM-R handoff.
- **A real plan IR** (`param_seed`, `materialize`, `KeyJoinNode`,
  `NegationNode`, `AggregateNode`, `AggregateSubplanNode`). WAM-R has no
  plan layer -- the WAM bytecode *is* the plan, emitted directly by the
  Prolog compiler, with whatever evaluation order the source clauses imply.

### Where WAM-R goes further

Things WAM-R does that the C# runtime does not:

- **Full Prolog operational semantics**: choice points, backtracking,
  cut, negation-as-failure on first-class WAM terms, dynamic predicates
  (assertz/retract), exception handling. The C# runtime is a
  declarative-Datalog evaluator extended with parameterized inputs; it
  doesn't carry the full Prolog runtime weight. Different design point.

- **Native R kernels** for graph traversal (BFS, Dijkstra, A*) that bypass
  the WAM dispatch entirely. The C# runtime has the `KeyJoinNode` /
  `AggregateNode` plan nodes for similar specialisation but does it through
  the plan IR, not via runtime fast-path detection.

These are different design axes; the indexing subsystem is one of the few
places where the two converge on the same design.

## Limitations and worst case

### Things we explicitly accept

1. **No composite indexes.** Queries that bind multiple args still iterate the
   smallest single-arg bucket and post-filter. If two args are *individually*
   non-selective but their conjunction is highly selective, we don't capture
   that gain.

2. **Floats not indexed.** Float-valued args are skipped at index-build time
   (no bucket key), and float-valued query args fall through to the unbound
   path (no bucket lookup, full scan). Acceptable because: (a) float equality
   is fragile in general; (b) Prolog facts rarely use float keys.

3. **Structs not indexed.** Compound terms (`f(a, b)`) at fact-arg positions
   fall through to the unbound path. Indexing structs would require either
   structural hashing or recursive decomposition; not justified by current
   workloads.

4. **No cardinality estimation at codegen time.** We could pre-compute and
   emit `<pred>_index_arg1_size`, `<pred>_index_arg2_size`, ... at codegen
   time and let dispatch pick by these statistics. Probably not worth it --
   the runtime `length(bucket)` call is cheap.

### Worst case

When *all* bound atom/int args are non-selective (every fact has the same value
at the queried positions), the smallest bucket is still O(F), so we scan
everything. There's no win over the old first-arg-only design (which would
have scanned for the same reason). A targeted composite index on the actual
binding pattern (option B above) would solve the specific case at O(F)
additional storage; an exhaustive composite-index scheme (option A) would
solve every binding pattern at O(F × 2^N) storage. We prefer to keep the
default linear in arity, with workload-driven composite indexes available as
opt-in future work.

In practice, on the WAM-R workloads we care about (genealogy, graph traversal,
benchmark fact lookups), at least one bound arg is highly selective and the
smallest-bucket pick wins.

## Future directions

In rough priority order, with the most useful first:

1. **Range / interval indexes (sorted per-arg).** Sort each per-arg bucket
   key list at codegen time (or use a balanced BST at build time for runtime-
   loaded sources). Adds support for `X > 5`, `X between A B` queries which
   currently fall through to the per-tuple scan. Modest perf win on
   range-heavy workloads. Listed as follow-up #8 in the WAM-R handoff.

2. **Mode-driven index pruning.** When mode analysis arrives (handoff
   follow-up #4), use the mode info to skip building indexes for positions
   that will only ever be outputs. Saves O(F) memory per never-input
   position. The C# parameterized runtime already does this via mode
   declarations.

3. **Workload-driven composite indexes.** A `r_fact_composite_index([p/3,
   [1,2]])` directive would emit one extra `(arg1, arg2)` index for queries
   that bind both. Opt-in, declared per-workload, so no exponential blowup
   in the default case. Would need codegen for composite-key buckets and a
   dispatch hook to use them when the right binding pattern shows up.

4. **Codegen-time cardinality stats.** Emit `<pred>_arg<K>_max_bucket_size`
   constants alongside the indexes. Dispatch could prefer args with smaller
   *known* max-bucket size when bucket sizes tie at runtime. Marginal win;
   useful if profiling shows the runtime `length()` is ever a hot path.

5. **Bitmap representation for high-cardinality buckets.** When a bucket
   exceeds some threshold (e.g. 50% of F), switch from `c(1L, 2L, 3L, ...)`
   to a logical bitmap. Allows efficient AND-with-other-buckets fallback for
   the worst-case scenario above. Adds complexity for an edge case.

6. **Cross-target indexing IR.** Both the C# parameterized runtime and WAM-R
   now emit per-column hash indexes. A shared Prolog-side description (e.g.
   `index_spec(Pred, [hash(1), hash(2), hash(3)])`) consumed by both
   targets would let workload-specific index choices be declared once and
   honoured everywhere. Requires aligning the two indexing models more
   carefully than they're aligned today.

## Glossary

- **Fact table** -- a Prolog predicate whose every clause is `get_constant +
  proceed` only (no body, no compound args). The WAM-R classifier identifies
  these and the codegen emits a flat tuple list + indexes instead of WAM
  bytecode.
- **Bucket** -- the integer vector under one key in one per-arg index; the
  set of tuple indices that match that (position, value) pair.
- **Selective** -- a bound arg whose bucket is small relative to the total
  fact count. The runtime picks the most selective bound arg by bucket size.
- **Per-tuple unify** -- the inner loop of `fact_table_iter_subset`: for each
  candidate tuple from a bucket, run unification on every arg of the query
  vs. every arg of the tuple. This is the post-filter that makes per-arg
  indexes work without composite keys.
- **Lowered dispatch** -- the WAM-R `program$lowered_dispatch` env consulted
  by `Call` / `Execute` / `dispatch_call` before label dispatch. Fact tables
  register here so internal Prolog-to-Prolog calls reach the indexed fast
  path, not just direct R-API calls.

## References

- `WAM_R_TARGET.md` -- user-facing reference for the WAM-R target. See
  the **Fact-table lowering** section.
- `docs/handoff/wam_r_session_handoff.md` -- session handoff for the
  WAM-R campaign. See the **Follow-up ideas** section for context on
  where this fits.
- `src/unifyweaver/targets/csharp_query_runtime/QueryRuntime.cs` -- the
  parameterized C# query runtime. See `IIndexedRelationProvider`,
  `IIndexedRelationBucketProvider`, `IRelationCardinalityProvider`,
  `KeyJoinNode`.
- `docs/development/proposals/PARAMETERIZED_QUERIES_PROPOSAL.md` and
  `PARAMETERIZED_QUERIES_STATUS.md` -- design and status of the C#
  parameterized queries work, including the per-predicate / per-column
  fact-index cache that aligns with WAM-R's per-arg indexes.
- Standard query-engine references for further reading: Garcia-Molina /
  Ullman / Widom *Database Systems*; Abiteboul / Hull / Vianu
  *Foundations of Databases* (chapters on Datalog evaluation, magic sets,
  semi-naive); Soufflé documentation (compile-time Datalog with rich
  per-attribute indexing).
