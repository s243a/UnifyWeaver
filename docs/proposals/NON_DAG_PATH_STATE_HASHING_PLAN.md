# Non-DAG Path-State Hashing: Implementation Plan

## Phase 0: Preserve the Split

Before implementation work, keep the benchmark and planner framing clear:

- DAG workloads are optimized as DAGs
- non-DAG simple-path workloads use exact path-state machinery

This avoids mixing the two problem classes again.

## Phase 1: Runtime Instrumentation

Add measurement around the current exact non-DAG path-state paths:

- frontier candidate count
- exact comparison count
- subset-check count
- bucket sizes if any preliminary indexing already exists

Goal:

- identify the real hot spots before changing representation

## Phase 2: Integer-State Normalization

Normalize relevant non-DAG recursive executors to use:

- stable integer ids for graph nodes

This is a prerequisite for compact exact path-state representation and
cheap hashing.

Status:

- counted `PathAwareTransitiveClosureNode` traversal now uses integer node ids
  and `CompactVisitedPath` instead of cloning per-state `HashSet<object?>`
  instances
- weighted `Min` frontier fallback already uses compact integer path state

## Phase 3: Fingerprint-Carrying State

Extend the relevant recursive state structures to carry:

- exact visited state
- deterministic fingerprint
- optional compact summaries

At this phase, correctness should remain unchanged. The fingerprint is
added first as metadata, not yet as the sole lookup route.

Status:

- implemented for weighted `Min` frontier fallback states in the C# query
  runtime
- compact visited paths now carry deterministic set fingerprints alongside
  exact node-id arrays and summary masks
- counted transitive closure uses the same compact exact path representation
  for cycle checks, but does not use frontier fingerprint lookup because it
  has no subset-dominance frontier

## Phase 4: Bucketed Lookup

Introduce fingerprint-indexed buckets for:

1. exact deduplication
2. `Min` frontier candidate lookup

Workflow:

- derive candidate bucket from the fingerprint key
- run exact verification inside the bucket
- keep fallback correctness behavior intact

Status:

- implemented for same-cardinality weighted `Min` frontier dominance checks
  and dominated-state removal
- target-local frontiers now keep path-count buckets and fingerprint counts,
  using fingerprints only to prune candidates before exact subset checks

## Phase 5: Dominance Prefilters

Add cheap prefilters before exact subset / equality checks, such as:

- visited cardinality checks
- summary mismatches
- node-local bucket partitioning

Only after those pass should exact dominance verification run.

Status:

- implemented for weighted `Min` frontier fallback states
- same-cardinality checks use fingerprint indexes
- lower-cardinality checks use lazy representative-node indexes only for
  larger path-count buckets
- small buckets direct-scan because the index overhead is higher than the
  candidate reduction on the current benchmark shape

## Phase 6: Benchmark Validation

Use cyclic simple-path benchmarks to compare:

- previous exact implementation
- hashed path-state indexed implementation

Measure:

- runtime
- memory
- candidate bucket sizes
- exact verification counts

Status:

- weighted `Min` fallback now reports `min_frontier_*` counters for
  dominance candidates, exact subset checks, and retained frontier buckets
- counted simple-path closure now reports `path_state_*` counters for raw
  traversal work: seeds, stack pops, successor candidates, cycle skips,
  depth-limit skips, best-known pruning, enqueued states, output rows, max
  stack size, and max path length

## Phase 7: Planner Integration

Make the runtime selection explicit:

- DAG fast path when structural acyclicity is known
- hashed non-DAG path-state strategy when simple-path cyclic semantics
  are required

## Immediate Follow-Up After Weighted-Min Frontier Metrics

The weighted `Min` frontier fallback metric survey now points back to this
plan. On benchmark-scale cyclic fallback cases, dominance-candidate scans
dominate the counters, retained buckets grow, and exact subset checks are not
the main count driver.

The requested comparison is now in place:

- positive multiplicative weighted `Min` is now handled separately when every
  factor is finite and at least `1`
- counted simple-path shortest-path runs expose `path_state_*` metrics from
  `PathAwareTransitiveClosureNode`
- preserve the current rule that fingerprints, masks, and representatives are
  only candidate filters; exact subset verification remains authoritative

Local survey:

| Workload | Scale | All | Min | Match | Main Counter |
| --- | ---: | ---: | ---: | --- | --- |
| counted shortest path | 300 | 0.634s | 0.214s | yes | `982,581` all-mode successor candidates |
| counted shortest path | 1k | 0.450s | 0.180s | yes | `592,698` all-mode successor candidates |
| negative additive weighted `Min` | 300 | 0.871s | 1.478s | yes | `20,404,270` dominance candidates |
| negative additive weighted `Min` | 1k | 0.563s | 1.217s | yes | `16,522,183` dominance candidates |

Counted-closure phase split after typed row buffering, pre-sized
materialization, edge-state node-id preindexing, node-id keyed retained-min
tracking/flush, concrete-array `nodeValues` replay on the counted-path
materialization path, cached boxed depth reuse for counted-path row
construction, split `All` versus retained-min successor loops in the counted
path hot traversal, per-row timing removal, a compact `(target, depth)`
buffered row shape, O(1) parent-linked visited-path extension, a dedicated
counted-path traversal frame stack, and direct-write seed-batch materialization
with a packed target/depth row buffer and node-id driven traversal/replay:

| Scale | Mode | Traversal | Row Creation | Result Materialization | Best-Known Flush/Sort |
| --- | --- | ---: | ---: | ---: | ---: |
| 300 | All | 95.476ms | 0.000ms | 52.799ms | n/a |
| 300 | Min | 33.764ms | 0.000ms | 5.895ms | 4.619ms |
| 1k | All | 62.974ms | 0.000ms | 27.954ms | n/a |
| 1k | Min | 11.950ms | 0.000ms | 0.925ms | 1.835ms |

Interpretation:

- counted closure is dominated by raw successor expansion and depth-limit
  skips, not exact subset dominance
- compact visited state reduces counted-closure allocation overhead while
  preserving the same traversal counters and exact output
- result materialization is significant enough to justify typed row buffering
  and pre-sizing, but traversal remains the largest counted-closure phase
- edge-state node-id preindexing removes the per-successor candidate node-id
  dictionary lookup from traversal while preserving output hashes and
  `path_state_*` counters
- row-buffer recording no longer starts a stopwatch for every emitted path
  row; the explicit `path_state_row_creation` phase is now `0`, and row-buffer
  work is included in traversal timing
- the counted-path `All` buffer now stores only `(target, depth)` per row,
  because `seed` is constant for each per-seed traversal; this reduces buffer
  footprint and lowers final `object[]` materialization cost
- `CompactVisitedPath.Extend` now uses a parent-linked immutable node instead
  of copying the full visited-node array on every successor push, which
  reduces traversal-time allocation/copy overhead while preserving exact
  cycle checks and frontier semantics
- counted-path traversal now uses a dedicated frame struct and an explicit
  initial stack capacity, which trims hot-path stack overhead without changing
  the reported `path_state_*` counters
- when the destination is a `List<object[]>`, counted-path `All` materialization
  now grows the list once and writes the new seed batch directly into the new
  slots via `CollectionsMarshal`, avoiding per-row `Add` bookkeeping on the
  final output path
- the counted-path `All` staging buffer now stores targets and depths in
  parallel packed lists instead of shuttling a tiny struct per row, reducing
  replay overhead on the final materialization path
- counted-path `All` now reports replay-shape survey phases and metrics:
  `path_state_result_replay_setup`, `path_state_result_replay_write`,
  `path_state_result_replay_batch_count`, and
  `path_state_result_replay_max_batch_size`; on the current benchmark shape,
  replay setup is small while replay writes dominate the remaining
  materialization cost
- counted-path `All` replay write is now further split into
  `path_state_result_replay_value_lookup` and
  `path_state_result_replay_row_alloc`; on the current benchmark shape, row
  allocation is the larger write-side cost, with value lookup still visible
- counted-path `All` row allocation is now documented as a broader runtime
  boundary rather than a purely local replay-loop choice, because execution
  surfaces, caches, wrappers, and tests all currently expect `object[]` rows
- counted-path traversal and buffered replay now operate on interned node ids
  with edge-state lookup tables, avoiding repeated object-key dictionary
  lookups on the hot path while preserving exact output values at final
  materialization time
- counted-path `Min` now keeps retained best depths keyed by interned target
  node id until the final target-sorted flush, cutting object-key hashing and
  reducing `best_known_flush_sort` plus final `Min` materialization cost while
  preserving the deterministic ordering contract
- counted-path replay/materialization now uses the concrete `object?[]`
  node-value table directly instead of an `IReadOnlyList<object?>` view, which
  trims lookup overhead on the hot replay path without changing output rows
- counted-path row construction now reuses cached boxed depth objects for
  common small path depths, reducing per-row boxing churn on the high-volume
  `All` materialization path
- counted-path traversal now uses a separate `All` hot-path successor loop so
  the high-volume `All` case no longer pays retained-min dictionary and mode
  branching checks on every successor candidate
- counted-path cycle checks on the `All` hot path now reuse precomputed
  node-id masks and fingerprints from edge-state construction, avoiding
  repeated per-successor summary derivation while preserving exact
  cycle-detection behavior
- weighted `Min` fallback remains the only measured shape where generic
  frontier candidate indexing is directly relevant
- the next optimization should not add another generic frontier index by
  default; if counted closure stays the target, the remaining work is in
  expansion/materialization overhead rather than dominance-frontier machinery

The optimization must remain exact: fingerprints and bucket keys should reduce
candidate scans, not replace the final simple-path dominance check.
