# Non-DAG Path-State Hashing: Specification

## Goal

Define a sound execution strategy for cyclic simple-path workloads that:

- preserves exact semantics
- reduces path-state comparison cost
- composes with existing `All` and `Min` recursive execution machinery

This proposal is for workloads where revisiting a node on the same path
is forbidden or otherwise semantically significant.

## Target Workloads

Representative shapes include:

1. counted simple-path closure on cyclic graphs
2. weighted `Min` simple-path recursion on cyclic graphs
3. future non-DAG dependency workloads with explicit simple-path
   semantics

Out of scope:

- DAG-only workloads that can be solved by topological dynamic
  programming
- approximate graph search

## State Model

Each active recursive state should carry:

1. current node or recursive position
2. current accumulator / depth if the workload uses one
3. exact visited-state representation
4. deterministic path-state fingerprint
5. optional compact summaries used only as prefilters

Examples of compact summaries:

- visited cardinality
- node-id bitset blocks where feasible
- min/max node id seen
- XOR / rolling hash summaries

These summaries may accelerate lookup but do not replace exact visited
state.

## Correctness Rule

Two states may be treated as identical or as dominance candidates only if
their exact semantics justify it.

Therefore:

- hash equality alone is insufficient
- summary equality alone is insufficient

Exact verification is required before:

1. deduplicating states
2. concluding dominance
3. rejecting a newly derived state

## Fingerprint Requirements

The path-state fingerprint should be:

- deterministic
- cheap to update incrementally when extending a path
- stable across execution order

Possible constructions:

- rolling hash over canonical path sequence
- hash over canonical visited-set encoding
- combined `(current_node, fingerprint)` key where needed

The proposal does not commit to one concrete hash function yet, but it
does require:

- no semantic dependence on collision-freeness

## Lookup Model

### Exact deduplication

Use:

- `(current node, fingerprint)` as the first lookup key

Then:

- verify candidate states exactly inside the bucket

### Dominance for `Min`

Use:

- node
- accumulator band or exact accumulator
- fingerprint bucket
- optional subset-friendly summaries

Then:

- verify exact dominance relation using the exact visited-state
  representation before pruning

## Visited-State Representation

The implementation should prefer more compact exact structures than the
earlier generic `HashSet<object?>` frontier representation.

Preferred direction:

- integer node ids
- compact exact visited encoding
- path fingerprint derived from that encoding

Current C# query-runtime status:

- counted `PathAwareTransitiveClosureNode` traversal uses integer node ids and
  compact exact visited paths for cycle checks
- weighted `Min` fallback uses compact exact visited paths plus fingerprints,
  masks, and exact subset verification for frontier dominance candidates

Exact structure options include:

1. sorted integer path vector
2. compact persistent bitset
3. hybrid structure:
   - compact exact membership structure
   - hash/fingerprint cache

## Runtime Applicability

This strategy should be used only for workloads that actually need
non-DAG simple-path semantics.

It should not replace:

- DAG-specific fast paths
- scalar closure/count strategies on acyclic graphs

The planner/runtime should continue to select the cheapest correct model
for the workload class.

## Trace Metrics

Runtime instrumentation should keep the main exact workload classes separate:

- counted simple-path traversal reports `path_state_*` counters for stack
  pops, successor candidates, cycle skips, depth-limit skips, best-known
  pruning, enqueued states, output rows, and maximum path/stack size
- weighted `Min` frontier fallback reports `min_frontier_*` counters for
  dominance candidates, subset checks, target buckets, and retained
  path-state partition sizes

These metrics are intentionally not normalized into one generic counter set.
The current measurements show different bottlenecks: counted closure is
expansion-heavy, while weighted `Min` fallback is dominance-candidate-heavy.

## Success Criteria

1. Exact output agreement with current exact non-DAG implementations
2. Reduced frontier lookup cost on cyclic simple-path benchmarks
3. Lower memory overhead than naive exact visited-set tracking
4. Clear separation from DAG-specialized execution paths
