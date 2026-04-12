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

## Phase 3: Fingerprint-Carrying State

Extend the relevant recursive state structures to carry:

- exact visited state
- deterministic fingerprint
- optional compact summaries

At this phase, correctness should remain unchanged. The fingerprint is
added first as metadata, not yet as the sole lookup route.

## Phase 4: Bucketed Lookup

Introduce fingerprint-indexed buckets for:

1. exact deduplication
2. `Min` frontier candidate lookup

Workflow:

- derive candidate bucket from the fingerprint key
- run exact verification inside the bucket
- keep fallback correctness behavior intact

## Phase 5: Dominance Prefilters

Add cheap prefilters before exact subset / equality checks, such as:

- visited cardinality checks
- summary mismatches
- node-local bucket partitioning

Only after those pass should exact dominance verification run.

## Phase 6: Benchmark Validation

Use cyclic simple-path benchmarks to compare:

- previous exact implementation
- hashed path-state indexed implementation

Measure:

- runtime
- memory
- candidate bucket sizes
- exact verification counts

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

The next concrete coding step should start here:

- normalize weighted `Min` frontier states around stable integer node ids
- carry deterministic fingerprints with exact visited paths
- partition candidate buckets before exact subset/dominance verification

The optimization must remain exact: fingerprints and bucket keys should reduce
candidate scans, not replace the final simple-path dominance check.
