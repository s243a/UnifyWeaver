# F# CSR Reverse-Index: Philosophy

**Status**: Phase 1 reader implemented. Deeper integration pending.
**Date**: 2026-05-25
**Companion**: `WAM_REVERSE_INDEX_ARTIFACTS.md` (format spec),
`WAM_FSHARP_CSR_PARALLEL_PLAN.md` (implementation plan)

## 1. What CSR is for

The effective-distance algorithm currently traverses *upward* only
(child -> parent via `category_parent`). CSR enables *downward*
traversal (parent -> children) for:

- **Bidirectional search**: meet-in-the-middle from seed and root
- **Demand-set expansion**: find all nodes reachable from root within
  K hops, then compute distances only for those
- **Non-carrot-shaped paths**: paths that go down before going up

The existing Phase 1 reader (`CsrLookupSource`) provides the
`ILookupSource` interface for parent -> children lookup from packed
binary files.

## 2. CSR and LMDB: complementary, not exclusive

There are three relationships between CSR and LMDB:

### 2.1 CSR alongside LMDB (current)

CSR provides reverse (parent -> children) lookup while LMDB provides
forward (child -> parent) lookup. Both coexist in `WcLookupSources`:

```
WcLookupSources = {
    "category_parent" -> TwoLevelCachedLookupSource(LmdbCursorLookup(...)),
    "category_child"  -> TwoLevelCachedLookupSource(CsrLookupSource(...))
}
```

This is the simplest composition. Each direction has its own optimal
backing store. The cost analyzer doesn't need to choose — both are
available.

### 2.2 CSR with LMDB offset index (Rust precedent)

The CSR `.idx` file uses binary search (O(log n)) to find a parent's
offset. For sparse parent ID spaces (e.g., Wikipedia page IDs with
gaps), an LMDB offset index provides O(1) amortized lookup:

```
category_child.csr.offsets.lmdb:
    key: int32_le parent_id
    value: uint64_le offset_edges, uint32_le count_edges
```

The Rust `build_reverse_csr_artifact.py` already supports
`--index-backend lmdb_offset`. The F# reader would open this LMDB
for the index lookup while reading values from the flat `.val` file.

### 2.3 CSR replacing LMDB for forward lookup

If we build a CSR grouped by children (child -> parents), it could
replace the `category_parent` LMDB DUPSORT database entirely:

```
category_parent.csr.idx   child -> (offset, count) records
category_parent.csr.val   packed int32 parent IDs
```

**Trade-off**: CSR is read-only and requires preprocessing. LMDB
supports concurrent writes and dynamic updates. CSR wins when:
- The graph is static (typical for benchmark/batch workloads)
- The working set exceeds OS page cache (CSR is more compact)
- Sequential scan patterns dominate (CSR is contiguous)

## 3. Dual-CSR: both directions as CSR

For static graphs (the common case in effective-distance), we can
build *both* directions as CSR:

```
category_parent.csr.idx/val   child -> [parents]   (upward walk)
category_child.csr.idx/val    parent -> [children]  (downward walk)
```

This eliminates LMDB entirely for the hot path. The cost model needs:

- **Preprocessing cost**: time to build both CSR artifacts from the
  raw edge list (O(E log E) for sorting + O(E) for writing)
- **Per-query savings**: CSR binary search + flat read vs LMDB
  B-tree cursor traversal
- **Expected query count**: amortizes preprocessing

### 3.1 When dual-CSR wins

```
break_even_queries = preprocessing_cost / per_query_savings
```

For simplewiki (6k edges): preprocessing ~50ms, savings ~0.5ms/query
-> break-even at ~100 queries. For enwiki (10M edges): preprocessing
~10s, savings ~1ms/query -> break-even at ~10k queries.

Batch workloads (computing effective distance for all articles)
easily exceed break-even. Interactive single-query workloads don't.

### 3.2 When LMDB wins

- Dynamic graphs (edges added/removed between queries)
- Single-query workloads (preprocessing not amortized)
- Memory-mapped shared access (multiple processes reading same DB)
- When the working set fits in OS page cache (LMDB B-tree
  traversal is ~same cost as CSR binary search when pages are hot)

## 4. Cost analyzer integration

The cost analyzer (`src/unifyweaver/core/cost_model.pl`) needs these
inputs to choose materialization strategy:

| Input | Source | Used for |
|---|---|---|
| `expected_query_count` | caller/manifest | Break-even calculation |
| `expected_lookups_per_query` | workload analysis | Total lookup volume |
| `edge_count` | corpus manifest | Preprocessing cost estimate |
| `graph_mutability` | caller declaration | CSR eligibility |
| `direction_needed` | kernel analysis | Which CSR(s) to build |
| `working_set_ratio` | demand filter output | Cache effectiveness |

### 4.1 Decision tree

```prolog
resolve_edge_store(Options, Store) :-
    option(expected_query_count(Q), Options),
    option(expected_lookups_per_query(L), Options),
    option(edge_count(E), Options),
    TotalLookups is Q * L,
    PreprocessCost is E * 0.00001,  % ~10us per edge for CSR build
    PerLookupSavings is 0.000005,   % ~5us saved per lookup (CSR vs LMDB)
    BreakEven is PreprocessCost / PerLookupSavings,
    (   TotalLookups > BreakEven
    ->  Store = csr
    ;   Store = lmdb_cached
    ).
```

### 4.2 Materialization options (resolved by cost analyzer)

| Option | Store | When |
|---|---|---|
| `materialisation(lmdb_cached)` | LMDB + L1/L2 cache | Default, dynamic, few queries |
| `materialisation(csr_parent)` | CSR child->parents | Static, many queries, upward only |
| `materialisation(csr_child)` | CSR parent->children | Descendant exploration needed |
| `materialisation(dual_csr)` | Both CSR directions | Bidirectional search, many queries |
| `materialisation(lmdb_eager)` | LMDB -> IntMap at startup | Small corpus, all edges fit in RAM |

## 5. Implementation roadmap

### Phase 1 (done): F# CSR reader
- `CsrLookupSource` implementing `ILookupSource`
- `csr_path(Path)` codegen option
- Binary search on `.idx`, positioned reads on `.val`
- Wraps with `TwoLevelCachedLookupSource`

### Phase 2: LMDB offset index backend
- Add `index_backend(lmdb_offset)` support to `CsrLookupSource`
- Open companion LMDB for O(1) parent -> (offset, count) lookup
- Useful for sparse parent ID spaces (Wikipedia page IDs)

### Phase 3: Forward CSR (child -> parents)
- Build `category_parent.csr.idx/val` from LMDB
- New `CsrParentLookupSource` or reuse `CsrLookupSource` with
  different artifact path
- Register in `WcLookupSources["category_parent"]`
- Benchmark vs LMDB cached on effective-distance workload

### Phase 4: Kernel integration
- Wire `CsrLookupSource` (category_child) into the
  effective-distance kernel for descendant-path exploration
- Bidirectional search: upward from seed + downward from root,
  meet in the middle
- Demand-set expansion: enumerate root's descendants within K hops

### Phase 5: Cost analyzer
- `resolve_edge_store/2` predicate in `cost_model.pl`
- Inputs: query count, lookups/query, edge count, mutability
- Outputs: materialization strategy per relation per direction
- Wire into `write_wam_fsharp_project/3` option resolution

## 6. References

- Format spec: `docs/design/WAM_REVERSE_INDEX_ARTIFACTS.md`
- Plan: `docs/design/WAM_FSHARP_CSR_PARALLEL_PLAN.md`
- Parity audit: `docs/design/WAM_FSHARP_PARITY_AUDIT.md`
- Cost model: `src/unifyweaver/core/cost_model.pl`
- Python builder: `examples/benchmark/build_reverse_csr_artifact.py`
- F# reader: `templates/targets/fsharp_wam/csr_reader.fs.mustache`
- Benchmark: `tests/core/test_wam_fsharp_csr_bench.pl`
