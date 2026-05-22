# WAM Reverse-Index Artifacts

**Status**: Design discussion. Companion to
[`WAM_LMDB_LAZY_SPECIFICATION.md`](WAM_LMDB_LAZY_SPECIFICATION.md),
focused on the reverse `category_child/2` artifact and how it composes
with parent-edge LMDB access.

**Snapshot date**: 2026-05-22.

The lazy LMDB docs define when a WAM target reads the hot
`category_parent(child, parent)` relation: eager materialisation, lazy
cursor lookup, or lazy lookup with a bounded cache. This document adds a
separate axis: whether we also build a reverse child index for
`category_child(parent, child)`, how that artifact is laid out, and when
the runtime is allowed to touch it.

The short version:

- `category_parent/2` is the hot edge relation for upward category
  walks. It should remain the primary memory-mapped LMDB relation.
- `category_child/2` is useful for planning, demand-set discovery,
  cache warmup, and workloads that repeatedly ask for descendants.
- Interleaving both relations during the hot algorithm can make two
  backing stores compete for the OS page cache, even if they are in
  separate files.
- A CSR-style reverse artifact can keep child lookup out of the
  memory-mapped path by using explicit `pread` slices and a bounded
  user-space cache.

This is not a fourth lazy/eager mode. It is an artifact and lifecycle
choice that composes with the L0/L1/L2 parent-edge tier.

## 1. Problem

Wikipedia's `categorylinks` export gives the raw material for both
directions:

```
cl_from       = source page/category id
cl_type       = page | subcat | file
cl_target_id  = target category id
```

For `cl_type = subcat`, the canonical parent edge is:

```
category_parent(cl_from, cl_target_id)
```

For `cl_type = page`, the canonical article edge is:

```
article_category(cl_from, cl_target_id)
```

The Phase 1 LMDB conversion can derive:

```
category_child(parent, child)
```

by reversing every `category_parent(child, parent)` edge.

That reverse map is smaller in key cardinality because there are far
fewer parent categories than child categories in the relevant workload.
It is therefore attractive for finding descendants or building a demand
set. The problem is not storage size; it is runtime locality. If the hot
algorithm alternates between parent lookups and child lookups, the OS
may keep paging between two B-trees or two files. The reverse index then
pollutes the page cache that the parent-edge walk needed.

So the design needs two decisions:

1. Should a reverse artifact exist at all?
2. If it exists, when may the runtime touch it?

## 2. User-facing options

Initial Prolog shape:

```prolog
:- wam_lmdb_source(enwiki_categories, [
    parent_edge(category_parent/2),
    reverse_index(none)
]).

:- wam_lmdb_source(enwiki_categories, [
    parent_edge(category_parent/2),
    reverse_index(lmdb([
        phase(planning_only)
    ]))
]).

:- wam_lmdb_source(enwiki_categories, [
    parent_edge(category_parent/2),
    reverse_index(csr([
        ordering(parent_sort),
        phase(cache_warmup),
        cache_bytes(67108864)
    ]))
]).

:- wam_lmdb_source(enwiki_categories, [
    parent_edge(category_parent/2),
    reverse_index(csr([
        ordering(root_bfs),
        block_size_edges(65536),
        phase(runtime_available)
    ]))
]).
```

`reverse_index(auto)` is allowed later, once the cost model has enough
measurements to make the choice defensible.

### 2.1 Reverse-index phases

| Phase | Meaning | Page-cache posture |
| --- | --- | --- |
| `planning_only` | Use the reverse artifact before the hot kernel to choose demand sets, roots, or query batches. Close it before measured runtime. | No deliberate competition during parent walk. |
| `cache_warmup` | Use the reverse artifact before the hot kernel to prefetch or populate an explicit parent-edge cache. Stop touching it during the kernel. | Competition is bounded to warmup. |
| `runtime_available` | The algorithm may call reverse lookup during the hot kernel. | Highest risk; only appropriate when descendant lookup is truly part of the workload. |

These phases are independent of `lmdb_materialisation(eager|lazy|cached)`
for parent lookup. For example, `lmdb_materialisation(lazy)` plus
`reverse_index(csr([phase(planning_only)]))` means parent edges are read
lazily during the kernel, while child edges are only used before the
kernel starts.

## 3. Artifact choices

### 3.1 `reverse_index(none)`

Build only the parent-edge LMDB. This is the right default when:

- the workload is a small number of one-shot upward queries;
- descendant lookup is not required;
- preprocessing time must be minimal;
- the corpus will not be reused enough to amortise extra artifacts.

### 3.2 `reverse_index(lmdb(...))`

Build a separate LMDB file or named database for
`category_child(parent, child)`. This is convenient and reuses the
existing ingestion path.

Pros:

- simple to build from Phase 1 conversion;
- supports sorted duplicate values per parent;
- easy to inspect with existing LMDB tooling.

Cons:

- ordinary LMDB access is memory-mapped;
- touching this DB during the hot kernel can compete with the
  parent-edge LMDB for page cache;
- it gives less control over block packing than a custom artifact.

Recommended phase: `planning_only` or `cache_warmup`.

### 3.3 `reverse_index(csr(...))`

Build a compressed sparse row style artifact:

```
idx[parent] = offset,count
val[offset : offset + count] = child ids for that parent
```

The artifact can be represented as:

```
category_child.csr.idx   fixed-width parent -> offset/count records
category_child.csr.val   packed int32 child ids
category_child.csr.meta  format version, ordering, id width, checksums
```

For dense parent IDs, `.idx` can be a direct array. For sparse parent
IDs, `.idx` is sorted by parent ID and looked up with binary search or a
small in-memory key index.

Lookup is:

1. find the parent's `(offset, count)` record;
2. `pread` exactly `count * sizeof(child_id)` bytes from `.val`;
3. optionally cache that slice in a bounded user-space cache.

The important property is that the CSR files do not have to be
memory-mapped. The runtime can use ordinary reads or `pread`, keeping
reverse-edge caching explicit and bounded instead of letting the OS page
cache make the trade-off implicitly.

### 3.4 `reverse_index(csr([block_size_edges(N), ...]))`

For high-reuse corpora, the values can be split into related blocks:

```
category_child.blocks.meta
category_child.blocks.idx
category_child.blocks.00000
category_child.blocks.00001
...
```

or kept as one `.val` file with block metadata:

```
block_id, file_offset, edge_count, parent_min, parent_max, checksum
```

Block CSR lets the builder group parents whose child lists are likely to
be queried together. It also gives the runtime a natural cache unit:
cache or drop one block at a time, rather than caching arbitrary slices.

## 4. Ordering strategies

The ordering strategy controls how related rows are packed. This is
similar in spirit to defragmenting a graph artifact: move rows and
columns so that queries touch fewer distant pages or blocks. It is not
free, and it only pays when the same dataset sees enough queries.

| Ordering | Cost | Locality target | Notes |
| --- | ---: | --- | --- |
| `parent_sort` | `O(E log E)`, or near `O(E)` if input is already parent-sorted | children for one parent contiguous | Baseline CSR build. |
| `root_bfs` | `O(V + E)` after adjacency exists | parents near the same root clustered | Good cheap default for repeated root-scoped workloads. |
| `component_degree` | `O(V + E)` plus sorting components/degrees | high-fanout parents and components grouped | Useful when hot queries start at common ancestors. |
| `multi_root_bfs` | roughly `O(V + E)` per root frontier, usually bounded by one shared traversal | several known roots clustered | Good when query roots are known at ingest time. |
| `graph_partition` | near-linear practical cost, high constants | blocks with low cross-block traffic | METIS-style partitioning; valuable only for large reusable corpora. |
| `spectral` | expensive eigensolver iterations | global bandwidth reduction | Mostly completeness; likely too costly for ordinary benchmarks. |
| `bandwidth_heuristic` | NP-hard exact problem; heuristic cost varies | block diagonal shape | Use only when query volume is massive. |
| `embedding_cluster` | expensive unless embeddings already exist | semantic locality | Useful if topic embeddings are already part of the corpus. |

The cheap and moderate options are close enough to linear that they are
reasonable for corpora that will be queried repeatedly. The high-cost
options belong behind explicit configuration because their preprocessing
cost can dominate unless query volume is truly massive.

## 5. Break-even model

The artifact should be selected from measured terms, not a fixed rule:

```
break_even_queries =
    reverse_preprocess_wall_seconds /
    per_query_savings_wall_seconds
```

Where:

```
per_query_savings =
    baseline_parent_only_query_time -
    query_time_with_reverse_artifact
```

If the caller expects only a few queries, parent-only LMDB usually wins
because `reverse_preprocess_wall_seconds` is not amortised. If the same
dataset will support thousands or millions of queries, even a linear
preprocessing pass can become cheap.

The cost model needs these inputs:

| Field | Source | Meaning |
| --- | --- | --- |
| `expected_query_count_per_process` | caller | Query count inside one process. |
| `expected_query_count_per_artifact` | caller or corpus manifest | Query count over the artifact's lifetime. |
| `needs_descendant_lookup` | caller or planner | Whether child lookup is semantically required. |
| `reverse_index_phase` | caller | Whether child lookup can stop before the hot kernel. |
| `reverse_preprocess_cost_estimate` | builder telemetry | Cost to build the chosen artifact. |
| `reverse_lookup_savings_estimate` | benchmark telemetry | Per-query win after the artifact exists. |
| `memory_map_parent_only` | caller/default true | Whether parent LMDB should be the only mmap hot file. |

## 6. Recommended defaults

| Workload | Recommended option | Reason |
| --- | --- | --- |
| Few upward queries | `reverse_index(none)` | Avoid preprocessing and page-cache competition. |
| Planning needs descendants, hot kernel does not | `reverse_index(lmdb([phase(planning_only)]))` | Simple artifact; close before parent walk. |
| Many queries on one corpus | `reverse_index(csr([ordering(parent_sort), phase(cache_warmup)]))` | Cheap build, non-mmap reverse reads, bounded warmup. |
| Many root-scoped queries | `reverse_index(csr([ordering(root_bfs), phase(cache_warmup)]))` | Better locality with roughly linear preprocessing. |
| Runtime descendant traversal | `reverse_index(csr([ordering(root_bfs), phase(runtime_available), cache_bytes(B)]))` | Descendant lookup is explicit and bounded. |
| Massive repeated workload | partitioned/block CSR with `graph_partition` or stronger ordering | High preprocessing cost may amortise. |

The default for `auto` should remain conservative until benchmarks say
otherwise:

```prolog
resolve_reverse_index(Options, none) :-
    option(expected_query_count_per_artifact(Q), Options, 1),
    Q < 100,
    \+ option(needs_descendant_lookup(true), Options).
```

Actual thresholds should come from measured benchmark data, not from
this sketch.

## 7. Interaction with parent-edge LMDB

The parent-edge store remains the primary hot artifact:

```
category_parent(child, parent)
```

If parent lookup is memory-mapped, the reverse child artifact should not
be touched during the hot kernel unless `phase(runtime_available)` is
explicit. This gives the planner a clear rule:

```prolog
hot_kernel_mmap_sources([category_parent]).
reverse_index_phase(planning_only).
```

or:

```prolog
hot_kernel_mmap_sources([category_parent]).
reverse_index_phase(cache_warmup).
reverse_index_runtime_access(disabled).
```

If a workload genuinely needs both directions during runtime, the target
should prefer CSR with explicit cache sizing over a second mmap source:

```prolog
reverse_index(csr([
    phase(runtime_available),
    cache_bytes(134217728)
])).
```

## 8. Testing and measurement plan

Correctness:

- Build `category_parent/2` only, LMDB reverse, CSR reverse, and block
  CSR reverse from the same fixture.
- Verify identical descendant sets for sampled parents.
- Verify identical demand sets and effective-distance outputs when a
  reverse artifact is used only for planning or warmup.

Performance:

- Record preprocessing wall time and output size for each artifact.
- Record hot-kernel runtime with `planning_only`, `cache_warmup`, and
  `runtime_available`.
- Record minor and major page faults where the platform exposes them.
- Compare parent-only LMDB, reverse LMDB, `parent_sort` CSR, and
  `root_bfs` CSR.
- Compute break-even query counts from observed preprocessing cost and
  per-query savings.

Regression guard:

- Add a telemetry field for "reverse artifact touched during hot
  kernel". It should be zero for `planning_only` and `cache_warmup`.

## 9. Implementation plan

Phase A - design and option validation:

- Land this design doc.
- Add Prolog option parsing tests for `reverse_index(none|lmdb|csr|auto)`.
- Reject `phase(runtime_available)` when no runtime reverse lookup API
  exists for the target.

Phase B - baseline builder:

- Extend the existing Phase 1 conversion to optionally write reverse
  LMDB into a separate file or named database.
- Emit manifest telemetry: edge count, parent key count, build wall
  time, ordering, artifact bytes.

Phase C - CSR prototype:

- Build `parent_sort` CSR from `category_parent/2`.
- Add a small reader with direct-index or binary-search lookup.
- Use `pread` instead of mmap by default.
- Add bounded slice/block cache.

Phase D - planner integration:

- Add `resolve_reverse_index/2`.
- Add phase enforcement so `planning_only` and `cache_warmup` cannot be
  touched during the hot kernel.
- Wire telemetry into the benchmark reports.

Phase E - ordering refinements:

- Add `root_bfs` and `component_degree`.
- Add block CSR metadata and cache-block sizing.
- Evaluate high-cost graph partitioning only if measured query volume
  justifies it.

## 10. References

- `WAM_LMDB_LAZY_PHILOSOPHY.md` - lazy/eager tier vocabulary.
- `WAM_LMDB_LAZY_SPECIFICATION.md` - parent-edge lookup contract.
- `WAM_LMDB_RESIDENT_INTERNING_SPECIFICATION.md` - LMDB layout and
  resident ID conventions.
- `QUERY_PLAN_RUNTIME_PHILOSOPHY.md` - planner placement for runtime
  decisions.
- `CACHE_COST_MODEL_PHILOSOPHY.md` - cost-model vocabulary.
