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
  memory-mapped path by using explicit read policy, bounded user-space
  cache, and optional OS-cache bypass when the platform supports it.

This is not a fourth materialisation mode. It is an artifact and
lifecycle choice that composes with the `eager`/`lazy`/`cached`
parent-edge modes defined in `WAM_LMDB_LAZY_PHILOSOPHY.md`.

## 0. Current artifact surface on `main`

This design should be read against the artifact framework that already
exists on `main`, not as a replacement for it.

There are currently two relevant LMDB shapes:

- **C# query relation artifact**: `mysql_stream_lmdb` writes a `main`
  DUPSORT database plus a C#-query manifest. It stores the MediaWiki
  IDs as decimal UTF-8 keys/values so the C# `LmdbRelationProvider`
  can serve exact column-0 lookup without depending on the Phase 1
  resident layout.
- **Phase 1 resident layout**: `s2i`, `i2s`, `meta`,
  `category_parent`, `category_child`, and `article_category` named
  databases with int32 little-endian IDs. This is the layout expected
  by resident LMDB WAM paths and by the Phase 1 converter.

The C# query runtime also already has a dependency-free artifact routing
surface in `QueryRuntime.cs`:

| Storage kind | Current role |
| --- | --- |
| `binary_artifact` | Default exact two-column relation artifact. |
| `delimited_artifact` | Text/bucket artifact fallback and bucket path. |
| `lmdb_artifact` | Optional exact LMDB provider in the separate LMDB runtime assembly. |
| `mmap_array_artifact` | Memory-mapped int-array artifact for large page/category relations and column-1-heavy access shapes. |

`RelationArtifactAccessPolicy.ResolveEffectiveDistanceArtifactStorageKind`
already chooses a storage kind from relation name, row count, and access
shape. For large `article_category`, column-0 lookup prefers
`lmdb_artifact`, while column-1 lookup, buckets, scans, and storage
prefer `mmap_array_artifact`. For `category_parent` and smaller
effective-distance relations, `mmap_array_artifact` is currently the
preferred artifact kind.

So the reverse-index question is not "do we need an artifact
framework?" The current question is:

1. which existing artifact kind can represent reverse child lookup for
   a target today;
2. when should the planner allow that artifact to be touched;
3. whether we need a new non-mmap CSR storage kind for reverse lookup
   when page-cache competition matters.

Every reverse artifact MUST declare the ID convention it stores. A
target reader may translate between conventions before opening the
artifact, but it must not silently interpret one encoding as another.

```prolog
id_encoding(int32_le).       % Phase 1 resident IDs
id_encoding(decimal_utf8).   % C# query LMDB artifact IDs today
```

For CSR, the preferred implementation encoding is `int32_le`, because
CSR is an integer layout. A C# query workload that still uses decimal
UTF-8 IDs needs either a companion mapping table or a build step that
emits an artifact in the reader's declared convention.

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
        index_backend(sorted_array),
        phase(cache_warmup),
        cache_bytes(67108864)
    ]))
]).

:- wam_lmdb_source(enwiki_categories, [
    parent_edge(category_parent/2),
    reverse_index(csr([
        ordering(root_bfs),
        index_backend(lmdb_offset),
        block_size_edges(65536),
        phase(runtime_available)
    ]))
]).
```

`reverse_index(auto)` is allowed later, once the cost model has enough
measurements to make the choice defensible.

For targets that already use the relation-artifact policy layer, the
resolved option should include a storage kind:

```prolog
reverse_index(artifact([
    relation(category_child/2),
    storage_kind(mmap_array_artifact),
    id_encoding(int32_le),
    ordering(parent_sort),
    phase(cache_warmup)
])).

reverse_index(artifact([
    relation(category_child/2),
    storage_kind(csr_pread_artifact),
    id_encoding(int32_le),
    ordering(root_bfs),
    index_backend(sorted_array),
    phase(runtime_available),
    io_policy(auto),
    cache_bytes(67108864)
])).
```

`csr_pread_artifact` is a proposed storage kind, not a current one. It
names the important distinction from `mmap_array_artifact`: reverse
child rows are fetched by explicit reads with bounded user-space cache,
not by mapping another large relation into the process address space.

`io_policy(auto)` is the default for CSR. See §3.4.2 for how it
resolves.

`index_backend(auto|sorted_array|lmdb_offset|dense_direct)` selects how
the runtime finds a parent's `(offset, count)` record before reading
`.val`. If omitted, the current behavior is `sorted_array`: binary
search over the CSR index. Explicit `index_backend(auto)` is the
cost-analyzer hook. It should resolve to `sorted_array` until the cost
model has measured terms for ID density, index size, LMDB lookup cost,
and preprocessing cost.

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
`category_child(parent, child)`. This is convenient and can reuse the
existing ingestion path, but the exact layout matters:

- C# query artifact shape: `main` DUPSORT plus manifest, with decimal
  UTF-8 IDs unless the manifest says otherwise.
- Phase 1 resident shape: named `category_child` DUPSORT sub-db with
  int32 little-endian IDs.

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

### 3.3 `reverse_index(mmap_array(...))`

Use the existing `mmap_array_artifact` family for reverse child lookup.
This is the closest current runtime artifact to a compact reverse
integer layout. It already has manifests, provider routing, column
selection, row-count metadata, and access-shape policy in the C# query
runtime.

Pros:

- already implemented for the C# query runtime;
- efficient for large int-ID relation scans and column lookups;
- fits the current `RelationArtifactAccessPolicy` storage-kind model.

Cons:

- it is deliberately memory-mapped;
- using it during the hot parent-edge walk can still compete for page
  cache with parent LMDB pages;
- it is target/runtime-specific today, not yet the cross-target reverse
  artifact contract.

Recommended phase: `planning_only` or `cache_warmup` unless the workload
explicitly opts into `runtime_available`.

### 3.4 `reverse_index(csr(...))`

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
2. read exactly `count * sizeof(child_id)` bytes from `.val` using the
   configured `io_policy`;
3. optionally cache that slice in a bounded user-space cache.

The important property is that the CSR files do not have to be
memory-mapped. Avoiding mmap removes a second reverse-edge virtual
address mapping from the hot path. It does not, by itself, eliminate OS
page-cache pressure: ordinary `read` and `pread` still populate the
kernel page cache. True page-cache bypass requires platform-specific
direct I/O, and that comes with alignment and portability costs.

This proposed artifact should be treated as a new storage kind in the
same conceptual family as `binary_artifact`, `lmdb_artifact`, and
`mmap_array_artifact`, rather than as a replacement for the existing C#
artifact framework.

One concrete effective-distance use is a staged search for workloads
that allow non-carrot-shaped paths. The runtime can first run the hot
ancestor kernel over the parent-edge store, preserving the current
ancestor-first reuse path. Only after that search is exhausted should it
open the reverse CSR path to add child nodes and widen the frontier. In
that shape, CSR is not competing with the parent-edge LMDB as the primary
resident structure; it is a deferred expansion artifact used when the
query semantics justify leaving the ancestor-only search space.

### 3.4.1 CSR index backend

CSR has two lookups:

1. map `parent_id` to the row location for that parent;
2. read the child slice from `.val`.

The first step is controlled by `index_backend(...)`:

| Backend | Meaning | Tradeoff |
| --- | --- | --- |
| `sorted_array` | Keep `.idx` sorted by parent ID and use binary search. | Simple, compact, no second LMDB lookup; best when the index fits in memory. |
| `lmdb_offset` | Store `parent_id -> offset,count` or `parent_id -> idx_row` in an LMDB sub-db. | Handles sparse Wikipedia/page IDs without dense direct arrays; adds LMDB B-tree lookup and page-cache pressure. |
| `dense_direct` | Store a direct array indexed by numeric parent ID. | Fastest lookup when IDs are dense; wasteful or impossible for sparse page-id spaces. |
| `auto` | Let the cost analyzer choose. | Defaults to `sorted_array`; may select `lmdb_offset` when measured lookup savings amortize extra build cost and the offset index fits the memory budget. |

An LMDB offset index may store the final `.val` offset directly, or it
may store the row number in `.idx`. Storing the final offset can skip the
binary search and the `.idx` read, but it means the CSR builder must
write or rewrite LMDB offset metadata as rows are finalized. That is
extra preprocessing and another artifact to keep consistent with the CSR
files. The cost analyzer should only choose it when the saved lookup
work outweighs the additional build time, disk bytes, and page-cache
touches.

The first `auto` rule is deliberately narrow. It selects `lmdb_offset`
only when all of these are known:

```prolog
expected_child_lookups_per_query(NLookups).
expected_query_count_per_artifact(NQueries).
sorted_array_lookup_ms_per_1000(SortedMs).
lmdb_offset_lookup_ms_per_1000(OffsetMs).
sorted_array_build_seconds(SortedBuild).
lmdb_offset_build_seconds(OffsetBuild).
```

The rule computes:

```text
total_lookup_savings_seconds =
  (NLookups * NQueries / 1000) * ((SortedMs - OffsetMs) / 1000)

marginal_build_seconds =
  max(0, OffsetBuild - SortedBuild)
```

`lmdb_offset` is eligible only when lookup savings is positive and
`total_lookup_savings_seconds >= marginal_build_seconds`.

It must also pass a memory guard. The caller can either declare:

```prolog
lmdb_offset_memory_fits(true).
```

or provide:

```prolog
lmdb_offset_bytes(Bytes).
available_memory_bytes(Available).
csr_index_memory_fraction(Fraction).  % default 0.05
```

In the second form, `lmdb_offset` is eligible only when:

```text
Bytes <= Available * Fraction
```

This keeps `auto` target-neutral and evidence-driven. Targets can feed
the terms from local benchmark telemetry, artifact manifests, or
platform probes, while omitted or incomplete measurements keep the
current `sorted_array` behavior.

### 3.4.2 CSR I/O policy

CSR access should expose an explicit I/O policy:

```prolog
io_policy(auto).
io_policy(buffered_pread).
io_policy(buffered_pread_drop).
io_policy(direct_io).
```

| Policy | Meaning | Default use |
| --- | --- | --- |
| `buffered_pread` | Use `pread` or equivalent positional reads. No mmap, but OS page cache is allowed. | Safe portable baseline. |
| `buffered_pread_drop` | Use `pread`, then ask the OS to drop consumed reverse blocks where supported, e.g. `posix_fadvise(..., DONTNEED)`. | Planning/warmup phases that should not leave reverse pages resident. |
| `direct_io` | Use direct I/O, e.g. Linux `O_DIRECT`, when the filesystem, alignment, block size, and runtime bindings support it. | Large aligned runtime reads where page-cache isolation beats direct-I/O overhead. |
| `auto` | Choose from phase, platform support, block size, and measurements. | Default. |

`direct_io` should not be the unconditional default. It can avoid page
cache competition, but it requires aligned buffers, aligned offsets and
sizes, platform-specific open flags, and careful handling of short or
small reads. It may also defeat useful kernel readahead. The initial
resolver should be conservative:

```prolog
resolve_csr_io_policy(Options, buffered_pread_drop) :-
    option(phase(Phase), Options),
    memberchk(Phase, [planning_only, cache_warmup]).
resolve_csr_io_policy(Options, direct_io) :-
    option(phase(runtime_available), Options),
    option(block_size_edges(B), Options),
    B >= 65536,
    option(platform_supports_direct_io(true), Options),
    option(alignment_verified(true), Options),
    option(measured_direct_io_win(true), Options).
resolve_csr_io_policy(_Options, buffered_pread).
```

This is pseudocode using `option/3` in the `library(option)` style. The
actual resolver should live beside the other cost-model predicates.

### 3.5 `reverse_index(csr([block_size_edges(N), ...]))`

For high-reuse corpora, the values can be split into related blocks:

```
category_child.blocks.meta
category_child.blocks.idx
category_child.blocks.00000
category_child.blocks.00001
...
```

The default should be one `.val` file with block metadata:

```
block_id, file_offset, edge_count, parent_min, parent_max, checksum
```

Block CSR lets the builder group parents whose child lists are likely to
be queried together. It also gives the runtime a natural cache unit:
cache or drop one block at a time, rather than caching arbitrary slices.
Separate block files are a future option for platforms where operational
constraints make a single large file awkward. They should not be the
first implementation because they increase file-descriptor management
and manifest complexity.

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
| `embedding_cluster` | expensive unless embeddings already exist | semantic locality | Out of scope unless a separate embedding pipeline is integrated. |

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

If `per_query_savings <= 0`, the artifact is contraindicated regardless
of expected query count. Benchmark reports should flag that case
explicitly instead of producing a negative break-even query count.

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
| Existing C# artifact path, planning/warmup only | `reverse_index(artifact([storage_kind(mmap_array_artifact), phase(cache_warmup)]))` | Uses implemented artifact routing while avoiding hot interleaving. |
| Many queries on one corpus, page-cache competition matters | `reverse_index(csr([ordering(parent_sort), phase(cache_warmup)]))` | Cheap build, non-mmap reverse reads, bounded warmup. |
| Many root-scoped queries | `reverse_index(csr([ordering(root_bfs), phase(cache_warmup)]))` | Better locality with roughly linear preprocessing. |
| Runtime descendant traversal | `reverse_index(csr([ordering(root_bfs), phase(runtime_available), cache_bytes(B)]))` | Descendant lookup is explicit and bounded. |
| Massive repeated workload | partitioned/block CSR with `graph_partition` or stronger ordering | High preprocessing cost may amortise. |

The default for `auto` should remain conservative until benchmarks say
otherwise:

```prolog
resolve_reverse_index(Options, none) :-
    % option/3 is SWI-Prolog library(option) style pseudocode.
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

These predicates are design stubs. Phase D planner integration defines
where they live and how target runtimes enforce them.

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
- Verify artifact readers reject mismatched `id_encoding` declarations
  instead of silently interpreting decimal UTF-8 IDs as int32 or the
  reverse.

Phase enforcement:

- Verify `planning_only` and `cache_warmup` reverse artifacts are closed
  or flagged inaccessible before the hot kernel's first WAM instruction.
- Verify the "reverse artifact touched during hot kernel" telemetry is
  zero for every phase except `runtime_available`.

Performance:

- Record preprocessing wall time and output size for each artifact.
- Record hot-kernel runtime with `planning_only`, `cache_warmup`, and
  `runtime_available`.
- Record minor and major page faults where the platform exposes them.
- Compare parent-only LMDB, reverse LMDB, `parent_sort` CSR, and
  `root_bfs` CSR.
- Compare `sorted_array`, `lmdb_offset`, and dense-direct CSR index
  backends where the ID space makes each backend plausible.
- Compare `buffered_pread`, `buffered_pread_drop`, and `direct_io`
  where the platform supports direct I/O and the block layout can meet
  alignment requirements.
- Compute break-even query counts from observed preprocessing cost and
  per-query savings.

Regression guard:

- Add a telemetry field for "reverse artifact touched during hot
  kernel". It should be zero for `planning_only` and `cache_warmup`.
- Add telemetry for resolved CSR `io_policy`, direct-I/O support,
  alignment fallback, and any `DONTNEED` advisory calls.

## 9. Implementation plan

Phase A - design and option validation:

- Land this design doc.
- Add Prolog option parsing tests for
  `reverse_index(none|lmdb|mmap_array|csr|artifact|auto)`.
- Add option parsing for `id_encoding(...)` and
  `io_policy(auto|buffered_pread|buffered_pread_drop|direct_io)`.
- Add option parsing for
  `index_backend(auto|sorted_array|lmdb_offset|dense_direct)`.
- Reject `phase(runtime_available)` when no runtime reverse lookup API
  exists for the target.
- Map `reverse_index(artifact(...))` onto the existing storage-kind
  vocabulary where a target already supports relation artifacts.
- Define the minimum benchmark criteria before `reverse_index(auto)` can
  choose anything other than `none` or an existing target artifact.

Phase B - baseline builder:

- Extend the existing Phase 1 conversion to optionally write reverse
  LMDB into a separate file or named database.
- Emit manifest telemetry: edge count, parent key count, build wall
  time, ordering, artifact bytes.
- For C# query artifacts, decide whether reverse child lookup should be
  emitted as `mmap_array_artifact`, `lmdb_artifact`, or both, using the
  same access-shape policy vocabulary already in `QueryRuntime.cs`.

Phase C - CSR prototype:

- Build `parent_sort` CSR from `category_parent/2` with
  `id_encoding(int32_le)`. Prototype script:
  `examples/benchmark/build_reverse_csr_artifact.py`.
- Add a small Rust reader, co-located with the Rust LMDB sink/build path,
  with direct-index or binary-search lookup. C# binding can follow once
  the format and policy are stable. Prototype probe:
  `examples/benchmark/read_reverse_csr_artifact.py`.
- Implement `io_policy(buffered_pread)` and
  `io_policy(buffered_pread_drop)` first.
- Prototype `io_policy(direct_io)` only after block alignment and
  platform support are verified.
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
- `src/unifyweaver/targets/csharp_query_runtime/QueryRuntime.cs` -
  current relation artifact storage-kind policy and `mmap_array_artifact`
  provider.
- `src/unifyweaver/targets/csharp_query_runtime_lmdb/LmdbRelationProvider.cs`
  - optional C# LMDB artifact provider.
- `src/unifyweaver/runtime/rust/mysql_stream/src/lmdb_sink.rs` -
  current MediaWiki categorylinks LMDB artifact sink.
