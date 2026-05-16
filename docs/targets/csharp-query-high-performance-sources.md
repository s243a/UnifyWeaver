# C# Query High-Performance Relation Sources

This note maps the existing C# query relation-source surface to the next
high-performance backends: LMDB-backed artifacts, memory-mapped arrays, and
hash/block-sharded files. It is intentionally C# query-specific and should be
read with `docs/proposals/PREPROCESSED_PREDICATE_ARTIFACTS.md`.

## Current Surface

The C# query runtime already has the right coarse-grained provider seams:

- `IRelationProvider` for full relation scans.
- `IRetentionAwareRelationProvider` for streaming, replayable, or externally
  materialized bindings.
- `IIndexedRelationProvider` for bound-column lookups.
- `IIndexedRelationBucketProvider` and `IIndexedRelationBucketJoinProvider`
  for broad joins that can exploit sorted/indexed physical layouts.
- `IRelationCardinalityProvider` for planner cost estimates.

`ConfiguredDelimitedRelationProvider` currently resolves these source modes:

- `preload`: read delimited rows into `InMemoryRelationProvider`.
- `delimited`: stream directly from the delimited source.
- `artifact`: build/read runtime binary or delimited artifacts.
- `artifact-prebuilt`: use the same artifact providers while separating
  preprocessing cost from query execution in benchmarks.

There is also C# LMDB ingestion support at
`src/unifyweaver/runtime/csharp/lmdb_ingest/`. That code writes LMDB databases
with LightningDB and mirrors the Python ingest contract, but it is not yet a
C# query `IRelationProvider`. Separately, `examples/lmdb_relation_artifact/`
proves a local LMDB exact-relation artifact with a manifest, scan, lookup, and
dupsort support.

## Design Goal

Do not make LMDB a one-off query-engine special case. Treat it as one concrete
physical artifact backend under a common access-contract model:

- exact full scan
- single-column lookup
- grouped one-to-many lookup
- sorted bucket stream for joins
- cardinality and physical-size metadata

The runtime planner should ask for access by capability and key shape, not by
file extension or backend name. LMDB, memory-mapped arrays, and sharded
hash/block files can then compete on the same contract.

## Backend Options

| Backend | Best At | Weak At | Notes |
| --- | --- | --- | --- |
| Current binary artifact | exact binary relation lookup and bucket joins | wider arity, flexible manifests | Already wired into C# query runtime. Keep as the baseline. |
| Current delimited artifact | n-ary exact rows and covering buckets | large random seek workloads if no compact index | Already wired for wider relations. Useful control case. |
| LMDB artifact | mmap-backed B-tree lookup, MVCC readers, one-to-many dupsort | dependency and packaging complexity, cursor/transaction lifetime | C# ingest exists; runtime reader/provider is missing. |
| Memory-mapped sorted array | dense sequential scan, binary search, small fixed-width IDs | variable-width strings, updates | Good candidate for `int32`/`int64` interned relations. |
| Memory-mapped hash table | exact lookup with predictable O(1)-style probes | broad ordered joins, collision/layout tuning | Useful for high-selectivity predicates and point probes. |
| Hash/block-sharded files | bounded random I/O, parallel shard reads, streaming build | cross-shard joins and global ordering | Filename or block can be derived from hash prefix. Good large-scale option. |

LMDB is attractive because it gives a production mmap-backed B-tree quickly.
It should not preclude simpler mmap array/hash formats where the workload has
fixed-width IDs and predictable access patterns.

## Proposed Runtime Abstraction

Add an artifact/provider capability layer rather than adding many new
`RelationSourceMode` values immediately.

```text
RelationSourceMode.Artifact
  -> manifest declares physical_backend = binary|delimited|lmdb|mmap_array|mmap_hash|sharded_hash
  -> provider factory opens the matching backend
  -> planner sees capabilities, not backend-specific APIs
```

The C# runtime can start with:

```csharp
public interface IRelationArtifactProviderFactory
{
    bool TryOpen(string manifestPath, IRelationProvider? fallback, out IRelationProvider provider);
}
```

That factory can return providers implementing the existing interfaces:

- `IRelationProvider` for scans.
- `IIndexedRelationProvider` for lookup by indexed columns.
- `IIndexedRelationBucketProvider` for grouped/key-ordered buckets.
- `IIndexedRelationBucketJoinProvider` only when the backend can join without
  materializing both sides.
- `IRelationCardinalityProvider` from manifest metadata.

This keeps `QueryExecutor` mostly unchanged. The planner already has branches
for indexed providers and bucket providers; new backends should enter through
those branches.

## LMDB Path

The first LMDB C# query slice should be exact and narrow:

1. Define a C#-readable LMDB relation manifest shape based on the Rust
   prototype:
   - predicate name/arity
   - backend: `lmdb`
   - db name
   - dupsort flag
   - key/value encoding
   - access contracts
   - row count
   - source hash and format version
2. Add an optional C# query runtime integration project, not a hard dependency
   in `UnifyWeaver.QueryRuntime.Core.csproj`, because LightningDB is external.
   The default core runtime should not reference LightningDB. Instead,
   `UnifyWeaver.QueryRuntime.Lmdb` (or an equivalent optional integration
   project) should own the NuGet dependency and expose the provider/factory.
   A Prolog `declare_target` or generated-project option can then make that
   optional project a hard reference for programs that explicitly opt into
   LMDB-backed relations.
3. Implement `LmdbRelationProvider` for arity-2 rows with:
   - full scan
   - arg0 lookup
   - arg0 grouped lookup when dupsort is enabled
   - row-count metadata
4. Add a smoke benchmark that compares:
   - current binary artifact
   - current delimited artifact
   - LMDB artifact
   - preload
5. Only after measurements, decide whether `auto` should ever choose LMDB.

Important lifetime rule: hide LMDB transactions and raw memory from query
operators. Return owned C# values from provider methods unless a later measured
hot path justifies a scoped zero-copy reader.

## Memory-Mapped Array Path

Use this when relation values can be represented as fixed-width IDs and the
desired access is scan or binary-search lookup.

Candidate format:

```text
manifest.json
rows.uwa          fixed-width sorted pairs or tuples
index.col0.uwa    optional offsets/ranges by key
```

Initial access contracts:

- scan rows in physical order
- lookup by exact key on column 0
- bucket stream by column 0

This is simpler than LMDB and avoids external dependencies, but the ID strategy
must be explicit in the manifest. The same physical format should support at
least these strategies:

- `provided_id`: the input relation already carries stable numeric IDs.
- `position_id`: the preprocessor assigns IDs from row or intern-table
  position. This is compact and fast, but local to one artifact build.
- `hash_id`: the preprocessor derives IDs from a stable string hash. This is
  slower and needs collision policy metadata, but generalizes better to
  distributed preprocessing.
- `sidecar_string_table`: the array stores fixed-width IDs while a sidecar
  table maps strings to IDs for lookup and diagnostics.

The first implementation can require `provided_id` or `position_id`, but the
manifest should reserve the field from the start:

```json
{
  "physical_backend": "mmap_array",
  "id_strategy": "provided_id",
  "id_width": 32,
  "string_table": null
}
```

This is likely the right comparison point for effective-distance and graph
workloads once preprocessing is explicit.

## Hash/Block-Sharded File Path

Use this when point lookup or selective joins dominate and global ordering is
not required.

Candidate layout:

```text
manifest.json
shards/00.uwh
shards/01.uwh
...
```

The high bits of a hash select a shard. Each shard can contain either:

- sorted fixed-width keys with binary search
- a compact open-addressed hash table
- compressed blocks with a small in-shard index

This trades preprocessing/build cost and disk footprint for bounded lookup
cost. It is a natural fit for user-suggested streamed hash maps and for top-k
plans that can drop low-quality candidates between phases.

## Source-Mode Policy

Keep existing `RelationSourceMode` stable in the first implementation:

- `artifact` means "use the best available exact artifact backend for this
  manifest."
- `artifact-prebuilt` means the same, but do not charge build cost to runtime
  measurements.
- Backend choice lives in the manifest and provider factory.

Only introduce backend-specific config later if measurements show that users
need manual override:

- `UNIFYWEAVER_RELATION_ARTIFACT_BACKEND=lmdb|mmap-array|mmap-hash|auto`
- or a generated manifest/preprocess declaration with `backend(lmdb)`.

## Recommended Next Branches

1. `design/csharp-query-artifact-provider-factory`
   - Add the provider factory interface and manifest backend-dispatch sketch.
   - No external dependency.
   - Acceptance: current binary/delimited artifact providers can be opened via
     the factory in tests.
2. `feat/csharp-query-lmdb-provider-smoke`
   - Optional runtime project using LightningDB; core runtime stays
     dependency-free.
   - Read an LMDB artifact produced by existing C# ingest or the Rust prototype.
   - Acceptance: arity-2 scan and arg0 lookup match delimited facts.
3. `bench/csharp-query-lmdb-source-mode-sweep`
   - Add a focused benchmark comparing preload, binary artifact, delimited
     artifact, and LMDB on one measured workload.
   - Acceptance: report open time, lookup time, scan time, memory retained, and
     artifact size.
   - Status: implemented by
     `examples/benchmark/benchmark_csharp_query_lmdb_source_mode_sweep.py`.
4. `feat/csharp-query-mmap-array-provider-smoke`
   - Implement a dependency-free fixed-width mmap array provider for one
     manifest-declared ID strategy.
   - Acceptance: scan and lookup parity with delimited facts, plus benchmark
     comparison against LMDB.
   - Status: implemented by `MmapArrayRelationArtifactProvider` and covered by
     `tests/test_csharp_query_mmap_array_provider_smoke.py`.

## Non-Goals For The First Runtime PR

- Replacing existing binary/delimited artifacts.
- Making LMDB the default `auto` policy.
- Adding raw LMDB pointers or borrowed mmap spans to query operators.
- Supporting arbitrary n-ary LMDB layouts before arity-2 access is measured.
- Building a generalized preprocessing language before one backend is wired
  end-to-end.

## Open Questions

- The first C# LMDB reader uses a C#-owned manifest
  (`unifyweaver.lmdb_relation.v1`) that matches the existing C#/Python ingest
  key/value contract and can later converge with the shared artifact proposal.
- Effective-distance `category_parent/2` is the first real-workload artifact
  backend measurement because it exercises the same arity-2 graph support
  relation used by the established Wikipedia/SimpleWiki benchmark family.
  The benchmark now supports repeated 300/1k/5k/10k runs with summary output
  for best lookup, bucket, scan, and artifact-size backend.
  At these scales, mmap-array and preload are expected to beat LMDB; LMDB is
  still the larger-data comparison point once memory pressure starts to matter
  around larger 50k+ runs. The same benchmark can prepare and run the
  SimpleWiki category-only `50k_cats` and `100k_cats` fixtures with memory and
  optional idle guardrails. That mirrors the Haskell scaling surface where LMDB
  closed the gap near 100k category-only queries. `500k_cats` is the preferred
  intermediate checkpoint before a million-scale run, but both `500k_cats` and
  `1m_cats` are full-English-Wikipedia/enwiki fixtures and are fixture-driven
  rather than auto-generated by the C# wrapper. The fixture-preparation path
  should reuse the existing Rust MediaWiki MySQL parser in
  `src/unifyweaver/runtime/rust/mysql_stream/`, as used by
  `examples/streaming/enwiki_category_ingest.pl`, instead of adding another
  dump tokenizer. For the C# backend benchmark, the fixture-prep helper is
  `examples/benchmark/prepare_csharp_query_enwiki_category_fixture.py`; it
  writes capped TSV fixtures first. Direct Rust-parser-to-LMDB ingestion is a
  valid future optimization, but is intentionally not part of this first C#
  benchmark fixture path. The C# backend benchmark now accepts a persistent
  `--artifact-root` and reuses existing backend artifacts by default; use
  `--refresh-artifacts` to rebuild, matching the Haskell benchmark's explicit
  LMDB refresh convention.
  Haskell's LMDB cache modes remain a higher-level query-locality policy; this
  C# query benchmark intentionally isolates backend scan, lookup, and bucket
  access before adding cache-policy comparisons.
