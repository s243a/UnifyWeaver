# Proposal: Preprocessed Predicate Artifacts

**Status:** Draft
**Version:** 0.1
**Date:** 2026-04-19
**Proposed Implementation Phase:** Query runtime storage and planner follow-up

## Executive Summary

UnifyWeaver should treat preprocessed data as a first-class predicate access
option, not as a single storage format. A preprocessed artifact may be an
embedded database, a memory-mapped hash table, a sorted columnar run, a
distributed table, an inverted index, a vector index, or a model-derived corpus
such as a Cohere/Wikipedia semantic artifact. These are all different physical
representations of facts or scored relations, but the planner should reason
about them through the same logical boundary: which predicate access patterns
they can satisfy, with what semantics and cost.

The core design is to separate artifact format from runtime access contract.
Targets may continue to choose databases and storage engines that are native or
well-supported in that target language, while the query engine asks for scans,
point lookups, prefix/range probes, adjacency expansion, grouped summaries,
full-text search, vector search, or materialized recursive summaries. This keeps
the C# runtime work aligned with the existing relation-retention and
materialization-planner direction without forcing every target to standardize on
one database or binary layout.

## Motivation

Recent C# query-runtime work made relation retention explicit and started
measuring when the engine should stream, replay, externally materialize, or keep
compact cached rows. The next pressure point is larger and more diverse data:
for some predicates the best representation is not an in-memory `object[]` list
or a runtime-built adjacency map. It may be cheaper to preprocess once into an
artifact that is reused across many queries.

Examples include:

- A memory-mapped hash table for high-volume exact key lookups.
- A partitioned hash layout where part of the key hash selects a file or block.
- A memory-mapped adjacency index for transitive-closure seed expansion.
- A SQLite, LiteDB, DuckDB, bbolt, RocksDB, LMDB, Datomic, Redis, or similar
  backing store selected per target/runtime.
- A Hadoop/HDFS or object-store-backed distributed table for large batch data.
- An inverted index for text predicates.
- A vector/ANN index or model-derived embedding corpus for semantic predicates.
- A deterministic materialized closure or grouped-summary sidecar.

Without an explicit artifact contract, each of these becomes target-specific
glue. With a contract, the planner can compare them against current streaming
and in-memory paths and fall back cleanly when an artifact is unavailable.

## Design Goals

- Preserve predicate semantics first. Exact artifacts must behave like the
  relation they replace; approximate/model artifacts must declare scored or
  approximate semantics explicitly.
- Let targets choose storage engines that fit their ecosystems instead of
  forcing one universal database.
- Keep preprocessing optional. Existing source loading and current
  `IRelationProvider` behavior remain valid.
- Make invalidation explicit with manifests that bind artifacts to source
  hashes, schema hashes, format versions, target capabilities, and model
  versions when applicable.
- Expose access patterns, not implementation details, to the query planner.
- Support local, embedded, remote, and distributed artifact providers without
  hiding their different latency and consistency profiles.
- Start with a small exact C# prototype before broad cross-target work.

## Architecture Overview

The proposed boundary has three layers:

1. Predicate declaration layer

   A predicate may declare one or more preprocessing candidates. Each candidate
   describes the source, artifact builder, artifact format, required target
   capabilities, exactness, and supported access contracts.

2. Artifact manifest layer

   A built artifact has a manifest that records enough metadata to decide
   whether it can be opened safely:

   ```json
   {
     "predicate": "edge/2",
     "artifact_kind": "partitioned_hash",
     "format_version": 1,
     "exactness": "exact",
     "source_hash": "sha256:...",
     "schema_hash": "sha256:...",
     "access": ["point_lookup", "prefix_lookup", "scan"],
     "target_capabilities": ["mmap", "little_endian"],
     "files": ["edge_hash_00.bin", "edge_hash_01.bin"]
   }
   ```

   Semantic/model artifacts add corpus, embedding, model, quantization, and
   distance metadata. Approximate artifacts must not masquerade as exact
   relation providers.

   For exact artifacts, the target-neutral manifest seam should also preserve
   the declaration intent that selected the physical layout. The current
   Clojure benchmark path is the first proving ground for this narrower schema:

   ```json
   {
     "predicate": "article_category/2",
     "resolved_mode": "artifact",
     "file_format": "grouped_tsv_arg1",
     "access": ["scan", "arg1_grouped_lookup"],
     "declaration": {
       "source": "shared_preprocess",
       "mode": "artifact",
       "kind": "exact_hash_index",
       "format": "exact_hash_index",
       "access_contracts": [
         "exact_key_lookup",
         "arg_position_lookup(1)",
         "grouped_values_lookup([2])",
         "scan"
       ],
       "options": ["key([1])", "values([2])"]
     }
   }
   ```

   Not every runtime needs to emit the exact same file names or storage
   encodings. The important cross-target seam is:

   - resolved storage mode
   - physical format
   - supported runtime access contracts
   - originating preprocess declaration kind/options
   - source and schema invalidation metadata

   That is enough for a planner or provider to explain both what artifact was
   chosen and why it was considered semantically valid.

3. Runtime access layer

   The runtime opens artifacts through a narrow provider interface and asks for
   access patterns:

   ```csharp
   public interface IPredicateArtifactProvider
   {
       PredicateArtifactDescriptor Descriptor { get; }
       bool Supports(PredicateAccessRequest request);
       IEnumerable<object[]> Scan(PredicateId predicate);
       bool TryLookup(PredicateId predicate, ReadOnlySpan<object?> key, out IReadOnlyList<object[]> rows);
       bool TryRange(PredicateId predicate, PredicateRangeRequest request, out IEnumerable<object[]> rows);
       bool TryVectorSearch(PredicateId predicate, VectorSearchRequest request, out IEnumerable<ScoredRow> rows);
   }
   ```

   The exact method shapes should be smaller in the first implementation, but
   the important point is the planner requests capabilities rather than naming
   SQLite, Redis, mmap, or a model file directly.

## Artifact Families

### Embedded Database Artifacts

Embedded databases are useful when a target has a mature local database story.
They support durable reuse, ad hoc indexing, transactions during build, and
standard operational tooling. The downside is dependency management, startup
cost, and varying behavior across target ecosystems.

Candidate uses:

- Small to medium lookup tables.
- Multi-index relations.
- Development-friendly artifact inspection.
- Targets where the database driver is already idiomatic.

### Immutable And Temporal Database Artifacts

Datomic-like systems are interesting because they are closer to UnifyWeaver's
logic-programming shape than a generic key/value store. They can represent facts
as immutable datoms, preserve historical database values, and expose a
Datalog-adjacent query interface. In this design they should be modeled as
database-backed predicate providers with additional temporal and snapshot
metadata, not as the default artifact store.

Candidate uses:

- Versioned predicate snapshots.
- Auditable derived facts.
- Time-travel or "as-of" query modes.
- Interop with existing Datalog-like data systems.

The planner should still reason about concrete access contracts. A temporal
database is useful only when it can satisfy the requested predicate access
pattern at an acceptable cost, or when its snapshot semantics are part of the
query.

### Memory-Mapped Binary Artifacts

Memory-mapped layouts are useful when the relation is large but stable and the
query pattern is predictable. A partitioned hash table can map hash prefixes to
files or blocks so a lookup touches only a small portion of the artifact. An
adjacency artifact can map a node to a contiguous span of targets for closure
operators.

Candidate uses:

- Exact point lookups.
- Probe-heavy joins.
- Seeded graph expansion.
- Read-mostly deployments where build time can be amortized.

### Sorted And Columnar Artifacts

Sorted runs and columnar files make sense for scans, range filters, grouping,
and projection-heavy workloads. They are less attractive for arbitrary joins
unless paired with secondary indexes.

Candidate uses:

- Range predicates.
- Prefix scans.
- Aggregates and grouped summaries.
- Late row materialization experiments.

### Remote Service Artifacts

Remote systems such as Redis can represent preprocessed predicate state even
when the artifact is not local. Redis-like providers are most useful for
low-latency key/value, set, sorted-set, or adjacency probes. The planner must
model network latency, batching behavior, failure modes, and consistency.

Candidate uses:

- Shared hot lookup sets across workers.
- Mutable or frequently refreshed support predicates.
- Distributed query runtimes that cannot assume local disk affinity.

### Distributed Batch Artifacts

Hadoop/HDFS, object-store tables, and similar systems are preprocessed data
surfaces for large batch workloads. They should not be treated like local
lookup indexes. Their useful contract is usually scan, partitioned scan,
coarse predicate pushdown, or precomputed grouped summaries.

Candidate uses:

- Large offline relations.
- Batch scans and aggregations.
- Precomputed summaries consumed by smaller online runtimes.
- Cross-target pipelines where the generated target reads a produced table
  rather than rebuilding state.

### Text And Vector Artifacts

Text indexes and vector/model artifacts are important because they are
preprocessed predicates, but not always exact relations. A Cohere/Wikipedia
embedding model or derived vector corpus can expose a semantic predicate like
`related(Document, Query, Score)` or `nearest(Entity, Neighbor, Distance)`.
That is different from an exact fact table and must be represented as scored or
approximate unless the declaration intentionally materializes fixed top-k facts.

Candidate uses:

- Semantic source predicates.
- Full-text lookup.
- Approximate nearest neighbor search.
- Hybrid symbolic/semantic query plans.

### Materialized Recursive Artifacts

Some recursive results can be precomputed when inputs are stable. Examples are
closure pairs, source-seeded reachability, target-seeded reachability, SCC
condensation, shortest path minima, or grouped path summaries. These artifacts
should declare their derivation and invalidation inputs because they are not
raw source relations.

Candidate uses:

- Repeated closure queries over stable graphs.
- Interactive workloads where build latency dominates.
- Graph-derived summaries that are cheaper to load than recompute.

## Planner Integration

The planner should compare artifact access against current paths:

- Current source scan or streaming ingestion.
- Replayable relation buffering.
- External in-memory materialization.
- Runtime-built indexes and operator-owned retained state.
- Preprocessed artifact access.

The first planner decision can be conservative:

1. If an artifact is declared, valid, exact, and supports the requested access
   pattern, allow it as a candidate.
2. If no compatible artifact exists, use the current runtime path.
3. If multiple candidates exist, use static cost hints first.
4. Add measured cost buckets later, matching the existing retention-planner
   direction.

Access requests should be explicit enough to avoid false matches:

- Full scan.
- Point lookup by one or more bound columns.
- Prefix lookup by leading columns.
- Range lookup.
- Source adjacency expansion.
- Target adjacency expansion.
- Pair reachability probe.
- Grouped summary lookup.
- Full-text query.
- Vector nearest-neighbor query.

## Cross-Target Strategy

UnifyWeaver already supports diverse target ecosystems, so artifact support
should be capability-driven:

- C# can prototype the planner boundary because the current query runtime has
  explicit relation-retention and materialization-planner hooks.
- Go may prefer bbolt, native maps serialized to files, or target-specific
  database/sql adapters depending on deployment.
- Rust may prefer memory-mapped binary layouts, sled/RocksDB-style embedded
  stores, or zero-copy row views.
- Python may prefer SQLite, DuckDB, NumPy/Arrow-style files, FAISS-like vector
  indexes, or model-native stores.
- SQL targets may treat a database table or index as the native artifact.
- Browser/WASM targets may need embedded, VFS, or compact binary artifacts
  rather than OS-level mmap.

The proposal does not require all targets to support all artifact kinds. It
requires targets to declare what they support and provide predictable fallback
behavior.

## Predicate Declaration Sketch

A source-level declaration should describe intent without hardcoding one target
implementation:

```prolog
:- preprocess edge/2 as exact_hash_index([
    key([1]),
    values([2]),
    access([point_lookup, adjacency]),
    partition(hash_prefix(8)),
    fallback(scan)
]).

:- preprocess article_embedding/3 as vector_index([
    key([1]),
    vector_column(2),
    score_column(3),
    metric(cosine),
    exactness(approximate),
    model("cohere-wikipedia-..."),
    fallback(disabled)
]).
```

The exact syntax is intentionally open. The important fields are predicate,
access pattern, exactness, invalidation inputs, and fallback.

## Shared Manifest Schema Notes

The current C# prototype and the current Clojure benchmark generator are using
different concrete manifest encodings, but they are already converging on the
same logical fields. A small shared schema should standardize the metadata
shape before attempting a common binary format:

- `predicate`
- `resolved_mode`
- `artifact_kind` or declaration `kind`
- `format_version`
- `physical_format`
- `exactness`
- `source_hash`
- `schema_hash`
- `target_capabilities`
- `access`
- `declaration`
  - `source`
  - `mode`
  - `kind`
  - `format`
  - `access_contracts`
  - `options`
- `files`

This should stay intentionally target-neutral:

- C# may keep JSON and binary sidecars.
- Clojure may keep EDN manifests with TSV or EDN sidecars.
- Haskell may eventually pair this with LMDB, mmap, or FactSource-backed
  providers.
- Elixir `external_source` can consume the same declaration metadata even if
  the first runtime adaptor is TSV, ETS, or SQLite.

The shared declaration layer should therefore stabilize before the physical
artifact formats do.

## Build And Invalidation Lifecycle

Artifact build should be a separate lifecycle from query execution:

1. Resolve predicate sources and preprocessing declarations.
2. Build artifacts into a target-specific directory.
3. Write manifests with source, schema, declaration, format, and tool hashes.
4. At runtime, validate manifests before opening artifacts.
5. Fall back or fail according to the declaration when validation fails.

For model-derived artifacts, manifests must also include model identifier,
embedding dimensions, distance metric, corpus version, quantization settings,
and whether results are exact, deterministic top-k, or approximate.

## Semantics And Safety

Exact artifacts must preserve row identity, row order only when declared, null
handling, term encoding, and equality behavior expected by the target runtime.

Approximate or scored artifacts must expose different predicate semantics. A
vector index should not silently replace an exact relation unless the build
materializes fixed facts and declares them exact. Queries that combine symbolic
and semantic predicates need to know where scores enter the plan.

Remote and distributed artifacts introduce deployment concerns:

- Authentication and secrets must not be embedded in generated code by default.
- Network failures need explicit fallback or failure policy.
- Mutable remote state can break reproducibility unless versioned or pinned.
- Batch stores may have high startup and scan latency that must be modeled.

## Initial C# Prototype

The recommended first implementation is intentionally narrow:

1. Add a manifest format for one exact binary relation artifact.
2. Build a two-column partitioned hash or adjacency artifact for benchmark
   graph predicates.
3. Add a C# provider that can open the artifact and satisfy point lookup,
   source adjacency, and scan fallback if practical.
4. Add planner selection behind an explicit `QueryExecutorOptions` override
   before enabling auto-selection.
5. Benchmark against current in-memory relation indexes, replayable sources,
   and compact seeded-cache rows.

This prototype should avoid vector search, Redis, Hadoop, and general database
integration until the exact local artifact boundary is proven.

Prototype status:

- The first runtime seam is a string-row binary relation artifact with a JSON
  manifest, `.uwbr` data file, and per-column offset index sidecars.
- `BinaryRelationArtifactBuilder` can build an artifact from a delimited
  relation source and records predicate, arity, row count, source length, and
  source SHA-256 metadata.
- `BinaryRelationArtifactProvider` exposes artifacts through the existing
  `IRelationProvider` and `IRetentionAwareRelationProvider` boundaries, so the
  current scan materialization planner can compare artifact replayable and
  external-materialized access without a new planner enum yet. It also exposes
  indexed single-column lookups through `IIndexedRelationProvider`, letting
  parameterized fact scans probe preprocessed offsets without first
  materializing the whole relation. Single-key joins can also use that provider
  seam when one side is a selective probe and the other side is an indexed
  artifact-backed scan.
- Binary relation index sidecars now include an appended key directory for
  small lookup sets. The reader binary-searches the fixed directory table on
  disk, seeks from key hash to index entry, and still falls back to the
  original sequential index scan for old artifacts or broad probe sets where
  sequential access is cheaper than many random seeks.
- The binary artifact builder also emits per-column covering bucket sidecars
  for broad indexed joins. The current runtime can merge those sorted sidecars
  directly and only deserialize rows for matching keys, which avoids
  materializing the full bucket stream for broad joins. When bucket sidecars
  are unavailable, it falls back to the smaller-side hash build path when
  relation cardinalities are known, and then to the older generic bucket path.
- `benchmark_scan_materialization.py` can now compare `preload`, `delimited`,
  `artifact`, and `artifact-prebuilt` source modes against the existing
  scan-family workloads, including `bound_scan` and `selective_join` modes for
  indexed parameter probes. The prebuilt mode keeps artifacts in a stable
  benchmark directory keyed by the current runtime source so runtime
  measurements can separate query execution from preprocessing cost without
  reusing stale artifact formats. The benchmark now resolves source-family
  choices through runtime-owned `RelationSourceMode` and
  `RelationSourceModePolicy` helpers instead of a benchmark-local string switch,
  which is the current seam for moving source selection out of Python-only
  configuration.
- Runtime-owned `ConfiguredDelimitedRelationProvider` now encapsulates
  delimited-source registration for `preload`, `delimited`, `artifact`, and
  `artifact-prebuilt` modes. The scan benchmark and generated C# pipeline
  programs use that helper, and pipeline programs can now take
  `UNIFYWEAVER_RELATION_SOURCE_MODE` plus `UNIFYWEAVER_RELATION_ARTIFACT_DIR`
  without re-implementing artifact registration in each generated template.
- Termux viability is now less speculative for LMDB than before. A local probe
  using the native `liblmdb` Termux package plus a Rust client confirmed a real
  open/write/read round-trip in this environment. The current prototype lives
  at `examples/lmdb_relation_artifact/` and builds a small exact two-column
  LMDB artifact with a sidecar `manifest.json`. That does not settle the final
  cross-target artifact format, but it does confirm that LMDB-backed exact
  artifact experiments are feasible here without deferring all work to desktop.

### N-ary delimited artifact direction

The current C# artifact implementation is deliberately binary. Wider
delimited relations can now use the same runtime-configured source registration
path, but `artifact` and `artifact-prebuilt` modes fall back to preloaded rows
for arities other than 2. That is the correct interim behavior: it preserves
answers and source-mode wiring without pretending the `.uwbr` sidecars support
general row shapes.

A general delimited artifact should be a new format revision, not an implicit
reinterpretation of the binary files. The row file should keep the existing
length-prefixed string-cell encoding, but record `arity > 0` as first-class
metadata and avoid names that imply binary-only relations. A possible naming
scheme is:

- `.uwrel` for row data
- `.uwrel.json` for the manifest
- `.colN.uwreli` for single-column index entries
- `.colsA_B.uwreli` for composite-key index entries
- `.colsA_B.uwrelb` for sorted covering bucket sidecars

The manifest needs to describe index coverage explicitly:

```json
{
  "format": "unifyweaver.delimited-relation",
  "version": 1,
  "predicate": "sale_item/3",
  "arity": 3,
  "row_count": 100000,
  "source_hash": "sha256:...",
  "row_encoding": "length_prefixed_utf8_cells",
  "indexes": [
    { "columns": [0], "kind": "offset_directory" },
    { "columns": [0, 1], "kind": "covering_bucket" }
  ]
}
```

The planner should request access by key shape rather than by artifact file:

- full scan: any exact artifact can replay rows
- single-column lookup: requires an index for `[column]`
- composite lookup: requires an index for the exact bound column list
- join bucket merge: requires sorted covering buckets for the join key columns
- projection-heavy plans: prefer covering buckets only when they include all
  columns needed downstream, otherwise account for row-file seeks

The first implementation should target one measured case: a TSV relation with
arity 3 and a composite lookup or join on columns `[0, 1]`. That exercises the
real reason to go beyond binary artifacts without committing to every possible
index combination. Builder defaults should remain conservative: emit the row
file plus indexes only for columns the generated query or benchmark actually
uses. A future declaration syntax can request additional indexes up front.

Fallback policy stays the same as the current configured provider policy:
binary artifacts use the proven binary path, N-ary artifacts are used only when
the manifest supports the requested access, and unsupported or stale artifacts
fall back to preload unless the caller explicitly asks to fail.

## Success Criteria

- A valid artifact can replace runtime-built state for a narrow exact predicate
  without changing query answers.
- Invalid or missing artifacts reliably fall back or fail according to policy.
- Benchmarks report build time, artifact size, open time, cold lookup latency,
  hot lookup latency, scan cost, and memory retained by the runtime.
- The planner can explain why it selected or rejected the artifact.
- The design remains compatible with existing `IRelationProvider`,
  `IRetentionAwareRelationProvider`, and `IReplayableRelationSource` behavior.

## Risks

- Artifact sprawl: too many target-specific formats can become unmaintainable.
- Semantic confusion: approximate/model artifacts may be mistaken for exact
  relations.
- Stale artifacts: missing invalidation metadata can produce wrong answers.
- Deployment complexity: remote stores and distributed filesystems add
  operational requirements beyond local generated code.
- Premature abstraction: a broad interface before one measured prototype may
  hide the real hot-path costs.

## Open Questions

- Should preprocessing declarations live in source files, build manifests, or
  target options first?
- How much of the artifact manifest should be target-neutral versus
  target-specific?
- Should exact binary artifacts use a common row encoding across targets or let
  each target define its own encoding?
- Which fallback policies are needed: always fallback, fail if missing, rebuild
  if stale, or warn and continue?
- How should scored semantic predicates compose with symbolic query planning?
- Can remote providers participate in measured planning without making every
  query pay probe overhead?

## Related Documents

- `docs/proposals/QUERY_ENGINE_MATERIALIZATION_SPEC.md`
- `docs/proposals/ROW_CONTAINER_ABSTRACTION_SURVEY.md`
- `docs/design/INPUT_SOURCE_DESIGN.md`
- `docs/proposals/semantic_source_integration.md`
- `docs/proposals/partitioning_transpilation_roadmap.md`
