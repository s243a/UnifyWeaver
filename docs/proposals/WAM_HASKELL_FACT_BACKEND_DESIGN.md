# WAM Haskell Fact Backend Design

## Summary

The Haskell WAM target needs pluggable fact backends that go beyond the
current strict IntMap and inline literal list options. At 10k+ scale,
neither compiled WAM instructions nor inline Haskell literals are viable
(GHC compilation time is prohibitive for 100k-line arrays). The target
needs backends that can serve indexed lookups without full in-memory
materialization while remaining compatible with the `parMap rdeepseq`
parallelism model.

This document designs a unified `FactBackend` interface for the Haskell
WAM target, covering memory-mapped binary files, embedded databases,
and future backend types. It builds on the `FactSource` abstraction
from Phase F4 and draws from the C# query runtime's
`IRelationProvider` / `IRetentionAwareRelationProvider` hierarchy and
the cross-target `PREPROCESSED_PREDICATE_ARTIFACTS.md` proposal.

## Current state

The Haskell WAM target has four fact access paths today:

| Path | Introduced | Scale | Parallelism | Access |
|------|-----------|-------|-------------|--------|
| Compiled WAM instructions | Original | Small (<100) | Safe (immutable code array) | O(1) SwitchOnConstant dispatch |
| Strict IntMap (FFI) | Phase E | Any | Safe (immutable) | O(log n) IntMap lookup |
| inline_data literals (F3) | Phase F3 | Medium (100-1000) | Safe (immutable list) | O(n) scan or O(log n) with index |
| FactSource (F4) | Phase F4 | Any (abstract) | Depends on backend | Via fsScan / fsLookupArg1 |

The FactSource interface (F4) provides the right abstraction boundary
but currently only has two concrete implementations:

- `tsvFactSource` — lazy IO TSV reader. Not useful for repeated lookups
  (caches everything on first access anyway).
- `intMapFactSource` — wraps existing strict IntMap. No improvement over
  the direct IntMap path.

Neither solves the 10k+ scale problem where we need indexed lookups
without full materialization.

## Design principle

**Separate the access contract from the storage engine.**

The WAM interpreter asks for facts through a narrow interface:
- Scan all rows
- Lookup by first argument (indexed)
- Close / release resources

The backend decides how to serve those requests. The planner (compile
time + runtime) decides which backend to use based on:
- Predicate scale (clause count, estimated row count)
- Access pattern (scan-only, repeated indexed lookups, graph expansion)
- Platform capabilities (mmap, threading, available databases)
- User declarations (`fact_layout/2`, `environment/1`)

This mirrors the C# query runtime's approach where `IRelationProvider`
defines the contract and concrete providers (`BinaryRelationArtifactProvider`,
`InMemoryRelationProvider`, `ConfiguredDelimitedRelationProvider`) serve it.

## The FactBackend typeclass

The F4 `FactSource` record is adequate for the current two-column case
but becomes awkward for wider arities and richer access patterns. We
extend it to a `FactBackend` typeclass that backends implement:

```haskell
-- | Access pattern descriptors for the planner.
data AccessPattern
  = Scan                     -- full sequential scan
  | PointLookup !Int         -- lookup by column N (0-indexed)
  | PrefixLookup ![Int]      -- lookup by leading columns
  | AdjacencyExpand !Int     -- graph expansion from column N
  deriving (Show, Eq)

-- | Backend capabilities reported to the planner.
data BackendCaps = BackendCaps
  { bcSupports     :: [AccessPattern]
  , bcParSafe      :: !Bool           -- safe under parMap?
  , bcEstimatedRows :: !(Maybe Int)   -- hint for planner
  } deriving (Show)

-- | Unified fact backend interface.
-- All methods return IO because some backends (mmap, database) need it.
-- The WAM interpreter bridges to pure code via unsafePerformIO in
-- streamFacts, which is safe because backends are read-only after the
-- force barrier.
class FactBackend a where
  fbScan       :: a -> IO [(Int, Int)]
  fbLookup     :: a -> Int -> IO [(Int, Int)]  -- lookup by arg1
  fbCaps       :: a -> BackendCaps
  fbClose      :: a -> IO ()
```

The existing `FactSource` record becomes one implementation strategy
(existential wrapper for the typeclass). The `wcFactSources` field in
WamContext continues to work unchanged — backends are wrapped in a
`FactSource` at registration time.

## Backend: MmapFactSource

Memory-mapped binary files for indexed point lookups. The ideal backend
for desktop/server where mmap is available.

### Why mmap

- **No resident memory**: the OS page cache handles loading. Only pages
  touched by queries are faulted in. A 100MB relation occupies zero
  application heap.
- **Parallelism-safe**: each thread reads by dereferencing independent
  virtual addresses. No file handle contention, no locks. This is why
  the philosophy doc identified mmap as the ideal parallel data source.
- **Startup cost**: near-zero. `mmap()` returns a pointer immediately;
  pages load on demand.
- **GHC integration**: `bytestring` provides `ByteString` backed by
  `ForeignPtr` which can point into an mmap'd region. The `mmap`
  package (or `System.Posix.IO.ByteString`) handles the mapping.

### Binary format

Consume the C# target's `.uwbr` (UnifyWeaver Binary Relation) format,
which already defines:
- Row-based data with offset indexing
- Partitioned hash index for point lookups
- Per-column covering bucket sidecars
- JSON manifest with predicate, arity, source hash, capabilities

For the Haskell consumer, the minimum viable format is:

```
[Header: 16 bytes]
  magic:    4 bytes  "UWBR"
  version:  4 bytes  uint32 LE
  nrows:    4 bytes  uint32 LE
  ncols:    4 bytes  uint32 LE  (2 for current use)

[Key directory: nkeys * 12 bytes]
  key:      4 bytes  int32 LE (interned atom ID)
  offset:   4 bytes  uint32 LE (byte offset into data section)
  count:    4 bytes  uint32 LE (number of rows for this key)

[Data section: nrows * ncols * 4 bytes]
  Each row is ncols int32 LE values (interned atom IDs)
```

Point lookup: binary search the key directory (O(log n)), then read
`count` rows starting at `offset`. All via pointer arithmetic on the
mmap'd region — no allocation, no IO syscalls.

### Haskell implementation sketch

```haskell
data MmapFactSource = MmapFactSource
  { mfsPtr     :: !(ForeignPtr Word8)  -- mmap'd region
  , mfsLen     :: !Int                 -- total bytes
  , mfsNRows   :: !Int
  , mfsNCols   :: !Int
  , mfsKeyDir  :: !(Ptr Word8)         -- start of key directory
  , mfsNKeys   :: !Int
  , mfsData    :: !(Ptr Word8)         -- start of data section
  }

instance FactBackend MmapFactSource where
  fbScan mfs = return $ readAllRows mfs
  fbLookup mfs key = return $ binarySearchAndRead mfs key
  fbCaps _ = BackendCaps
    { bcSupports = [Scan, PointLookup 0]
    , bcParSafe = True
    , bcEstimatedRows = Nothing  -- read from header
    }
  fbClose _ = return ()  -- GC handles munmap via ForeignPtr
```

### Platform constraints

- **Linux/macOS/Windows**: full mmap support. Default on desktop.
- **Termux/Android**: filesystem-dependent. May work on internal
  storage but fail on SD card (FAT32). Environment predicate
  `environment(mmap(false))` disables this backend.
- **WASM**: no mmap. Must use database or in-memory backend.

## Backend: SqliteFactSource

SQLite for platforms where mmap is unavailable or restricted.

### Why SQLite

- **Universal availability**: works on Termux, Android, iOS, desktop,
  server. No filesystem restrictions.
- **Indexed lookups**: CREATE INDEX on arg1 gives O(log n) lookups
  without loading everything into memory.
- **WAL mode**: concurrent readers with single writer. Adequate for
  read-only fact access during queries.
- **Mature Haskell bindings**: `sqlite-simple` or `direct-sqlite`.

### Embedding vs system library

SQLite can be statically embedded into the Haskell binary or linked
against the system library. The tradeoff:

- **Embedded** (`direct-sqlite` with bundled amalgamation): no runtime
  dependency, works on any platform, but adds ~1MB to the binary and
  memory footprint. Good for Termux/Android where the system SQLite
  may be an older version.
- **System library** (`sqlite-simple` linking to `-lsqlite3`): smaller
  binary, shared memory with other processes using SQLite, but requires
  the library to be installed. Good for desktop/server.

The choice should be gated by a build option (e.g., `embedded_sqlite(true)`
in the cabal generation) so the user can decide based on their deployment.

### Parallelism strategy

SQLite in default journal mode has a single-writer lock that blocks
readers. Under `parMap rdeepseq`:

**Option A: Read-once-then-split** (recommended for first implementation)
Load needed facts into an immutable IntMap before the parallel section.
The SQLite connection is used only during context construction, not
during the parallel seed loop.

**Option B: Per-worker connections**
Each spark opens its own read-only SQLite connection. Works with WAL
mode but multiplies connection overhead. Better for very large datasets
where full materialization is too expensive.

**Option C: Single pre-parallel query**
Query all facts once, partition by seed assignment, pass each partition
to its spark. The database is accessed once; parallelism operates on
the partitioned data.

### Schema

```sql
CREATE TABLE facts_2 (
  predicate TEXT NOT NULL,
  arg1 INTEGER NOT NULL,    -- interned atom ID
  arg2 INTEGER NOT NULL,    -- interned atom ID
  PRIMARY KEY (predicate, arg1, arg2)
);
CREATE INDEX idx_facts_2_arg1 ON facts_2(predicate, arg1);
```

For wider arities, the schema generalizes to `arg1..argN` columns.

### Haskell implementation sketch

```haskell
data SqliteFactSource = SqliteFactSource
  { sfsConn    :: !Connection          -- sqlite-simple connection
  , sfsPred    :: !String              -- predicate name
  }

instance FactBackend SqliteFactSource where
  fbScan sfs = query_ (sfsConn sfs)
    "SELECT arg1, arg2 FROM facts_2 WHERE predicate = ?"
  fbLookup sfs key = query (sfsConn sfs)
    "SELECT arg1, arg2 FROM facts_2 WHERE predicate = ? AND arg1 = ?"
    (sfsPred sfs, key)
  fbCaps _ = BackendCaps
    { bcSupports = [Scan, PointLookup 0]
    , bcParSafe = False  -- not safe under parMap without Option A/B
    , bcEstimatedRows = Nothing
    }
  fbClose sfs = close (sfsConn sfs)
```

## Backend: Immutable/append-only databases (Datomic-style)

Datomic (Clojure ecosystem) is interesting because its data model
aligns naturally with logic programming:

- **Facts are immutable datoms**: `[entity attribute value tx]`. Once
  asserted, never modified — only retracted by new assertions.
- **Time-travel**: every query runs against a stable database value
  (`db`). No locks needed because the value is immutable.
- **Datalog-adjacent**: Datomic's query language IS Datalog. The
  mapping to Prolog fact access is natural.
- **Parallelism-safe by design**: immutable snapshots can be shared
  across threads with zero coordination.

### Relevance to Haskell

Haskell's immutability model maps perfectly to Datomic's:
- A `db` value is like a Haskell value — referentially transparent
- Multiple threads can hold the same `db` with no locking
- Queries are pure functions over an immutable snapshot

### Candidate implementations for Haskell

Datomic itself is JVM-only. Haskell equivalents:

| System | Status | Notes |
|--------|--------|-------|
| **Datahike** | Clojure (JVM/JS) | Open-source Datomic-compatible, Datalog query engine |
| **Datascript** | ClojureScript | In-memory Datalog database, portable |
| **Custom Haskell** | Not yet | Build a simple datom store with HAMT-backed indexes |
| **XTDB v2** | JVM | Bitemporal, SQL + Datalog |

For a first implementation, the pragmatic path is a **datom-shaped
IntMap store**: facts stored as `(entity, attribute, value)` triples
with IntMap indexes on each component. This gives Datomic-like
semantics (immutable, snapshot-queryable, parallelism-safe) without
a JVM dependency.

```haskell
data DatomStore = DatomStore
  { dsEAV :: !(IM.IntMap (IM.IntMap [Int]))  -- entity -> attr -> [values]
  , dsAEV :: !(IM.IntMap (IM.IntMap [Int]))  -- attr -> entity -> [values]
  , dsAVE :: !(IM.IntMap (IM.IntMap [Int]))  -- attr -> value -> [entities]
  }
```

For the fact access use case (2-arg predicates like `category_parent/2`),
the datom model maps to: entity=arg1, attribute=predicate, value=arg2.
The AVE index gives O(log n) lookup by arg1.

This is future work — the immediate priority is MmapFactSource and
SqliteFactSource.

## Why not LiteDB or bbolt?

Other UnifyWeaver targets use LiteDB (C#) and bbolt (Go) as embedded
databases. These are not viable for the Haskell target because they are
tightly coupled to their host language runtimes — LiteDB is a .NET
library with no C API, and bbolt is a Go library with no C API. There
are no Haskell bindings for either, and FFI bridging to .NET or Go
runtimes is impractical.

The Haskell equivalents that fill the same niches:

| C#/Go choice | Haskell equivalent | Why |
|---|---|---|
| LiteDB (C#, NoSQL) | **LMDB** (C lib, key-value, mmap-backed) | C FFI, zero-copy reads, parallelism-safe |
| bbolt (Go, key-value) | **LMDB** or **RocksDB** (C libs) | Both have Haskell bindings via C FFI |

LMDB is particularly interesting as a middle ground between raw mmap
and SQLite: it is mmap-backed internally, supports concurrent readers
with zero locking (MVCC), and has mature Haskell bindings
(`lmdb-simple`). These are candidates for Phase B3 or later.

## Other backends to consider

| Backend | Ecosystem | Use case | Priority |
|---------|-----------|----------|----------|
| **LMDB** | C via FFI (`lmdb-simple`) | Key-value, mmap-backed, zero-lock reads | Medium — LiteDB/bbolt equivalent for Haskell |
| **RocksDB** | C++ via FFI | Sorted key-value, range scans | Medium — when range predicates matter |
| **DuckDB** | C via FFI | Analytical queries, columnar | Low — overkill for point lookups |
| **Redis** | Network | Shared hot lookups across workers | Low — network latency |
| **Arrow/Parquet** | Cross-platform | Columnar batch scans | Low — better for Python/Rust |

## Planner integration

The compile-time planner (Prolog codegen) already classifies predicates
and selects layouts. The backend selection extends this:

```prolog
% User declares backend preference
:- fact_layout(category_parent/2, external_source(mmap("data/cp.uwbr"))).
:- fact_layout(category_parent/2, external_source(sqlite("data/facts.db"))).

% Environment constraints gate backend availability
:- environment(mmap(false)).    % Termux: disable mmap
:- environment(sqlite(true)).   % Termux: SQLite available
```

The runtime planner checks:
1. Is the declared backend available on this platform?
2. Does it support the required access pattern?
3. Fall back to the next option (SQLite → IntMap → compiled).

## Interaction with existing paths

The `FactBackend` typeclass does NOT replace:
- **FFI kernel path** (`wcFfiFacts`): native kernels need strict IntMaps
  for maximum throughput. The backend abstraction serves the WAM
  interpreter path.
- **Compiled WAM path**: small predicates (<100 clauses) continue to
  use SwitchOnConstant dispatch. No overhead, no indirection.
- **inline_data path** (F3): medium predicates (100-1000) use Haskell
  literals. No IO, no backend indirection.

The backend path activates for:
- Large predicates (>1000 clauses) where GHC compilation and in-memory
  lists are impractical
- External data sources declared with `fact_layout(..., external_source(...))`
- Runtime-loaded facts that don't benefit from compile-time indexing

## Implementation phases

### Phase ordering rationale

The original plan had raw MmapFactSource (custom binary format) first,
but we revised the ordering to **LMDB → SQLite → raw mmap** for
practical reasons:

1. **LMDB is a faster design path.** It handles storage layout, index
   construction, and crash safety internally. We write a thin wrapper
   instead of designing a binary format from scratch. The `lmdb-simple`
   Haskell package wraps the mature C library.

2. **LMDB delivers equivalent mmap performance.** LMDB uses mmap
   internally — reads are zero-copy pointer dereferences into the OS
   page cache, exactly like a raw mmap'd file. Concurrent readers use
   MVCC with no locks. The performance profile is the same as raw mmap
   but with less implementation risk.

3. **The C# `.uwbr` format is not yet stable.** The preprocessed
   artifacts proposal (PREPROCESSED_PREDICATE_ARTIFACTS.md) is still
   draft v0.1. Implementing a consumer for an unstable format risks
   rework. By the time we need cross-target artifact sharing, the
   format will have stabilized and we can add a raw mmap consumer then.

4. **SQLite provides the universal fallback.** It works everywhere
   (desktop, Termux, Android, CI) and has mature tooling. It fills the
   gap on platforms where LMDB's C dependency is inconvenient, though
   LMDB itself is very portable.

### Phase B1: LmdbFactSource (primary backend)

- Implement `LmdbFactSource` using `lmdb-simple` (Haskell bindings to
  the C library)
- Store interned `(arg1, arg2)` pairs keyed by predicate name + arg1
- Use LMDB's read-only transactions for parallelism safety (MVCC,
  zero-lock concurrent readers)
- Add fact ingestion tool (TSV → LMDB database)
- Wire into `wcFactSources` registration
- Benchmark against IntMap at 10k and 100k
- Add `lmdb` to cabal dependencies (conditional on `use_lmdb(true)`)

LMDB-specific advantages for this use case:
- **mmap-backed**: reads are pointer dereferences, no allocation
- **MVCC readers**: each parMap spark can hold its own read transaction
  with no contention — better than the read-once-then-split strategy
  needed for SQLite
- **Sorted keys**: range scans and prefix lookups come for free
- **Crash-safe**: ACID transactions protect against partial writes
  during ingestion
- **Small footprint**: the C library is ~40KB, no external daemon

### Phase B2: SqliteFactSource (universal fallback)

- Implement `SqliteFactSource` using `sqlite-simple` or `direct-sqlite`
- Add read-once-then-split strategy for parallelism (load into IntMap
  before parMap section)
- Add fact ingestion tool (TSV → SQLite database)
- Wire into `wcFactSources` with environment-gated selection
- Benchmark on Termux vs desktop
- Embedding option: bundled SQLite amalgamation (~1MB binary cost) or
  link to system `-lsqlite3`

### Phase B3: MmapFactSource (cross-target artifact consumer)

- Consume the C# `.uwbr` binary relation format once it stabilizes
- Implement using `bytestring` + `mmap` (or `System.Posix.IO`)
- Binary search over key directory for point lookups
- Primarily for cross-target artifact sharing (C# builds artifacts,
  Haskell consumes them)
- Only needed when the artifact format is stable and the use case
  (shared preprocessed data across targets) is proven

### Phase B4: DatomStore (future, exploratory)

- Implement triple-indexed IntMap store (EAV/AEV/AVE indexes)
- Expose as FactBackend with immutable snapshot semantics
- Explore integration with Datahike or custom Datalog engine
- Consider LMDB as the backing store for persistent datom storage

### Phase B5: Planner-driven backend selection

- Extend the compile-time planner to emit backend selection code
- Add runtime capability detection
- Environment predicates for cross-compilation

## Dependencies

- `WAM_HASKELL_FACT_ACCESS_SPEC.md` — FactSource interface (F4)
- `WAM_HASKELL_FACT_ACCESS_PHILOSOPHY.md` — laziness and parallelism
- `PREPROCESSED_PREDICATE_ARTIFACTS.md` — artifact format (Phase B3)
- C# `BinaryRelationArtifactProvider` — .uwbr format reference (Phase B3)

## What this does not change

- FFI kernel path (stays on strict IntMap)
- Compiled WAM path for small predicates
- inline_data path for medium predicates
- The `parMap rdeepseq` parallelism model
- The `run`/`step`/`backtrack` core loop
