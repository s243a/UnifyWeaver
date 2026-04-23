# WAM Haskell Scaling Insights

## Summary

The Haskell WAM target scales competitively from small (300 facts)
to large (10k+ facts) workloads, with a clear path to 1M+ via LMDB.
This document ties together the benchmarking results from atom
interning, FFI kernels, and fact backend work, and explains why
Haskell's language properties make this performance profile difficult
to achieve in other languages.

## The performance story so far

### Baseline → FFI → Atom Interning → LMDB

| Phase | Scale | query_ms | Speedup | What changed |
|-------|-------|----------|---------|-------------|
| WAM interpreter only | 10k | ~15,000 | 1.0x | Pure WAM step/backtrack loop |
| + FFI kernels | 10k | 532 | 28x | Native DFS, skip WAM for hot predicates |
| + atom interning | 10k | 431 | 35x | Int comparison instead of String hashing |
| + LMDB raw (warm) | 10k | 684 | 22x | On-demand mmap, no eager materialization |

The FFI kernel optimization delivered the largest single improvement
(28x) because it replaces the WAM interpreter's per-instruction
dispatch loop with a native Haskell DFS function that GHC can optimize
aggressively. Atom interning added another 19% by eliminating string
comparison from the hot loop.

LMDB is slightly slower than IntMap at 10k (1.29x) but trades
in-memory materialization for on-demand mmap reads. At scales where
the IntMap doesn't fit in available RAM, LMDB wins decisively.

### In-memory (IntMap) vs database-backed (LMDB)

The two paths serve different scale regimes:

| Property | IntMap | LMDB raw |
|----------|--------|----------|
| Lookup speed | O(log n) in-memory | O(log n) B+ tree on mmap'd pages |
| Memory | Full dataset in GHC heap | OS page cache, zero GHC heap |
| Startup cost | TSV parse + intern + fromListWith | One-time ingestion, then instant open |
| GC pressure | Proportional to dataset size | Zero (data not on GHC heap) |
| Parallelism | Immutable, zero-copy sharing | MVCC read snapshot, no locks |
| Scale limit | Available RAM (~2GB on this WSL) | Disk size (practically unlimited) |

At 10k (25k edges, ~600KB), IntMap is 29% faster. At 1M+ (25M edges,
~600MB), IntMap would consume most available RAM and GC pauses would
dominate. LMDB stays flat — the OS page cache evicts cold pages
transparently.

### Projected crossover point

With 2GB available RAM on this WSL instance:

- **10k**: IntMap wins (600KB, trivial)
- **100k**: IntMap still wins (~6MB, comfortable)
- **1M**: Toss-up (~60MB, GC pressure starts)
- **5-10M**: LMDB wins (~300-600MB, IntMap approaches RAM limit)
- **Full Wikipedia (30M edges)**: LMDB wins decisively (~720MB)

The `EdgeLookup = Int -> [Int]` abstraction makes the transition
seamless — the kernel code is identical for both backends.

## Why Haskell is uniquely suited for this

### The in-memory story: persistent data structures

Most languages struggle with the WAM's core challenge: cheap state
snapshots for backtracking. Every `TryMeElse` choice point needs a
snapshot of the current state so `backtrack` can restore it.

**Haskell's IntMap** (a hash array mapped trie, HAMT) provides O(1)
snapshots via structural sharing. When a choice point is created, the
saved registers, bindings, and stack share subtrees with the current
state. Only the modified paths are copied.

This is the property that made the Haskell WAM viable where the Rust
WAM failed (343x slower than SWI-Prolog). The Rust implementation
used `Vec<Value>` for registers and bindings, requiring O(n) clones
on every choice point. The Haskell implementation uses `IntMap Value`,
which shares unmodified subtrees across snapshots.

**Impact on scaling**: A million-entry IntMap with 1000 concurrent
choice points doesn't use 1000x memory — the shared subtrees mean
total memory is proportional to the number of *modified* entries
across all snapshots, not the total entries times the number of
snapshots. This is fundamentally different from languages with
mutable data structures.

### The parallelism story: immutability for free

`parMap rdeepseq` parallelizes across 888 seed categories with zero
synchronization code. Each spark gets an independent `WamState` but
shares the immutable `WamContext` (including the fact IntMap or LMDB
EdgeLookup). No locks, no concurrent hash maps, no per-thread copies.

Compare this to other languages:

| Language | Parallel fact access approach | Overhead |
|----------|------|---------|
| **Haskell** | Share immutable IntMap/EdgeLookup across sparks | Zero |
| **Go** | `sync.Map` or per-goroutine copies | Lock contention or copy cost |
| **Rust** | `Arc<HashMap>` with careful lifetime management | Reference counting + borrow checker constraints |
| **Python** | `multiprocessing` with pickle serialization | Serialization cost per process |
| **Java** | `ConcurrentHashMap` or thread-local copies | CAS overhead or copy cost |
| **C#** | `ConcurrentDictionary` or `ImmutableDictionary` | CAS overhead or allocation pressure |

Haskell's approach is not just simpler — it's fundamentally cheaper
because immutability is the default, not an opt-in pattern.

### The database story: unsafePerformIO as an abstraction boundary

The LMDB integration uses `unsafePerformIO` to bridge from IO (mmap
reads) to the pure kernel code. This is safe because:
1. The LMDB database is read-only during queries
2. The long-lived read transaction provides a consistent snapshot
3. LMDB's MVCC means concurrent reads don't interfere

This pattern — hiding IO behind a pure interface when the side effects
are observationally pure — is a Haskell idiom that most other languages
can't express cleanly:

- **Go/Rust**: Would need explicit error handling at every call site
- **Python**: No mechanism to guarantee the IO is safe to hide
- **Java**: Could use `Supplier<T>` caching but loses type-level purity

The result is that `lmdbRawEdgeLookup txn dbi` has the same type as
`intMapEdgeLookup intmap` — `Int -> [Int]`. The kernel DFS code
literally cannot tell the difference. This is what allows us to swap
backends without changing any query code.

### The GC story: why LMDB's zero-heap-pressure matters at scale

At 10k, IntMap's GC cost is negligible. At 1M+, the story changes:

- GHC's generational GC promotes long-lived data (the fact IntMap) to
  the old generation. But major GC still needs to traverse it.
- Each major GC pause is proportional to the live heap size. A 600MB
  IntMap means 600MB of live data that the GC must walk.
- With LMDB, the fact data lives in the OS page cache, completely
  invisible to GHC's GC. The GHC heap contains only the query
  computation state (registers, bindings, choice points) — typically
  a few MB regardless of dataset size.

This is why the projected crossover at 5-10M edges is real: GC pause
time grows linearly with IntMap size, while LMDB's GC impact is
constant.

## The CBOR vs raw binary lesson

The initial LMDB implementation used `lmdb-simple` with CBOR
serialization (Codec.Serialise). At 10k, this was 9.6x slower than
IntMap — dominated by per-lookup CBOR deserialization.

Switching to raw `lmdb` with packed `Int32` arrays eliminated the
serialization overhead entirely. The read path becomes:
1. `mdb_get'` returns a `(Ptr Word8, CSize)` pointing into the mmap
2. `peekElemOff` reads `Int32` values directly from the pointer
3. No allocation, no deserialization, no copies

Combined with a long-lived read transaction (eliminating per-lookup
`mdb_txn_begin`/`mdb_txn_abort`), this brought LMDB to within 29%
of IntMap with warm OS page cache.

| Backend | query_ms (10k) | Ratio |
|---------|---------------|-------|
| IntMap (in-memory) | 532 | 1.0x |
| LMDB-simple (CBOR) | 5099 | 9.6x |
| LMDB raw (cold pages) | 1018 | 1.9x |
| LMDB raw (warm pages) | 684 | 1.29x |

The lesson: **serialization format matters enormously for database-backed
lookups in hot loops**. CBOR's flexibility (arbitrary Haskell types) is
not worth the cost when the data is just arrays of integers.

## Design principles validated

1. **Engine decides retention, not the loader** (from
   WAM_HASKELL_FACT_ACCESS_PHILOSOPHY.md): The `EdgeLookup` abstraction
   lets the engine choose IntMap or LMDB based on scale, without the
   kernel code knowing or caring.

2. **Lazy by default, strict where parallelism demands** (from the
   same doc): IntMap fields in WamContext use strict `!` annotations;
   LMDB data stays lazy (loaded on demand via mmap). The force barrier
   before `parMap` ensures consistency.

3. **Facts as data, not code** (from the fact access spec): The F1-F5
   infrastructure separates fact representation from query execution.
   The same `category_ancestor` kernel works with compiled WAM facts,
   inline literals, TSV-loaded IntMaps, or LMDB-backed lookups.

4. **Purity enables materialization freedom** (from the philosophy
   doc): Because the kernels are pure functions over an immutable
   snapshot, the materialization strategy (eager IntMap vs on-demand
   LMDB) can change without affecting correctness. This is the same
   principle that enables goal reordering in the purity certificate
   system.

## What this means for the project

The Haskell WAM target is not just a transpilation experiment — it
demonstrates that a logic programming engine can achieve competitive
performance at scale by leveraging the host language's strengths
rather than fighting them:

- **Persistent data structures** for cheap backtracking
- **Immutability** for zero-cost parallelism
- **Lazy evaluation** for demand-driven materialization
- **Type-safe IO abstraction** for database-backed fact access
- **GHC's optimizer** for aggressive inlining and specialization

The combination of IntMap (fast, in-memory) and LMDB (scalable,
on-demand) gives the target a performance envelope from hundreds
of facts to potentially millions, with a seamless transition
controlled by a single option (`use_lmdb(true)`).

## Related documents

- `WAM_HASKELL_TRANSPILATION_PHILOSOPHY.md` — why Haskell over Rust
- `WAM_HASKELL_PERF_PHILOSOPHY.md` — optimization principles
- `WAM_HASKELL_FACT_ACCESS_PHILOSOPHY.md` — fact materialization design
- `WAM_HASKELL_FACT_BACKEND_DESIGN.md` — LMDB/SQLite/mmap backend design
- `WAM_RUST_STATE_MANAGEMENT_RETROSPECTIVE.md` — Rust WAM failure analysis
- `WAM_PERF_OPTIMIZATION_LOG.md` — benchmark history
