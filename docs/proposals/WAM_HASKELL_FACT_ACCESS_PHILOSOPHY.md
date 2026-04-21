# WAM Haskell Fact Access Philosophy

## Summary

The Haskell WAM target needs a principled approach to how facts enter
and are accessed by the query engine. The current design eagerly loads
all facts from TSV into strict IntMaps at startup, then either compiles
them into WAM instructions (SwitchOnConstant dispatch) or passes them
as interned IntMaps to FFI kernels. This works well at 10k scale but
sacrifices Haskell's laziness — one of the language's key strengths —
and will not scale to larger data without rethinking the access layer.

This document establishes the principles that should guide fact access
design for the Haskell WAM target.

## Core position

**The engine should decide what to retain, not the loader.**

The current flow is: TSV parser → strict list → strict IntMap → either
WAM instructions or FFI kernel input. Every fact is eagerly materialized
before any query runs. The parser and the engine have no conversation
about what the query actually needs.

The preferred flow is: fact source → lazy stream → engine retains what
the query plan requires. The fact source may be a TSV file, a
memory-mapped binary artifact, a database cursor, or a compiled literal
table. The engine decides whether to build an index, scan linearly, or
stream on demand.

## When facts should be compiled

Compiling facts into WAM instructions (the `compiled` layout) has real
advantages:

- **Compile-time indexing.** SwitchOnConstant maps are built once by the
  Prolog codegen. At runtime, first-argument dispatch is an O(1) IntMap
  lookup with zero startup cost. No runtime index construction needed.
- **GHC optimization.** Known constants in WAM instruction arrays let GHC
  specialize and inline. Strict IntMap lookups on known keys can be
  optimized at compile time.
- **No runtime dispatch overhead.** The WAM interpreter treats fact
  instructions identically to rule instructions — no special FactSource
  protocol, no typeclass dispatch, no indirection.

Compiled facts are the right choice when:

- The fact count is small (< 100-200 clauses)
- The facts are known at code generation time (not loaded from external data)
- The query pattern benefits from first-argument indexing
- Startup latency matters more than memory footprint

## When facts should be data

Facts should be treated as data (not compiled code) when:

- **Scale.** Thousands of facts generate thousands of WAM instructions,
  bloating the instruction array and slowing code generation. The Elixir
  target hit this at 6000 clauses; Haskell's GHC handles it better but
  the principle holds.
- **External data.** Facts loaded from TSV, databases, or network sources
  are not known at code generation time. Compiling them into WAM
  instructions at startup (`buildFact2Code`) is an eager-materialization
  tax.
- **Laziness.** Haskell can naturally stream facts through the query
  pipeline. Eager `IM.fromListWith (++)` over all TSV rows throws away
  this advantage. A lazy fact source would only materialize rows the
  query actually touches.
- **Memory.** At 100k+ facts, holding everything in strict IntMaps may
  not be viable. Memory-mapped files or database cursors can serve the
  same lookups without resident memory for every row.

## How Haskell's laziness fits

The C# query runtime explicitly manages retention modes (Streaming,
Replayable, ExternalMaterialized) because C# is eager by default. The
engine must explicitly choose to buffer or discard.

Haskell inverts this: laziness is the default. A `[Row]` is not
materialized until forced. An `IM.IntMap [Int]` built from a lazy list
only constructs entries as they are demanded. This means:

- **Streaming is free.** A lazy TSV reader yields rows on demand. If the
  query only touches 10% of facts, 90% are never parsed.
- **Retention is explicit.** A `BangPattern` or `deepseq` forces
  materialization. The engine chooses when to force — not the parser.
- **Parallelism requires strictness.** Our `parMap rdeepseq` seed loop
  forces everything in the WamContext to be fully evaluated before
  forking. This is the right tradeoff for the parallel path, but it
  means lazy fact sources must be forced before the parallel section.

The design principle: **lazy by default, strict where parallelism or
repeated access demands it.**

## The FactSource interface

Following the Elixir fact-shape spec (WAM_FACT_SHAPE_SPEC.md), the
Haskell target should define a FactSource abstraction:

```haskell
data FactSource = FactSource
  { fsScan        :: IO [Row]            -- full scan (lazy)
  , fsLookupArg1  :: Int -> IO [Row]     -- indexed by first arg (interned)
  , fsClose       :: IO ()               -- release resources
  }
```

This mirrors the Elixir `FactSource` behaviour (`open/3`,
`stream_all/2`, `lookup_by_arg1/3`, `close/2`) but uses Haskell idioms
(lazy IO, interned Int keys).

Concrete implementations:

- **TsvFactSource**: Lazy TSV reader. First access parses the file;
  subsequent accesses use the cached parse. Index built on demand when
  `fsLookupArg1` is first called.
- **IntMapFactSource**: The current strict IntMap, wrapped in the
  interface. Used when facts are loaded eagerly (the `buildFact2Code`
  path or explicit configuration).
- **MmapFactSource** (future): Memory-mapped binary artifact (`.uwbr`
  or similar). Point lookups via mmap'd hash table. No resident memory
  for the full relation.

The WamContext gains a `wcFactSources :: Map String FactSource` field
alongside the existing `wcFfiFacts`. The FFI kernel path continues to
use `wcFfiFacts` (interned IntMap) for maximum performance; the fact
source abstraction serves the WAM interpreter path and future non-FFI
access patterns.

## Alignment with the broader system

### Elixir fact-shape work (Phases A-E)

The Elixir target already landed this design: `compiled` for small
predicates, `inline_data` for medium, `external_source` for large. The
Haskell target should adopt the same classification and layout vocabulary
so all WAM targets speak the same language.

### C# materialization direction

The C# query runtime's `IRetentionAwareRelationProvider` and
`RelationRetentionMode` are the same idea at a different layer. The
Haskell FactSource interface is the WAM-level equivalent.

### Preprocessed artifacts

The `PREPROCESSED_PREDICATE_ARTIFACTS.md` proposal describes binary
artifacts with manifests, source hashes, and access-pattern declarations.
The Haskell `MmapFactSource` would be a consumer of these artifacts,
providing point lookups and adjacency expansion without full in-memory
materialization.

## What this does not change

- The FFI kernel path (`nativeKernel_category_ancestor` etc.) continues
  to use strict `IM.IntMap [Int]` for maximum performance. The fact
  source abstraction does not replace the FFI hot path.
- Recursive and rule-bearing predicates stay in their current WAM or
  lowered forms. Fact access is orthogonal to recursion handling.
- The `parMap rdeepseq` parallelism model is unchanged. Lazy fact
  sources are forced before the parallel section.
- Existing benchmarks continue to work. The strict IntMap path is
  wrapped in the FactSource interface but behaves identically.

## Connection to purity, order-independence, and parallelism

Fact access design does not exist in isolation. It connects to three
other concerns that form a feedback loop:

**Purity and order-independence** (see `PURITY_CERTIFICATE_SPECIFICATION.md`):
When the purity certificate declares goals as `pure` and
`order_independent`, the engine gains freedom to reorder them. Goal
reordering directly affects which fact predicates are accessed first,
how selective each access is, and therefore which materialization
strategy is optimal. A predicate probed 1000 times benefits from an
indexed layout; one scanned once benefits from streaming. The ordering
determines the access pattern, which determines the layout.

The Rust WAM target has taken purity and order-independence analysis
further than C#, providing more reordering freedom and thus more
materialization options for its planner.

**Parallelism constrains data structures**: The `parMap rdeepseq`
model requires all shared state to be immutable. This constrains which
fact sources are parallelism-safe:

| Source type | Parallelism-safe? | Why |
|---|---|---|
| Strict IntMap | Yes | Immutable, no locks needed |
| Lazy Haskell list | Yes (once forced) | Immutable after evaluation |
| Memory-mapped file | Yes | No file handle contention (see below) |
| SQLite (default) | No | Single-writer lock; readers block writers |
| SQLite (WAL mode) | Partial | Concurrent reads OK, single writer |
| Mutable IORef | No | Race conditions without MVar/STM |

**Why memory-mapped files are the ideal parallel data source:**
Memory-mapped files avoid file handle contention entirely. The OS maps
the file into virtual memory and each thread reads by dereferencing
pointers at different addresses — there is no `read()` syscall with a
shared file offset. Page faults are handled transparently by the kernel:
the first access to a page loads it from disk; subsequent accesses from
any thread hit the page cache. Multiple threads reading different
offsets is fully concurrent with zero coordination. In Haskell, a
`ByteString` backed by an `mmap`'d region is just a pointer + length,
safe to share across GHC's green threads and OS threads. This makes
mmap'd artifacts the natural choice for large fact sources under
parallelism — they combine the immutability of an IntMap with the
memory efficiency of not loading everything at once.

**Contrast with traditional file IO:** A single file descriptor with
`seek` + `read` has a shared offset, so concurrent reads require
locking or per-thread handles. `mmap` sidesteps this because each
thread accesses the mapped region via independent virtual addresses.

For sources that are not parallelism-safe (like SQLite in default
mode), the engine must either:

1. **Read once, then split.** Load the needed facts into an immutable
   structure before the parallel section, then share it. This is what
   the current eager IntMap approach does — it works, but forces full
   materialization.
2. **Per-worker connections.** Each spark gets its own database handle.
   Works for read-only workloads but multiplies connection overhead.
3. **Stream and partition.** Read the source sequentially, partition
   rows by seed/worker, pass each partition to its spark. The source
   is accessed once; parallelism operates on the partitioned data.

The Elixir target's recent SQLite data source (via external_source)
would face this exact constraint under parallelism. The fact that
SQLite isn't good at multi-process access means the materialization
planner must account for concurrency when choosing a layout — not just
access pattern and scale.

**The feedback loop:**

```
purity analysis → goal reordering freedom
                        ↓
              materialization planner
              (which predicates to index, stream, or precompute)
                        ↓
              data structure selection
              (IntMap, lazy list, mmap, database)
                        ↓
              parallelism constraints
              (immutable? lockable? per-worker?)
                        ↓
              back to materialization
              (must materialize before parallel section
               if source isn't parallelism-safe)
```

**Environment awareness:** The planner also needs to know about the
deployment environment. Not all platforms support all strategies:

- **Termux / Android userland**: May have restricted `mmap` support
  (filesystem-dependent, executable mapping limits). Parallelism is
  constrained by thermal throttling and limited cores. The planner
  should fall back to sequential scan or in-memory IntMap.
- **WebAssembly**: No `mmap`, no threads (unless SharedArrayBuffer is
  available). Single-threaded streaming only.
- **Embedded / low-memory**: Large IntMaps may not fit. External
  streaming or small-batch loading is required.
- **Server / cloud**: Full mmap, many cores, large memory. All
  strategies available; cost-based selection pays off.

The artifact proposal's `target_capabilities` field
(`PREPROCESSED_PREDICATE_ARTIFACTS.md`) already handles this: each
artifact declares required capabilities (`mmap`, `little_endian`,
`threads`), and the runtime checks what is available before selecting
a provider. The fact access layer should respect the same capability
declarations, falling back gracefully when a preferred strategy is
unavailable.

**Compile-time environment predicates:** Environment constraints
should be declarable at compile time via Prolog-side predicates:

```prolog
% Explicit declarations — user knows the target platform
:- environment(mmap(false)).
:- environment(max_cores(2)).
:- environment(platform(termux)).

% Auto-detect from the build machine (default: enabled)
:- environment(auto_detect(true)).

% Disable auto-detect for cross-compilation scenarios
:- environment(auto_detect(false)).
```

When `auto_detect(true)`, the codegen probes the build machine at
generation time: OS type, available memory, core count, filesystem
capabilities. This feeds the compile-time planner's decision space.

Auto-detect must be disableable because the build machine is not
always the target machine — cross-compilation, CI pipelines building
for mobile, generating Haskell on a desktop that runs on embedded.
When disabled, only explicit `environment/1` declarations are used;
absent any declaration, the planner assumes conservative defaults
(no mmap, single core, limited memory) so generated code is safe
everywhere.

Explicit declarations override auto-detect on a per-capability
basis: `environment(mmap(false))` disables mmap even if auto-detect
would find it available.

**Compile-time vs runtime planning:** The planner is not a single
runtime decision. It operates at two stages:

*Compile time* (Prolog codegen): The emitter already knows clause
structure, mode declarations, purity certificates, kernel detection
results, fact-only classification, and first-arg groundness. It can
trim logical paths — eliminating layout options that are provably
wrong or suboptimal before any code is generated:

- Impure predicate → don't consider parallel streaming layouts
- Clause count below threshold → compiled layout, no further options
- First-arg always ground → compile-time index is viable
- No mode declaration → cannot infer selectivity, keep all options open
- Kernel-detected predicate → FFI path, skip fact layout entirely

The compile-time planner emits code that supports the remaining
options (e.g., generates both a flat literal and an indexed map so
the runtime can choose).

*Runtime* (startup / query time): The actual data characteristics are
not known until load time — file sizes, row counts, available memory,
number of cores, whether mmap works on this filesystem. The runtime
planner selects from the options the compile-time planner left open:

- Row count exceeds memory budget → stream, don't materialize
- Multiple cores available → prefer immutable indexed structures
- mmap available → use artifact provider
- Single query → scan is fine; repeated probes → build index

This two-stage design means the compile-time planner trims the
decision space and the runtime planner selects within it. Neither
stage operates alone — compile time without runtime would over-commit
to a strategy that doesn't fit the data; runtime without compile time
would waste time considering options that the program structure
already rules out.

## Non-goals

- Join planning or cost-based layout selection. The default policy is
  intentionally simple: compiled below threshold, data above. A richer
  cost model is future work.
- Replacing the FFI kernel path. The native kernels are already 28x
  faster than the WAM interpreter; adding a fact source indirection
  there would be a regression.
- Cross-target artifact format standardization. Each target picks the
  host-native representation that fits best. The shared part is the
  predicate-level declaration and manifest format.
