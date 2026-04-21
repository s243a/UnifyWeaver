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
