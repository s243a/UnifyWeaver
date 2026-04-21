# WAM Haskell Fact Access Specification

## Scope

This document specifies the contract between the Haskell WAM emitter
and the runtime for representing and accessing fact-only predicates.
It mirrors the WAM_FACT_SHAPE_SPEC.md (designed for Elixir, applicable
to all WAM targets) but addresses Haskell-specific concerns: lazy
evaluation, strict parallelism boundaries, and GHC compilation.

## Terminology

- **fact-only predicate** — every clause has a body of `true` (or no
  body). Head arguments are ground terms.
- **layout** — the host-language representation: `compiled`,
  `inline_data`, or `external_source`.
- **fact source** — a runtime adaptor that yields tuples. Analogous to
  the Elixir `WamRuntime.FactSource` behaviour and the C# query
  runtime's `IRetentionAwareRelationProvider`.
- **fact stream** — per-call iteration state carried in choice points
  while enumerating a fact-only predicate's solutions.

## Classification predicates

The emitter consults these to choose a layout. The same vocabulary as
the Elixir spec (WAM_FACT_SHAPE_SPEC.md section "Classification
predicates") applies:

- `fact_only(+PredIndicator)` — true when every clause body is `true`.
- `clause_count(+PredIndicator, -N)` — number of clauses.
- `first_arg_groundness(+PredIndicator, -Status)` — `all_ground`,
  `all_variable`, or `mixed`.
- `fact_layout(+PredIndicator, -Layout)` — user override.
- `fact_count_threshold(-N)` — default 100; above this, `inline_data`.

## Layout contracts

### Layout: `compiled` (current default)

Each fact becomes WAM instructions: `SwitchOnConstant` + `GetConstant`
+ `Proceed` with `TryMeElse`/`RetryMeElse`/`TrustMe` choice points.

Haskell-specific advantages:
- SwitchOnConstantPc is an `IM.IntMap Int` lookup (O(log n), effectively
  O(1) for typical sizes) with zero startup cost after `resolveCallInstrs`.
- GHC can optimize the strict instruction array access patterns.
- No runtime dispatch or typeclass overhead.

Applies when: small fact count, or explicit user override.

### Layout: `inline_data`

The emitter emits fact tuples as a Haskell literal in Predicates.hs:

```haskell
categoryParentFacts :: [(Int, Int)]
categoryParentFacts =
    [ (atomId_cat1, atomId_parent1)
    , (atomId_cat2, atomId_parent2)
    , ...
    ]
```

With optional compile-time index:

```haskell
categoryParentIndex :: IM.IntMap [(Int, Int)]
categoryParentIndex = IM.fromList
    [ (atomId_cat1, [(atomId_cat1, atomId_parent1)])
    , ...
    ]
```

The WAM runtime gains a `streamFacts` handler that iterates the literal
list, pushing a single `FactStream` choice point for backtracking:

```haskell
data BuiltinState
  = ...
  | FactStream ![(Int, Int)] !Int  -- remaining rows, return PC
```

The `step` function handles `CallFactStream` by matching the current
row against registers and advancing the stream. `backtrack` recognises
`FactStream` CPs and resumes iteration.

Haskell-specific considerations:
- The literal list is lazy by default. GHC will not allocate the full
  list until forced. This is the right behaviour for single-pass scans.
- The index (`IM.fromList`) is strict on construction. If defined at
  top level in Predicates.hs, GHC evaluates it once on module load.
  This matches the Elixir `@facts_by_arg1` pattern.
- For parallelism: the `WamContext` must force fact data before the
  `parMap` section. A `BangPattern` on the context field suffices.

### Layout: `external_source`

The emitter emits a thin wrapper that delegates to a FactSource
registered at startup in Main.hs:

```haskell
-- In WamTypes.hs
data FactSource = FactSource
  { fsScan       :: IO [(Int, Int)]       -- full scan (lazy)
  , fsLookupArg1 :: Int -> IO [(Int, Int)] -- indexed by first arg
  , fsClose      :: IO ()
  }

-- In WamContext
data WamContext = WamContext
  { ...
  , wcFactSources :: !(Map.HashMap String FactSource)
  }
```

Concrete implementations:

**TsvFactSource:**
```haskell
tsvFactSource :: InternTable -> FilePath -> IO FactSource
tsvFactSource tbl path = do
    rows <- lazy $ parseTsv tbl path   -- lazy IO
    let index = buildIndex rows        -- built on first demand
    return FactSource
      { fsScan = return rows
      , fsLookupArg1 = \key -> return $ IM.findWithDefault [] key index
      , fsClose = return ()
      }
```

**IntMapFactSource** (wraps current strict IntMap):
```haskell
intMapFactSource :: IM.IntMap [Int] -> FactSource
intMapFactSource im = FactSource
  { fsScan = return [(k, v) | (k, vs) <- IM.toList im, v <- vs]
  , fsLookupArg1 = \key -> return $ map (\v -> (key, v)) $ IM.findWithDefault [] key im
  , fsClose = return ()
  }
```

**MmapFactSource** (future):
- Memory-mapped binary artifact with offset index
- Point lookups via binary search on key directory
- No resident memory for full relation
- Requires `bytestring` + `mmap` packages

The WAM interpreter accesses external facts via the same `FactStream`
CP mechanism as `inline_data`, but obtains the row list from
`wcFactSources` instead of a compiled literal.

## Interaction with the FFI kernel path

The FFI path (`executeForeign` → `nativeKernel_*`) continues to use
`wcFfiFacts :: Map String (IM.IntMap [Int])` directly. The FactSource
abstraction does NOT replace this path. The FFI kernels need strict,
fully-evaluated, interned IntMaps for maximum throughput.

When both paths exist for the same predicate (e.g., `category_parent/2`
is both an FFI kernel's edge predicate AND a callable fact predicate):
- FFI kernel uses `wcFfiFacts` (strict IntMap, interned)
- WAM interpreter uses `wcFactSources` (potentially lazy, via FactStream)
- Both are populated from the same underlying data at startup

## Strictness boundary

The parallelism model (`parMap rdeepseq`) requires all shared state to
be fully evaluated before forking. The strictness protocol:

1. `WamContext` fields are strict (`!` annotations).
2. `wcFfiFacts` is forced via `BangPattern` at construction (current).
3. `wcFactSources` entries that back lazy IO must be forced before
   `parMap`. The Main.hs template includes a
   `seq (deepseq factSources ()) ()` barrier between context
   construction and the parallel seed loop.
4. Within a single seed's computation (inside the `parMap` lambda),
   lazy fact access is safe — no sharing across threads.

## Configuration

Users control layout via Prolog-side declarations:

```prolog
% Override layout for a specific predicate
:- fact_layout(category_parent/2, external_source(tsv("data/category_parent.tsv"))).

% Change the threshold for inline_data
:- fact_count_threshold(200).

% Control indexing policy
:- fact_index_policy(category_parent/2, first_arg).
```

The emitter reads these when deciding how to emit each fact-only
predicate. Absent any declaration, the default policy applies:
- N <= threshold → `compiled`
- N > threshold, facts known at codegen → `inline_data`
- Facts from external source → `external_source`

## Backward-compatibility

- Existing benchmarks and generated projects continue to work without
  any configuration changes. The default layout for the effective-
  distance benchmark predicates remains `compiled` (small predicates)
  or the current `buildFact2Code` path (TSV-loaded facts).
- The `buildFact2Code` function in Main.hs is the `compiled` layout
  for runtime-loaded facts. It is not removed; it becomes one strategy
  among several.
- No changes to the `run` loop, `step` function, or `backtrack`
  handler beyond adding the `FactStream` CP type.

## Non-goals

- Cost-based automatic selection between layouts. The default policy is
  simple and predictable; richer planning is future work.
- Replacing the FFI kernel path. Native kernels are already optimal.
- Streaming SGML/XML/JSON parsing. The FactSource interface is for
  tabular fact data. Document-oriented sources are a separate concern.
