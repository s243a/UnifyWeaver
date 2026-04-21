# WAM Haskell Fact Access Implementation Plan

## Goal

Move the Haskell WAM target toward the "facts as data" model described
in WAM_FACT_SHAPE_PHILOSOPHY.md, while preserving Haskell's laziness
advantage and the existing FFI kernel performance. This is the Haskell
port of Phase F from the WAM fact-shape plan.

## Current state

The Haskell WAM target has three fact access patterns today:

| Pattern | Where | How |
|---|---|---|
| Compiled WAM instructions | `buildFact2Code` in Main.hs | SwitchOnConstant + GetConstant + Proceed; built at startup from TSV |
| Strict IntMap for FFI | `wcFfiFacts` in WamContext | `IM.IntMap [Int]`, interned, passed to native kernels |
| Eager TSV load | `loadTsvPairs` in Main.hs | `readFile` → strict list of `(String, String)` pairs |

All three eagerly materialize all facts before any query runs. The
TSV parser returns a strict list; the IntMap construction forces every
entry; the WAM instruction builder appends every fact to the code array.

## Phased rollout

### Phase F1: Classification infrastructure (no behaviour change)

Add `classify_fact_predicate/4` to the Haskell emitter pipeline. For
each predicate, compute and record:
- `fact_only` status
- Clause count
- First-arg groundness
- Chosen layout (always `compiled` in this phase)

Emit the layout choice as a comment in Predicates.hs so it is
observable. All predicates still emit as `compiled`.

Acceptance: existing benchmarks produce identical output. New tests
verify classification is correct for small/large/rule predicates.

### Phase F2: FactStream choice point type

Add a `FactStream` constructor to `BuiltinState` in WamTypes.hs:

```haskell
| FactStream ![(Int, Int)] !Int  -- remaining rows, return PC
```

Add `streamFacts` and `resumeFactStream` handlers to WamRuntime.hs.
The `backtrack` function recognises `FactStream` CPs.

This is pure runtime infrastructure — no emitter changes yet. Tested
by manually constructing a FactStream CP in a unit test and verifying
correct iteration and backtracking.

Acceptance: existing benchmarks unaffected (FactStream is never
constructed by the emitter yet). New tests verify FactStream iteration.

### Phase F3: `inline_data` emission

For fact-only predicates above the threshold, emit a Haskell literal
list in Predicates.hs instead of WAM instructions. Emit a
`CallFactStream` instruction that the runtime dispatches to the
`streamFacts` handler.

With optional first-arg index (IntMap grouping by arg1) when
`fact_index_policy = first_arg` or `auto`.

Acceptance: effective-distance benchmark with `no_kernels(true)` at 1k
and 10k produces identical output. `inline_data` predicates load and
execute correctly.

### Phase F4: FactSource abstraction and external_source layout

Add the `FactSource` record type to WamTypes.hs and `wcFactSources`
to WamContext. Implement `TsvFactSource` with lazy IO.

The emitter gains an `external_source` path: for predicates declared
with `fact_layout(P/A, external_source(tsv(Path)))`, emit a wrapper
that dispatches to `wcFactSources` at runtime.

Key Haskell design: use `unsafeInterleaveIO` or lazy `readFile` for
the scan path, so facts are parsed on demand. Build the index lazily
(forced on first `fsLookupArg1` call via `unsafePerformIO` + `IORef`
+ `once` pattern).

Acceptance: benchmark with `external_source` layout produces correct
output. Memory profile shows lazy loading at scale (facts not all
resident at once for scan-only queries).

### Phase F5: Force barrier for parallelism

Ensure `wcFactSources` entries are fully evaluated before `parMap`.
Add a force barrier in the Main.hs template between context
construction and the parallel seed loop. Document the strictness
contract.

This phase is small but critical for correctness under parallelism.

Acceptance: parallel benchmark (`+RTS -N4`) produces correct output
with external_source facts. No thunk leaks across spark boundaries.

### Phase F6: MmapFactSource (future, deferred)

Memory-mapped binary artifact consumer. Depends on artifact format
stabilization from the C# side. Uses `bytestring` + `mmap` to provide
point lookups without resident memory. Would consume `.uwbr` artifacts
produced by the C# `BinaryRelationArtifactBuilder`.

Deferred until the artifact format is stable and the scale pressure
justifies the complexity.

## Risks and mitigations

- **Risk: lazy IO + parallelism.** Lazy IO is famously tricky with
  concurrent access. Mitigation: force all fact sources before `parMap`;
  within each seed computation, facts are accessed single-threaded.

- **Risk: `inline_data` slower than `compiled` for indexed lookups.**
  Compiled SwitchOnConstant is O(1) IntMap lookup. FactStream is O(n)
  scan (or O(log n) with index). Mitigation: keep `compiled` as the
  default for small predicates; only switch to `inline_data` above
  threshold where compilation cost dominates.

- **Risk: correctness drift between layouts.** Mitigation: golden-output
  tests that run the same query with each layout and diff results.

- **Risk: GHC compilation time.** Large Haskell literal lists can slow
  GHC. Mitigation: for very large predicates (>10k facts), prefer
  `external_source` over `inline_data` to keep generated code small.

## Dependencies

- WAM_FACT_SHAPE_SPEC.md — the cross-target contract this implements.
- Atom interning (landed) — `inline_data` and FactSource use interned
  Int keys, not String.
- PREPROCESSED_PREDICATE_ARTIFACTS.md — Phase F6 depends on artifact
  format stabilization.

## What this does not change

- FFI kernel path (`wcFfiFacts`, `executeForeign`, native kernels).
- Recursive predicate compilation (WAM or lowered).
- Parallelism model (`parMap rdeepseq`).
- The `run`/`step`/`backtrack` core loop (extended, not replaced).
