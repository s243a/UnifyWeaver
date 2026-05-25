# F# CSR + Parallelization + Haskell Speedup Plan

**Status**: Phase 1 done. Phases 2-3 pending.
**Date**: 2026-05-25
**Context**: This session delivered F# ISO support (6 PRs) + LMDB full
stack (11 PRs). F# achieves 11 ms query at scale 300 (beating Haskell
107 ms single-core, 32 ms 4-core). LMDB cached mode is 3.1× faster
than in-memory Map for the DFS hot loop.

## Phase 1: F# CSR Reader  [DONE]

### Motivation

The effective-distance algorithm currently only traverses *upward*
(child→parent via `category_parent`). Adding descendant-path
exploration (parent→child) enables new algorithms:
- Bidirectional search (meet-in-the-middle from source and root)
- Demand-set expansion (find all nodes reachable from a root within
  K hops, then compute distances only for those)
- Non-carrot-shaped path finding (paths that go down before going up)

CSR (Compressed Sparse Row) is the right format for child-edge lookup
because it gives packed, scan-friendly arrays — one contiguous read
per parent's child list, vs LMDB DUPSORT's B-tree cursor hops.

### Design

From `docs/design/WAM_REVERSE_INDEX_ARTIFACTS.md`:

```
category_child.csr.idx   fixed-width parent → (offset, count) records
category_child.csr.val   packed int32 child IDs
category_child.csr.meta  format version, ordering, id width, checksums
```

F# implementation:

```fsharp
/// CSR-backed child lookup. Memory-mapped .val file with direct
/// indexing into .idx for O(1) per-parent access.
type CsrLookupSource(idxPath: string, valPath: string) =
    // For dense parent IDs: idx is a direct array.
    // For sparse parent IDs: idx is sorted, binary-search lookup.
    let idx : (int * int) array = loadIdx idxPath  // (offset, count) per parent
    let vals : int array = loadVals valPath         // packed child IDs

    interface ILookupSource with
        member _.Lookup(parentId: int) : int list =
            let (offset, count) = idx.[parentId]  // or binary search for sparse
            [ for i in offset .. offset + count - 1 -> vals.[i] ]
```

Key decisions:
- Use `System.IO.MemoryMappedFiles` for the .val file (zero-copy)
- Or: load .val into a flat `int array` at startup (simpler, fast
  for simplewiki/enwiki scale — 10M × 4 bytes = 40 MB)
- The `ILookupSource` interface already supports this — `CsrLookupSource`
  plugs in alongside `LmdbCursorLookup` and `TwoLevelCachedLookupSource`

### Prerequisites

- Rust Phase C must stabilize the `.csr.idx`/`.csr.val`/`.csr.meta`
  format. OR: we define the format ourselves for F# and reconcile
  with Rust later. The format is simple enough (~20 lines of spec)
  that parallel development is low-risk.
- A CSR builder (Python script) to convert from the Phase 1 LMDB
  `category_child` DUPSORT database to the packed CSR files.

### Files created

- `templates/targets/fsharp_wam/csr_reader.fs.mustache` -- CsrLookupSource
- `tests/core/test_wam_fsharp_csr_smoke.pl` -- E2E smoke test
- `tests/core/test_wam_fsharp_csr_bench.pl` -- CSR vs LMDB benchmark
- Updated `wam_fsharp_target.pl` -- `csr_path(Path)` option + conditional .fsproj
- Builder already existed: `examples/benchmark/build_reverse_csr_artifact.py`

### Benchmark results (500 parents x 6 children)

- CSR raw: 0.55 ms (2.7x faster than LMDB cursor)
- CSR cached (L1/L2): 0.06 ms (24.5x faster than LMDB cursor)
- LMDB cursor: 1.47 ms (baseline)

---

## Phase 2: F# Parallel Kernel Execution  [DONE]

### Motivation

F# already has parallel infrastructure (`runNegationParallel`,
`forkParBranches`, `Async.Choice`). What's missing: parallel
*seed-level* execution for effective-distance -- running multiple
seeds concurrently against the same read-only graph.

### Design

```fsharp
/// Parallel seed dispatch. WamContext is read-only and safely shared.
/// Each seed gets its own copy of the L1 cache (ThreadLocal handles
/// this automatically). L2 is ConcurrentDictionary (already safe).
let runSeedsParallel (ctx: WamContext) (seeds: int array)
                     (kernel: int -> int) : int array =
    seeds
    |> Array.Parallel.map kernel
```

The `TwoLevelCachedLookupSource` already supports this:
- L1 is `ThreadLocal` — each parallel task gets its own cache
- L2 is `ConcurrentDictionary` — shared across tasks, lock-free

Key decisions:
- Use `Array.Parallel.map` (TPL-backed) for seed-level parallelism
- The WAM state is per-seed (no sharing) — same as Haskell's `-N4`
- Parallelism granularity: per-seed (coarse), not per-DFS-branch
  (fine). Coarse is simpler and matches the Haskell baseline.

### Expected speedup

At scale 300 with 100+ seeds: ~2-3x on 4 cores (matching Haskell's
107 ms -> 32 ms at `-N4`). The limiting factor is work balance across
seeds (some seeds have deep ancestor paths, others don't).

### Measured results (depth-12 tree, branching=3, 2000 seeds, 4 cores)

| Mode | median_ms | speedup |
|---|---:|---|
| Sequential | 11.1 | 1.0x |
| Array.Parallel.map | 6.3 | 1.77x |
| Parallel.For(N=4) | 5.6 | 1.98x |

Correctness verified: all modes produce identical hit counts.
TwoLevelCachedLookupSource confirmed thread-safe under parallel access
(L1 ThreadLocal + L2 ConcurrentDictionary).

### Test

- `tests/core/test_wam_fsharp_parallel_seeds.pl` -- E2E benchmark

---

## Phase 3: Haskell Speedup via Mutable Registers

### Why F# beats Haskell

F# WAM: 11 ms query. Haskell WAM: 107 ms single-core. The 9.7×
gap is NOT language overhead — it's data structure choice:

| Component | F# | Haskell | Impact |
|---|---|---|---|
| Registers | `Value array` (mutable, O(1)) | `Data.IntMap` (persistent, O(log n)) | **Dominant** |
| putReg | In-place array write | IntMap.insert (allocates new node) | ~10× per write |
| Choice point snapshot | `Array.copy` (one memcpy) | IntMap shared (O(1) but lookup is O(log n)) | Trade-off |
| Bindings | `Map<int, Value>` (persistent) | `Data.IntMap` (persistent) | Same |
| Hot-loop GC pressure | Low (array is one object) | High (IntMap allocates per step) | Significant |

### The fix: State monad + mutable arrays

The Haskell WAM step function currently uses pure IntMap:

```haskell
step :: WamState -> Instruction -> Maybe WamState
-- where WamState = { wsRegs :: IntMap Value, ... }
```

Proposed: switch to `ST` monad with mutable unboxed array for regs:

```haskell
step :: STRef s WamState -> Instruction -> ST s (Maybe ())
-- where WamState = { wsRegs :: STUArray s Int Value, ... }
```

Benefits:
- `putReg` becomes O(1) array write (no allocation)
- `getReg` becomes O(1) array read (no tree traversal)
- Choice point snapshot: `freeze` the array (one memcpy, same as F#)
- Backtrack restore: `thaw` + copy from snapshot

### Compatibility approach

Use a typeclass or type family to abstract the register store:

```haskell
class MonadReg m where
    getReg :: Int -> m Value
    putReg :: Int -> Value -> m ()
    snapshotRegs :: m RegSnapshot
    restoreRegs :: RegSnapshot -> m ()

-- Pure implementation (current, for testing):
instance MonadReg (State WamState) where ...

-- Mutable implementation (fast, for production):
instance MonadReg (ST s) where ...
```

This lets you swap implementations without changing the step function
logic — test with pure, ship with mutable.

### Expected speedup

- Register operations dominate the hot loop (5-10 per WAM step)
- Switching from IntMap to array: ~5-8× speedup on register-heavy code
- Overall query: 107 ms → ~15-25 ms (approaching F#'s 11 ms)
- With parallelism on top: would match or beat F#

### Risk

- `ST` monad threading changes the step function signature
- Choice point save/restore semantics must be carefully preserved
- The persistent IntMap advantage (O(1) snapshot) is lost; need
  explicit Array.copy on every TryMeElse

### Estimated effort: 2-3 sessions

---

## Sequencing

```
Phase 1: F# CSR reader          [DONE]
    |
Phase 2: F# parallel seeds      [DONE]
    |
Phase 3: Haskell mutable regs   [next -- separate sessions, independent of 1+2]
```

Phase 3 is independent -- can be started anytime, doesn't depend on
CSR or F# parallel work. The value is bringing Haskell back into
competition with F# so both targets can serve as viable production
choices.

## References

- `docs/design/WAM_REVERSE_INDEX_ARTIFACTS.md` — CSR format spec
- `docs/design/WAM_LMDB_LAZY_PHILOSOPHY.md` — lazy/cached motivation
- `docs/design/WAM_CROSS_TARGET_BENCHMARK_RESULTS.md` — current numbers
- `docs/design/WAM_FSHARP_PARITY_AUDIT.md` — F# status
- `docs/design/WAM_PERF_OPTIMIZATION_LOG.md` Phase L — Haskell measurements
- `templates/targets/fsharp_wam/lmdb_fact_source.fs.mustache` — ILookupSource
- `templates/targets/haskell_wam/lmdb_fact_source.hs.mustache` — Haskell L1/L2
