# Haskell WAM Mutable Registers: Philosophy

**Status**: Default in generated projects. 52/55 step arms converted.
**Date**: 2026-05-25
**POC benchmark**: commit `9889825` (PR #2471) --
`examples/benchmark/haskell_register_bench/Main.hs`

## 1. The problem

The Haskell WAM uses `Data.IntMap.Strict` for register storage
(`wsRegs :: IntMap Value`). Every `putReg` allocates a new tree node.
Every `getReg` traverses the tree. With 5-10 register operations per
WAM instruction and hundreds of thousands of instructions per query,
this dominates both runtime and GC pressure.

F# uses a flat mutable `Value array` for the same purpose. The result:
F# 11 ms vs Haskell 107 ms (single-core) on the same workload --
a 9.7x gap that is almost entirely data structure overhead.

## 2. The alternatives (measured)

The POC benchmark (`9889825`) measured four register implementations
under a realistic WAM workload: 500k steps, 200 registers, 7
read/write ops per step, choice point snapshot every 50 steps,
restore every 200 steps.

| Implementation | median_ms | speedup | notes |
|---|---:|---|---|
| `IntMap` (current) | 178.3 | 1.0x | O(log n) per op, O(1) snapshot via structural sharing |
| `STArray` (proposed) | 22.6 | **7.9x** | O(1) per op, pure outside `runST` |
| `IOArray` | 22.6 | 7.9x | O(1) per op, requires IO monad |
| `Array` (//) | 281.1 | 0.63x | O(1) read, O(n) copy-on-write -- **worse than IntMap** |

### Why immutable Array (//) loses

Immutable `Array.//` copies the entire array on every single-element
update. With 200 registers and 4 writes per step, that is 4 full
array copies per instruction -- 800 words allocated per step vs
IntMap's ~35 words (7 tree nodes of ~5 words each). The constant
factor of copying dominates the O(1) index advantage.

This rules out the "just use an immutable flat array" approach. The
only path to matching F# is mutable arrays.

### Why STArray over IOArray

Both achieve identical performance (22.6 ms). The difference is
semantic safety:

- **`STArray` + `runST`**: the mutable computation is encapsulated
  in a pure interface. `runST` is referentially transparent and can
  be freely sparked in parallel (matching the existing `parMap
  rdeepseq` infrastructure for parallel clause evaluation).

- **`IOArray`**: requires the `IO` monad, which loses referential
  transparency. Parallel execution needs explicit thread management
  instead of `par`/`rseq` strategies.

For a WAM runtime that already uses `parMap` and `Async.Choice` for
parallel negation and forked branches, `ST` is the natural fit.

## 3. The trade-off: snapshot cost

IntMap's persistent structure gives O(1) choice point snapshots
(structural sharing -- just copy the pointer). STArray requires
`freeze` which copies all 200 elements.

For a typical effective-distance query:
- ~500k WAM instructions
- ~200 choice points (TryMeElse)
- Each snapshot copies 200 elements = 40k total elements copied

The snapshot cost is ~40k copies vs ~3.5M IntMap operations saved
(500k steps x 7 ops). The ratio is ~87:1 in favour of mutable arrays.

The trade-off only reverses in workloads with very frequent choice
points and very few register operations per step -- the opposite of
effective-distance.

## 4. Parallel safety

`runST` guarantees that the mutable state does not escape. Each
parallel seed gets its own `runST` block with its own `STArray`.
There is no shared mutable state between seeds, which means:

- No locks, no contention, no cache-line bouncing
- Each seed's register file is local to its core
- The pattern matches F#'s `Array.Parallel.map` where each seed
  creates fresh `Value array` registers

This is strictly better than `IOArray` for the parallel case because
the type system prevents accidental sharing.

## 5. Implementation scope

The conversion touches 90+ sites in `wam_haskell_target.pl` where
the generated Haskell code references `wsRegs`:

- `step` function: ~70 pattern match arms doing `IM.lookup`/`IM.insert`
- `backtrack`: register restore from choice point
- `getReg`/`putReg` helpers
- `addToBuilder`: struct/list builder finalisation
- Lowered emitter: 4 sites in `wam_haskell_lowered_emitter.pl`

The approach: wrap the `run` loop (including `step` and `backtrack`)
inside `runST`. All register access becomes `readArray`/`writeArray`.
The outer interface stays pure: `run :: WamContext -> WamState -> Maybe WamState`.

## 6. Future work

### 6.1 Remaining bridge arms (~7)

The following step arms still use the freeze/thaw bridge (converting
between STArray and IntMap per call). These are rare in the hot loop
but would eliminate the last conversion overhead:

- `member/2` (choice point creation inside builtin)
- `\+/1` (negation-as-failure with parallel fast path)
- `BeginAggregate` / `EndAggregate` (aggregate frames)
- `SwitchOnConstant` (label-based, pre-resolution form)
- `functor/3`, `=../2`, `copy_term/2` (term manipulation)
- `CallFactStream` (Phase F2 fact iteration)

### 6.2 Lowered emitter (4 sites)

`wam_haskell_lowered_emitter.pl` generates functions that access
registers directly via `IM.lookup` and `IM.insert`. When a workload
uses lowered predicates, these bypass `stepST` and miss the ST
speedup. Converting these 4 sites requires the lowered functions to
accept an `STArray` parameter or to be called from within the same
`runST` block.

### 6.3 Simplewiki-scale benchmark

The dev fixture (198 edges) is too small for timing. A benchmark on
simplewiki (6k+ edges, 300+ seeds) would measure the actual speedup
on the effective-distance workload. The POC prototype predicts 7.66x;
the real number depends on how often the bridge arms are hit and on
the overhead of `freeze` at choice points.

### 6.4 Bindings store

`wsBindings` (also IntMap) is the second-hottest data structure after
registers. Converting it to a mutable hash table or flat array would
give additional speedup. The trade-off is more complex: bindings
grow unboundedly (unlike registers which are fixed at 200), and trail
undo needs efficient deletion. A segmented approach (mutable array
for recently-bound variables, IntMap for overflow) may work.

### 6.5 Parallel seed dispatch via runST

Each seed currently runs through `parMap rdeepseq` with `run` (pure).
With `runMutableRegs` (also pure via `runST`), each seed still gets
its own isolated mutable register file — the type system prevents
sharing. The parallel infrastructure (`parMap`, `async`) should work
unchanged. Testing with `-N4` on a larger fixture would confirm.

## 7. References

- POC benchmark: `examples/benchmark/haskell_register_bench/Main.hs`
  (commit `9889825`, PR #2471)
- Plan doc: `docs/design/WAM_FSHARP_CSR_PARALLEL_PLAN.md` Phase 3
- Perf log: `docs/design/WAM_PERF_OPTIMIZATION_LOG.md` Phase L
- F# register design: `src/unifyweaver/bindings/fsharp_wam_bindings.pl`
  (WamState with `Value array`)
- Haskell step function: `src/unifyweaver/targets/wam_haskell_target.pl`
  `step_function_haskell/1`
