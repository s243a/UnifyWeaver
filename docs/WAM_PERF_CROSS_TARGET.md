# WAM Runtime Perf — Cross-Target Notes

Notes on per-target WAM runtime performance characteristics, recording
what's been learned during profiling-and-fix work.  Use this when
you're considering whether a perf win on one target backports to
another, or when starting profile work on a new target.

The big takeaway upfront: **WAM runtime perf is target-specific.  A
fix on one target rarely backports cleanly to others.**  Each
target's host-language idioms (mutable vs immutable state, reference
vs value semantics, choice of data structure) drive its own
performance shape.

## Case study: the F# `putReg` fix (PR #2428, ~2-3x speedup)

`dotnet-trace` on the F# parser-heavy benchmark
([`tests/core/test_wam_fsharp_lowered_bench.pl`](../tests/core/test_wam_fsharp_lowered_bench.pl))
showed `WamTypes.putReg` accounting for **~32% of total CPU time**.
Root cause: the F# runtime represented `WsRegs` as a fixed-size
`Value array` (512 slots) inside an *immutable* `WamState` record,
so the only way to "update" it while preserving record-immutability
was

```fsharp
let r = Array.copy s.WsRegs
r.[n] <- v
{ s with WsRegs = r }     // new record, fresh 512-element array
```

— an O(512) `Array.copy` per register write.

The fix (PR #2428) was specific to that pattern: switch to in-place
mutation of `s.WsRegs`, relying on the single-owner state semantics
of the run loop (each `step` returns the new state, the previous
reference is dead by the next instruction).  Choice points already
snapshotted via explicit `Array.copy` at creation; the fix also
copies on `backtrack`-restore so future in-place writes can't
corrupt the restored CP's saved registers.

## Cross-target audit

After the F# win, every other WAM target was checked for the same
pattern.  None of them have it.

| target | register-update pattern | per-write cost | analog of F# bug? |
| --- | --- | --- | :---: |
| **Python** | `state.regs[n] = val` (Python `list` mutation) | O(1) | no — direct mutation |
| **C++** | `*it->second = std::move(v)` (cell deref through `std::unordered_map`) | O(1) avg | no — direct mutation |
| **Haskell** | `s { wsRegs = IM.insert rid val (wsRegs s) }` | O(log n) | no — `IntMap` structural sharing |
| **Elixir** | `%{state \| regs: Map.put(state.regs, reg, val)}` | O(log₃₂ n) | no — `Map` (HAMT) structural sharing |
| **R** | `state$regs2[[idx]] <- val` | O(1) | no — R env reference semantics |
| **Rust** | `self.regs[idx] = val` (`Vec<Value>`) | O(1) | no — `&mut self` enforces single-owner |
| **Go** | `vm.Regs[idx] = val` (slice mutation, already has `MaxYReg` opt) | O(1) | no — already optimized |
| **Clojure** | `(assoc-in state [:regs reg] value)` | O(log₃₂ n) | no — persistent map sharing |
| **Lua** | `state.regs[idx] = val` (Lua `table` mutation) | O(1) | no — direct mutation |

The F# bug needed **all three** of:

1. A fixed-size mutable array (not a persistent data structure),
2. An *immutable* surrounding state record forcing a new array per
   "update",
3. The conventional F# idiom of `Array.copy` + `with` for both.

No other target has this combination.  The mutable-state targets
(Python, C++, R, Rust, Go, Lua) mutate registers in place at the
language level; the immutable-state targets (Haskell, Elixir,
Clojure) use persistent data structures whose updates are O(log n)
with structural sharing, so the snapshot is essentially free.

## How to find a real bottleneck in target X

Profile first.  Don't backport.  Specifically:

- **.NET targets (F#, C#)** — use `dotnet-trace collect --format Speedscope -- <invocation>`.  PR #2428 walks through this; the speedscope JSON has events with timings, and a small Python parser (in the PR body) tabulates self-vs-total time per function.

- **Python target** — `python -m cProfile -o profile.out <script>` then `pstats` or [snakeviz](https://jiffyclub.github.io/snakeviz/).

- **C++ target** — `perf record` on Linux (g++ -O2 builds), or `Instruments` on macOS.  Note: `wam_runtime.cpp` is ~6.7k lines and compiles in seconds; `generated_program.cpp` for parser-bundled projects is ~42k lines and takes 11+ min at `g++ -O0`, so build benchmarks like
  [`tests/test_wam_cpp_generator.pl`](../tests/test_wam_cpp_generator.pl)
  cache `wam_runtime.o` (PR #2442).

- **Go target** — `go test -cpuprofile` is the standard.

- **Other targets** — host-language-native tools.

**The actual bottleneck is rarely where the equivalent target had
theirs.**  F#'s was `Array.copy`; Python's might be `Map.add` on the
bindings; Clojure's might be `assoc-in` cumulative cost; C++'s might
be `std::move` on `Value` variants.  Profile to know.

## What stays the same across targets

A few things ARE universal and *are* worth keeping in mind:

- **Choice-point snapshots cost O(stuff-in-state) regardless of
  target.**  Targets that snapshot more eagerly (e.g., full register
  array copy at every `TryMeElse`) lose more.  Targets with
  persistent data structures get the snapshot "for free" (pointer
  copy) but still need to ensure backtrack restore reuses the
  snapshot rather than rebuilding from a trail.

- **The trail-undo pattern** (walk the trail prefix, undo each
  binding) is the same across all targets.  In immutable-data
  targets it's mostly a no-op (because `cp.CpBindings` already IS
  the restore target — see the
  [F# experiment from PR #2428's predecessor work](WAM_FSHARP_TARGET.md#key-runtime-invariants)
  where removing the seemingly-redundant trail fold regressed parser
  benchmarks ~34% for non-obvious JIT-inlining reasons).  In
  mutable-state targets the trail-undo is load-bearing.

- **`step`'s instruction-dispatch shape** is universal — every target
  has a big `match instr with | ...` (or equivalent) that gets
  invoked once per WAM instruction executed.  How efficiently the
  host language compiles that dispatch is a big factor in raw
  throughput, but it's hard to influence from the target codegen
  side.

## Case study: the LLVM arena-cleanup fix (~18% per-query speedup)

The WAM-LLVM target's `@wam_cleanup` was advertised in three places as
a *rewind* (e.g. `wam_llvm_target.pl:1298` "free() is a no-op — memory
is reclaimed by @wam_cleanup via arena rewind") but the implementation
in `templates/targets/llvm_wam/state.ll.mustache` actually called
`@wam_arena_destroy`, which `free()`s the entire 1 MiB arena buffer.
The WASM-export wrapper invokes `@wam_cleanup` after *every* exported
predicate call, so each query paid a `free()` plus the next iteration's
1 MiB `malloc()` to re-establish the arena.

The fix splits the two operations:

- `@wam_arena_reset` — zeros the bump pointer (`@wam_arena_pos`),
  keeps the buffer mapped. All previously-allocated `%Compound` /
  `%List` pointers become invalid, which is the documented and
  expected semantics.
- `@wam_cleanup` — now calls `@wam_arena_reset`. Per-query path
  pays one `store i64 0`.
- `@wam_full_shutdown` (new) — calls `@wam_arena_destroy` for hosts
  that actually want to reclaim the 1 MiB on module unload.

Numbers from `/tmp/wam_dispatch_bench.pl` (single-clause arithmetic
predicate, single `%WamState` reused across iterations, `-O2` build,
median of 3 runs, 10M iterations):

| variant                              | total   | per-iter |
| ------------------------------------ | ------: | -------: |
| baseline (`free` + next-iter `malloc`) | 2362 ms |   236 ns |
| arena-reset (this PR)                | 1937 ms |   194 ns |

This is independent of the LLVM optimization level — the win comes
from removing the per-iter `malloc(1 MiB)` syscall path, not from
LLVM IR shape.

## LLVM hot-path inline attributes

A second LLVM-target change: every register / heap / trail / deref
helper in `templates/targets/llvm_wam/{value,state}.ll.mustache`
(roughly 25 functions, e.g. `@wam_get_reg`, `@wam_set_reg`,
`@wam_inc_pc`, `@wam_deref_value`, `@wam_trail_binding`,
`@value_is_unbound`, `@value_tag`) is now tagged
`alwaysinline nounwind`, plus `readnone` on the pure constructors
and inspectors. `@value_equals` (recursive) uses `inlinehint
readonly` instead of `alwaysinline` so LLVM can still inline the
non-recursive call sites without rejecting the IR.

**Perf impact at `-O2`: essentially zero.** Modern clang's inliner is
aggressive enough that it already inlines these small helpers without
hints — `opt -O2` of the generated IR with the hints stripped still
yields the same `@step` body. The hints matter for:

1. **`-O0` / `-O1` development builds** — the always-inline pass runs
   independent of the inliner heuristic, so the inlined dispatch loop
   stays consistent across optimization levels.
2. **IR review** — the dispatch loop's intent ("these are inline
   primitives, not external calls") is now legible in the source
   template, matching what the C runtime gets from `static inline`
   in `wam_c_runtime/wam_runtime.h:474+` (`resolve_reg`,
   `trail_binding`, `wam_deref_ptr`) and what F# gets from
   `let inline`.

## Case study: Kotlin recursive `snapshotForNative` (KT-DISPATCH-SNAPSHOT-OPT)

Same **copy-the-world-per-operation** class as the F# `putReg` bug, different
trigger. BENCH-KOTLIN showed `emit_mode(functions)` **append** regressing
(~0.55–0.80× vs interpreter) and worsening with depth. Manual
`System.nanoTime` around `snapshotForNative` on `append_500` attributed
**~31% of timed wall** to snapshots, with `snap_count == native_entries`
(one full register/heap map copy per recursive `execute`→`tryRun` hop).
Heap vars (`H<n>`) accumulate within a query, so cost ≈O(depth²).

**Fix (pattern, not F# code):** keep the top-level WAT-style
snapshot+bytecode fallback (T5 unbound / incomplete lowering), but skip
snapshot+fallback on **recursive** native hops (`nativeDepth > 0`). Clause
backtracking stays on the lowered T4 `_t4` snapshot. Toggle:
`skipRecursiveNativeSnapshot` (default true); A/B via
`SKIP_RECURSIVE_SNAP=0`.

**Result:** append_100 → ~1.03×; append_500 → ~0.85× (from ~0.55×);
functions-only append_500 self-speedup ~1.55×. Remaining gap is T4’s
per-entry `_t4` map copy — see
[`design/WAM_KOTLIN_OPTIMIZATION_HISTORY.md`](design/WAM_KOTLIN_OPTIMIZATION_HISTORY.md)
and [`WAM_KOTLIN_BENCH.md`](WAM_KOTLIN_BENCH.md).

**Cross-target lesson (unchanged):** profile first; the bottleneck is
rarely where the sibling target’s was. Here it was fallback-safety
snapshots on a hot recursive seam, not per-register immutability.

## Related

- [WAM_FSHARP_TARGET.md](WAM_FSHARP_TARGET.md) — the F# WAM target's
  full usage guide, including the runtime invariants that drove the
  perf work.
- [WAM_RUNTIME_PARSER_STATUS.md](WAM_RUNTIME_PARSER_STATUS.md) —
  cross-target runtime-parser status (different concern from this
  doc but adjacent).
- [`tests/core/test_wam_fsharp_lowered_bench.pl`](../tests/core/test_wam_fsharp_lowered_bench.pl)
  — the F# benchmark harness; methodology (5 warm-up rounds, forced
  GC, median-of-3) is reusable for other targets.
- [`tests/core/test_wam_llvm_benchmark.pl`](../tests/core/test_wam_llvm_benchmark.pl)
  — the LLVM foreign-kernel BFS / Dijkstra harness. Per-iter timings
  for that bench did not move with the arena-reset fix because the
  kernels run inside their own helpers and do not exercise the
  per-query `@wam_cleanup` path.
