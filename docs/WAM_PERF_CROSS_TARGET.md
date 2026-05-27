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
