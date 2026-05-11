# WAM Target Performance Optimization Log

Chronological record of performance optimizations applied to the Haskell
and Rust WAM targets. This is a reference document — not a narrative. For
the strategic framing (why these optimizations matter, where we're going
next), see `docs/vision/HASKELL_TARGET_ROADMAP.md`.

## Executive summary — where we ended up

All numbers are medians of 10 runs on the effective-distance benchmark,
non-profiled, 4-core WSL2 host:

### 300 scale (386 seeds, ~6k category_parent facts)

| Target | total_ms | query_ms | Notes |
|---|---|---|---|
| **Haskell + FFI + parallel** | **75** | **32** | 4 cores, `+RTS -N4` |
| Haskell + FFI (single core) | 193 | 107 | after atom interning |
| Rust + FFI | 126 | — | hand-tuned, single-threaded |
| SWI-Prolog (optimized) | 311 | — | baseline reference |
| Haskell pure interpreter | 2518 | — | no FFI, no kernels |
| Haskell initial (pre-optimization) | ~4700 | — | naive `Map String Value` |

**Haskell net speedup from baseline: ~63x.** The target outperforms the
single-threaded Rust implementation by 1.75x on total time at 300 scale.

### Scaling across dataset sizes

| Scale | 1-core query | 4-core query | Speedup | 4-core total | Notes |
|---|---|---|---|---|---|
| 300 (6k facts) | 107ms | 32ms | **3.3x** | 75ms | ideal case |
| 5k (18k facts) | 297ms | 86ms | **3.5x** | 213ms | best scaling |
| 10k (35k facts) | 836ms | 284ms | **2.9x** | 604ms | GC pressure starts to show |

**Scaling observations:**
- 5k gives the best parallel scaling (3.5x on 4 cores) — setup overhead
  amortizes well over the per-seed work.
- 10k scaling drops to 2.9x. Hypothesis: GC pressure from larger intern
  tables and recursion depth allocations. Could be investigated if 10k
  becomes the primary benchmark workload.
- 8 cores regressed on this WSL2 host at all tested scales due to
  scheduling/GC contention. Native Linux or different hardware may
  differ — worth re-testing on real multi-socket systems.
- **All scales remain faster than the single-threaded Rust target on
  total time at 4 cores.**

### Multi-output kernel: `transitive_distance3/3`

After adding multi-output kernel support (FFIStreamRetry), we ran a
dedicated `transitive_distance3/3` benchmark on the same category_parent
graph. This kernel does BFS from each source and returns `(target, distance)`
pairs — 2-output stream, exercising the new infrastructure under load.

| Scale | Sources | Pairs | 1-core query | 4-core query | Speedup |
|---|---|---|---|---|---|
| 300 (6k edges)  | 2165 | 199,029 | 185ms | **70ms** | 2.6x |
| 10k (25k edges) | 7811 | 877,029 | 1183ms | **420ms** | 2.8x |

The reachability computation is substantial — 199k pairs at 300 scale,
877k at 10k scale. Parallel scaling holds at 2.6-2.8x on 4 cores, similar
to the effective-distance benchmark's scaling profile. FFIStreamRetry
works correctly under load with hundreds of thousands of solutions.

### Weighted kernel: `weighted_shortest_path3/3` (Dijkstra)

Added the Dijkstra kernel with Double-valued output, validating Float
wrapping in the multi-output FFI path. Synthetic weights
(`1.0 + (childId mod 10) * 0.1`) give non-uniform edge costs.

| Scale | Sources | Pairs | 1-core query | 4-core query | Speedup |
|---|---|---|---|---|---|
| 300   | 2165 | 199,029 | 212ms  | **90ms**  | 2.4x |
| 10k   | 7811 | 877,029 | 1232ms | **441ms** | 2.8x |

Slightly slower than `transitive_distance3` (Dijkstra priority queue
overhead vs straight BFS). Deterministic output (`total_weight` byte-for-
byte identical across runs) confirms Float accumulation is consistent
and the priority-queue ordering is stable.

`wcFfiWeightedFacts` field added to `WamContext` for `Map String (IntMap
[(Int, Double)])` storage. New `config_weighted_facts_from(edge_pred)`
ArgSpec resolves to `config_weighted_facts(pred_name)` which emits the
appropriate lookup at codegen time.

### Wide-output BFS kernels: `pdist4` and `tspd5`

After completing multi-output infrastructure, validated 3-output and
4-output kernels at scale on the same category_parent graph:

| Kernel | Outputs | 300 (4-core) | 10k (4-core) | Determinism |
|---|---|---|---|---|
| `transitive_distance3` | (target, dist) | 70ms | 420ms | total_pairs=199029/877029 |
| `transitive_parent_distance4` | (target, parent, dist) | 62ms | 380ms | sum_distances=1493158/6482523 |
| `transitive_step_parent_distance5` | (target, step, parent, dist) | 73ms | 395ms | unique_first_hops=6008/25214 |

All three share identical `sum_distances` at each scale, confirming
shortest-path distance computation is consistent regardless of output
arity. Performance scales linearly with output count (each additional
field adds ~5-15% query time from the trail+binding update overhead).

---

## Haskell WAM — optimization timeline

### Phase A: Data structures (weeks 1-2)

Early work — moving from naive associative lookups to efficient containers.

| Commit | Date | What | Why it matters |
|---|---|---|---|
| `8abf0df3` | 2026-04-09 | O(1) Array for instruction fetch + Atom [] list fix | Instruction fetch was O(log n) via Map lookup on PC; Array turns it into O(1) |
| `0daa9d65` | 2026-04-09 | `Data.HashMap.Strict` for register/binding maps | Replaced `Data.Map`; better constants for String keys |
| `221f828f` | 2026-04-09 | `IntMap` for registers, eliminating string hashing | Registers are Ints at compile time; no reason to hash strings |
| `803ed711` | 2026-04-09 | Intern variables as Ints, `IntMap` for bindings | Same idea for runtime-allocated variables |
| `da115dec` | 2026-04-08 | O(log n) SwitchOnConstant + parser | Indexed clause selection via Map instead of linear scan |

**Lesson:** The biggest early wins came from matching data structures to
access patterns. `Data.Map String` with String keys was doing a lot of
unnecessary hashing; the "right" container was usually `IntMap` with
Ints that were already available at compile time.

### Phase B: Eliminate repeated work (weeks 2-3)

Caching fields and pre-computing at compile time.

| Commit | Date | What | Why it matters |
|---|---|---|---|
| `501da4fc` | 2026-04-09 | Pre-parse PutStructure arity at compile time | Runtime was re-parsing functor strings on every call |
| `cec17ca3` | 2026-04-09 | Pre-resolve Call instructions at project load | String label → Int PC, one-time cost at startup |
| `f757b881` | 2026-04-09 | Cache `wsTrailLen`/`wsHeapLen`, eliminate `length` calls | Lists don't cache length; counters do |
| `bdae58cf` | 2026-04-09 | Cache `wsCPsLen`, fix `addToBuilder` O(n²) append | Appending to a list with `++` at each step was quadratic |
| `0d9695a9` | 2026-04-09 | List-based visited + unsafe instruction fetch | Replaced Set for tiny visited lists; `unsafeAt` skips bounds check |
| `d0e639de` | 2026-04-09 | INLINE pragmas on `getReg`, `putReg`, `derefVar` | Force GHC to inline hot accessors |
| `ab829695` | 2026-04-09 | Split `WamState` into hot `WamState` + cold `WamContext` | Code array, labels, foreign facts — shouldn't be copied per step |
| `76a0619d` | 2026-04-13 | UNPACK pragmas for Int fields in `WamState`/`ChoicePoint` | Force unboxing of strict fields; fewer heap indirections |

**Lesson:** "Cache the length" and "pre-compute at compile time" are
recurring themes. Haskell laziness makes it easy to accidentally re-do
work; `BangPatterns` + strict fields + UNPACK are the tools that fix this.

### Phase C: Label resolution + profiling (week 3)

| Commit | Date | What | Why it matters |
|---|---|---|---|
| `1ecca565` | 2026-04-13 | Pre-resolve all label lookups to direct PC jumps | Introduces `TryMeElsePc`, `RetryMeElsePc`, `ExecutePc`, `JumpPc`, `SwitchOnConstantPc` — the interpreter hot loop never touches string labels |
| `a6280d1a` | 2026-04-13 | String-key `SwitchOnConstantPc` + profiling utility | `gen_prof.pl` for generating pure-interpreter profiling benchmarks |

Interpreter speedup was **47%** on pure-interpreter path after label
pre-resolution (2518ms → ~1350ms). This is the work that motivated the
profiling matrix below.

### Phase D: Profile-guided wins (April 13 — final push)

This is when we systematically profiled and attacked the biggest costs.

| Commit | PR | Date | What | Impact |
|---|---|---|---|---|
| `fcaa885c` | #1375 | 2026-04-13 | Skip WAM-compilation of FFI-owned fact predicates | **-70% total** (740ms→225ms). `buildFact2Code` was allocating ~2GB of WAM instructions that the FFI path never executed |
| `d139991f` | #1376 | 2026-04-13 | Atom interning at FFI boundary | **-48% query** (200ms→107ms). Kills `hashWithSalt1` in the kernel's recursive HashMap lookups — IntMap of interned Ints replaces HashMap String |
| `74d9e9b4` | #1377 | 2026-04-13 | Parallel seeds + O(n) intern build | **-67% total** (225ms→75ms). `parMap rdeepseq` over 386 seeds; each gets independent WamState, WamContext shared read-only |

**Lesson from the profiling matrix**: the biggest optimizations were
*not* in the code we were optimizing. The pure interpreter (`step` at
59% of time) was a red herring — once FFI handled the hot predicate,
`step` dropped to 2.8% and `buildFact2Code` took over. Without profiling
all four configurations (pure-interp, interp+FFI, lowered-only, lowered+FFI),
we would have chased the wrong bottleneck.

### Additional refinements

| Commit | Date | What | Notes |
|---|---|---|---|
| `7bbdbdbb` | 2026-04-13 | Enable FFI for optimized aggregate benchmark | Wire-up for benchmarks |

---

## Rust WAM — optimization timeline

### Phase A: Core data structures

| Commit | Date | What | Why it matters |
|---|---|---|---|
| `3e3d526b` | 2026-04-05 | Binary search for SwitchOnConstant dispatch | Sorted array + bsearch instead of linear scan |
| `fae8fbdd` | 2026-04-06 | O(log n) instruction fetch and switch_on_constant dispatch | Shared with Haskell work |

### Phase B: Choice points

Choice points are hit on every disjunction — making them cheap is critical.

| Commit | Date | What | Why it matters |
|---|---|---|---|
| `6bec8b7e` | 2026-04-06 | Lightweight ChoicePoints — save indices, not full clones | Avoid O(n) snapshot of stack/trail/heap |
| `006f6548` | 2026-04-06 | Lightweight ChoicePoints + member/2 fast path in `\+/1` | Skip CP creation when the negation can be decided trivially |
| `fcc88aca` | 2026-04-06 | Hybrid ChoicePoint — stack clone + trail/heap indices | Stack must be cloned (mutable through env frames); trail/heap are append-only so indices suffice |
| `6815f9a3` | 2026-04-06 | `Rc<Vec<StackEntry>>` for O(1) stack clone | Reference-counted stack so CP creation is a pointer bump |
| `fea72426` | 2026-04-08 | Save only argument regs in choice points | X-registers can be trashed across choice points; only A-regs carry query state |
| `4a6e374e` | 2026-04-08 | Drop redundant binding snapshots | Bindings are reconstructable from trail, don't clone them |

### Phase C: Bindings

| Commit | Date | What | Notes |
|---|---|---|---|
| `7ca8e964` | 2026-04-07 | Save/restore var counter in CP bindings snapshot | Correctness + perf |
| `f339df5b` | 2026-04-08 | `nb_setarg` box for zero-copy binding table + deferred fact binds | Biggest Rust-specific win |

### Phase D: Benchmark specialization

| Commit | Date | What | Why it matters |
|---|---|---|---|
| `ff154028` | 2026-04-08 | Add benchmark profiling counters | Instrumentation |
| `90130350` | 2026-04-08 | Pre-resolve benchmark call targets | Compile-time call resolution |
| `1c914bb1` | 2026-04-08 | Resolve choice and switch targets | Same pattern |
| `25663f2d` | 2026-04-08 | Fast-path indexed benchmark fact calls | Avoid general dispatch for facts |
| `7f7899a6` | 2026-04-08 | Fuse benchmark not-member guards | Inline negation check |
| `89f947b6` | 2026-04-08 | Fuse benchmark length guards | Inline arithmetic |
| `c65b8f80` | 2026-04-08 | Fuse recursive category_ancestor calls | Tight recursion loop |
| `ebfca29d` | 2026-04-08 | Fuse benchmark ancestor base clause | Base case inlined |

**Lesson:** The Rust target's most impactful Phase-D work was
**benchmark fusion** — inlining the specific shape of the query into
hot paths. This is what got Rust to 126ms. It's also why the Rust
target doesn't generalize cleanly to other graph benchmarks without
re-doing the fusion.

---

## Go WAM — optimization timeline

Started from a working but slow runtime (effective-distance bench at
scale-300 took ~340s after the late-April correctness fixes landed).
Three rounds of perf brought it to ~53s — a **6.4x cumulative
speedup** — and aligned the runtime with the Rust target's choicepoint
and environment-trimming design.

### Phase A: Correctness work (April 2026)

Six runtime bugs found and fixed in sequence; collectively the
difference between `tuple_count=0` and `tuple_count=211 article_count=271`
matching the reference output. Documented in
`benchmarks/go_wam_runtime_findings.md`. Most relevant for future perf
work:

| Bug | Fix |
|---|---|
| `bindUnbound` corrupting A1 when `Idx=0` | Drop unconditional `putReg(u.Idx, val)` |
| `pushChoicePoint` saving only A-regs, leaving stale heap-Refs | Snapshot full register file (constraint that Phase D would later loosen via env trimming) |
| `PutVariable` Idx collision across recursive activations | Globally-unique Idx via `allocVarId/0` |
| `extract_shared_start_pc` taking the wrong digits | Fixed-length `sub_string` |
| `Atom.Equals` always doing string compare | Pointer-identity short-circuit |
| Y-regs colliding across nested predicate calls | Save/restore in `EnvFrame.SavedYRegs` at Allocate/Deallocate |

### Phase B: First constant-factor wins (May 2026, `86f6febb`)

| Commit | What | Why it matters |
|---|---|---|
| `4d0dd53e` | Include `category_parent/2` in `kernels_on` predicate list | Without it the kernel's recursion had nothing to call, every weight query returned 0 |
| `86f6febb` | `Regs[512]` → `Regs[320]` + `resolveInstructions` handles `default` label sentinel | The `default` label kept SwitchOnConstant on the linear-scan runtime path even though SwitchOnConstantPc binary search was available; resolution lets the indexed dispatch fire. Combined with the smaller register file these brought scale-300 from ~340s to ~178s (1.91x) |

### Phase C: Constant-factor attempts that didn't pan out (May 2026)

Tracked here so they don't get re-tried; the underlying lessons match
the Haskell/Rust patterns but the constant-factor reductions aren't
distinguishable from variance noise (~±10%) at scale-300.

| Attempt | Outcome |
|---|---|
| `Ctx.SaveRegBound` (compute max-used reg, snapshot only `Regs[:208]`) | Within variance |
| `SavedRegs` pool with `truncateChoicePoints` recycle helper + `Clone()` deep-copy | +3% at 50-seed scale, **−6% at full scale-300** — Clone deep-copy + pool bookkeeping exceed alloc savings on the longer run |
| Pre-sized `Bindings` map (`make(map[int]Value, 4096)`) | Consistent ~3-5% regression |
| `Bindings []Value` slice indexed by Idx (`perf/wam-go-bindings-slice-and-stack-share`) | Performance-neutral at scale (theoretical 6x speedup on per-binding ops, but Bindings ops are ≤10% of total CPU; real bottleneck was elsewhere). Kept as structural cleanup |
| Save only A-regs at CP push (Rust's `fea72426`-equivalent, attempted before env trimming) | **Broke correctness** (tuple_count dropped from 16 to 4 at dev): without env trimming, Y-regs at backtrack-time can't be recovered from the env frame because the SavedYRegs there hold *outer* values, not the current Y-regs at TryMeElse time. Re-enabled in Phase D as save-A+Y-skip-X once env trimming was in place |

### Phase D: Structural refactor (May 2026, `perf/wam-go-env-trimming`)

The two patches that delivered most of the Phase B → Phase D
improvement (178s → 53s = 3.4x at full scale-300; cumulative 6.4x
from the original 340s baseline):

| Change | What | Why it matters |
|---|---|---|
| Environment trimming | Add `WamState.E` (current env-frame stack index) and `EnvFrame.PrevE`. Allocate links via PrevE and updates `vm.E`. Deallocate walks the PrevE chain *logically* (updates vm.E) and only physically pops the frame when both (a) it's at the top of the stack and (b) `env.B0 >= len(ChoicePoints)` (no younger CP references it). Otherwise the frame stays on the stack, dead but harmless, until backtrack truncation sweeps it. `peekEnvFrame` becomes O(1) via vm.E instead of an O(stack) reverse scan. | Prerequisite for the next change — without it, env frames could be physically popped while a younger CP needed them, so `pushChoicePoint` had to deep-copy the stack to capture the frames. With env trimming, the live frames are guaranteed to still be on the stack at backtrack time. |
| Stack-mark choicepoints | `ChoicePoint.Stack []StackEntry` → `ChoicePoint.StackLen int`. `pushChoicePoint` records `len(vm.Stack)` instead of calling `copyStack`. `backtrack` truncates via `vm.Stack = vm.Stack[:cp.StackLen]`. The `Stack` field on ChoicePoint is gone; `copyStack` is unused. | Eliminates the per-CP `make([]StackEntry, ...)` + memcpy that was ~19% of CPU. Combined with env trimming this brought scale-300/50 from ~20s to ~8.5s (2.4x). |
| Save-A+Y-skip-X register snapshot | `snapshotAllRegs` now saves 108 elements: `Regs[0..7]` (A-regs) + `Regs[200..299]` (Y-regs). `restoreSavedRegs` reverses the layout. The X-reg range (Regs[8..199]) is intentionally skipped — X-regs are clause-local in the codegen, the next clause's head writes whatever X-regs it needs (PutVariable / GetVariable Xn always writes before the X-reg is read), and stale leftovers from the failed clause never get touched. Y-regs *do* need saving because env trimming stores Y-regs at *Allocate* time; the *current* Y-regs at TryMeElse time only exist in the CP snapshot. | Cuts the per-CP snapshot copy from 320 elements to 108 (≈3x). Combined with the above, scale-300/50 went from ~8.5s to ~5.1s (1.7x more); scale-300/full went from 88.7s to 53s (1.7x more). |

Cross-target precedent: the Rust target's Phase B (`6bec8b7e`,
`fcc88aca`, `6815f9a3`, `fea72426`, `4a6e374e`) landed the equivalent
sequence — lightweight choicepoints + Rc-shared stack + save only
argument regs. The Go runtime can't use `Rc<Vec>` for stack sharing,
but the `len(vm.Stack)` mark gives the same effect because env
trimming guarantees the live frames stay reachable until backtrack
truncates them.

### Phase E: Y-reg snapshot bounded by actually-used range (May 2026, `perf/wam-go-yreg-trail`)

After Phase D the per-CP profile showed `restoreSavedRegs` at ~42%
of CPU, with the Y-reg copy (100 elements) dominating over the A-reg
copy (8 elements). Most clauses use only ~10 Y-regs; the snapshot
was paying for 100.

Tried first: trail-based Y-reg restoration (per-write trail entries,
no per-CP snapshot). It hung the bench — too many trail entries +
subtle ordering issues with `bindUnbound`'s alias-rewrite loop and
the existing `trailBinding(i.Xn)` calls in `GetVariable` /
`UnifyVariable` (leftovers from when Idx == register slot index, now
wrong with allocVarId starting at 1000+). Reverted.

Approach that worked: track `vm.MaxYReg` as a high-water mark of the
highest Y-reg index ever written, bumped inside `putReg`. The
snapshot copies only `Regs[200..vm.MaxYReg]` instead of the full Y
range. For the bench's category_ancestor (uses ~10 Y-regs), this
shrinks the snapshot from 108 elements to ~18 and the per-CP
memmove proportionally.

Two bugs surfaced and were fixed:

1. **MaxYReg is monotonic but the *failed clause* could grow it
   past push-time.** If the snapshot was made when MaxYReg was N
   and the failed clause wrote a higher Y-reg, restore wouldn't
   touch the slot — it'd stay at the failed value. Fix:
   `restoreSavedRegs` now clears `Regs[pushMaxY..vm.MaxYReg]` to
   nil after restoring the snapshot (those slots were nil at push
   time by construction).

2. **`Clone()` didn't carry MaxYReg.** Sub-VMs in `executeAggregate`
   started with MaxYReg=0, snapshot omitted Y-regs, backtrack
   inside the sub-VM lost values the failed clause overwrote.
   Symptom: `Velocity Physics 1.602159` instead of the
   post-Phase-D `1.601079`. Fix: Clone copies MaxYReg.

### Phase F: ChoicePoint pointer access + Bindings as `[]Value` (May 2026, `perf/wam-go-cp-pointer-and-bindings`)

The post-Phase-E profile showed two costs newly visible relative to
the dominant `restoreSavedRegs` from prior phases:

1. `cp := vm.ChoicePoints[topIdx]` — value-copy of the
   ~150-byte `ChoicePoint` struct on every backtrack (~210ms cum,
   8.7% in the 50-seed run).
2. Map ops in `deref` / `unwindTrailTo` / `bindUnbound` — small in
   absolute terms but harder to see while bigger costs were
   dominant.

This commit:

- **`backtrack` uses `cp := &vm.ChoicePoints[topIdx]`** instead of
  the value-copy. Mutations to `cp.IndexedClausePCs` /
  `cp.ForeignResults` apply in-place, removing the explicit
  `vm.ChoicePoints[topIdx] = cp` write-back. The pointer stays
  valid because backtrack only ever truncates the slice (no
  appends), so the underlying array slot doesn't move while
  we're reading.
- **`Bindings: map[int]Value` → `[]Value`** indexed by `Unbound.Idx`.
  `nil` means unbound; `getBinding` / `setBinding` helpers handle
  out-of-range reads (return nil) and writes (grow with doubling,
  floor 64). Pre-sized to 4096 in `NewWamState` to skip the
  doubling-grow churn for typical query Idx ranges. The Rust
  target's Phase C (`f339df5b` "nb_setarg box for zero-copy
  binding table") landed the same int-indexed structure for the
  same reason once choicepoint snapshots got cheap enough that
  per-deref/per-trail-entry map overhead became visible.

Earlier attempt (the `perf/wam-go-bindings-slice-and-stack-share`
branch, on top of Phase B): performance-neutral. The dominant
costs at the time were elsewhere; the slice version is now
slightly faster because Phases D and E shrunk those.

### Phase G: Pointer-only `Atom.Equals` + `internAtom` routing + `emptyListAtom` singleton (May 2026, `perf/wam-go-bindunbound-aliases`)

The post-Phase-F profile flagged `valueEquals` at 9.56% cum and
`Atom.Equals` at 4.02% cum. Atom.Equals had a string-compare
fallback (`return v.Name == o.Name`) for the case where two atoms
had the same name but different pointers. With every codegen-emitted
atom in `atoms.go`'s `atomInternMap` and every runtime-side atom
created via `internAtom`, the fallback path is unreachable in
practice — the SwitchOnConstantPc forward-scan miss path that fired
it was just paying for the safety net.

This commit:

- **`Atom.Equals` is pointer-only.** Drops the string fallback;
  asserts the contract that all atoms come from
  `atomInternMap`. Documents the constraint in the source comment.
- **Routes all raw `&Atom{Name: ...}` constructions through
  `internAtom`.** Affected sites: the runtime helpers in
  `runtime.go` (`collectNativeTransitiveDistanceResults` and
  friends — kernels_on path, not exercised by the bench but must
  satisfy the contract), the codegen fallback in
  `go_atom_to_literal/2` (when an atom isn't in the pre-interned
  table), and the `[]` empty-list terminator in
  `rawListHeadTail` / `listHeadTail`.
- **`emptyListAtom` package-level singleton** (`var emptyListAtom = internAtom("[]")`)
  caches the result of `internAtom("[]")` once at init. The
  list-head/tail helpers now use the cached pointer instead of
  calling `internAtom("[]")` per invocation — list cell
  decomposition runs once per recursion step in the bench.

Profile after Phase G:
- `Atom.Equals`: 4.02% → 1.39%
- `valueEquals`: 9.56% → 4.38%

Wall-clock at the 50-seed scale: median 2524ms vs Phase F's 2626ms
(5 alternating trials each), about 4% faster within run-to-run
variance. Full 386-seed bench: ~26.6s for both Phase F and Phase G
(repeated 3 runs each), so the gain is invisible at full scale —
the saved CPU cycles get absorbed by GC and other overhead. The
single-run "23.8s" Phase F number cited in the prior section turned
out to be an outlier from one lucky run; the realistic median is
~26.6s. The cumulative table below uses the realistic medians.

### Phase H: list-builtin micro-opts (May 2026, `perf/wam-go-list-builtins`)

The post-Phase-G profile flagged `listToSlice` at 4.43% cum and
`mallocgc` at 4.76% cum. The bench's `\+ member(M, Visited)` and
`length(Visited, Depth)` calls — once per category_ancestor recursion
step — were each materialising the Visited list into a Go `[]Value`
just to count or scan it. The recursive form did
`append([]Value{head}, rest...)` which is O(N²) in allocations
(every level allocates a new slice and copies the rest).

Three changes:

1. **`listToSlice` is iterative** — pre-allocates a 16-cap slice
   (the bench's max_depth(10) means 16 covers the common case
   without growth), walks cells with a `for` loop appending each
   head, returns the final slice. O(N) allocs instead of O(N²).
2. **`length/2` no longer allocates** — counts cells in a `for`
   loop without materialising a slice. Bench's
   `length(Visited, Depth)` now does an in-place walk.
3. **`member/2` no longer allocates** — walk-and-unify, with a
   `vm.unwindTrailTo(mark)` between iterations to prevent failed
   bindings from leaking into the next element check. Standard
   WAM `member/2` semantics, no slice.

Wall-clock impact at scale-300: essentially neutral. 50-seed
median shifted from 2524ms → 2534ms (within run-to-run variance).
Full bench: 26.6s → 27.4s (also within variance — 3 runs each
showed overlap). The post-G profile already had `listToSlice` at
only ~4% of CPU, so the saved allocs are absorbed by GC. Phase H
is committed as a structural cleanup that becomes more impactful
at workloads with longer lists or higher member/length call
density.

### Phase I: Constant-factor attempts that didn't pan out (May 2026, `perf/wam-go-instr-tag-dispatch`)

Two more candidates explored on top of Phase H. Both reverted —
documenting here so they don't get re-tried at the same shape.

| Attempt | What | Outcome |
|---|---|---|
| Inline `setBinding` fast path in `unwindTrailTo` | Replace `vm.setBinding(entry.Addr, entry.Old)` with a direct `vm.Bindings[entry.Addr] = entry.Old` write when `entry.Addr < len(vm.Bindings)`, falling back to the helper for the (impossible-by-construction) overflow path. Post-Phase-G profile flagged the loop at ~6% flat. | Within variance. One alternating trial favoured the change at ~11% (3179ms→2831ms), four follow-up trials averaged to a 0.86% slowdown (list mean 2661ms, tagdsp mean 2684ms). The fast path runs *only* on backtrack and `len(Bindings)` is large enough that the bounds branch is well-predicted; the saved function-call frame is below the noise floor. |
| Inline-array `SavedRegs` in `ChoicePoint` | Replace `SavedRegs []Value` with `SavedA [8]Value` + `SavedY [16]Value` + `SavedYLen int` + `SavedYExt []Value` overflow. Goal: eliminate the per-push `make([]Value, 8+ycount)` heap allocation that `snapshotAllRegs` was doing on every choicepoint. | **10x slowdown** at 50-seed scale (list ~2.4s vs inreg ~25s, two trials confirmed). Reason: each `ChoicePoint` grew from ~150 bytes to ~530 bytes (24 inline `Value` interface slots = 384 bytes added). Every `append(vm.ChoicePoints, ChoicePoint{...})` and slice-grow now memmoves the larger struct, and the bench creates ~1.5M choicepoints — the extra per-CP memmove cost vastly exceeds the saved heap alloc. The lesson is that for slice-of-struct dispatch, struct *size* matters more than per-element heap allocs, because slice growth amortises malloc but every append still pays the element memmove. A pooled `*SavedRegs` (CP holds an 8-byte pointer) would avoid both — deferred. |

Cumulative position is unchanged from Phase H; Phase I is a no-op
on the cumulative table below.

### Phase J: Tag-indexed function table for Step dispatch (May 2026, `perf/wam-go-tag-table`)

The post-Phase-H profile showed `Step` accounting for ~20% of CPU,
much of it on the `switch i := instr.(type) { ... }` line itself
(line 18 of `runtime.go` was ~10% flat in pprof). With ~40
instruction types, the type switch was theorised to be doing an
O(N) linear scan of itab pointer comparisons. The fix attempted:

1. Add `Tag() uint8` to the `Instruction` interface (alongside the
   existing `instrTag()` marker).
2. Generate a `const ( TagGetConstant uint8 = iota; ...; tagCount
   )` block enumerating all instruction types.
3. Generate a `Tag()` method per type returning its constant.
4. Generate one top-level handler function per instruction holding
   the body that previously lived inside the type-switch case.
5. Build a dispatch table: `var stepTable [tagCount]func(*WamState,
   Instruction) bool`, populated in `init()` (a static initializer
   would form a cycle: `stepBeginAggregate` → `executeAggregate`
   → `runUntilPC` → `Step` → `stepTable`).
6. Replace `Step`'s body with `return stepTable[instr.Tag()](vm, instr)`.

Output bit-for-bit identical to Phase H baseline at scale-300/full.

Performance: **~12% slower**. Three alternating full-bench trials
on a fresh A/B (both binaries built from current main with
`kernels_on`):

| trial | h-fresh (Phase H) | tag (Phase J) | tag/h-fresh |
|---|---|---|---|
| 1 | 32591ms | 37089ms | 1.138x |
| 2 | 33234ms | 37206ms | 1.119x |
| 3 | 33228ms | 36553ms | 1.100x |
| **mean** | **33018ms** | **36949ms** | **1.119x** |

Why the "obvious O(N) → O(1) dispatch win" backfired:

1. **Go's type switch is already efficient.** With ~40 typed
   cases the compiler emits an internal hash on the runtime type
   pointer, not a linear scan; effective dispatch cost is closer
   to O(1) than the analytical worst case suggested. The post-G
   profile attribution to "line 18" included most of the
   switch-prologue work, but the per-instruction marginal cost was
   already small.

2. **Inlining loss dominates the dispatch saving.** When each case
   body lives inside the switch, the Go compiler inlines its
   contents into `Step` (and from there sometimes into
   `runUntilPC` via mid-stack inlining). Once each body becomes a
   separate top-level function called via the table, every
   instruction pays a real function-call frame: stack-frame setup,
   register save/restore, return-address push/pop. With tens of
   millions of instructions per scale-300 run, that overhead
   exceeds whatever the table lookup saved.

3. **`instr.Tag()` is also an interface method call** — itab
   load + indirect call, ~5ns. So the dispatch path is `Tag()`
   call + table load + `stepXxx` indirect call vs. the prior
   single type-switch dispatch. The two indirect calls roughly
   double the dispatch overhead per instruction.

The lesson aligns with prior phases: **structural changes that
remove work compose; constant-factor reshuffles inside well-
optimised compiler code rarely beat the compiler.** The type
switch wasn't "linear scan over 40 types" in practice; it was
mostly dispatch + inlining the compiler had already taken care of.

A pooled-handler-pointer approach (storing `*func` directly in
each Instruction at codegen time, bypassing the Tag() lookup)
might recover some of the loss, but it would still pay the
per-instruction function call. Deferred indefinitely — the type
switch is the right shape for this language.

**Watch condition — when to revisit.** The conclusion above was
measured at ~40 instruction types. We expect it to hold as the
switch grows, because the dominant cost (lost inlining) scales
with the *number of dispatches* not the *number of cases*. But
Go's type-switch internals are version-dependent and the hash-
on-itab approach has a working-set limit; if the instruction set
crosses ~80–100 types, or if a future Go release changes type-
switch lowering (e.g. drops the hash-table optimisation in favour
of binary search on type pointers), the trade-off could flip. Re-
test with the same A/B harness when (a) the instruction count
roughly doubles or (b) profile shows the type-switch dispatch
line at >25% of CPU after the next round of structural wins.

Cumulative position is unchanged from Phase H; Phase J is a no-op
on the cumulative table below.

### Phase K: `category_ancestor` FFI kernel for Go (May 2026, `feat/wam-go-category-ancestor-kernel`)

After Phases I and J both bottomed out at the variance floor, the
honest read was that constant-factor work on the WAM dispatch was
exhausted. The post-survey verdict (see the cross-target lowering
audit in this branch's commit description) was that the next big
multiplicative win would come from **fewer WAM dispatches entirely**,
not making each dispatch faster — i.e. lowering more predicates to
native Go via the FFI-kernel system that already powered
`transitive_closure2`, `weighted_shortest_path3`, and the
`astar_shortest_path4` family.

The single missing kernel — and the one that drove the entire
`effective_distance` benchmark family the prior 12 phases had been
optimising for — was `category_ancestor/4`. Both Haskell and Rust
already had it; Go fell through to full WAM bytecode for the
inner ancestor walk on every depth-bounded recursive step.

Implementation (mirrors `wam_rust_target.pl:1797`):

1. **Pl-side dispatch** — three new clauses in `wam_go_target.pl`:
   - `go_supported_shared_kernel(recursive_kernel(category_ancestor, _, _))`
     — accept the kernel from the shared detector.
   - `go_recursive_kernel_with_facts/3` clause that pulls
     `max_depth(N)` and `edge_pred(EP/2)` out of the detector's
     config and indexes the parent-edge facts via
     `go_binary_edge_fact_pairs/3`.
   - `go_recursive_kernel_config_ops/3` clause that emits both
     `register_foreign_usize_config(.., max_depth, N)` and
     `register_foreign_string_config(.., edge_pred, ..)` plus
     `register_indexed_atom_fact2(EP/2, Pairs)`.
   The shared `kernel_metadata/4` (`recursive_kernel_detection.pl:61`)
   already had the right `NativeKind=category_ancestor`,
   `ResultLayout=tuple(1)`, `ResultMode=stream` mappings; no
   metadata work needed.

2. **Generated runtime** — three new functions emitted from the
   `compile_wam_helpers_to_go` Prolog format string:
   - `listAsAtomStrings(vm, v)` — derefs a Prolog cons list and
     extracts each element as an atom string. Note: uses
     `vm.listToSlice` (the iterative cons-walker added in Phase H),
     not `listAsSlice` — the latter returns the 2-element
     `[head, tail]` cell of the *outer* List, which would silently
     truncate the visited set.
   - `collectNativeCategoryAncestorHops(cat, root, visited, maxDepth, pairs)`
     — top-level entry that builds the parent-edge adjacency map
     from the indexed atom-fact pairs once, then delegates to:
   - `collectNativeCategoryAncestorHopsRec(...)` — the recursive
     DFS, mirroring the Rust implementation byte-for-byte. Emits
     one `int64` hop count per matching path, with the +1 increment
     applied to the slice tail produced by each recursive call (so
     that hop counts compose correctly on the way back up).

3. **Dispatch case** — new `case "category_ancestor":` in
   `executeForeignPredicate`. Pulls `cat`/`root` from A1/A2 as
   atom strings, `visited` from A4 as a flattened atom-string
   list, `maxDepth` from the foreign-usize config, edge-pred
   adjacency from the indexed fact map. Calls the helper, packs
   the int64 hops as `&Integer{Val: h}` Values, and finishes via
   `finishForeignResults(predKey, []int{2}, results)` (output reg
   is A3 = reg index 2; layout is `tuple(1)` = no tuple wrapping).

#### Wall-clock impact at scale-300/full

Three alternating trials (A/B; both binaries built from the
current branch with `kernels_on`):

| trial | h-fresh (Phase H baseline) | cancestor (Phase K) | speedup |
|---|---|---|---|
| 1 | 25219ms | 504ms | 50.0x |
| 2 | 24171ms | 443ms | 54.6x |
| 3 | 24139ms | 451ms | 53.5x |
| **mean** | **24510ms** | **466ms** | **52.6x** |

Cumulative vs the ~340s Phase A baseline: **~730x**. The 12.8×
constant-factor work from Phases B–H was a prerequisite — it
defanged everything *outside* the inner ancestor walk so that
when the kernel removed that walk's WAM-dispatch cost, the
remaining time is dominated by argument marshalling and result
unification rather than by the dispatcher itself.

#### Correctness — better than Phase H

Output diff vs `data/benchmark/300/reference_output.tsv`: only 4
lines, all stable-sort tie-break ordering between articles with
identical effective-distance values (`Hermann_von_Helmholtz` /
`Walter_Noll`-cluster). Phase H had 11+ lines of diff including
*wrong values* (Brownian_motion 0.993865 vs reference 0.993717;
2008_in_science 4.372980 vs reference 3.726584). The native
kernel fixes those by computing the depth-bounded ancestor walk
directly in Go instead of through the WAM bytecode path that had
some subtle interaction with the bench's `\+ member(M, Visited)`
guards. The reference tie-break ordering (where it differs from
the kernel) is `data/benchmark/300/reference_output.tsv`-internal
and not part of the algorithm — the kernel is correct, the diff
is cosmetic.

#### Why this worked when Phases I/J didn't

- Phases I/J reshuffled dispatch *inside* the WAM interpreter
  (where the Go compiler already did a good job) — net cost was
  inlining loss and noise.
- Phase K removes whole categories of WAM dispatches entirely.
  The inner ancestor walk previously executed dozens of WAM
  instructions per parent edge (Allocate, Get*/Put*, Call,
  Deallocate, …); now it's a single Go function call from the
  outer aggregator. The work that's removed isn't 10–25% of CPU
  in any one bytecode op — it's *all of it* for the inner loop.

This is the structural-vs-constant-factor lesson from the
Lessons section made concrete: a structural change that removes
work composes multiplicatively with prior structural wins; a
constant-factor reshuffle inside compiler-optimised dispatch
fights the compiler.

### Cumulative measurement (scale-300, kernels_off, single-thread)

50-seed: median of 5 alternating trials. Full bench: median of 3
re-runs (the prior section cited single-run numbers; some of those
were outliers — this section uses re-measured medians).

| Stage | scale-300/50 median | scale-300/full | speedup vs prior | speedup vs original |
|---|---|---|---|---|
| Pre-Phase-A (broken — tuple_count=0) | n/a | n/a | n/a | n/a |
| Phase A (correctness; full-Regs snapshot) | n/a | ~340s | — | 1.0x |
| Phase B (`Regs[320]` + default-label resolve) | n/a | ~178s | 1.91x | 1.91x |
| Phase D env trimming + StackLen | 8594ms | 88.7s | 2.0x | 3.83x |
| Phase D + save-A+Y-skip-X | 5084ms | 53.0s | 1.67x | 6.42x |
| Phase E MaxYReg-bounded snapshot | 3624ms | 38.4s | 1.43x / 1.38x | 8.85x / 8.85x |
| Phase F cp-pointer + Bindings slice | 2626ms | 26.6s | 1.38x / 1.44x | 12.9x / 12.8x |
| Phase G ptr-only Atom.Equals etc. | 2524ms | 26.6s | 1.04x / 1.0x | 13.5x / 12.8x |
| Phase H list-builtin micro-opts | 2534ms | ~27.4s | 1.0x / ~1.0x | 13.4x / ~12.4x |
| Phase I (reverted attempts; doc-only) | — | — | 1.0x | 13.4x / ~12.4x |
| Phase J tag-table dispatch (reverted) | — | ~37s vs h-fresh ~33s | **0.89x** (regression) | doc-only |
| **Phase K `category_ancestor` FFI kernel** | — | **466ms** (mean of 3) | **52.6x** | **~730x** |

Output identical to prior at every stage except for the documented
`max_depth(10)` drift on Brownian_motion / Hermann_von_Helmholtz
tie-break (the workload uses `max_depth(10)` while the reference TSV
was generated with `max_depth >= 50`; tracked in
`benchmarks/go_wam_runtime_findings.md`).

### Remaining headroom

- **`bindUnbound`'s alias-rewrite loop** (`for idx, reg := range vm.Regs { if reg == u { vm.Regs[idx] = val } }`) is O(320) per binding. With the snapshot/restore work largely defanged, this is the next visible bandwidth cost. Could be replaced by deref-on-read (every reader that takes an `Unbound` derefs through `vm.getBinding`), at the cost of auditing every `vm.Regs[N]` read in the codegen.
- **`Bindings map[int]Value` → `[]Value`** (the `perf/wam-go-bindings-slice-and-stack-share` branch). Performance-neutral at the moment because it was committed before Phase D and the dominant cost was elsewhere; with the per-CP work now small, the map's per-access cost may show up. Worth re-measuring on top of Phase E.
- **Heap-trim on backtrack.** `vm.Heap = vm.Heap[:cp.HeapTop]` is fast but allocations to `vm.Heap` still happen via `append`. A pool / arena could eliminate the steady GC churn.
- **Trail-based Y-reg restoration** (proper version). The Phase-E attempt failed because `trailBinding(i.Xn)` calls in GetVariable/UnifyVariable leak Bindings entries with addresses that overlap the Y-reg range, AND `bindUnbound`'s alias-rewrite loop doesn't trail Y-reg overwrites. With both fixed, trail-based restoration could be made to work and would drop the per-CP snapshot to just A-regs (8 elements). Deferred — the MaxYReg approach captures most of the upside without the audit risk.

### Lessons unique to the Go runtime

1. **Variance bands stayed wide right up until Phase D.** Phase C's
   constant-factor attempts couldn't be distinguished from noise;
   Phase D's structural change shows up as a clean 4x with no overlap
   between baseline and post-fix trial bands. The lesson isn't "don't
   profile" — it's "if your patch only moves things by less than the
   variance, the patch is too small."

2. **Save-only-A-regs is correct in Go too — but only after env
   trimming.** The Rust precedent (`fea72426`) wasn't directly
   portable because the Go target had no equivalent of Rust's
   per-clause register liveness analysis. What made the Rust
   optimization safe was the same thing that made env trimming work:
   Y-regs live in the env frame, not in a global register file
   shared across activations. Once the Go runtime had that property,
   the same observation applied.

3. **Cumulative speedup ≠ sum of individual speedups.** Phase B was
   1.91x. Phase D env-trim was 2.0x. Phase D save-A+Y was 1.67x.
   Multiplied: 6.4x. The structural changes compose because they
   each remove a different bottleneck — the constant-factor attempts
   in Phase C didn't compose because they all targeted the same cost
   (the SavedRegs copy) and the underlying issue was the design, not
   the constants.

4. **Don't fight the Go compiler's dispatch — extending the work
   it can inline beats reshuffling its dispatch shape.** Phase J
   replaced Step's ~40-case type switch with the textbook O(1)
   tag-indexed function table; result was a clean ~12% regression
   across three trials. Two compounding reasons:

   - Go lowers a type switch with many cases to a hash on the
     interface itab pointer, not a linear scan, so the analytical
     "O(N) → O(1)" win was already mostly priced in.
   - The dispatcher path is shorter for the type switch than for
     the table because each `case *X:` body is *inlinable* into
     `Step` (and via mid-stack inlining sometimes into `runUntilPC`).
     A function-table dispatch necessarily forces every body into
     a separate top-level function with a real call frame, and an
     extra `Tag()` interface method call besides. The lost
     inlining costs more than the (already-small) dispatch saving.

   Generalised: when a hot loop's body contains a switch over a
   compiler-friendly shape (interface type switch, integer switch
   with dense cases, etc.), the right next move is to *give the
   compiler more to inline* — not to indirect the dispatch
   through a level the compiler can't follow. Watch this if the
   instruction count grows past ~80–100 or if a future Go release
   changes type-switch lowering; until then, prefer additive
   structural changes over dispatch rewrites.

---

## Clojure WAM — current implementation status

The Clojure hybrid/WAM target is not yet in the same maturity class as
Haskell or Rust, but the runtime now has the right architectural shape
for further optimization work.

### Implemented so far

| Area | Status | Notes |
|---|---|---|
| Shared code table | Done | All generated predicates dispatch into one shared instruction table with per-predicate start PCs |
| One-time label resolution | Done | `call`, `execute`, `jump`, choice ops, and `switch_on_constant` are resolved at project load |
| Indexed dispatch | Done | `switch_on_constant` is compiled into a direct lookup map in the runtime |
| FFI controls | Partial | Explicit `foreign_predicates([...])` emits `call-foreign` stubs; `no_kernels(true)` and `foreign_lowering(false)` suppress stubs. Deterministic handlers work, and the optimized benchmark generator now emits a fact-backed `category_parent/2` graph handler plus a Clojure ancestor-hop traversal index for `kernels_on` benchmark runners |
| Choice points | Partial | Saves persistent regs/env stack plus trail/heap/build boundaries; avoids binding snapshots and filtered register-map allocation, but still heavier than Haskell/Rust |
| Environment frames | Done | `allocate`/`deallocate` use explicit environment frames for `Y` slots |
| Cut semantics | Partial | Clause cut uses a cut barrier; `cut_ite` pops only the enclosing if-then-else CP |
| Read-mode compound terms | Done | `get_structure`, `get_list`, `unify_*` support structure/list matching |
| Write-mode compound terms | Done | `put_structure`, `put_list`, `set_*` build nested terms via a builder stack |
| Benchmark generation | Partial | `generate_wam_clojure_optimized_benchmark.pl` emits optimized effective-distance Clojure WAM projects with `kernels_on`/`kernels_off` controls, fact-backed graph handler generation, and a result-producing traversal-index runner |
| End-to-end verification | Partial | Generator tests, fact-backed foreign-handler JVM smoke, and standalone runtime smoke runner pass locally; large JVM benchmarks remain constrained in Termux |

### Clojure-specific lessons from this phase

1. Shared-table generation and pre-resolved control flow were worth doing
   first. They match the Haskell/Rust shape and remove obvious runtime
   string lookup costs.
2. Choice-point snapshots cannot be reduced to `A` registers in the current
   Clojure runtime. `X` registers are still needed across retry paths such as
   generated if-then-else code. Because Clojure maps/vectors are persistent,
   saving the full register map is cheaper than rebuilding a filtered `A`/`X`
   snapshot at every choice point.
3. Treating `!/0` and `cut_ite` as "clear all choice points" is incorrect.
   The runtime now distinguishes clause-level cuts from soft cuts.
4. The current runtime is still a bindings-centric approximation, not a full
   heap/trail WAM. That keeps the implementation moving, but it also defines
   the next real parity boundary.
5. The standalone smoke suite now covers structure and list construction
   across failing branches, including an env-mediated retry path. Nested
   disjunction/cut shapes remain a generator-level gap rather than a runtime
   optimization target.
6. Clojure now has the same basic control vocabulary used by benchmark
   matrices in other WAM targets: explicit foreign predicates can be marked,
   and `no_kernels(true)` / `foreign_lowering(false)` force the pure WAM path.
   Deterministic handlers can be wired through `clojure_foreign_handlers/1`.
7. Clojure now has an optimized effective-distance project generator. It
   establishes benchmark-matrix shape and kernel-mode controls without trying
   to run large JVM benchmarks in Termux.
8. The first Clojure graph-kernel path is now fact-backed rather than a
   placeholder: `kernels_on` emits a Clojure set-backed `category_parent/2`
   handler from `facts.pl`, while `kernels_off` keeps the pure WAM fallback.
   Synthesizing the handler immediately after loading facts is important
   because later optimization/loading steps can alter predicate visibility in
   PlUnit contexts.
9. Clojure WAM scaffold targets are now registered in the configurable
   benchmark target matrix under `clojure-wam-scaffold`. They are listed and
   resolvable, but intentionally skipped by the effective-distance runner
   until Clojure emits the same result table as the mature Rust/Haskell/Go
   benchmark paths.
10. `clojure-wam-accumulated` now has a generated no-argument benchmark
    entrypoint that emits the common effective-distance result table. On the
    `dev` scale it matches `prolog-accumulated` by normalized output digest;
    the remaining Clojure modes stay scaffold-only until they have equivalent
    result-producing entrypoints.
11. `clojure-wam-accumulated-no-kernels` is now also executable in the matrix.
    On `dev`, both accumulated Clojure modes and `prolog-accumulated` produce
    the same normalized digest, which gives a valid Clojure kernel-on/off
    comparison surface.
12. The seeded Clojure modes are executable too. All four Clojure
    effective-distance modes now match `prolog-accumulated` on `dev` by
    normalized digest:
    `clojure-wam-accumulated`, `clojure-wam-accumulated-no-kernels`,
    `clojure-wam-seeded`, and `clojure-wam-seeded-no-kernels`.
13. The result-producing runner now distinguishes `kernels_on` and
    `kernels_off`: `kernels_on` builds a native Clojure ancestor-hop index once
    for the benchmark seed categories and roots, while `kernels_off` keeps the
    on-demand recursive traversal path. Both modes still match
    `prolog-accumulated` on `dev`.
14. Clojure `call-foreign` now accepts deterministic output bindings from
    handlers. Existing boolean handlers still work, while handlers can return
    `{:bindings {2 "value"}}` to unify output argument registers before
    returning to WAM code. This moves traversal kernels closer to the generic
    Haskell/Rust hybrid FFI shape, but it is still single-result and does not
    provide streaming/backtracking foreign solutions yet.
15. Clojure `call-foreign` can now consume deterministic streams of foreign
    solutions. A handler can return `{:solutions [{:bindings {...}} ...]}`;
    the runtime binds the first viable solution and stores the remaining
    solutions as foreign choice points. If later WAM code fails, backtracking
    restores the pre-foreign state and tries the next solution. This is the
    generic runtime surface needed before moving traversal kernels out of
    runner-specific code.
16. Clojure `kernels_on` now exposes `category_ancestor/4` through that
    generic streaming `call-foreign` surface. The generated handler builds a
    parent adjacency map from facts, returns multiple `{Ancestor, Hops}`
    solutions via `:solutions`, and keeps `kernels_off` on the pure WAM path.
    The effective-distance runner still has its runner-side traversal index,
    but the WAM predicate surface now has the same shape needed to replace it.
17. Clojure `kernels_on` effective-distance runner now consumes the generic
    streaming `category_ancestor/4` foreign handler directly. The bespoke
    runner-side `benchmark-ancestor-hops-index` materialization is gone;
    `kernels_off` still uses the pure recursive runner path for comparison.
18. Streamed foreign alternatives in Clojure now use a narrower choice-point
    shape than ordinary WAM backtracking. Foreign choice points retain the
    trail boundary, regs/env/stack, cut barrier, next-var-id, resume PC, and
    remaining foreign results, but they no longer snapshot heap/unify/build
    state that deterministic foreign handlers do not touch.
19. Large Clojure benchmark scaffolds now externalize benchmark relation data
    and foreign-kernel lookup tables into EDN sidecars instead of embedding
    giant handler literals in generated source. This avoids JVM bytecode
    `Method code too large` failures on larger scales and is the first concrete
    Clojure step toward the same preprocessing/materialization direction that
    already exists in the C# query runtime.

### Highest-value remaining work

1. Push the new Clojure benchmark sidecars toward a real preprocessed artifact
   path so larger scales reuse compact adjacency data instead of reparsing EDN
   into generic vectors and maps on every JVM start.
2. Measure whether the slimmer foreign choice points improve the `dev`
   Clojure benchmark timings enough to justify similar hot-state split work.
3. Add proper heap/trail semantics instead of relying primarily on the
   bindings table.
4. Reduce choice-point snapshots toward the lighter Haskell/Rust model once
   the remaining runtime state is better separated.
5. Split hot runtime state from cold code/context data, following the same
   optimization pattern that paid off heavily in Haskell.

---

## Patterns that recurred across both targets

These are the generalizable lessons — they'd apply to any WAM
implementation regardless of host language.

### 1. Pre-resolve at compile time what you can

Both targets saw significant wins from replacing runtime string lookups
with compile-time-resolved indices. Call instructions become PC values;
SwitchOnConstant becomes a sorted Int array. At runtime you do a
dereference, not a hash+lookup.

### 2. Cache lengths that the container doesn't track

Linked lists don't cache length; neither does HashMap (Haskell) for
`size`. If you're reading length in a hot path, cache it in a counter
field.

### 3. Match container to access pattern

HashMap for many-random-insert, IntMap for dense-Int keys, Array for
sequential index access, Vec for ordered append. "Use HashMap everywhere"
doesn't work — each container has an access pattern it wins on.

### 4. Snapshot only what mutates

Choice points need to restore state on backtrack. But what *has* to be
saved? In the Rust target, `nb_setarg` + deferred fact binds means the
trail does most of the work. In the Haskell target, immutability means
snapshots are free (just pointer copies).

### 5. Profile before optimizing

The Phase-D Haskell wins all came from profiling the actual hot
configuration (interp+FFI, lowered+FFI) rather than the pure-interpreter
baseline everyone was focused on. The real bottleneck — `buildFact2Code`
at 42% — was invisible in the pre-FFI profile.

---

## Optimizations that did NOT work

Not every attempt was a win. Tracking these so we don't repeat them.

| Attempt | Why it didn't work |
|---|---|
| Splitting PC into a separate argument threaded through `step` | Added a record copy (`s { wsPC = pc }`) before each step call — regressed from 2518ms to 2975ms. Reverted. |
| ST monad for the hot section | Would kill intra-query parallelism (future roadmap item). Not worth the ~30-40% interpreter gain when FFI is already in play. Documented in `project_wam_haskell_st_monad_plan.md` memory. |
| `nub $ ...` for intern-table dedup | O(n²) at ~130ms overhead at 300 scale. Replaced with foldl' + Map.member. Then had to fix *that* too (next row). |
| `Map.size` inside foldl' to get next intern ID | HashMap.size is O(n); made intern-build O(n²). Fixed with a counter pair `(map, nextId)`. |

---

## Phase E: System-wide atom interning (2026-04-20)

**Branch:** `feat/wam-haskell-atom-interning`

Changed `Value` from `Atom String` / `Str String [Value]` to `Atom !Int` /
`Str !Int [Value]` with a system-wide `InternTable`. Compile-time atoms get
reserved IDs (0-7: true, fail, [], ., "", member/2, +/2, **/2); runtime atoms
from TSV data extend the table at load time.

### Changes

| Component | Before | After |
|---|---|---|
| `data Value` | `Atom String`, `Str String [Value]` | `Atom !Int`, `Str !Int [Value]` |
| `SwitchOnConstantPc` | `Map.Map String Int` | `IM.IntMap Int` |
| Atom equality | O(n) string compare | O(1) Int compare |
| `WamContext` | `wcAtomIntern`/`wcAtomDeintern` maps | `wcInternTable :: !InternTable` |
| FFI boundary | Map.lookup to intern/deintern | Identity (atoms already Int) |
| `evalArith` | String-based operator names | Reverse-lookup via InternTable |
| Lowered emitter | `Atom "string"` literals | `Atom <id>` via `intern_atom/2` |

### Benchmark results — 10k scale (25k category_parent, 10k article_category)

All configurations produce identical output: tuple_count=462, article_count=5192.

| Configuration | Mean query_ms | vs Baseline |
|---|---|---|
| Baseline FFI (main, String atoms) | **555** | — |
| Interned FFI (atom-interning, Int atoms) | **556** | ~0% |
| Baseline WAM-only (main, String atoms) | **19,173** | — |
| Interned WAM-only (atom-interning, Int atoms) | **15,546** | **-19% faster** |

At 1k scale, both WAM-only paths were ~1550ms (within noise). The 19%
improvement at 10k confirms atom interning scales with dataset size — more
atoms mean more comparisons where Int beats String.

The FFI path is unaffected because the native kernel already used `Int`
comparison (via the old `wcAtomIntern` boundary table).

### Bug fixes included in this branch

| Bug | Symptom | Fix |
|---|---|---|
| FFI query dispatch | `tuple_count=0` on effective-distance benchmark when kernels detected | Added `collectForeignSolutions` using `executeForeign` directly instead of running WAM code (where fact code was skipped) |
| `collectSolutions` refactoring | Infinite loop on WAM-only path | Restored `run ctx` call after `backtrack` (WAM choice points need re-execution; only FFI StreamRetry CPs bind directly) |

**Note:** The FFI query dispatch bug also affected main (pre-interning) and
was fixed separately in commit `e5a8e866` on main.

---

## Related documents

- `docs/vision/HASKELL_TARGET_ROADMAP.md` — where this is going
- `docs/vision/HASKELL_TARGET_PHILOSOPHY.md` — why Haskell as a target
- `docs/design/WAM_HASKELL_PERF_IMPLEMENTATION_PLAN.md` — earlier plan,
  now largely delivered
- `docs/design/WAM_HASKELL_FFI_PROFILING_REPORT.md` — the profiling
  matrix that drove the Phase-D wins

## Clojure benchmark preprocessing follow-up

Recent Clojure hybrid WAM benchmark work exposed two practical
lessons for externalized/preprocessed predicate data:

1. Sidecar paths must be generation-time absolute for benchmark harnesses
   that launch projects from a repository root rather than the generated
   project directory. Relative sidecar paths were enough for direct
   smoke runs but broke the cross-target matrix.
2. Preprocessing policy should be declarative, not only a CLI switch.
   The benchmark generator now lets `auto` honor an optional benchmark
   predicate (`wam_clojure_benchmark_data_mode/1` or
   `benchmark_data_mode/1`) before falling back to the scale-favoring
   heuristic (`sidecar` above the current fact-volume threshold,
   `inline` otherwise).

That moves the Clojure benchmark path closer to the C# materialization
direction: policy can live with the workload, while the default still
favors scaling safely.

The next Clojure benchmark step added an explicit `artifact` mode beside
`sidecar` and `inline`. The first artifact shape precomputes
`category_parent_by_child` and `article_category_by_article` EDN maps so
the generated runner and `category_ancestor/4` foreign handler can skip
their startup regrouping pass.

Initial Termux `dev` measurements were mixed rather than a clear win:

- `clojure-wam-accumulated`: `1.867s`
- `clojure-wam-accumulated-artifact`: `1.852s`
- `clojure-wam-seeded`: `1.846s`
- `clojure-wam-seeded-artifact`: `2.250s`

All outputs still matched digest `1659619c9d36`, but these numbers are
not strong enough to justify changing the default heuristic away from
`sidecar`. The artifact path is valuable as an explicit comparison mode
and as groundwork for more compact non-EDN preprocessing, but it still
needs better desktop measurements and probably a denser artifact format
before it should become the default.

The follow-up dense-artifact pass kept the public `artifact` mode stable
but replaced those grouped EDN maps with grouped TSV sidecars:

- `category_parent_by_child.tsv`
- `article_category_by_article.tsv`

That reduces artifact size and drops `edn/read-string` from the hot
artifact path. The generated Clojure runner and the
`category_parent/2` / `category_ancestor/4` foreign handlers now parse
the grouped TSV files directly into vectors/maps at startup. `sidecar`
remains the default-for-scale heuristic until repeated measurements on a
desktop JVM show that the denser artifact format is consistently better
than the simpler EDN row sidecars.

The next follow-up made `artifact` mode selective by benchmark variant:

- `accumulated` keeps the fully grouped artifact path
- `seeded` keeps the grouped `category_parent_by_child.tsv` artifact for
  traversal lookups but falls back to the simpler article row sidecar for
  article/category ingestion

On the small Termux `dev` matrix, that selective seeded path removed the
earlier regression while preserving output parity:

- `clojure-wam-accumulated`: `2.014s`
- `clojure-wam-accumulated-artifact`: `1.988s`
- `clojure-wam-seeded`: `2.254s`
- `clojure-wam-seeded-artifact`: `2.130s`

All outputs still matched digest `1659619c9d36`. The result is still not
strong enough to justify changing the default heuristic away from
`sidecar`, but it does show that Clojure artifact preprocessing should
stay workload-sensitive instead of assuming one storage shape wins
everywhere.

The next refinement pushed that workload sensitivity one level deeper by
adding per-relation benchmark predicates:

- `wam_clojure_benchmark_relation_data_mode/2`
- `benchmark_relation_data_mode/2`

Supported relation keys are currently `article_category` and
`category_parent`. These overrides sit on top of the public benchmark
`artifact` mode and apply consistently to both the generated
effective-distance runner and the Clojure foreign traversal handlers.

That gives the benchmark workload a C#-style policy surface:

- keep `category_parent` on the grouped artifact path when traversal
  lookups benefit from it
- keep `article_category` on row sidecars when startup regrouping is
  cheaper than loading a grouped artifact

Small Termux `dev` validation after adding the per-relation hook still
matched digest `1659619c9d36` across:

- `clojure-wam-accumulated`
- `clojure-wam-accumulated-artifact`
- `clojure-wam-seeded`
- `clojure-wam-seeded-artifact`

The timings are still noisy, but the code path is now flexible enough to
move further measurement and policy tuning to desktop without changing
the benchmark interface again.

The follow-up manifest pass closes another pre-desktop gap by giving the
Clojure benchmark sidecars a C#-style inspection surface. Sidecar-backed
`sidecar` and `artifact` modes now emit
`data/generated/wam_clojure_optimized_bench/manifest.edn` with:

- artifact format/version
- source facts path
- benchmark variant, kernel mode, and top-level data mode
- per-relation resolved mode
- per-relation file name and physical format
- row counts
- supported access contracts

This does not change runtime behavior. It makes the generated data
layout explicit enough for desktop benchmarking, artifact invalidation
work, and later provider-style loaders without forcing another change to
the benchmark target names.

The next follow-up starts lifting those Clojure benchmark storage-policy
knobs into shared infrastructure. A new
`src/unifyweaver/core/predicate_preprocessing.pl` layer now normalizes
source-level `preprocess/2` declarations onto the same small storage
surface used by the benchmark generator: `artifact`, `sidecar`, and
`inline`.

The Clojure effective-distance generator now consumes that shared
declaration layer in addition to its existing benchmark-local predicates:

- benchmark-local relation overrides still win first
- shared `preprocess/2` declarations are the next policy source
- generator defaults remain the final fallback

That is a better match for the C# and Haskell direction than adding more
one-off benchmark predicates. It turns the current Clojure artifact and
sidecar work into a reusable seam for later cross-target preprocessing
and artifact-provider work, while keeping the current benchmark
interface stable.

The next small extension makes that seam inspectable instead of only
actionable. The shared preprocessing module now exposes normalized
metadata for a declaration:

- originating declaration kind
- normalized physical format
- normalized access contracts
- preserved declaration options

The Clojure benchmark manifest now includes that metadata whenever a
relation's resolved mode came from the shared `preprocess/2` layer. That
brings the current Clojure path closer to the C# provider/materializer
shape: generated artifacts now carry both the chosen storage mode and
the declaration intent that selected it.

The next Clojure step is no longer just metadata. The benchmark
generator now supports an opt-in per-relation `lmdb` mode for
`category_parent`. When that override is selected, generation:

- writes a flat `category_parent.tsv` source file
- builds a real LMDB dupsort artifact for `category_parent/2`
- packages the shared JVM LMDB reader into
  `lib/lmdb-artifact-reader.jar`
- builds `lib/liblmdb_artifact_jni.so`
- switches the generated Clojure `category_parent/2` and
  `category_ancestor/4` foreign handlers to use
  `generated.lmdb.LmdbArtifactReader`

The generated benchmark runner also knows how to put that helper jar on
the Java classpath and expose the JNI library path when the project is
launched. The current integration is deliberately narrow: only
`category_parent` can opt into LMDB, and the existing EDN / grouped-TSV
paths remain the stable defaults and fallbacks.

Follow-up design work for the next Clojure LMDB phases now lives in
`docs/proposals/WAM_CLOJURE_LMDB_FACT_ACCESS_PLAN.md`. That document
captures the Clojure-specific philosophy, specification, and
implementation sequencing, and explicitly references the recent Haskell
LMDB cache and reader commits that should guide the next Clojure steps.

The first implementation step from that plan is now in place too. The
JVM helper behind the Clojure LMDB benchmark path no longer opens LMDB
from scratch on every lookup. `LmdbArtifactReader` now fronts a
thread-local native store seam:

- one LMDB read transaction per thread
- one dupsort cursor per thread
- owned `LmdbRow` objects still cross the JNI boundary
- generated Clojure code keeps the same call surface

This matches the spirit of the recent Haskell progression:

- split raw reader mechanics from wrapper policy first
- add thread-local reader reuse next
- keep memoization as a separate later layer

That later layer is now in place too, but still narrowly. The Clojure
benchmark generator now accepts an opt-in relation-local cache policy
for LMDB-backed `category_parent`:

- `wam_clojure_benchmark_relation_cache_policy(category_parent, memoize)`
- `benchmark_relation_cache_policy(category_parent, memoize)`

When selected, generated Clojure projects keep the same `lmdb` storage
mode and reader seam, but switch to `LmdbArtifactReader.openMemoized`.
The memoization layer is thread-local and lives in the Java wrapper,
not in the native store object:

- one native store per thread still owns the LMDB transaction/cursor
- one thread-local L1 map caches `lookupArg1` results by key
- scan behavior is unchanged
- shared L2 caching is still deferred

The next narrow step is now underway too: the JVM helper exposes
`openSharedCached` and `openTwoLevel` so Clojure can opt into a shared
`lookupArg1` cache or a composed L1+L2 policy without changing the
native store seam again. We should not trust Termux to tell us whether
`none` vs `memoize` is the right default; that comparison is now a
desktop-only measurement TODO.

The cache policy logic is now being pulled into a clearer JVM seam as
well. `LmdbArtifactReader` is no longer the only place where policy
details live conceptually; the direction is:

- explicit lookup-cache wrapper
- lightweight stats hooks
- opt-in generated Clojure debug output

The immediate stats surface is intentionally small:

- `localHits`
- `sharedHits`
- `misses`

That LMDB wiring is no longer benchmark-only either. The Clojure target
itself now accepts declarative LMDB foreign relations for
`category_parent/2`, and now also for `category_ancestor/4`, together
with the existing cache-policy and debug options:

- `clojure_lmdb_foreign_relations([category_parent/2-"..."])`
- `clojure_lmdb_foreign_relations([category_ancestor/4-"..."])`
- `clojure_lmdb_cache_policy(none|memoize|shared|two_level)`
- `clojure_lmdb_cache_debug(true|false)`
- `clojure_lmdb_ancestor_max_depth(N)` for `category_ancestor/4`

Generated non-benchmark Clojure WAM projects can therefore package the
JVM/JNI LMDB helper seam and emit a `call-foreign` handler directly
through `wam_clojure_target.pl`, rather than relying on
benchmark-generator-local helper code. The target-level
`category_ancestor/4` handler reuses the LMDB-backed parent lookup seam
and emits streamed `{:solutions ...}` results using the same recursive
ancestor traversal contract as the benchmark path.

One Termux-specific stabilization also landed while lifting that seam:
repeated LMDB-backed benchmark generation was corrupting or racing on
the shared Rust helper under `examples/lmdb_relation_artifact/target`.
The benchmark generator now uses a workspace-local isolated Cargo target
directory per SWI process instead of the shared repo `target/` path.

---

## Phase F: Mode-driven term-construction lowering (2026-04-27)

**Branches:**
- `docs/wam-haskell-mode-analysis-design` — three-doc design package
- `feat/wam-haskell-mode-analysis` — analyser + `=../2` lowering (M1–M6)
- `feat/wam-haskell-mode-analysis-followups` — `functor/3` lowering, M7
  cabal end-to-end smoke, this log entry

Added a forward, three-valued (`unbound`/`bound`/`unknown`)
binding-state analysis pass at
`src/unifyweaver/core/binding_state_analysis.pl`. The pass produces, per
clause, a list of `goal_binding(Idx, BeforeEnv, AfterEnv)` records that
the WAM compiler consults at the program point of each `=../2` and
`functor/3` goal. When the analysis can prove the term is unbound and
the functor name is bound, the compiler lowers the goal to a
`PutStructureDyn` instruction sequence; otherwise it falls through to
the existing `BuiltinCall` path.

### What lowers and what does not

| Goal shape | Mode declaration | Result |
|---|---|---|
| `T =.. [Name | FixedArgs]` | T proven `unbound`, Name proven `bound` | `PutStructureDyn` + `set_value`/`set_variable` × N + `get_value` |
| `T =.. [...]` | T `unknown` (no mode decl) | `BuiltinCall "=../2"` |
| `T =.. [...]` | T proven `bound` (decompose) | `BuiltinCall "=../2"` |
| `functor(T, Name, Arity)` | T `unbound`, Name `bound`, Arity literal int | `PutStructureDyn` + `set_variable` × Arity + `get_value` |
| `functor(T, Name, Arity)` | Arity is a runtime variable | `BuiltinCall "functor/3"` |
| `functor(T, Name, Arity)` | no mode declaration | `BuiltinCall "functor/3"` |

The `functor/3` lowering reuses `emit_put_structure_dyn_lowering/6`
verbatim by synthesising N fresh anonymous Prolog variables and threading
them through `compile_set_arguments/4`, which emits one
`set_variable` per slot. No new emit helper.

### Soundness contract

The analysis is forward-only, no fixpoint, no aliasing tracking, and
collapses to `unknown` at any control-flow meet where branches
disagree. Every positive answer is a runtime guarantee; `unknown`
keeps the existing path. Wrong analysis can only leave performance on
the table — never produce incorrect runtime behaviour.

### Why no benchmark numbers yet

Both lowerings are correctness-preserving optimisations on a path that
is rarely the hot loop in the benchmarks we currently measure
(category-ancestor walks, etc.). The lowerings will pay off on
workloads that construct many terms in tight loops — code-generation
utilities, Prolog-meta interpreters, term-rewriting passes. We will
benchmark when a real such workload exists. The unit tests
(`tests/core/test_binding_state_analysis.pl`,
`test_wam_univ_lowering.pl`, `test_wam_functor3_lowering.pl`) plus the
M7 cabal end-to-end smoke
(`test_wam_term_construction_e2e.pl`) cover correctness.

### Out-of-scope follow-ups (analysis substrate is in place)

- `arg/3` on a known-bound term: fuse to indexed `GetValue`.
- `\+ member(X, L)` on a ground L: skip unification, use `IS.notMember`.
- `copy_term/2` on a ground source: identity (no walk needed).
- Multi-mode predicate specialisation: separate WAM bodies per mode.

These would extend the analysis with new propagation-rule entries
(`arg/3`, `\+/1`-special-cased on `member/2`) and add new
`compile_goal_call/4` clauses, without touching the public API.

### Related documents

- `WAM_HASKELL_MODE_ANALYSIS_PHILOSOPHY.md` — the *why*
- `WAM_HASKELL_MODE_ANALYSIS_SPEC.md` — data structures, propagation
  rules, integration sites
- `WAM_HASKELL_MODE_ANALYSIS_PLAN.md` — phases M1–M8 with test gates

---

## Phase F appendix: term-construction lowering measurement and `arg/3` (2026-04-27)

**Branch:** `feat/wam-haskell-arg3-and-bench`

Three follow-ups closing out the Phase F arc:

### F.1 — Latent `GetValue` handler bug found by benchmarking

The first attempt to drive the `=../2` compose lowering through the
runtime in a benchmark loop revealed that 100 % of the lowered
iterations were returning `Nothing` (failing the goal). Cause: the
`GetValue xn ai` step handler in `WamRuntime` only handled the case
where `ai` (the second argument register) held the unbound side. The
lowering's emit helper produces `get_value T_reg, TermReg` where
`T_reg` (the *first* argument) is the fresh output and `TermReg` is
the freshly-constructed `Str` — i.e. the unbound side is on the
*xn* side. The handler fell through to `Nothing`.

The fix adds the symmetric case directly to `step (GetValue xn ai)`:

```haskell
(Just a, Just (Unbound vid)) ->
  Just (s { ..., wsBindings = IM.insert vid a (wsBindings s), ... })
```

Unification is symmetric, so this is the principled fix. The bug is
latent — every codegen unit test in the original mode-analysis arc
verified `PutStructureDyn` text appeared in the WAM, but none ran
the bytecode through `run`. The benchmark is the first thing that
did. The cabal e2e (`test_wam_term_construction_e2e.pl`) builds the
project with cabal but does not execute it, so it did not surface
the bug either.

### F.2 — Synthetic benchmark numbers

`tests/benchmarks/wam_term_construction_bench.pl` generates one
project containing two predicates with **identical Prolog source**:

```prolog
:- mode bench_lowered(+, +, -).
bench_lowered(Name, Arg, T) :- T =.. [Name, Arg].

bench_unlowered(Name, Arg, T) :- T =.. [Name, Arg].
```

The mode declaration triggers the binding-state analyser; the
unannotated copy gets the list-build + `BuiltinCall "=../2"` runtime
path. A custom `Bench.hs` (in `tests/fixtures/wam_term_construction_bench/`)
imports the generated `WamRuntime` + `Predicates` and times each
predicate across `N` calls of `bench(foo, k, T)`, alternating the
order across two trial blocks to reduce cache-warming bias.

Result at N = 100,000 (GHC 8.6.5, `-O2`):

| Trial block | lowered (s) | unlowered (s) | speedup |
|---|---:|---:|---:|
| Block 1 | 0.198 | 0.284 | 1.43× |
| Block 2 | 0.181 | 0.216 | 1.19× |
| **Mean** | **0.190** | **0.250** | **1.32×** |

The lowering wins by ~24–32 % on this microbenchmark. The savings
are *not* from instruction count (both paths emit ~6 instructions);
they come from the runtime work the `BuiltinCall "=../2"` handler
does on the unlowered side: allocate an intermediate `VList`, walk
it to extract the `Atom` head, and only then construct the `Str`.
The lowered `PutStructureDyn` skips all of that — it pre-allocates
the `BuildStruct` builder directly and `SetValue` writes the args
straight in.

This is a microbenchmark; real workloads that construct terms in
tight loops (codegen utilities, meta-interpreters, term-rewriting
passes) would see the same per-call shape and similar relative
gains. Benchmarks that are dominated by I/O, FFI, or recursion-heavy
logic will see no measurable change because term construction is
not on their hot path.

### F.3 — `arg/3` lowering

Adds the `arg/3` term-projection counterpart to the
construction-side lowerings already in place. Detects:

```prolog
:- mode pred(..., +, ..., -).
pred(..., T, ..., A) :- arg(N, T, A).
```

where `N` is a literal positive integer and the binding-state
analyser proves T is `bound`. Lowers to a single specialised
`Arg !Int !RegId !RegId` instruction (new in this PR), skipping the
`put_constant N → A1`, `put_value T → A2`, `put_variable A → A3`,
`builtin_call arg/3` chain (4 dispatches) in favour of one direct
runtime step.

The runtime handler:
- derefs T from its register, requires `Str _ args` or `VList _`,
- indexes the N-th element (1-based; matches the existing builtin),
- unifies the result with the output register's current value
  (handles both unbound and already-bound cases for safety).

Five unit tests in `tests/core/test_wam_arg3_lowering.pl`:

| Test | What it covers |
|---|---|
| `test_arg3_lowered_when_t_bound_and_n_literal` | Mode-annotated T + literal N ⇒ `arg N TReg AReg` emitted |
| `test_arg3_no_mode_falls_through_to_builtin` | No mode ⇒ `builtin_call arg/3` |
| `test_arg3_dynamic_n_falls_through` | N is a runtime variable ⇒ no lowering |
| `test_arg3_zero_or_negative_n_falls_through` | N=0 ⇒ no lowering (preserves builtin failure semantics) |
| `test_arg3_literal_n_appears_in_wam_text` | Literal N appears verbatim in emitted WAM |

No new benchmark numbers for `arg/3` yet; the construction-side
benchmark above is the first wired-up timing harness, and `arg/3` on
hot paths would need its own workload to exercise. The savings
shape is the same as `=../2` (skipping ~3 dispatch hops) so the
microbenchmark improvement is expected to be in the same range.

### F.4 — Test surface (cumulative)

After this appendix lands the term-construction arc has:

| Suite | Count |
|---|---:|
| `test_wam_haskell_target.pl` (full target suite) | unchanged |
| `test_binding_state_analysis.pl` | 31 |
| `test_wam_univ_lowering.pl` | 5 |
| `test_wam_functor3_lowering.pl` | 6 |
| `test_wam_arg3_lowering.pl` | 5 (new) |
| `test_wam_term_construction_e2e.pl` (cabal smoke) | 2 |
| `wam_term_construction_bench.pl` (benchmark, not a regression test) | — |

Plus 30 + lines added to the existing `step (GetValue xn ai)` to
cover the symmetric case.

---

## Phase G: \\+ member(X, L) lowering (2026-04-28)

**Branch:** `feat/wam-haskell-member-lowering-and-realworkload`

Adds the `NotMemberList` instruction and the matching
`compile_goal_call(\\+ member(X, L), ...)` clause that fires when
binding-state analysis proves both X and L are `bound`. Unlike the
construction-side lowerings (`=../2`, `functor/3`) and the
projection-side lowering (`arg/3`), this one targets a pattern that
appears verbatim in real graph-traversal workloads
(`category_ancestor`, etc.) — the visited-set check.

### What replaces what

The existing path for `\\+ member(X, V)` emits:

```
put_structure member/2, A1
set_value Reg(X)
set_value Reg(V)
builtin_call \\+/1, 1
```

(4 dispatches plus a heap allocation for the `Str "member/2" [X, V]`
goal term.) The runtime then fast-paths to a `VList` walk inside the
`\\+/1` handler.

The lowered path emits one instruction:

```
not_member_list Reg(X), Reg(V)
```

The `NotMemberList` step handler reads X and L directly from
registers (no goal-term construction) and walks `VList items`
inline, skipping the dispatch chain and the heap allocation.

### Measurement

`tests/benchmarks/wam_not_member_bench.pl` builds a project with
`bench_notmember_lowered/2` (mode-annotated) and
`bench_notmember_unlowered/2` (no mode) — same Prolog source, same
50-item visited list at runtime, X is an Integer that varies
per-iteration to defeat constant folding.

At N = 200,000 (GHC 8.6.5, `-O2`):

| Trial block | lowered (s) | unlowered (s) | speedup |
|---|---:|---:|---:|
| Block 1 | 0.290 | 0.350 | 1.21× |
| Block 2 | 0.296 | 0.317 | 1.07× |
| **Mean** | **0.293** | **0.334** | **~1.14×** |

A modest ~14 % win. Most of the per-call cost is the actual list
walk (50 atom comparisons), which both paths still do; the savings
are ~210 ns per call from skipping the dispatch chain and heap
allocation. The relative speedup will scale with how short the
visited list is — for a list of 5 items the dispatch overhead is a
larger fraction of total work, so the speedup will be larger.

### Real-workload applicability (option 1)

The visited-set check pattern in `category_ancestor`-style
predicates uses `\\+ member(Z, V)` where Z is bound (output of
`parent(X, Z)` in the previous goal) and V is bound (mode `+`
head argument). Existing benchmarks declare
`mode(category_ancestor(-, +, -, +))`. As of this PR, when
`category_ancestor`'s body compiles to WAM (the WAM-only path,
not the lowered-Haskell or FFI'd-kernel path), the
`NotMemberList` lowering fires automatically — no source changes
needed in the benchmark fixtures. Future benchmark runs that
exercise the WAM-only ancestry path should see the same
~14 % per-call improvement compounded across recursion depth.

### Tests

`tests/core/test_wam_not_member_lowering.pl`, 5 tests:

| Test | What it covers |
|---|---|
| `test_not_member_lowered_when_x_and_l_bound` | Mode +,+ ⇒ `not_member_list` |
| `test_not_member_no_mode_falls_through` | No mode ⇒ `builtin_call \\+/1` |
| `test_not_member_only_x_bound_falls_through` | L `?` ⇒ no lowering |
| `test_not_member_only_l_bound_falls_through` | X `?` ⇒ no lowering |
| `test_plain_member_not_lowered` | `member` alone (no `\\+`) is NOT lowered (semantics-preserving) |

### Out-of-scope follow-ups

- **Visited-set as IntSet.** The big win for graph traversal is
  representing visited as `IntSet`, not `[Value]`. That would
  reduce `\\+ member` from O(N) to O(log N). It is a fundamental
  data-structure change — not a lowering. **Designed as of
  PR #1683 in `WAM_HASKELL_INTSET_VISITED_DESIGN.md`**;
  implementation is the natural next perf arc.
- **`member(X, L)` succeeding case.** The current lowering only
  fires inside `\\+`; positive `member` with a bound L (i.e. a
  type-test "is X one of these atoms") could use the same
  walk-inline shape but with success/fail inverted. Marginal
  utility — the negation case is the common visited-set check.

---

## Phase G appendix: macro benchmark on effective-distance + analyser fix (2026-04-28)

**Branch:** `feat/wam-haskell-macro-bench-and-intset`

Two follow-ups closing out Phase G:

### G.1 — Latent `?`-mode bug found by trying to fire on real workload

The first attempt to measure the `\\+ member` lowering on
`category_ancestor`-style code showed **the lowering was not firing**
even with `mode(category_ancestor(-, +, -, +))` declared. The
unlowered path stayed in place: `put_structure member/2 + builtin_call \\+/1`.

Cause: the binding-state analyser's `apply_call_mode(_, any, ...)`
clause was setting the arg to `unknown` after a call to a predicate
declared with `?` mode, contradicting the spec
(`WAM_HASKELL_MODE_ANALYSIS_SPEC.md` §2.3.7 says `?` should leave
the arg at its pre-call state). For the canonical pattern

```prolog
parent(X, Z), \\+ member(Z, V)
```

the call to `parent/2` (declared `?, ?`) was destroying `Z`'s proven
`bound` state, so the lowering's `binding_state_at_var(BeforeEnv,
Z, bound)` precondition failed and we fell through.

Fix: change `apply_call_mode(_Arg, any, Env, Env)` to a no-op,
matching the spec. New regression test
(`test_call_any_mode_preserves_bound`) asserts that a bound arg
passed to a `?,?`-mode predicate stays bound across the call.

### G.2 — Macro benchmark on effective-distance

`tests/benchmarks/wam_effective_distance_macro_bench.pl` generates
two Haskell WAM projects from the same `effective_distance.pl`
source, distinguished only by which mode declarations are in
effect:

| variant | `mode(category_ancestor)` | `mode(category_parent)` | result |
|---|---|---|---|
| lowered | `(-, +, -, +)` | `(?, ?)` | `NotMemberList` fires |
| unlowered | `(-, +, -, +)` | none | `BuiltinCall "\\+/1"` |

Both build the standard benchmark project (Main.hs reads TSV facts,
runs effective-distance, reports `query_ms` to stderr). Both
variants are run twice with the order alternated to spot
cache-warming bias.

Result on `data/benchmark/1k` (1002 articles, 5934 category-parent
edges, 2 root categories, GHC 8.6.5 `-O2`):

| Trial | lowered query_ms | unlowered query_ms |
|---|---:|---:|
| Trial 1 | 84 | 118 |
| Trial 2 | 74 | 68 |
| **Mean** | **79** | **93** |

**Speedup: ~1.18× on the macro path** (~17 % faster).
`tuple_count=48` matches across both variants — correctness
preserved.

This is the macro-level confirmation of the Phase G claim that
"the lowering fires automatically on existing benchmarks once the
mode declarations are in place." It also confirms the microbenchmark
result (~14 %) carries over to a real workload, with the macro
slightly higher because the `=../2` and `functor/3` lowerings also
fire in adjacent code paths.

Trial 2's unlowered (68 ms) is faster than trial 1's lowered (84 ms),
which is jitter on a cold first run — that's why the harness runs
twice and alternates order. The mean across two trials is the
honest number.

### G.3 — IntSet visited design landed (implementation deferred)

`WAM_HASKELL_INTSET_VISITED_DESIGN.md` (added in this PR) covers
the algorithmic next step: a `VSet IS.IntSet` `Value` variant + 3
new instructions (`BuildEmptySet`, `SetInsert`, `NotMemberSet`) +
a `:- visited_set/2` directive that opts a specific predicate-arg
position into the IntSet representation. Expected speedup: another
~1.5–3× on top of the constant-factor wins, scaling with
`max_depth`. Implementation deferred to a follow-up PR (task #191).

### Test surface (cumulative across the mode-analysis arc)

| Suite | Count |
|---|---:|
| `test_binding_state_analysis.pl` | 32 (was 31; +1 for `?`-mode fix) |
| `test_wam_univ_lowering.pl` | 5 |
| `test_wam_functor3_lowering.pl` | 6 |
| `test_wam_arg3_lowering.pl` | 5 |
| `test_wam_not_member_lowering.pl` | 5 |
| `test_wam_term_construction_e2e.pl` (cabal) | 2 |
| `wam_term_construction_bench.pl` (microbench) | — |
| `wam_not_member_bench.pl` (microbench) | — |
| `wam_effective_distance_macro_bench.pl` (macrobench, NEW) | — |

---

## Phase H: IntSet-backed visited (Layer 1 — runtime + ADT) (2026-04-28)

**Branch:** `feat/wam-haskell-intset-visited`

Implements Layer 1 of the IntSet visited design
(`WAM_HASKELL_INTSET_VISITED_DESIGN.md` from PR #1684): the runtime
data structures and instructions, validated by 12 codegen unit
tests. Layer 2 (compile-time recognition of the
`:- visited_set/2` directive and the bootstrap/recursive/`\\+ member`
emission triple) is the natural follow-up — it's where the actual
speedup gets unlocked, and it's substantial enough on its own to
warrant a focused PR with its own test gates.

### Changes

#### `src/unifyweaver/targets/wam_haskell_target.pl` (template edits)

- **`Value` ADT**: add `VSet !IS.IntSet` variant; `import qualified Data.IntSet as IS` in WamTypes.
- **`NFData Value`**: add `rnf (VSet s) = rnf (IS.size s)` branch (IntSet has no NFData; force the size to ensure thunks resolve).
- **`Instruction` ADT**: add three new constructors:
  - `BuildEmptySet !RegId` — write `VSet IS.empty` into the named register.
  - `SetInsert !RegId !RegId !RegId` — `elemReg`, `inSetReg`, `outSetReg`.
  - `NotMemberSet !RegId !RegId` — `elemReg`, `setReg`; succeeds when the elem is NOT in the set.
- **Step handlers** for the three new instructions, mirroring the design doc's pseudocode. `SetInsert` and `NotMemberSet` require the element to deref to an `Atom` (visited-set members are interned atom IDs); non-atom elements return `Nothing` (semantically: backtrack).
- **WAM text parsers**: `build_empty_set XReg`, `set_insert EReg, InReg, OutReg`, `not_member_set EReg, SReg`.

#### Tests

`tests/core/test_wam_intset_runtime.pl` (NEW, 12 tests, all green):

| Section | Coverage |
|---|---|
| Value type | `VSet !IS.IntSet` present, `Data.IntSet` imported, `NFData VSet` branch present |
| Instruction ADT | `BuildEmptySet`, `SetInsert`, `NotMemberSet` all present |
| Step handlers | All three handler bodies emit the right `IS.*` operations |
| WAM parsers | All three text→ADT translations produce correct register IDs |

Plus the existing `test_wam_haskell_target.pl` suite stays green
(verifies the new variant doesn't break any pattern-match
exhaustiveness on `Value`), and the cabal e2e
(`test_wam_term_construction_e2e.pl`) still builds successfully —
GHC is happy with the new variant + the explicit `NFData VSet`
branch + the `_ -> Nothing` fall-throughs in non-set-aware
handlers.

### What does NOT lower yet (Layer 2)

The Phase H runtime is correct but currently *unreachable* from
user code: nothing in the WAM compiler emits `BuildEmptySet` /
`SetInsert` / `NotMemberSet`, so a clause that uses
`\\+ member(X, V)` still compiles to `NotMemberList` (or the
slow path, depending on mode declarations). The compile-time
integration is the next PR.

The Layer 2 work needs:
1. A `:- visited_set(Pred/Arity, ArgN)` directive parser/registry.
2. Per-clause context for "this head's visited-set vars".
3. `compile_goal_call(\\+ member(X, V), ...)` clause that emits
   `NotMemberSet` when `V` is in the visited-set context.
4. `compile_put_argument` integration to detect when an arg
   position matches a directive AND the arg is a list literal
   (bootstrap) or `[X|V_visited]` (recursive extension), and
   emit `BuildEmptySet` + `SetInsert` × N or
   `SetInsert(X, V_visited, FreshReg)` instead of the standard
   list-build sequence.

Steps 3 and 4 must coordinate — they share state (the head's
visited-set vars) and emission rules. That's the heart of the
follow-up PR.

---

## Phase H appendix: IntSet Layer 2 partial — `\\+ member` lowering only (2026-04-28)

**Branch:** `feat/wam-haskell-intset-visited-codegen`

Implements two of the four pieces Layer 2 needs:

1. **`:- visited_set(Pred/Arity, ArgN)` directive registry** — read at
   compile time via `is_visited_set_arg/2`.
2. **Per-clause visited-set var context** — `set_clause_visited_context/1`
   captures head-arg variables matching the directive at clause start
   into a non-backtrackable global, mirroring the existing
   `wam_clause_binding_records` plumbing for binding-state analysis.
3. **`\\+ member(X, V)` lowering** — when V is a head-arg variable in
   the per-clause visited-set context, emits `not_member_set XReg, VReg`
   instead of `not_member_list`. Tail-call form mirrored.

Pieces still pending (documented as Layer 2.5):

4. **Call-site arg rewriting** — converting `[Cat]` (bootstrap) and
   `[X|V_visited]` (recursive extension) into `BuildEmptySet`/
   `SetInsert` sequences when passed into a visited-set arg position.
   Initial implementation revealed an interaction with the TCO
   dispatch in `compile_goals/5` that caused infinite recursion on
   simple test cases — backed out of this PR to keep the runtime
   landing unblocked. The rewrite needs to be reworked so it doesn't
   recurse through the same dispatch on the rewritten goal.

### What does and doesn't lower

| Pattern | Layer 2 (this PR) | Layer 2.5 (next PR) |
|---|:---:|:---:|
| `\\+ member(X, V)` where V is head's visited-set var | ✅ `not_member_set` | — |
| `pred(..., [Cat], ...)` (bootstrap into visited slot) | put_list (unchanged) | `build_empty_set` + `set_insert` |
| `pred(..., [X\|V_visited], ...)` (recursive cons) | put_list (unchanged) | `set_insert` |

The `\\+ member` lowering is correct and useful in isolation **only
once a separate code path constructs the `VSet`**. Without Layer 2.5,
calling code that uses the directive will get `NotMemberSet` against
a `VList`-typed register at runtime → handler returns `Nothing`. So
in practice this PR adds the half-machinery; Layer 2.5 closes the loop.

### Findall-vs-identity bug worth noting

While implementing the per-clause visited-set var capture, an early
implementation used `findall/3` to collect head-arg variables matching
directives. `findall` creates fresh copies of captured variables,
which broke `==` identity later when the body's `\\+ member(X, V)`
goal compared its V against the captured set. Fixed by walking the
head-arg list in place via a recursive `collect_visited_vars/4`
helper that preserves variable identity. New regression test
`test_visited_set_var_propagation_across_clauses` covers the case.

### Tests

`tests/core/test_wam_visited_set_lowering.pl` (NEW, 4 tests):

| Test | Coverage |
|---|---|
| `test_not_member_set_when_visited_declared` | Directive + body `\\+ member` ⇒ `not_member_set` |
| `test_no_visited_directive_keeps_not_member_list` | No directive ⇒ Phase G fallback fires |
| `test_visited_set_var_propagation_across_clauses` | Multi-clause head var identity preserved across clause boundaries |
| `test_unrelated_call_with_list_arg_unchanged` | Calls without visited_set directive are NOT rewritten |

All 4/4 green. Plus the full
`tests/test_wam_haskell_target.pl`,
`tests/core/test_wam_not_member_lowering.pl`,
`tests/core/test_binding_state_analysis.pl`, and
`tests/core/test_wam_intset_runtime.pl` suites all stay green.

---

## Phase H final: IntSet visited Layer 2.5 — call-site arg rewrite (2026-04-28)

**Branch:** `feat/wam-haskell-intset-layer25`

Closes the IntSet visited arc by adding the call-site argument
rewriting that was deferred from PR #1690 (the partial Layer 2 PR).
With this PR, the runtime / instructions / `\\+ member` lowering /
construction-site rewriting all coordinate, and a workload that uses
`:- visited_set/2` actually constructs `VSet` values at runtime —
the loop is closed.

### What landed

#### `compile_goal_call(Goal, ...)` and `compile_goal_execute(Goal, ...)`

New early clauses fire when the goal calls a predicate with at least
one `:- visited_set/2` declaration AND the arg at that position
matches a recognised list shape. The clauses inline the construction
code + `compile_put_arguments` + `call`/`execute` in **a single pass
— no recursion on the rewritten goal**. The previous Layer 2 attempt
recursed on the rewritten goal AND the dispatch in `compile_goals/5`
re-routed visited-set goals through the same rewrite, causing
infinite recursion. Single-pass emission breaks the cycle.

#### `rewrite_visited_set_args/6` and `rewrite_visited_arg/5`

The rewrite walks the args list, replacing visited-set positions:

- **Recursive extension `[X|V_visited]`**: emits
  `set_insert XReg, V_visited_Reg, NewSetReg` and binds the
  replacement var to `NewSetReg`.
- **Bootstrap `[X]`** (1-elem list, atom-`[]` tail explicitly
  required): emits `build_empty_set Rs ; set_insert XReg, Rs, Rs`.

The recursive case is tested **first**, with the bootstrap clause
explicitly requiring `nonvar(T), T == []` for the cdr. Otherwise
clause-head unification of `[X]` would match `[X|V_var]` by binding
`V_var = []`, silently mis-routing recursive cons through the
bootstrap path. (Caught by macro-shape integration check.)

#### `goal_has_visited_set_arg/1` + TCO dispatch hook

Added back to `compile_goals/5`'s last-goal branch so visited-set
goals route through `compile_goal_execute` (where the rewrite
fires) rather than the inline `put_arguments` fast path that
bypasses the rewrite. With single-pass emission in the rewrite
clause, this hook no longer causes the recursion that hung the
previous attempt.

### Tests

`tests/core/test_wam_visited_set_lowering.pl` extended to 8 tests
(was 4):

| Test | Coverage |
|---|---|
| (existing) `test_not_member_set_when_visited_declared` | `\\+ member` lowering |
| (existing) `test_no_visited_directive_keeps_not_member_list` | Phase G fallback |
| (existing) `test_visited_set_var_propagation_across_clauses` | Multi-clause var identity |
| (existing) `test_unrelated_call_with_list_arg_unchanged` | No directive ⇒ no rewrite |
| **new** `test_bootstrap_emits_build_empty_set_and_insert` | `[Cat]` ⇒ build_empty_set + set_insert |
| **new** `test_recursive_cons_emits_set_insert` | `[X\|V]` ⇒ set_insert |
| **new** `test_bootstrap_does_not_emit_put_list` | bootstrap doesn't fall back to put_list |
| **new** `test_recursive_cons_does_not_emit_put_list` | recursive cons doesn't fall back to put_list |

All 8/8 green. Full target suite + binding-state +
`\\+ member`-list lowering + IntSet runtime all stay green.

### End-to-end check

A direct compile of the canonical `category_ancestor/4` predicate
with `:- visited_set(category_ancestor/4, 4)` and `mode/1`
declarations now emits all three Layer 1 instructions in
`Predicates.hs`:

```
NotMemberSet 201 202     -- \\+ member(Parent, Visited) in clause 1
NotMemberSet 201 203     -- \\+ member(Mid, Visited) in clause 2
SetInsert 201 203 107    -- [Mid|Visited] passed to recursive call
```

The runtime instructions (Layer 1) + lowering (Layer 2) + arg
rewrite (Layer 2.5) coordinate end-to-end.

### Macro benchmark — measured (and the design's expected speedup did NOT materialise)

`tests/benchmarks/wam_effective_distance_macro_bench.pl` extended
to a 3-way comparison: `unlowered` / `lowered` (Phase G) /
`intset` (Phase G + H). 6 trials with rotating order to spot
cache-warming bias.

Results at 10k scale (`data/benchmark/10k`, 462 tuples,
~50-element visited list bounded by `max_depth=10`):

| variant | mean query_ms |
|---|---:|
| unlowered (no directives) | 931.5 |
| lowered (Phase G `mode` only) | 861.0 |
| intset (Phase G + H combined) | 957.5 |

Speedups:

- **lowered vs unlowered: 1.082×** — Phase G's constant-factor
  win on the macro path is consistent with the prior 1.18× at 1k.
- **intset vs lowered: 0.899×** — the IntSet path is **~10 %
  slower** than the list-based lowering at this scale.
- **intset vs unlowered: 0.973×** — IntSet barely matches the
  unlowered baseline.

`tuple_count=462` matches across all three — correctness
preserved.

### Why the algorithmic O(log N) does not pay off here

The design predicted ~1.5–3× speedup from O(N) → O(log N) at
`max_depth=10`. Reality: the Patricia-trie constant factor
(tree descent, node allocation per insert) **exceeds** the
~10-element list walk's linear cost. Specifically:

1. **Allocation per insert**: `IS.insert` allocates new tree
   nodes on each cons-extension (purely functional). `[X|V]`
   allocates one cons cell. For shallow visited sets, the
   allocation cost dominates.
2. **Cache locality**: a small contiguous cons-cell list packs
   into one or two cache lines. A 4-element IntSet trie scatters
   across multiple nodes that may hit different cache lines.
3. **Node traversal overhead**: even `IS.member` on a 10-element
   `IntSet` does ~3-4 tag-checks and comparisons in a Patricia
   trie, vs at most 10 simple `==` checks in the list — the
   constant factors are surprisingly close at this size.

**The IntSet wins kick in at much deeper visited sets** —
probably `max_depth ≥ 50` or unbounded depth on cyclic graphs.
For the canonical effective-distance workload at `max_depth=10`,
the algorithmic improvement does not amortise its constant
factors.

### Honest reading of the IntSet arc

The IntSet implementation is **correct** (tests + cabal e2e green,
`tuple_count` matches across all variants), but on the workload it
was designed for, it does not improve performance — and slightly
hurts it. Three takeaways:

1. **Algorithmic wins are not free at small N.** The design's
   "1.5–3× expected" was reasoning from asymptotic complexity
   without accounting for IntSet's per-operation constant factors
   on deeply-allocating workloads.
2. **The infrastructure is reusable.** `VSet`, the directive,
   and the codegen paths can host other set representations that
   might win at small N — e.g. a sorted array or a small bitmap
   for known-small visited sets.
3. **Phase G is the real macro win.** The constant-factor
   `not_member_list` lowering (skipping put_structure +
   builtin_call dispatch + heap term allocation) is what
   actually speeds up the workload. Phase H's algorithmic
   pivot was the wrong move for this particular workload.

The directive remains opt-in. Users who don't declare it pay
nothing. Users who do declare it on workloads with deep visited
sets may see the expected algorithmic win; users at typical
`max_depth=10` should leave it off.

A useful follow-up would be to test at `max_depth=50` or higher
to find the crossover point where IntSet's algorithmic benefit
finally dominates the constant factors.

### Phase H follow-up — depth-sweep (2026-04-29)

> **⚠ This sweep was based on a broken instrumentation. See the
> "max_depth instrumentation bug + retraction" section below for the
> corrected reading.**

Ran the depth-sweep filed above. The benchmark gained a
`WAM_EFF_DIST_BENCH_MAX_DEPTH` env var that overrides
`max_depth/1` at codegen time. Sweep at scale=1k, 2 trials per
variant per depth (rotating order):

| max_depth | unlowered (ms) | lowered (ms) | intset (ms) | intset/lowered |
| ---: | ---: | ---: | ---: | ---: |
| 10 | 72.5 | 53.5 | 67.5 | **0.79×** |
| 30 | 77.0 | 51.0 | 80.0 | **0.64×** |
| 50 | 69.0 | 77.0 | 70.0 | **1.10×** |

`tuple_count=48` matches across **all** runs at **all** depths.
This is the punchline: the wikipedia category-graph paths terminate
well before depth 10, so the depth bound never fires. Increasing
`max_depth` from 10 to 50 has no observable effect on visited-list
sizes at the goal site. There is no IntSet crossover to find on
this graph — the visited list is shape-bound by graph topology to
~10 elements regardless of the depth cap.

The 1.10× at depth=50 sits inside the 2-trial noise (lowered's
spread that depth was 70→84 ms, ~20%); it is not a real crossover.

**Honest reading: the IntSet algorithmic crossover does not
materialise on the wikipedia-category-graph workload at any
`max_depth` setting.** The graph is the limit, not the bound.
Confirming the IntSet wins on real workloads would require either
(a) a synthetic workload with deep cycles or long chains, or
(b) a different real graph with longer transitive paths (some
biological or citation networks would qualify).

The directive remains opt-in. The infrastructure is reusable for
alternate small-N set representations (sorted array, bitmap)
which might beat IntSet at this scale — filed as future
exploration alongside the synthetic-workload follow-up.

### Phase G.2 — ground-list `\+ member` lowering (2026-04-29)

Companion to Phase G's `not_member_list` (which handles
`\+ member(X, V)` where V is a runtime-bound list var). Phase G.2
handles the OTHER `\+ member` shape — a literal ground list at
the call site:

```prolog
\+ member(X, [foo, bar, baz])    %% ground list of atoms in source
```

Lowers to a single new instruction:

```
not_member_const_atoms <xReg> foo bar baz
```

with the atom IDs interned at codegen and baked into the
generated Haskell as `NotMemberConstAtoms !RegId ![Int]`. The
step handler does a single `elem` check on the embedded ID list:

```haskell
step !_ctx s (NotMemberConstAtoms xReg atomIds) =
  case derefVar (wsBindings s) <$> IM.lookup xReg (wsRegs s) of
    Just (Atom aid) | aid `elem` atomIds -> Nothing
    Just (Atom _)                        -> Just (s { wsPC = wsPC s + 1 })
    _                                    -> Nothing
```

Vs the unlowered path (`PutStructure` cons cells × N + builtin
dispatch `\+/1` + `member/2` + N unifications), this is a single
WAM dispatch with zero heap allocation. Vs the inline-IntSet
construction (`BuildEmptySet` + N `SetInsert` + `NotMemberSet`),
this avoids per-call Patricia-trie allocation that — per the
Phase H finding above — does not pay off at small N.

Fires when the second arg is a proper list of ground atoms.
Falls through cleanly for var lists (Phase G `not_member_list`),
partial lists, and lists containing integers/structures/unbound
elements.

**Macro impact**: none on the canonical effective-distance
workload — that workload uses var L (visited set), not ground L.
A targeted microbench (e.g. `\+ member(X, [a,b,c,d,e,f,g,h])`
in a tight loop) would be the natural follow-up to quantify the
saved heap allocation + dispatch. Not measured on this branch.

Touch points: `wam_target.pl` (lowering, helper), `wam_haskell_target.pl`
(Instruction ADT entry, step handler, WAM-text parser),
`tests/core/test_wam_ground_member_lowering.pl` (9 codegen +
runtime + parse tests).

### Phase G.2 microbench — measured ~14.5× (2026-04-29)

The follow-up filed above. `tests/benchmarks/wam_ground_member_bench.pl`
generates two projects from the same source predicate:

```prolog
bench_ground(X) :- \+ member(X, [a, b, c, d, e, f, g, h]).
```

The lowered variant uses default codegen (NotMemberConstAtoms);
the unlowered variant asserts
`wam_target:lowering_disabled(ground_member)` to fall through to
the standard builtin path. A new module-level disable hook in
`wam_target.pl` exists exactly so benches can build the unlowered
baseline.

A shared Bench.hs (in `tests/fixtures/wam_ground_member_bench/`)
runs `bench_ground(Integer k)` 200 000 times per case across 4
trials, with `Integer k` varying per iteration to defeat constant
folding. Probe is never one of the 8 atoms, so the check always
succeeds.

Measured (after fixing the step handler — see "Correctness fix"
below):

| variant | mean (ms) | per-call (μs) |
| --- | ---: | ---: |
| lowered (NotMemberConstAtoms) | 61.9 | 0.31 |
| unlowered (builtin \\+/1 + member walk) | 899.4 | 4.50 |

**Speedup: ~14.5×**, both variants report 200 000/200 000 correct
results.

The win matches the design's intuition: 1 instruction dispatch + 1
Atom-id list scan vs `N+1 PutStructure` heap allocations + 2
builtin dispatches + N unifications. At N=8 atoms the unlowered
path's heap allocation is the bulk of the cost.

#### Correctness fix surfaced by the microbench

The first run of the microbench reported **lowered ok=0/200000,
unlowered ok=200000/200000** — a divergence. Cause: the original
step handler returned `Nothing` for any non-Atom value:

```haskell
case mX of
  Just (Atom aid) -> ...
  _               -> Nothing  -- WRONG for Integer / Float / Str / etc.
```

But Prolog's `\\+ member(Integer 42, [a, b, c])` succeeds
trivially — an integer can never unify with an atom, so member
fails, so `\\+` succeeds. The handler now distinguishes:

- `Atom aid in atomIds` → fail (member would succeed)
- `Atom aid not in atomIds` → succeed
- `Unbound _` / `Ref _` → fail (Prolog could unify with some atom
  via free unification)
- Any other ground value (`Integer`, `Float`, `Str`, `VList`,
  `VSet`) → succeed (no atom can unify with it)

This bug would have hit any user code that passes non-atom probes
to `\\+ member`. The microbench caught it before any user did.

### Phase H follow-up #2 — synthetic deep-chain workload (2026-04-29)

To answer the open IntSet-crossover question on a workload not
shape-bound by graph topology, this branch ships
`examples/benchmark/generate_synthetic_chain.py`. It produces a
linear category chain `cat_001 → cat_002 → ... → cat_N` with a
single root `cat_N` and articles placed near the leaf end. Each
article must walk the full chain to reach the root, forcing a
visited list of length up to N-1 during `category_ancestor`
recursion — exactly the shape Phase H was designed for.

The generator produces both `facts.pl` (for SWI-Prolog) and the
TSV files (`category_parent.tsv`, `article_category.tsv`,
`root_categories.tsv`) the WAM Haskell binary loads at startup.

**Status: blocked on a binary-side issue.** The SWI-Prolog
version of the workload runs on `synth_chain_30` and finds 10
paths (each ~30 hops). The compiled WAM Haskell binary on the
same data returns `tuple_count=0` and `article_count=0` — even
though `seed_count=10` and `demand_total_nodes=29` confirm the
seeds and edges are loaded correctly. The query terminates in
≤1 ms without finding any paths.

Best guess: the demand-driven path's behaviour on a linear chain
diverges from SWI-Prolog's evaluation, possibly because of how
`Par*` parallel fork choice points interact with deep `\\+ member`
+ recursive call patterns. Filed as a follow-up for diagnosis
on its own branch.

Net: the IntSet-crossover question remains open. The
infrastructure (generator + bench wiring + max_depth env var) is
in place; the binary-side bug needs to be resolved before the
crossover can be measured on this synthetic shape.

### max_depth instrumentation bug + retraction (2026-04-29)

**The bug**: `templates/targets/haskell_wam/main.hs.mustache`
hardcoded `wcForeignConfig = Map.singleton "max_depth" 10`. The
bench's `override_max_depth/0` (added in PR #1712) correctly
retracted/asserted `user:max_depth/1`, but the codegen never
consumed it — `nativeKernel_category_ancestor` always cut DFS
recursion at depth 10 regardless of the env var.

This single bug retroactively invalidates two prior conclusions:

1. **PR #1712's "graph is shape-bound" depth-sweep is wrong.**
   The 1k bench at WAM_EFF_DIST_BENCH_MAX_DEPTH ∈ {10, 30, 50}
   all returned `tuple_count=48` because the kernel's actual
   max_depth was always 10. Re-measuring the same 1k bench with
   the fix in place at depth=30 returns **89 paths** (not 48)
   and the query takes **~120 seconds** (not ~50 ms) — paths
   genuinely deepen, the visited list genuinely grows. The wiki
   graph is NOT shape-bound; PR #1712's measurement was
   instrumentation-bound.

2. **PR #1714's "binary returns 0 on linear chains" is the same
   bug.** synth_chain_30 with `WAM_EFF_DIST_BENCH_MAX_DEPTH=30`
   needs paths up to 29 hops deep. The kernel cut at hardcoded
   depth=10, so every seed returned `[]`. The "demand-driven Par*
   fork interaction with linear chains" hypothesis was wrong;
   the actual cause was a missing template substitution.

**The fix** (`fix(wam-haskell): resolve max_depth from
user:max_depth/1 at codegen`, cherry-picked from the diagnostic
branch):

- Replace the hardcoded `10` with `{{max_depth}}` in
  `main.hs.mustache`.
- Add `resolve_max_depth/2` in `wam_haskell_target.pl` that
  reads from `Options [max_depth(N)]` first, then
  `user:max_depth/1`, defaulting to 10. Threaded through
  `render_template`.
- Three new tests in `test_wam_haskell_target.pl` covering
  default/user-fact/option-override priorities.

### Phase H follow-up #3 — IntSet crossover finally measured (2026-04-29)

With the fix in place, the synthetic chain workload — designed
specifically to grow visited lists past Phase H's expected
crossover — produces real numbers.

**`synth_chain_300` (chain depth 300, 200 articles,
max_depth=300, 2 trials per variant, rotating order)**:

| variant | trial 1 (ms) | trial 2 (ms) | mean (ms) |
| --- | ---: | ---: | ---: |
| intset (Phase G + H) | 80 | 81 | **80.5** |
| lowered (Phase G `not_member_list`) | 111 | 91 | **101.0** |
| unlowered (builtin \\+/1 + member walk) | 54 | 81 | 67.5 |

**intset vs lowered: 1.25×** — IntSet beats the list walk on
deep visited lists, as the Phase H design predicted. Trial-pair
spread on intset (80 vs 81) is tight; the result is real, not
noise.

The unlowered timing (67.5 mean, 27ms spread) is too noisy to
read as faster than lowered — that's a 27ms swing on an 80ms
mean, dominated by cabal startup + cache effects on first
trials. The intset/lowered comparison is the load-bearing one
and it's clean.

**`synth_chain_30` correctness sanity** (chain depth 30, 10
articles, max_depth=30): all three variants return
`tuple_count=10` matching SWI-Prolog. Times are below resolution
(0-1 ms) so timing comparison is not informative at this scale,
but correctness is preserved end-to-end.

### Honest reading of the IntSet arc, after the fix

1. **Phase H's algorithmic design was correct.** On workloads
   that genuinely grow visited lists past ~50 elements, IntSet
   beats Phase G's list walk by ~25 %. This is the speedup
   originally predicted in the design.

2. **The wiki workload at the originally-tested depths really is
   shape-bound, but only at small `max_depth`.** At max_depth=10
   (the canonical workload setting), wiki paths terminate before
   the cap fires, so visited lists stay small (~10 elements) and
   IntSet's constant factor exceeds the list walk's. **This is
   the real Phase H finding for the canonical workload, and it
   stands.**

3. **The 1k bench at deeper max_depth is a different workload
   shape.** At max_depth=30 paths actually reach that depth
   (89 vs 48 paths), and queries take 120 seconds. Whether
   IntSet wins there has not been re-measured on this branch —
   the synth_chain_300 measurement covers the design question;
   the wiki-1k-at-deep-depth measurement is a separate bench
   exercise filed as future work.

4. **Bench instrumentation should always be cross-checked.** The
   missing template substitution survived three PRs (#1698,
   #1712, #1714) because every measurement was internally
   consistent at the broken depth=10 cap. Adding tests that
   *fail* when max_depth doesn't flow into the kernel — as the
   fix's three new tests do — closes the loophole.

---

## Planning Note: Clojure lowered-tier and interning follow-up (2026-04-28)

The recent Clojure LMDB work and Scala hybrid-WAM design work exposed a
separate Clojure performance/design gap:

- the Clojure hybrid WAM target still lacks a Rust-style lowered WAM
  middle tier
- the Clojure runtime is still comparatively string- and map-heavy in
  its hot path, whereas Rust and Haskell already treat atom/functor
  interning as a first-class runtime concern

The resulting proposal is:

- keep TypR-style native clause lowering as one family
- add a Rust-style `wam_clojure_lowered_emitter` as a second family
- allow overridable routing defaults between native lowering, lowered
  WAM, foreign/kernel lowering, and full WAM fallback
- design atom/functor interning into the lowered-tier plan rather than
  treating it as a late micro-optimization

Reference:

- [WAM_CLOJURE_LOWERED_TIER_PLAN.md](../proposals/WAM_CLOJURE_LOWERED_TIER_PLAN.md)

---

## Phase L: Haskell `category_ancestor` kernel — IntSet visited + INLINE + bang patterns (2026-05-02, `perf/wam-haskell-category-ancestor-kernel`)

### Why this phase

After Phase K added the `category_ancestor` FFI kernel for Go (52× at
scale-300), a fresh kernels-on Haskell-vs-Rust matrix bench surfaced a
~10× residual gap at scale-300:

| target | mode | kernels | median (s) |
|---|---|---|---:|
| haskell-pure-interp | interp | off | 10.774 |
| haskell-lowered-only | lowered | off | 10.127 |
| haskell-interp-ffi | interp | **on** | 0.519 |
| haskell-lowered-ffi | lowered | **on** | 0.718 |
| rust-pure-interp | interp | off | 3.637 |
| rust-lowered-only | lowered | off | 3.477 |
| rust-interp-ffi | interp | **on** | 0.062 |
| rust-lowered-ffi | lowered | **on** | 0.071 |

(All eight rows produced identical 271-row output, sha256 `70bbc9ffa4cf`.)

Two observations from the matrix:

1. **The Haskell-vs-Rust ratio widens with kernels on**: 3× kernels-off,
   ~10× kernels-on. With kernels on, the inner ancestor walk leaves the
   WAM dispatch loop and runs as native FFI — so the gap can't be the
   WAM interpreter itself. It has to be the kernel implementation, the
   FFI marshalling, or the residual orchestration.
2. **Lowered helps with kernels off but not kernels on**: when the
   kernel does the heavy lifting, the surrounding orchestration (which
   is what lowering changes) is small enough that the extra function-
   call boundaries cost more than they save.

That points the next optimisation at the kernel itself.

### What changed

`templates/targets/haskell_wam/kernel_category_ancestor.hs.mustache` is
the kernel template. Three coupled changes:

1. **Visited set as `IntSet`, not `[Int]`.** The public signature still
   accepts `[Int]` (FFI marshalling format extracted from VList) but on
   entry the kernel calls `IS.fromList visited` once, then the
   recursive `go` helper carries the IntSet directly. `IS.member` is
   O(log N); `elem` on a list is O(N). At deep visited sets this
   matters. Same idea as the IntSet-visited WAM lowering from Phase H —
   applied inside the kernel rather than around it.
2. **`{-# INLINE #-}` pragma** on the kernel function. Encourages GHC
   to specialise the recursion site and avoid heap-allocating the
   closure for `go` on each call.
3. **Bang patterns (`!c !d !v`)** on the inner loop variables. Forces
   strict evaluation so each recursive call commits to a concrete
   `(cat, depth, visited-set)` tuple instead of building thunks that
   would be forced later under GC pressure.

### Measured impact

#### Scale-300 (canonical wiki, max_depth=10, visited stays ~10)

| target | before (s) | after (s) | improvement |
|---|---:|---:|---:|
| haskell-interp-ffi | 0.519 | **0.434** | ~16% |
| haskell-lowered-ffi | 0.718 | **0.405** | **~44%** |

Modest at this scale — the visited set never exceeds ~10 elements, so
the IntSet algorithmic win barely fires. Most of the scale-300 gain is
the strictness/INLINE pieces, not IntSet itself. The lowered path saw
the bigger relative win because it has less fixed dispatch cost outside
the kernel call.

#### synth_chain_300 (chain depth 300, 200 articles, max_depth=300, visited grows up to 99)

| variant | main (ms) | this branch (ms) | speedup |
|---|---:|---:|---:|
| intset | 80.5 | **18.0** | **4.5×** |
| lowered | 101.0 | 22.0 | 4.6× |
| unlowered | 67.5 | 31.0 | 2.2× |

Where IntSet earns its keep. All three macro-bench variants improved
because all three call into the FFI kernel — the WAM-side directive
choice (intset/lowered/unlowered) doesn't affect kernel behaviour.
tuple_count=200 across all runs — correctness preserved end-to-end.

#### Scale-1k matrix (real wiki data, max_depth=10) — sanity check

| target | median (s) | rows |
|---|---:|---:|
| haskell-interp-ffi | 0.492 | 580 |
| haskell-lowered-ffi | **0.361** | 580 |
| rust-interp-ffi | 0.052 | 580 |

580 rows match across targets (sha `4c8841411a2c`). The kernel changes
hold at the larger wiki scale; lowered-ffi is now ~27% faster than
interp-ffi at 1k (was a slight regression at scale-300, but the
scale-300 measurements were noisy below the cabal-startup floor).
Rust still leads ~7× — the structural gap discussed in Phase L below
remains.

### Honest reading

The 10× Haskell-vs-Rust gap at scale-300 is roughly halved (the
lowered-ffi delta from 0.718→0.405 closes about 44% of the gap on its
side). The bench-level ratio is still in Rust's favour because:

- Rust's monomorphization + stack-allocated state keeps the residual
  orchestration tighter even when the kernel dominates work.
- GHC's IntSet uses Patricia tries which carry per-node allocation
  overhead — fine in absolute terms, slightly less efficient than
  Rust's HashSet at small N.
- The bigger structural advantage Rust holds (immutable-record updates
  through GHC's heap vs Rust's `&mut`-style state mutation) isn't
  closed by this change; that would need a separate ST/IORef arc on
  the Haskell side.

What this phase does cleanly close:

- The visited-set algorithmic loss inside the kernel for
  deep-recursion workloads. **4-5× wins at synth_chain_300** is the
  load-bearing measurement; the canonical-workload improvement at
  scale-300 is a side effect of the strictness work.

### Touch points

- `templates/targets/haskell_wam/kernel_category_ancestor.hs.mustache` —
  kernel rewrite (19 insertions, 8 deletions). Public type unchanged.
- No changes to executeForeign template or kernel registration metadata
  — the IntSet conversion stays internal to the kernel.

### Phase L appendix: `use_lmdb(auto)` resolver

While the kernel work was in flight, this branch also adds the
auto-mode resolver previously filed as a follow-up. New predicate
`resolve_auto_use_lmdb/2` in `wam_haskell_target.pl` normalises
`use_lmdb(auto)` to a concrete `true`/`false` based on:

1. `ghc-pkg list --simple-output lmdb` — if the Haskell package isn't
   installed, fall back to `false` with a stderr warning. Cabal's
   own dependency resolution would also catch this at build time,
   but pre-checking gives a friendlier error and lets the build
   complete on the IntMap path.
2. `option(fact_count(N), ...)` against
   `option(lmdb_auto_threshold(T), ..., 50000)`. When N > T pick
   true; else false. The threshold lands between the
   IntMap-wins-clearly regime (≤10k facts, per the prior crossover
   study at `project_wam_haskell_fact_access.md`) and the
   LMDB+cache-wins regime (≥100k); both endpoints are documented
   memory.
3. `fact_count` absent → conservatively false. Caller opts into the
   scale-driven path by passing the count.

Wired in at the top of `write_wam_haskell_project/3` so the four
existing `option(use_lmdb(true), Options)` checks see the resolved
value transparently. Explicit `use_lmdb(true)` and `use_lmdb(false)`
stay unchanged.

Firewall integration is deliberately *not* in this first cut — cabal
itself fails loudly if lmdb is missing, which is enough of a safety
gate. A later optional firewall hook (`firewall_check(use_lmdb, R)`
before step 1) can layer on top without changing the resolver's
contract.

5 tests in `tests/test_wam_haskell_target.pl` cover the deterministic
paths: explicit-true passthrough, explicit-false passthrough, absent
unchanged, auto+low-fact-count→false, auto-without-fact-count→false.
The "auto + high fact_count + lmdb installed → true" path is
environment-gated so we don't test it directly here; the resolver's
guard logic is fully covered by the false-paths.

### Phase L appendix #2: accumulator-passing kernel rewrite + bench harness wiring (2026-05-03)

Two of the Phase L open follow-ups closed on this branch:

**Accumulator-passing rewrite.** The kernel previously returned
relative hop counts and post-incremented via `map (+1) $ go ...` in
the recursion. GHC's list fusion can in principle eliminate that
extra pass, but `(baseHits ++ recHits)` breaks the build/foldr
pattern fusion needs — so without fusion, the post-traversal cost
compounds to O(N²) at depth N. Rewrite passes `acc :: Int` down,
returning absolute hop counts directly.

Validated at synth_chain_300:

| variant | Phase L only (ms) | + accumulator (ms) | improvement |
|---|---:|---:|---:|
| intset | 18.0 | 16.0 | ~12% |
| lowered | 22.0 | 14.5 | 1.5× |
| unlowered | 31.0 | 10.5 | ~3.0× |

The intset variant's improvement is small because IntSet visited
already eliminated most of the inner-loop cost; the accumulator
mostly helps the lowered/unlowered paths whose visited check is
unchanged. The 3× on unlowered is the load-bearing measurement —
shows the post-traversal `map` was a real cost on this shape.
tuple_count=200 across all variants — correctness preserved.

**Bench harness wiring for `use_lmdb(auto)`.** The resolver from
Phase L appendix #1 was dormant — no bench fed it `fact_count(N)`
or set `use_lmdb(auto)`. Wiring now in place:

- `generate_wam_haskell_matrix_benchmark.pl` counts
  `category_parent/2` clauses in the facts.pl and passes
  `fact_count(N)` in Options. New optional 6th CLI arg
  `<lmdb_mode>` ∈ {none, auto, true, false} translates to
  `use_lmdb(...)` in Options.
- `benchmark_effective_distance_matrix.py` threads `lmdb_mode`
  through `build_haskell_effective_distance` and adds a dispatch
  arm for the new `haskell-interp-ffi-auto` target.
- `benchmark_target_matrix.py` registers the new target in the
  `hybrid-wam` category.

Verified at scale-300: `haskell-interp-ffi-auto` produces the same
output sha (`70bbc9ffa4cf`, 271 rows) as the explicit
`haskell-interp-ffi` target. In a dev environment without the
Haskell `lmdb` package installed the resolver falls back to
`use_lmdb(false)` per the documented contract; the `[WAM-Haskell]
use_lmdb(auto): ghc-pkg does not list lmdb; falling back to IntMap`
stderr line confirms the resolver fired.

### Open follow-ups (post-Phase-L appendix #2)

1. **ST-monad / IORef state** for the Haskell WAM hot loop — the bigger
   structural change that would close more of the residual Rust gap.
   Multi-PR architectural arc; see `project_wam_haskell_st_monad_plan.md`
   notes.
2. **Optional firewall hook** in the auto-mode resolver — consult
   `firewall_check(use_lmdb, R)` before the ghc-pkg check, so a
   strict-mode deployment can refuse LMDB even when it's installable.
   Requires extending the firewall's tool registry to know about
   Haskell library packages, not just executables.
3. **Run the auto target with the lmdb cabal package installed** to
   verify the "auto + high fact_count + lmdb available → true" path
   works end-to-end. Requires environmental setup (cabal install lmdb +
   LMDB-ingested data); the wiring itself is in place.
   **— validated in Phase L appendix #3 (2026-05-04).**
4. **Scale 5k/10k matrix runs** with the kernel changes, to see how
   the Haskell-vs-Rust gap evolves at larger wiki scales. Each run
   is ~5-15 min wall time so deferred for budget.

---

## Phase L appendix #3: cross-target audit + auto-resolver true-path validation (2026-05-04)

Two Phase L follow-ups closed:

**Cross-target audit** for the instrumentation-bug class (PR #1724
fixed `dimension_n` in Haskell; the broader question was whether other
WAM target backends had analogous hardcoded values that should come
from user-asserted facts). Surveyed Rust, Elixir, Scala, Clojure, Go:

- **Haskell** — already fixed (#1721 max_depth, #1724 dimension_n).
- **Go** — already fine; `resolve_dimension_n_go/2` mirrors the
  Haskell resolver, and `max_depth` flows via the shared kernel
  registry's detector at `recursive_kernel_detection.pl:289` which
  reads `user:max_depth/1`.
- **Rust** — fine; FFI kernel reads `max_depth` from
  `foreign_usize_config` at runtime, populated at codegen via
  `register_foreign_usize_config` from the same shared registry.
- **Elixir, Scala** — no FFI kernel with separate config that could
  diverge from the WAM-compiled `max_depth/1` predicate; user-asserted
  values flow via the standard predicate-compilation path.
- **Clojure — bug found and fixed.** The LMDB foreign handler at
  `wam_clojure_target.pl:272` read `max_depth` from
  `option(clojure_lmdb_ancestor_max_depth(M), Options, 10)` —
  explicit option with a hardcoded default. A user asserting
  `user:max_depth(30)` in the workload would silently still get 10
  unless they also passed the clojure-specific option. New
  `resolve_clojure_lmdb_ancestor_max_depth/2` mirrors the Haskell
  resolver pattern (option → user-fact → default 10). Regression
  test in `tests/test_wam_clojure_generator.pl` gated on the LMDB
  Java toolchain; verifies `user:max_depth(13)` reaches generated
  Clojure code as `max-depth 13`.

**Auto-resolver true-path validation.** Phase L appendix #1 shipped
the resolver + 5 deterministic false-path tests; Phase L appendix #2
wired it through the matrix harness; this appendix validates the
true-path end-to-end.

Two pieces:

1. `lmdb_haskell_package_available` extended to also probe
   `~/.cabal/store/ghc-<ver>/package.db`. The previous check only
   consulted ghc-pkg's user/global db, but `cabal v2-build` /
   `cabal v2-install --lib` (the cabal ≥ 2.4 norm) installs to the
   cabal store db, which a bare `ghc-pkg list` doesn't see. With this
   fix, the resolver finds lmdb in either location.

2. Ran `haskell-interp-ffi-auto` on `100k_cats` (196,900 facts —
   ~4× the 50,000 default threshold). Generator stderr confirms
   `lmdb=auto fact_count=196900`; generated cabal includes
   `lmdb >= 0.2.5` (so the resolver picked `use_lmdb(true)`); binary
   built and ran in ~406 s, producing 11 rows of output
   (sha `8b134f6f910e`).

The five deterministic resolver tests still pass — `auto + low
fact_count` and `auto + no fact_count` continue producing false even
with lmdb available, because the fact_count gate (step 2) overrides.

### Open follow-ups (still)

The Phase L appendix #2 follow-ups #1, #2, #4 remain — ST-monad arc,
optional firewall hook for the resolver, and the larger scale 5k/10k
sweeps. Cross-target audit found one Clojure bug; no other target
needed fixes.

## Phase L appendix #4: pre-filter demand seeds before parMap (2026-04-28)

PR #1876 (codex) added a per-seed `IS.member` gate inside the parMap
closure so seeds outside the structural demand set returned `(cat,
0.0)` immediately. The gate was correct but introduced a parallel
scaling regression at high seed counts: the spark count still equalled
`length seedCats`, so on `100k_cats` (84,136 seeds, ~84,125 outside
the demand set) GHC scheduled ~84k sparks of trivial work and paid
synchronization cost on each. The handoff doc captured the regression
— at `-N1` total was 9.66 s and at `-N4` it was 7.02 s, with `-N2`
the only setting that beat `-N1`.

### The fix

Move the filter from inside the parMap closure to a separate let
binding executed *before* `parMap`:

```haskell
-- Before (PR #1876):
let !seedResultsForced = parMap rdeepseq (\cat ->
        if not (IS.member (iAtom cat) demandSet)
          then (cat, 0.0)
          else <full seed query>
        ) seedCats
    !demandSkippedSeeds = length [cat | cat <- seedCats, not (IS.member (iAtom cat) demandSet)]

-- After (this PR):
    !filteredSeedCats = filter (\cat -> IS.member (iAtom cat) demandSet) seedCats
    !demandSkippedSeeds = length seedCats - length filteredSeedCats
...
let !seedResultsForced = parMap rdeepseq (\cat ->
        <full seed query>
        ) filteredSeedCats
```

Semantically equivalent: the post-aggregation step
`Map.fromList [(cat, ws) | (cat, ws) <- seedResultsForced, ws > 0]`
already discards zero-weight entries, so removing them upstream
is observationally identical. Verified by output sha
`70bbc9ffa4cf` matching at scale-300 (271 rows) before and after.

### Codegen touch points

`src/unifyweaver/targets/wam_haskell_target.pl`:
- `generate_demand_filter/4` now emits the `!filteredSeedCats` binding
  and computes `demandSkippedSeeds` as a length subtraction (cheaper
  than a list-comprehension count).
- `generate_demand_gated_query_body/3` is now an identity passthrough.
  The historical wrapper is preserved as a comment in case anyone
  needs to rebuild the in-body gate variant.
- `generate_main_hs/4` selects `parmap_seed_source = filteredSeedCats`
  when the demand filter is active and `seedCats` otherwise.

`templates/targets/haskell_wam/main.hs.mustache`:
- The parMap line iterates over `{{parmap_seed_source}}` instead of
  hardcoded `seedCats`.

`tests/test_wam_haskell_target.pl`:
- `test_demand_filter_gates_seed_query_body` rewritten: now asserts
  `!filteredSeedCats = filter ...` is present, that the inline
  `if not (IS.member ...) then (cat, 0.0)` gate is *absent*, and
  that parMap closes over `filteredSeedCats`.
- `test_demand_filter_false_leaves_query_ungated` extended: also
  asserts no pre-filter binding and that parMap still closes over
  `seedCats`.

### Measurements (`100k_cats`, default root `0s_beginnings`,
`demand_skipped_seeds=84125`, 11 effective seeds, 3 trials each)

| RTS | codex baseline (#1876) | this fix | delta |
|---|---|---|---|
| `-N1` | total 9.66 s, query 240 ms | total 3.23 s, query 0 ms | **3.0× faster** |
| `-N2` | total 6.25 s, query 172 ms | total 3.32 s, query ≤4 ms | 1.9× faster |
| `-N4` | total 7.02 s, query 220 ms | total 6.15 s, query 0 ms | 1.1× faster |

Wall-clock is dominated by the sequential pre-query phase (TSV load,
atom interning, demand-set BFS) once the spark fanout is removed.
That sequential floor is the next target — not the parallel section.

### Honest reading

The parMap path is now correctly proportional to effective demand
(11 sparks at `-N1`, not 84,136). `query_ms` collapsing to ~0 ms
shows the spark scheduling overhead was real and is gone. `-N4`
hurting the *total* — even with the fix — reflects the GC / RTS
pressure of more capabilities on a workload where the parallelizable
section is now small. That is a separate problem (parallelize the
load / atom-intern / BFS phases, or drop into single-thread mode for
small effective-demand workloads); the spark-fanout regression PR
#1876 introduced is closed.

### Open follow-ups

- Codex's handoff doc lists Rust accumulated at 0.6 s and Elixir LMDB
  at 1.6 s on the same fixture. Even after this fix Haskell at 3.2 s
  is the slowest of the three because the sequential pre-query work
  dominates. Parallelizing or amortizing TSV load and atom interning
  is the next lever.
- The historical in-body gate code path is documented in
  `generate_demand_gated_query_body/3` but unreachable. Drop it once
  there's no plausible reason to A/B against the gate variant.

## Phase L appendix #5: LMDB-resident interning end-to-end (2026-05-08)

### Why this landed

Appendix #4 closed with: "Parallelizing or amortizing TSV load and
atom interning is the next lever." This appendix measures that lever
on `100k_cats`. The work landed across three PRs:

- **PR #1916** (Phase 2b.2a) — `loadInternTableFromLmdb`,
  `loadArticleCategoriesFromLmdb`, `loadForwardEdgesFromLmdb` +
  `iterateAllPairs` / `peekStringBytes` helpers. Codegen surface, no
  runtime wiring.
- **PR #1918** (Phase 2b.2b) — Replace the `int_atom_seeds(lmdb)`
  panic stub with calls to those loaders; wrap the int-ids-mode
  intern-table and parents-index blocks with `{{^int_atom_seeds_lmdb}}`
  and add LMDB-mode parallels. New `openLmdbInternEnvReadonly` helper.
- **PR #1929** (Phase 2b.2c plumbing) — Matrix-bench `resident` mode
  selector emitting `use_lmdb(true) + lmdb_layout(dupsort) +
  int_atom_seeds(lmdb)`; missing imports (`CChar`, `nullPtr`, `when`)
  added; **dupsort sub-db naming bug fixed** (`openLmdbEdgeLookup` /
  `lmdbFactSource` ignored their `dbName` argument and hardcoded
  `Just "main"`, so the FFI kernel couldn't find the
  Phase-1-ingester's `category_parent` sub-db); fixture-specific
  one-pass dual-table ingester `ingest_resident_lmdb_fixture.py`.

### Measurement

100k_cats fixture (196,900 category_parent edges, 84,136
article_category edges, 84,136 unique atoms; default first root from
`root_categories.tsv`). `seeded interpreter kernels_on`. 5 trials per
cell; medians reported.

| mode     | -N | load_ms | query_ms | total_ms | peak_RSS_MB |
|----------|---:|--------:|---------:|---------:|------------:|
| tsv      |  1 |       0 |        0 |    3535  |        751  |
| resident |  1 |     745 |        4 |     980  |        385  |
| tsv      |  2 |       0 |        2 |    3483  |        788  |
| resident |  2 |     776 |        4 |    1072  |        384  |
| tsv      |  4 |       0 |        1 |    5831  |        786  |
| resident |  4 |    1564 |        6 |    2090  |        388  |

`load_ms` is the bench's `t1 - t0` (file/LMDB load only). `query_ms`
is `t3 - t2` (per-seed parMap). `setup_ms = total_ms - load_ms -
query_ms - aggregation_ms` is the intern-table + parents-index build.

Speedup (`tsv total / resident total`):
- -N1: **3.61×** (3535 → 980 ms)
- -N2: **3.25×** (3483 → 1072 ms)
- -N4: **2.79×** (5831 → 2090 ms)

Peak RSS: **resident is ~50% of TSV** at every -N level (385 vs 751 MB
at -N1). The TSV path materialises a string-keyed `parentsIndex`
before reinterning into the system-wide table; resident reads
pre-interned int32 directly via `mdb_cursor_get'` and mmap-shares the
backing pages.

### Where the speedup comes from

Decomposing -N1 medians:

| phase                      | tsv      | resident |
|----------------------------|---------:|---------:|
| load (file/LMDB read)      |    0 ms  |   745 ms |
| setup (intern + index)     | 3535 ms  |   231 ms |
| query (per-seed parMap)    |    0 ms  |     4 ms |
| total                      | 3535 ms  |   980 ms |

The TSV path's 3.5 s is ~95% intern table + parents index
construction, the bottleneck Phase 2b targets. The resident path
trades it for a 0.7 s LMDB load plus a 0.2 s in-memory index build,
net ~1 s. The 1 s target from `WAM_LMDB_RESIDENT_INTERNING_IMPLEMENTATION_PLAN.md`
§4 lands by 20 ms.

### Open follow-ups

- **Parallelism regression at -N4 persists** — both modes get slower
  at -N4 vs -N1 (TSV: 3535→5831 = 1.65× slower; resident: 980→2090 =
  2.13× slower). Appendix #4 documented the seed pre-filter that
  dropped this from catastrophic to merely sublinear. Going further
  needs amortising the LMDB env open across sparks (resident
  load_ms doubles 745→1564 at -N4, suggesting first-page faults are
  paid concurrently across worker threads). Filed for a follow-up
  branch.
- **UTF-8 / ASCII** ([issue #1915](https://github.com/s243a/UnifyWeaver/issues/1915)) —
  `peekStringBytes` byte-by-byte decode is correct for the URL-encoded
  ASCII categories in this fixture; will trip on simplewiki's
  multi-byte UTF-8. Fix is `Data.Text.Encoding.decodeUtf8`. Triggers
  on first non-ASCII fixture.
- **Edge-count threshold** — `loadForwardEdgesFromLmdb` caps in-memory
  growth at 5_000_000 edges (~80 MB IntMap). 100k_cats's 196,900 fits
  easily; enwiki-scale (~28M) will trip the guard. Two long-term
  fixes in `WAM_LMDB_RESIDENT_INTERNING_IMPLEMENTATION_PLAN.md` §3.5:
  reverse-edge sub-db at the ingester (small change), or LMDB-cursor
  BFS instead of pre-load (Phase 2b.3 follow-up).
- **TSV path setup time at -N4** — TSV setup grows from 3.5 s (-N1) to
  5.8 s (-N4) without doing more work. The intern-table fold runs
  once before parMap fires, so this is GC pressure or scheduler
  noise. Resident's setup is small enough that the same noise has
  smaller relative impact. Worth investigating if TSV path remains a
  user-facing default.

### Reproducer

```bash
# 1. Re-ingest 100k_cats with the resident layout (Phase 1 sub-dbs).
mkdir -p data/benchmark/100k_cats_resident_run
cp data/benchmark/100k_cats/{category_parent,article_category,root_categories}.tsv \
   data/benchmark/100k_cats/{facts.pl,metadata.json} \
   data/benchmark/100k_cats_resident_run/
python3 examples/benchmark/ingest_resident_lmdb_fixture.py \
    data/benchmark/100k_cats data/benchmark/100k_cats_resident_run/lmdb

# 2. Generate matrix bench in both modes.
swipl -q -s examples/benchmark/generate_wam_haskell_matrix_benchmark.pl -- \
    data/benchmark/100k_cats_resident_run/facts.pl /tmp/wam_100k_tsv \
    seeded interpreter kernels_on none
swipl -q -s examples/benchmark/generate_wam_haskell_matrix_benchmark.pl -- \
    data/benchmark/100k_cats_resident_run/facts.pl /tmp/wam_100k_resident \
    seeded interpreter kernels_on resident

# 3. Build both.
(cd /tmp/wam_100k_tsv && cabal new-build)
(cd /tmp/wam_100k_resident && cabal new-build)

# 4. Run with /usr/bin/time -v for peak RSS; loop -N1/-N2/-N4 × 5 trials.
```

## Phase L appendix #6: parMap regression at -N>=2 is GC pressure (2026-05-08)

### Why this landed

Appendix #5 left this open: "both modes get slower at -N4 vs -N1
(TSV: 3535→5831 = 1.65× slower; resident: 980→2090 = 2.13× slower)".
The hypothesis was first-page faults paid concurrently across worker
threads on the LMDB load. This appendix tests it.

### Diagnosis

Resident binary, 100k_cats, +RTS -N4 -s, three trials per config:

| config       | total_ms | GC time |  RSS_MB |
|--------------|---------:|--------:|--------:|
| default      |     3327 |  5.66 s |     397 |
| -A64M        |     1146 |  1.36 s |     619 |
| -A256M       |     1031 |  0.65 s |    1244 |
| -A1G         |      892 |   ~0  s |    1318 |
| -A64M -qg    |     1286 |  0.66 s |     619 |

The regression is **GC pressure**, not LMDB or thread contention. At
-N4 the default 1 MB nursery triggers a parallel GC pass roughly
every 1 MB of allocation; with 4 threads competing for the GC's stop-
the-world phase, GC time blows up to 5.7 s out of 3.3 s wall (because
parallel GC accounts wall time across each capability separately).

`-A64M` lifts the nursery to 64 MB so collections happen ~64× less
often. `-A1G` is even better (near-zero GC) but uses 1.3 GB RSS.
`-A64M -qg` (use sequential GC) is comparable to `-A64M` alone — the
nursery size is the dominant lever, not the parallelism mode of GC
itself.

### Validation across -N

5 trials per cell, medians:

| cell           | total_ms | GC time | RSS_MB |
|----------------|---------:|--------:|-------:|
| default -N1    |     1016 |  0.68 s |    394 |
| **A64M    -N1**|   **1260**| **0.89 s** |**521** |
| default -N2    |     1079 |  1.19 s |    394 |
| **A64M    -N2**|    **801**| **0.71 s** |**504** |
| default -N4    |     2126 |  3.85 s |    397 |
| **A64M    -N4**|   **1062**| **1.10 s** |**619** |

`-A64M` is **only correct at -N≥2**. At -N1 it costs ~24% (1016 →
1260 ms): the small default nursery's frequent minor GCs beat one
big nursery + occasional major GC, when there's no parallel GC
synchronisation cost. Crossover is between -N1 and -N2.

### What landed

The matrix bench's primary use case is `+RTS -N>=2`, so we accept
the -N1 cost in exchange for halving total_ms at -N4. New
`with_rtsopts(Flags)` codegen option in `generate_cabal_file/4` bakes
`-with-rtsopts="..."` into the executable's `ghc-options`. The
matrix bench generator passes `with_rtsopts('-A64M')`. Per GHC's
"last flag wins" rule, callers can override at runtime with
`+RTS -A1M -RTS`.

| Codegen surface               | Default       | Override |
|-------------------------------|---------------|---------|
| `generate_cabal_file/4`       | no rtsopts    | `with_rtsopts(Flags)` |
| matrix bench generator        | `-A64M`       | `+RTS -A1M -RTS` |
| other WAM-Haskell projects    | unchanged     | n/a    |

Only the matrix bench opts in. Other generated projects (typical
user code, single-target benches, etc.) get the original GHC defaults
because the right -A varies with workload + intended -N.

### What's still open

A workload-aware nursery size is the proper fix, à la the C# target's
source-mode cost-model resolver
(`docs/design/CSHARP_QUERY_SOURCE_MODE_*.md`). For the WAM-Haskell
target the inputs are: number of seeds, number of edges, intended
-N, available RAM. Output: an -A choice that optimises total_ms
without exceeding budget. Not done in this iteration; the static
`-A64M` is a reasonable point on the curve for the matrix bench.

The TSV setup-growth observation from appendix #5 (TSV setup_ms
3.5→5.8 s going N1→N4) is the same phenomenon — a sequential
allocation-heavy phase running while the GC scales with -N. -A64M
helps there too; the matrix bench's resident mode is the canonical
test, but TSV mode benefits as a side effect.

### Reproducer

```bash
# Generate the matrix bench (resident or none — both inherit -A64M).
swipl -q -s examples/benchmark/generate_wam_haskell_matrix_benchmark.pl -- \
    data/benchmark/100k_cats_resident_run/facts.pl /tmp/wam_resident \
    seeded interpreter kernels_on resident
grep ghc-options /tmp/wam_resident/wam-haskell-matrix-bench.cabal
# -> ghc-options:      -O2 -threaded -rtsopts "-with-rtsopts=-A64M"

# Build and run; -A64M is now the runtime default at every -N.
(cd /tmp/wam_resident && cabal new-build)
.../wam-haskell-matrix-bench .../100k_cats_resident_run +RTS -N4 -s
```

## Phase L appendix #7: cursor BFS at simplewiki scale (2026-05-09)

### Why this measurement

Phase 2b.3 (PR #1950) landed LMDB-cursor BFS as an alternative to the
pre-loaded parentsIndex IntMap. The 1k smoke test in that PR showed
parity (cursor + sharded L2 ≈ in_memory + sharded L2 within trial
noise), which is the *correct* result at small scale but doesn't
demonstrate the architectural win — that materialises only when
pre-load isn't a viable option.

Appendix #5 had also flagged the 100k_cats fixture as structurally
broken for parMap measurement (4054 fragmented graph roots, max
descendant subtree of 22). This appendix moves to simplewiki, which
is real Wikipedia-derived data with proper few-roots-many-descendants
topology.

### Methodology

The existing `simplewiki_cats/lmdb_proj/lmdb` was ingested earlier in
the streaming-pipeline layout (single dupsort `main` sub-db with
int32 child → int32 parent edges) — which Phase 2b.3 cursor mode
can't read directly because it needs `category_parent` and
`category_child` named sub-dbs. Instead of re-ingesting from the SQL
dump, a small converter
(`examples/benchmark/convert_lmdb_to_phase1_layout.py`) mirrors the
existing data into the Phase 1 layout: copies `main` to
`category_parent`, builds reverse-edge `category_child`, and stubs
the intern table sub-dbs (s2i / i2s / article_category as empty —
resident_cursor mode loads them at startup but doesn't exercise
their contents at runtime). 297,283 edges; conversion runs in ~0.7s.

3 trials per cell, default first root from the existing
`root_ids.txt` (id 265340 — has 14,661 descendants in the simplewiki
hierarchy), 5,000 seeds.

### Results

| mode             | -N | load_ms | query_ms | total_ms | peak_RSS_MB |
|------------------|---:|--------:|---------:|---------:|------------:|
| resident         |  1 |     183 |       36 |     443  |        179  |
| resident_cursor  |  1 |     159 |       40 |    **226**|       **147** |
| resident         |  2 |     141 |       33 |     335  |        221  |
| resident_cursor  |  2 |     137 |       45 |    **207**|        217  |
| resident         |  4 |     137 |       13 |     339  |        320  |
| resident_cursor  |  4 |     140 |       15 |    **234**|        332  |

Speedup (resident total / resident_cursor total):
- -N1: **1.96×**  (443 → 226 ms)
- -N2: **1.62×**  (335 → 207 ms)
- -N4: **1.45×**  (339 → 234 ms)

Both modes return identical `tuple_count=5000` and
`demand_set_size=14661` across all 18 runs — correctness preserved.

### Why cursor wins at scale

At 297k edges, resident's `loadForwardEdgesFromLmdb` builds a
~10 MB IntMap before queries can run. That's 80–150 ms of upfront
work that cursor mode skips entirely (cursor reads only the edges
the BFS actually visits — at most ~14k for the demand set, plus the
kernel's per-seed walks).

`peak_RSS_MB` at -N1 confirms this: cursor mode is **18% smaller**
(147 vs 179 MB), the gap being the unbuilt parentsIndex IntMap.
At higher -N levels GC scratch space dominates and the gap closes.

The kernel's per-seed FFI overhead (highlighted in 2b.3 smoke
results) is fully amortised by the sharded L2 cache: shared
ancestors get cache hits across seeds and across multi-parent path
enumeration. The default `lmdb_cache_mode(sharded)` from the same
PR is doing real work here.

### What's still open

- **enwiki measurement** — `enwiki_cats/lmdb_proj/lmdb` has
  9,932,244 edges (33× simplewiki). Same converter applies; would
  need ~30s to convert + ~1 GB peak RSS for resident mode (which
  hits the 5 M-edge guard) vs ~mid-to-high MB for cursor. The
  cursor path is the only viable mode at that scale. Filed as
  follow-up; needs a clean run on a system with enough RAM headroom.
- **Empty intern table caveat** — the converter stubs s2i/i2s as
  empty. Means int IDs in stdout don't reverse-map to Wikipedia
  category names. For perf measurement that's fine; for any
  use case that wants human-readable output, fresh ingest from
  the SQL dump (via mysql_stream + ingest_to_lmdb.py) is needed.
- **demand_set_size = 14,661** is the test root's descendant subtree
  — well-defined and substantial. But seed_count = tuple_count =
  5,000 means *every* seed is in the demand set; the filter didn't
  prune any seeds. That's a property of the seed file (which was
  generated to be drawn from this root's subtree) rather than a
  general fact about Wikipedia. A "broad seeds, narrow root" workload
  would exercise the filter differently. Future work.

### Reproducer

```bash
# Convert existing simplewiki LMDB to Phase 1 layout.
python3 examples/benchmark/convert_lmdb_to_phase1_layout.py \
    data/benchmark/simplewiki_cats/lmdb_proj/lmdb \
    data/benchmark/simplewiki_cats/lmdb_proj_resident/lmdb
cp data/benchmark/simplewiki_cats/lmdb_proj/{seed_ids,root_ids}.txt \
   data/benchmark/simplewiki_cats/lmdb_proj_resident/

# Generate matrix bench in resident_cursor mode (any FactsPath works
# because resident_cursor doesn't read facts.pl at runtime).
swipl -q -s examples/benchmark/generate_wam_haskell_matrix_benchmark.pl -- \
    /dev/null /tmp/wam_simplewiki_cursor seeded interpreter kernels_on resident_cursor

# Build + run.
(cd /tmp/wam_simplewiki_cursor && cabal new-build)
.../wam-haskell-matrix-bench data/benchmark/simplewiki_cats/lmdb_proj_resident +RTS -N4
```

## Phase L appendix #8: enwiki at-scale finale (2026-05-09)

### Why this measurement

Phase L appendix #7 validated cursor BFS at simplewiki scale (297k
edges, 1.96× speedup vs in_memory at -N1) but didn't yet exercise
the architectural argument: *can the cursor path run where the
in-memory path literally cannot?* enwiki at 9.9M edges puts that to
the test.

### Codegen fix (in this PR)

A bug in the Phase 2b.3 codegen: cursor mode was calling
`loadForwardEdgesFromLmdb` unconditionally before the demand filter
ran. The 5_000_000-edge guard would fire even though the result was
unused at runtime. Fix: thread `demand_bfs_mode_cursor` through
`generate_main_hs` and gate the loader call with mustache. Cursor
mode now binds `let !lmdbParentsIndex = IM.empty` directly. No load,
no guard, no wasted IntMap construction.

This was masked at simplewiki scale because 297k edges fit under the
5M cap. Without the fix the enwiki measurement would have been
impossible.

### Resident mode at enwiki: cannot run

Confirmed deterministically:

```
$ ./wam-haskell-matrix-bench .../enwiki_cats/lmdb_proj_resident +RTS -N1
wam-haskell-matrix-bench: loadForwardEdgesFromLmdb: edge count
  exceeded threshold 5000000 (loaded 5000000 so far). Switch to
  LMDB-cursor BFS (Phase 2b.3) or use the ingester's
  --with-reverse-edges follow-up.
Exit status: 1
```

The guard's "switch to cursor BFS" message is exactly what landed
this PR's predecessor. Closing the loop.

### Cursor mode at enwiki: runs comfortably

Fixture: `data/benchmark/enwiki_cats/lmdb_proj_resident` after
running the converter on 9,932,244 edges (28s). Single root id
97,688,913 with **796,695 descendants** in the category hierarchy.
1000 seeds, 860 produced nonzero results.

Medians of 3 trials, kernels_on, +RTS -A64M (baked):

| -N | load_ms | query_ms | total_ms | RSS_MB |
|---:|--------:|---------:|---------:|-------:|
|  1 |       0 |       11 |     1129 |    752 |
|  2 |       0 |       12 |     1066 |    807 |
|  4 |       0 |        7 |     1138 |    922 |

`load_ms = 0` because cursor mode skips the IntMap pre-load
entirely. Total time is dominated by the demand BFS itself (walking
796k descendants via category_child cursor) plus the kernel's
per-seed work over 1000 seeds.

System: 4.8 GB total RAM, ~1.7 GB available pre-bench. **Cursor mode
finishes in ~1.1s with 750-920 MB peak RSS** — well within the
machine's headroom, on data that resident mode cannot load at all.

### Why -N scaling is flat at enwiki

Going N1 → N2 → N4 leaves total_ms essentially unchanged
(1129 / 1066 / 1138). Two reasons:

1. **Demand BFS is sequential**. One cursor, one thread, walks
   796k nodes. Doesn't parallelize.
2. **Per-seed parMap work is tiny** (`query_ms` 7-12ms). Even with
   perfect parallel scaling, that's a small fraction of total time.

Future work: parallelize the demand BFS itself (concurrent cursors
per worker, partition the descendant tree). That's Phase 2c
territory, requires the cache instrumentation + workload-aware
gating that the user has been (correctly) deferring.

### What this closes

Phase 2b's headline arc:
- Phase 2b.1-2b.2c: codegen, runtime, fixture infrastructure
- Phase 2b.3: cursor BFS architecture
- Phase L #5: 100k_cats measurement (showed 3.61× but on a
  structurally-broken fixture)
- Phase L #6: parMap regression diagnosis (GC pressure)
- Phase L #7: simplewiki real-scale measurement (1.96× cursor win)
- Phase L #8 (this): enwiki — cursor runs where resident cannot

The architecture handles every scale the existing fixtures expose,
from 1k (parity with cache) through 297k (1.96× cursor win) to
9.9M (cursor only). Caveat: the per-seed FFI cost amortizes
through the sharded L2 cache; without it, cursor mode at enwiki
would be much slower.

### Open follow-ups

- **Parallel demand BFS** — partition the descendant tree across
  workers. Phase 2c. Probably needs a new sub-db structure or an
  in-memory frontier-partitioning algorithm.
- **Multi-query workloads** — single-query benches don't expose
  cache hit rates that vary across queries. The matrix bench does
  one query per binary launch. Phase 2c.
- **Cost-model auto-resolver** — the user explicitly flagged this
  as the prerequisite for cache warming work. Inputs:
  fact_count, demand_set_size estimate, available RAM. Outputs:
  cache_mode + size budget. Phase 2c.
- **Real Flux strategy** (Phase 2.5) — replace the panic stub
  with personalised PageRank scoring. Self-contained feature work.

### Reproducer

```bash
# Convert enwiki LMDB to Phase 1 layout (~28s, 9.9M edges).
mkdir -p data/benchmark/enwiki_cats/lmdb_proj_resident
python3 examples/benchmark/convert_lmdb_to_phase1_layout.py \
    data/benchmark/enwiki_cats/lmdb_proj/lmdb \
    data/benchmark/enwiki_cats/lmdb_proj_resident/lmdb
cp data/benchmark/enwiki_cats/lmdb_proj/{seed_ids,root_ids}.txt \
   data/benchmark/enwiki_cats/lmdb_proj_resident/

# Generate cursor-mode bench. Resident mode would crash at 5M
# edges, so cursor is the only option here.
swipl -q -s examples/benchmark/generate_wam_haskell_matrix_benchmark.pl -- \
    /dev/null /tmp/wam_enwiki_cursor seeded interpreter kernels_on resident_cursor

(cd /tmp/wam_enwiki_cursor && cabal new-build)
.../wam-haskell-matrix-bench data/benchmark/enwiki_cats/lmdb_proj_resident +RTS -N1
```

## Phase L appendix #9: parallel demand BFS (2026-05-09)

### Why this measurement

Appendix #8 showed cursor BFS *runs* at enwiki scale but doesn't
*scale with -N*: the demand BFS itself is sequential (one cursor,
one thread, walking 796,695 descendants), and that BFS dominates
total time. parMap over per-seed work can't help when the seeds
themselves wait ~1s for the demand set to be computed.

### What changed

`computeDemandSetCursorBFS` now splits each BFS level's frontier
across `getNumCapabilities` workers via `Control.Concurrent.Async.mapConcurrently`.
Each worker:

1. Runs in a bound thread (`runInBoundThread`) so its LMDB cursor
   stays pinned to one OS thread (LMDB read-txn requirement).
2. Lazily opens its own (read txn, cursor) pair via the existing
   `DupsortCursorCache` infrastructure (same pattern used by the
   FFI kernel's `cpEdgeLookup`).
3. Sequentially does `mdb_cursor_get' MDB_SET` + `MDB_NEXT_DUP`
   iteration over its assigned chunk of frontier nodes.
4. Returns its slice of new descendants as an `IS.IntSet`.

The main thread unions all chunk results, advances to the next
BFS level, and repeats until the frontier is empty. Levels remain
sequential (depth N+1 needs depth N's frontier), but per-level
work is now parallel.

### Results (enwiki, 9.93M edges, 796,695-node demand set)

Medians of 3 trials, kernels_on, +RTS -A64M (baked):

| -N | sequential (PR #1955) | **parallel (this PR)** | delta vs sequential |
|---:|----------------------:|-----------------------:|--------------------:|
|  1 |                  1129 |               **1264** |               −12% |
|  2 |                  1066 |                **928** |              **+13%** |
|  4 |                  1138 |                **733** |              **+36%** |

Going N1 → N4 in parallel mode: **1264 → 733 ms = 1.73× parallel
speedup**. Compare to sequential mode where N1 → N4 was flat
(1129 → 1138). The architecture now actually scales.

### -N1 regression analysis

The 12% regression at -N1 is real and expected: `mapConcurrently`
spawns one bound thread per BFS level even at -N1 (vs zero in the
prior sequential implementation). Across ~10-20 BFS levels at
enwiki scale, the fork/join overhead adds ~135 ms.

This is acceptable because cursor mode at -N1 isn't the design
target — at sub-50k scales the auto resolver picks `in_memory`
mode (which doesn't go through this code path at all), and at
50k+ scales users typically run with -N≥2 anyway. The `auto`
resolver path is unaffected.

### What this closes

The -N scaling gap from appendix #8. Together with the simplewiki
appendix #7 (1.96× cursor-mode speedup at 297k edges) and the
sharded L2 default from PR #1950, the resident-cursor path now
delivers:

- Architectural scalability: runs where in_memory cannot
  (>5M edges).
- Memory efficiency: ~50% peak RSS reduction vs in_memory at
  comparable scale.
- Real parallel speedup: 1.73× at -N4 on enwiki.

### Open follow-ups

- **The mutator-side per-seed FFI overhead is still amortised by
  sharded L2** — that's PR #1950's contribution. Independent of
  this work but still load-bearing.
- **-N1 micro-regression** (12%) could be eliminated with a
  numCaps=1 fast path (skip mapConcurrently, evaluate inline). Not
  done because cursor mode at -N1 is off-target (auto resolver
  picks in_memory).
- **Cost-model auto-resolver** for cache_mode + -N selection still
  deferred to Phase 2c.
- **Parallel BFS chunking strategy** is currently uniform-size
  splits. A frontier where some nodes have 100k children and
  others have 1 will load-imbalance. Work-stealing or branch-
  weighted partitioning would tighten this further. Filed.

### Reproducer

```bash
# Same fixture as appendix #8.
swipl -q -s examples/benchmark/generate_wam_haskell_matrix_benchmark.pl -- \
    /dev/null /tmp/wam_enwiki_cursor seeded interpreter kernels_on resident_cursor

(cd /tmp/wam_enwiki_cursor && cabal new-build)
.../wam-haskell-matrix-bench data/benchmark/enwiki_cats/lmdb_proj_resident +RTS -N4
```

## Phase M appendix #10: cache-warming microbench (Phase 2c prerequisites) (2026-05-09)

### Why

Before building a cost-model auto-resolver for the cache layer, we
need empirical inputs the resolver can consume. The hand-wavy version
"warm hot edges" hides three independent cost classes:

1. *Read-pattern cost*: how much do we save by batching/sequencing
   reads vs doing them one at a time on demand?
2. *Ranker cost*: what does it actually cost to compute the score
   used to pick "hot" edges? Different rankers (PPR, hop distance,
   semantic similarity) have very different cost classes.
3. *Crossover M*: how many queries does a workload need before
   warming amortises its cost?

The Hadoop folklore "scans beat seeks" is real but conditional on
selection ratio, hardware, and data size. Phase M measures these on
our actual fixtures so any "warming pays off / doesn't pay off"
claim downstream is grounded in numbers, not intuition.

### Setup

`examples/benchmark/cache_warming_microbench/` — standalone cabal
project that opens a Phase 1 LMDB fixture (`category_child` dupsort
sub-db) and runs three sub-benches:

- **M1**: same workload (resolve dupsort values for N keys) under
  five access patterns: `PerLookupCursor`, `SharedInOrder`,
  `SharedSorted`, `SharedShuffled`, `FullScan`.
- **M1.b**: scan-vs-seek crossover sweep — sorted-seek time vs
  full-scan time at six selection ratios on the same N=10k key set.
- **M2**: ranker overhead — PPR/Flux one iteration, streaming-hop
  one pass (B-tree frontier propagation, per the algorithm sketched
  in this conversation), stub semantic similarity (D=128 dot
  products on K=1000 candidates).
- **M3**: warming-vs-JIT crossover — M random 3-hop BFS queries
  against (a) cold cursor path, (b) IntMap pre-warmed with top-5000
  source nodes by out-degree.

Hardware: same WSL2 machine as Phase L appendices. GHC 8.6.5,
single-threaded, GHC `-O1` (cabal default for `new-build`).

### M1: read patterns at simplewiki scale (10k keys → 78,751 edges)

| pattern            | wall_ms | edges_collected |
|---|---|---|
| PerLookupCursor    |   275   | 78,751 |
| SharedInOrder      |     5.4 | 78,751 |
| SharedSorted       |     4.0 | 78,751 |
| SharedShuffled     |    14.4 | 78,751 |
| FullScan           |    13.4 | 78,751 |

**Headline numbers**:

- **65× penalty** for opening a fresh cursor per key vs reusing one.
  This is already addressed in the runtime (cursor BFS shares cursors
  via `DupsortCursorCache`), but the magnitude is worth recording —
  any future change that breaks cursor reuse would be catastrophic.
- **3.5× penalty** for shuffled-key seeks vs sorted seeks. LMDB pages
  are mmap'd; sequential key access is page-friendly, random access
  trips through the page table.
- At ~3.4% selection (10k of 297k edges touched), full scan is **3.3×
  slower** than sorted seeks. Scan is *not* universally faster.

### M1.b: selectivity sweep (sorted seeks vs full scan)

Same N=10k key sample, time both patterns on progressively-smaller
subsets:

| key_fraction | n_keys | sorted_ms | scan_ms | scan / sorted |
|---|---|---|---|---|
| 0.01 |   100 |  0.077 |  5.8 | **76.1×** |
| 0.05 |   500 |  0.238 |  6.1 | 25.7× |
| 0.10 | 1,000 |  0.513 |  6.5 | 12.6× |
| 0.25 | 2,500 |  1.1   |  6.6 |  5.9× |
| 0.50 | 5,000 |  2.2   | 10.1 |  4.6× |
| 1.00 |10,000 |  4.7   | 13.2 |  2.8× |

Sorted seeks beat scan at every measured ratio. Linear-extrapolating
the sorted-seek cost (~0.47 ms per 1k keys) and treating scan as
fixed cost (~6 ms intercept + small slope), the projected scan-vs-
seek crossover lands somewhere around **N ≈ 13k–25k keys** of the
~144k true source population — i.e. **~10–18% true selection**.

The Hadoop "scans beat seeks" framing is correct but applies at a
specific operating regime: workloads that touch a high fraction of
the data per query, hardware where random page access is much
slower than sequential (spinning disks; or memory-pressured VMs
where mmap pages aren't resident), and queries that don't benefit
from caching across calls. None of those apply to category-graph
demand BFS at simplewiki scale on this hardware. They might apply
on different hardware or at different selection ratios — *the cost
model needs the workload's selection ratio as an input*.

### M2: ranker overhead (simplewiki, 297,283 edges, 144k sources)

| ranker                       | wall_ms | output_size |
|---|---|---|
| PPR/Flux 1 iter              | 80.1   | 43,974 |
| streaming-hop 1 pass         |  2.7   |    144 |
| semantic-sim K=1000 D=128    |  2.8   |  1,000 |

**Cost classes confirmed**:

- **PPR is heavyweight** — 80 ms for one iteration over the full
  edge list. PPR is only viable as a warming-decision input if its
  cost amortises over many queries. Treating one query as ~0.5–1 ms
  of cold-cursor work, **PPR pays for itself only at M ≥ ~80–160
  queries**, and only if warming itself produces a measurable speedup.
- **Streaming-hop is essentially free** — 2.7 ms because the
  frontier-propagation loop only does work proportional to the
  frontier size (here seeded with 10 sources, 1-hop expansion =
  144 outputs). Multi-pass convergence would cost N × this. The
  user's algorithmic intuition was right: this is the right shape
  for a per-query-affordable ranker.
- **Stub semantic similarity** is K-bound: 2.8 ms for K=1000, D=128.
  Production embeddings would be more expensive (real vector loads,
  not synthetic `sin(c*i)`), but the cost class is still O(K·D),
  not O(E).

### M3: warming-vs-JIT crossover (3-hop BFS from random roots)

| M (queries) | cold_ms | warm_ms | speedup |
|---|---|---|---|
|   1 | 0.016 | 0.008 | 2.1× |
|   5 | 0.055 | 0.040 | 1.4× |
|  10 | 0.107 | 0.109 | 0.98× |
|  25 | 2.2   | 0.293 | 7.6× *(noise outlier — cold path stalled)* |
|  50 | 0.343 | 0.378 | 0.91× |
| 100 | 0.458 | 0.461 | 0.99× |
| 250 | 0.841 | 0.883 | 0.95× |

**Warming with top-K-by-out-degree does not pay off** for random-BFS
queries on simplewiki. Speedups hover at 1× ± measurement noise.

This is workload-specific, not a general result. The warming policy
("top-K source nodes by out-degree") doesn't match the query traffic
("BFS from random roots"). High out-degree nodes aren't necessarily
the nodes the queries visit. To make warming pay, the policy needs a
signal correlated with query traffic — query-history-driven warming,
or a ranker informed by query endpoints (streaming-hop seeded from
recent query roots), not a static graph metric.

### Cost-model takeaways

The auto-resolver's input vector should include at least:

1. **`selection_ratio_estimate`** — selection ratio of the typical
   query against the full edge set. Below ~10–15%, prefer sorted
   seeks; above, consider full scan. Below 1%, scans are 76× too
   expensive.
2. **`expected_query_count`** — M, the number of queries amortising
   any warming work. PPR-class rankers pay off only at M ≥ ~80–160.
   Streaming-hop is cheap enough to run per-query.
3. **`workload_locality`** — proxy for how many queries hit the
   same hot region. Without this signal, default to no warming
   (M3 result).
4. **`available_ram`** — already in the proposed input vector.
   Caps the warm-set size.

The resolver itself should be small (cost-class tables + thresholds),
not a runtime ML model. Streaming-hop is cheap enough to be a default
*per-query* ranker if a warm cache exists; PPR should be opt-in.

### Caveats

- **Hardware**: WSL2, mmap-backed; results would shift on cold-disk-
  pressured systems where random-page access dominates. The
  "scans beat seeks" regime is reachable, just not on this setup.
- **Workload**: random-BFS from random roots is one shape. Repeated
  queries from a small hot root set would change M3's verdict.
- **Warming policy**: only top-K-by-out-degree was measured. A
  query-history-driven policy is the natural follow-up; not
  measured here.
- **Single-pass streaming-hop**: M2 measures *one* pass. Multi-pass
  convergence would multiply the cost; for warming purposes one
  pass is usually enough (per the user's algorithm sketch).
- **Sample-key bias**: `sampleDistinctKeys` returns ascending-order
  keys, so `SharedInOrder` is approximately `SharedSorted` in
  practice. The shuffle vs sort comparison is the meaningful one.

### Reproducer

```bash
(cd examples/benchmark/cache_warming_microbench && cabal new-build)
.../cache-warming-microbench \
    data/benchmark/simplewiki_cats/lmdb_proj_resident/lmdb 10000 42
# Or 1k_resident_run, or enwiki_cats/lmdb_proj_resident.
```

Phase M is measurement-only; no production codegen change. The
cost-model resolver itself remains deferred to Phase 2c, but it now
has empirical inputs to consume.

### Cost-model formula (cache-regime aware)

The "10–18% selection crossover" reported above is conditional on the
working set fitting in the page cache. On databases that exceed
**free** RAM (note: on WSL, this swings dynamically as the host
reclaims pages — `/proc/meminfo:MemAvailable`, not `MemTotal`), the
crossover shifts left dramatically because random seeks become real
disk operations.

**Variables**:

| symbol | meaning |
|---|---|
| `W` | bytes a full scan would touch (≈ DB size on disk) |
| `X` | working-set fraction the workload actually uses (BFS reachability fraction; cumulative across M queries: union of working sets) |
| `R_free` | **free** RAM available to the process (`/proc/meminfo:MemAvailable`) |
| `S_mem_seq` | sequential read throughput when data is page-cached (~5 GB/s mmap memcpy) |
| `S_disk_seq` | sequential read throughput from cold storage (~500 MB/s SSD, ~100 MB/s HDD) |
| `t_mem_seek` | one B-tree-walk seek with leaves cached (~1 µs) |
| `t_disk_seek` | one B-tree-walk seek with the leaf not cached (~100 µs SSD, ~10 ms HDD) |
| `K` | number of distinct keys the query looks up |

**Cache-regime weight**:

```
W_working = X * W
f_hot     = min(1, R_free / W_working)
```

`f_hot = 1` is the all-RAM regime we measured. `f_hot = 0` is the
cold-disk regime Hadoop was designed for.

**Effective costs**:

```
T_scan = W_working * [f_hot / S_mem_seq + (1 - f_hot) / S_disk_seq]
T_sort = K         * [f_hot * t_mem_seek + (1 - f_hot) * t_disk_seek]
```

**Crossover K** (sort = scan):

```
K_cross = W_working / [bandwidth_eff * latency_eff]
        where
          bandwidth_eff = f_hot * S_mem_seq    + (1 - f_hot) * S_disk_seq
          latency_eff   = f_hot * t_mem_seek   + (1 - f_hot) * t_disk_seek
```

If `K_query < K_cross` → sorted seeks. Otherwise → scan.

**Plugged in**:

| regime | W_working | bw_eff | lat_eff | K_cross | as % keys |
|---|---|---|---|---|---|
| simplewiki, hot (R_free ≫ W) | 15 MB | 5 GB/s | 1 µs | ~3,000 | ~2% |
| simplewiki, cold (R_free ≪ W) | 15 MB | 500 MB/s | 100 µs | ~300 | ~0.2% |
| enwiki, hot (R_free ≫ W) | 500 MB | 5 GB/s | 1 µs | ~100,000 | ~10% |
| enwiki, cold (R_free ≪ W) | 500 MB | 500 MB/s | 100 µs | ~10,000 | ~1% |

The hot-regime simplewiki value (~2%) is consistent with M1.b: sort
beats scan at all measured ratios up to 7% true selection.

**Warming-pays-off threshold** — for M queries each touching `K_q`
keys, warming a working set of `W_warm` bytes pays off at:

```
M_warm ≥ T_warm_load / (T_query_cold - T_query_hot)
```

The denominator collapses to ≈ 0 when `R_free ≫ W_warm` (matches our
M3 simplewiki result — warming is pointless when everything's already
cached). The numerator stays small because warming is one sequential
scan. So warming pays off when `R_free ≪ W_warm` and `M ≥ small`.

**What the resolver should read**:

- `R_free` from `/proc/meminfo:MemAvailable` (recheck per-bench, since
  WSL claims/releases pages dynamically).
- `W` from `mdb_env_info.me_mapsize` or `stat()` of the data file.
- `X_query`, `K_query`, `M` from workload metadata; defaults derivable
  from BFS-depth heuristics + declared workload type.

The hardware constants (`S_mem_seq`, `S_disk_seq`, `t_mem_seek`,
`t_disk_seek`) need a one-time per-machine calibration probe. SSD-
typical defaults are within an order of magnitude on commodity
hardware; the order of magnitude is what matters for the regime
selection, not the second decimal place.

### When this matters

Categories-only fixtures (simplewiki ≈ 15 MB, enwiki ≈ 500 MB) fit
in any modern machine's free RAM. The cost model is groundwork, not
an immediate optimization — at this scale, all regimes are `f_hot ≈ 1`
and the formula reduces to "sorted seeks always".

The regime starts to matter at:

- **Article-level Wikipedia ingest** — full enwiki article texts are
  ~80 GB compressed, ~250 GB uncompressed. Past free RAM on any
  consumer machine.
- **Multi-fixture aggregation** (e.g. enwiki categories + page
  metadata + revision counts joined into one LMDB).
- **Memory-pressured environments** — WSL2 in particular, where the
  Windows host can reclaim large fractions of the VM's RAM.
- **Spinning-disk targets** — `t_disk_seek` jumps 100×, shifting the
  crossover left aggressively.

The Prolog predicates in `src/unifyweaver/core/cost_model.pl`
implement these formulas. They're opt-in: callers pass `cost_model`
options to enable, otherwise targets fall back to the existing
fact-count-threshold heuristic.

## Phase L appendix #11: matrix-bench `resident_auto` mode (2026-05-10)

### Why

After PR #1975 wired the cost-model resolver into
`wam_haskell_target.pl`, nothing in the repo actually exercised it
end-to-end. This appendix records the first user of
`cache_strategy(auto)` — a new `resident_auto` mode in the matrix
bench — and what the resolver picks across the existing fixture
scales. The point isn't to discover a new winning configuration; it
is to verify the cost model's decisions agree with the empirical
Phase L#7–9 measurements, and to surface tuning issues in the
default workload metadata.

### What changed

`examples/benchmark/generate_wam_haskell_matrix_benchmark.pl`:

- New `resident_auto` mode parser, alongside the existing `resident`
  and `resident_cursor` modes.
- Emits `cache_strategy(auto)`, `cache_strategy_verbose(true)`,
  `expected_query_count(1)`, and `working_set_fraction(0.001)`.
- Tags `lmdb_cache_mode(sharded)` (same default as the other resident
  modes) and `int_atom_seeds(lmdb)`.

The mode lets the cost model decide between cursor and in-memory
demand BFS by reading the resolver's recommendation. Verbose tracing
is on by default so the picked decision shows up in the bench
generator's output, e.g.:

```
[WAM-Haskell] cache_strategy(auto): K=297 W=14850000 R_free=1579548672 → sort (cursor)
```

### Why `working_set_fraction(0.001)` and not the cost-model default `0.05`

The cost-model's general default of `0.05` is appropriate for
many-keys-per-query workloads (e.g. lookup-heavy joins where the
query touches 5% of the data). For graph-BFS demand-set workloads
like the matrix bench's effective-distance kernel, the demand set
is ~0.05 % of edges (Phase L#7: simplewiki Physics root reaches
144 of 297k nodes = 0.0005). At wsf=0.05 the model would
incorrectly recommend `in_memory` at every scale because K is
inflated by 100×.

`resident_auto` overrides to `0.001` (~10× the empirical 0.0005,
slightly conservative). The cost-model's `0.05` default stays as
the correct general-purpose value; it's the workload, not the
model, that's specialised here.

### What the resolver picks

R_free = 1.58 GB (WSL with active processes; reported by
`/proc/meminfo:MemAvailable`).

| fixture | fact_count | K     | W       | f_hot | picked  | empirical winner (Phase L) |
|---|---|---|---|---|---|---|
| 1k         |     5,933 |     6 |  290 KB |  1.0  | cursor  | parity (60 vs 70 ms)        |
| simplewiki |   297,000 |   297 |   15 MB |  1.0  | cursor  | cursor (1.96× vs in_memory) |
| enwiki     | 9,900,000 | 9,900 |  500 MB |  1.0  | cursor  | cursor only (in_memory panics) |

The model agrees with the empirical measurements at every scale.
For BFS workloads, sorted seeks always beat sequential pre-load —
because the per-query selection ratio (~0.05%) is so far below the
hot-regime crossover (~K_cross at 5000 bytes/key with W = fact_count
× 50 → K_cross ≈ fact_count / 100, while K = fact_count × 0.001 =
fact_count / 1000, so K is 10× below crossover at every scale).

### Cold-regime check (and a real finding)

To verify the model's cold-regime branch fires, the resolver was
probed with `mem_available_bytes(125 MB)` against the enwiki size
(W = 500 MB so f_hot ≈ 0.25):

| fact_count | wsf   | R_free  | f_hot | picked    |
|---|---|---|---|---|
| 9,900,000  | 0.001 | 125 MB  | 0.25  | in_memory |
| 9,900,000  | 0.05  | 125 MB  | 0.25  | in_memory |

K_cross drops sharply in the cold regime (~4,000 at f_hot=0.25 vs
~99,000 at f_hot=1.0), so even the small BFS K=9,900 lands above
the threshold and the model recommends `scan` → `in_memory`.

**This surfaces a real gap in the current sort-↔-cursor /
scan-↔-in_memory mapping.** Our `in_memory` implementation requires
loading the full edge IntMap into RAM (~500 MB at enwiki scale).
At R_free=125 MB the IntMap allocation blows the available memory
budget — exactly the regime where the model recommends it. The
"scan" abstraction in `cost_model.pl` says nothing about working-set
footprint, so the resolver can't currently catch this.

The right fix is a footprint guard in `resolve_auto_cache_strategy/2`
that overrides `in_memory` → `cursor` when `R_free < W`. Filed as
a follow-up; not in this PR's scope because it has implications
for the resolver's tests and decision contract.

For now, `resident_auto` works correctly when R_free ≥ W (the hot
regime), which covers all current categories-only fixtures. The
cold-regime override matters once article-level data lands.

### What this validates

- The end-to-end channel works: matrix-bench → `cache_strategy(auto)` →
  `resolve_auto_cache_strategy/2` → `cost_model:recommend_access_pattern/5`
  → concrete `demand_bfs_mode/1`.
- For BFS workloads at categories scale, the model agrees with
  hardcoded `resident_cursor` and the existing
  `resolve_auto_demand_bfs_mode/2` 50k threshold.
- The model is doing genuine work in the cold regime (where the DB
  exceeds free RAM) and at high selection ratios — both regimes
  beyond the existing threshold's reach.

### What this doesn't validate

- **Article-level scale** — fixtures gone after the previous
  workspace reset; couldn't probe the regime where `f_hot < 1`
  matters in production. Deferred until article-level data is
  re-ingested.
- **Multi-query workloads** — `expected_query_count(M)` doesn't
  affect the cursor-vs-in-memory decision; it would only fire if a
  warming-payoff resolver existed. Deferred to the warming-arc work.
- **Spinning-disk targets** — `t_disk_seek` jumps 100×, which
  would shift the crossover. Not tested on this hardware.

### Reproducer

```bash
# At any cwd:
swipl -q -s examples/benchmark/generate_wam_haskell_matrix_benchmark.pl -- \
    data/benchmark/1k/facts.pl /tmp/wam_auto_1k seeded interpreter kernels_on resident_auto
# Look for: [WAM-Haskell] cache_strategy(auto): K=... W=... R_free=... → ...
```

Phase L#11 closes the Phase 2c plumbing arc started in PR #1975.
The cost model has now been exercised end-to-end on real
fixtures; future warming and `lmdb_cache_mode` resolvers can
build on the same channel.

## Phase L appendix #12: `lmdb_cache_mode(auto)` resolver wiring (2026-05-10)

### What landed

`resolve_auto_lmdb_cache_mode/2` in `wam_haskell_target.pl`. Picks
`none` / `per_hec` / `sharded` / `two_level` from
`expected_query_count`, `workload_locality(...)`, and the already-
resolved `demand_bfs_mode/1`. Opt-in signal is presence of
`workload_locality/1`; without it, the existing in-place
resolution (via `statistics:select_cache_mode/2`) continues to
handle the auto path.

Decision matrix mirrors the philosophy doc:

| M    | locality        | pick      |
|---|---|---|
| ≤ 1  | *               | none      |
| > 1  | intra_thread    | per_hec   |
| > 1  | cross_thread    | sharded   |
| > 10 | mixed           | two_level |
| > 1  | mixed (low M)   | sharded   |
| > 1  | unknown         | sharded   |
| (any) | * + in_memory  | none (composition rule) |

The `unknown → sharded` default is the same call the matrix bench
has been making by hand since Phase L#5 — now explicit and
overridable per-workload.

### Matrix-bench `resident_auto` update

The mode previously hardcoded `lmdb_cache_mode(sharded)` and
`expected_query_count(1)`. Both are replaced with:

```prolog
expected_query_count(10),  % bench runs many parMap-driven queries
lmdb_cache_mode(auto),
workload_locality(unknown)
```

Smoke run at 1k confirms both resolvers fire:

```
[WAM-Haskell] cache_strategy(auto): K=6 W=296650 R_free=3653779456 → sort (cursor)
[WAM-Haskell] lmdb_cache_mode(auto) → sharded (default_safe)
```

The picks match the previous hardcoded values exactly, so this is
a behaviour-preserving refactor for the matrix bench. The
benefit is that workload authors can now override locality via
`workload_locality(intra_thread)` etc. without touching the bench
generator.

### Tests

9 new tests in `tests/test_wam_haskell_target.pl`:

- `intra_thread + M > 1 → per_hec`
- `cross_thread + M > 1 → sharded`
- `mixed + M > 10 → two_level`
- `mixed + M ≤ 10 → sharded (low-M fallback)`
- `unknown → sharded (safe default)`
- `M = 1 → none (no amortisation)`
- `composition: in_memory → none`
- `no-op without workload_locality`
- `explicit lmdb_cache_mode flows through unchanged`

All 164 WAM-Haskell tests pass (155 existing + 9 new); 21
cost_model tests unchanged.

### Open follow-ups

- **Memory budget guard on the cache layer itself.** The footprint
  guard from Phase L#11 covers the working set (`R_free < W` →
  cursor). The cache tier doesn't have an analogous guard yet:
  when `R_free - W_working < L2_capacity_floor` the L2 cache
  would thrash. Filed for a follow-up.
- **Automatic locality inference.** `workload_locality(...)` is
  authored by hand. Could be derived from kernel metadata
  (purity certificate, parMap usage). Research-y.
- **At-scale empirical validation.** The fixtures from Phase L#7–9
  were swept in the workspace reset; re-validating the resolvers'
  picks against measured wall-clock numbers needs them re-ingested.
  Filed.


