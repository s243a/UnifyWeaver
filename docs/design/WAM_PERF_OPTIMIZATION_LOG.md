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

## Related documents

- `docs/vision/HASKELL_TARGET_ROADMAP.md` — where this is going
- `docs/vision/HASKELL_TARGET_PHILOSOPHY.md` — why Haskell as a target
- `docs/design/WAM_HASKELL_PERF_IMPLEMENTATION_PLAN.md` — earlier plan,
  now largely delivered
- `docs/design/WAM_HASKELL_FFI_PROFILING_REPORT.md` — the profiling
  matrix that drove the Phase-D wins
