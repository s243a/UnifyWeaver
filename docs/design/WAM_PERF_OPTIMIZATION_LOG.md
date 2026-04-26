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
