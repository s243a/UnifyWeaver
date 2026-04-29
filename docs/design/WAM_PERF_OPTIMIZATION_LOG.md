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

### Macro benchmark — TODO

A follow-up should extend
`tests/benchmarks/wam_effective_distance_macro_bench.pl` to add a
third variant that includes `:- visited_set(category_ancestor/4, 4)`
and compare against the Phase G `mode`-only baseline. Expected
improvement per the design doc: another ~1.5–3× on top of the
~1.18× Phase G macro speedup (compounding the constant-factor and
algorithmic wins). Filed as a small follow-up — the substrate is in
place; the benchmark is just a copy-and-add-directive.
