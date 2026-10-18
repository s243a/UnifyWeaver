<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) 2026 John William Creighton (s243a) -->

# Rust WAM (uw-resolve) hotspot profile — FULL lowered tier ON

Where does the Rust WAM target spend its time **now** that the deterministic
lowering (regions 1–5 + the general recognizer, D78–D89, all default-ON) has
removed the old interpreter-dispatch hotspots? Measured on both representative
workloads, **B2** (2,600-case term differential) and **B3** (5,000-package
`resolve_layered`), and compared against the pre-lowering baseline the status
doc records (**~65 % step-dispatch / 21 % backtrack / 12 % restore_regs**).

**Content SHA.** `origin/claude/peerhailer-exploratory-docs-aodas5` @
`3f75765c2` (ledger D90 — regions 1–5 + genrec default-ON; 9 `region_*_dispatch`
arms in the generated `lib.rs`). MEASUREMENT ONLY: `resolver.pl`,
`resolver_store.pl`, and all target/spec/crate source were left unmodified; the
instrumentation lives in a throwaway copy of the crate under a scratch path,
never committed.

## TL;DR verdict

- **The old profile is gone.** Interpreter dispatch machinery (`step` + `run` +
  `backtrack` + `restore_regs` + register access, self-Ir) is now **~3.3 % of
  B2** and **~0.3 % of B3**. The lowering did what it set out to do: the deep
  interpreted recursion that was 65 % of the old profile no longer exists.
- **What replaced it is the runtime term/allocation subsystem, which lowering
  cannot touch.** By self-Ir: **B2 = ~61 % memory management (malloc/free/drop)
  + ~16 % term deref + ~11 % "f/N" functor-string handling**; **B3 = ~49 %
  memory management + ~24 % term deref + ~22 % functor-string + sort-comparator.**
- **B2 and B3 are bound by different things.** B2 is **backtrack-and-allocate**
  bound (479,531 backtracks over 2,600 small nondeterministic queries; `backtrack`
  inclusive-Ir = **72 %**). B3 is **sort-and-deref** bound (only 1,296 backtracks;
  the cost is one big `msort/2` whose comparator `term_compare` is **63 %**
  inclusive and whose `sort_by` is **46 %** inclusive).
- **The declined predicates are almost 100 % of the *remaining interpreter
  dispatch*, but that dispatch is only ~3 % of B2 / ~0 % of B3 of the time.** So
  **more genrec coverage has a ceiling of ≈3 % on B2 and ≈0 % on B3.** The lever
  is **not** more lowering.
- **Next lever (ranked): (1) intern functors** (u32 keys, kill the "f/N" string
  bucket — ~11 % B2, ~22 % B3, helps everything); **(2) cut term/Value
  allocation churn** — `deref_heap` (142 M calls in B2), the B2 backtrack
  saved-state clone, and the B3 `msort/2` decorate-sort key-caching; **(3) the
  store path for B3 scale** (already ~10× the term path). More genrec is #4 with
  a single-digit ceiling.

## Methodology

Two complementary instruments, both against the D90 default build (full tier ON,
release, this box, `LC_ALL=C.UTF-8`, nothing else heavy running):

1. **Per-predicate dispatch census + phase counters** (adapting the F11 census,
   `wam_rust_f11_census_and_stage1.md`). A throwaway copy of the generated crate
   was instrumented with a `UW_PROF`-gated, thread-local counter at the `Call`
   and `Execute` arms — split into **interp** (fell through to the interpreter
   label/builtin path) vs **lowered** (a `region_*_dispatch` fired) — plus scalar
   counters at `backtrack`, `restore_regs`, `save_regs`, `trail_binding`, `unify`,
   `deref_heap`. Run over the **full** B2 corpus (2,600) and B3 (5,000). The
   instrumented `uw_resolve` produced **byte-identical** B2 output to the default
   binary (`cmp` clean), so the counters are inert to semantics.
2. **Callgrind (`--cache-sim=no --branch-sim=no`)** on the **default** (clean,
   uninstrumented) binary — the same tool that produced the 65/21/12 baseline.
   Instruction-reads (Ir) are the cost proxy. B2 was profiled on a 150-case
   subset (per-function proportions are stable across cases, and the full-corpus
   census confirms the same predicate mix); B3 on the full 5k resolve.

**Timing reference** (clean default binary, this box): B2 rust leg **19.2 s**
(SWI ≈2.6 s, ≈7.2×); B3 5k `resolve_layered` **≈372–398 ms** resolve, ~23 ms
load; B3 store lane **≈40 ms** (≈10× the term lane). These match the D89 refresh
(`wam_rust_bench_refresh_full_tier.md`), so the profiled build is the shipping
one.

**Limits.** (a) Ir counts instructions, not cycles; cache/branch simulation was
off, so **memory-latency is *under*-counted** — the true wall-clock share of the
allocator is very likely *higher* than the Ir share shown. (b) Inclusive-Ir
percentages overlap (a callee's cost appears under every caller); the phase
*rollups* below are computed from **self-Ir** to avoid double-counting, with
inclusive figures given separately for the named functions. (c) The dispatch
counters are thread-local; the resolve/differential paths run single-threaded
(rayon is only on the aggregate/matrix kernels, unused here), verified by the
counters landing on the main thread.

## B2 — hotspot breakdown (term differential, 2,600 cases)

### Phase rollup (callgrind self-Ir, 6.17 B Ir on the 150-case subset)

| phase | self-Ir share | components |
|---|---:|---|
| **Memory management** | **61.5 %** | alloc/free (libc+rust shim) 51.6 %, drop `Value` trees 7.1 %, memcpy (clone/realloc) 2.8 % |
| **Term deref** | **16.2 %** | `deref_heap` 10.9 %, `deref_var` 3.7 %, `same_cell` 1.7 % |
| **"f/N" functor strings** | **10.8 %** | `String` clone 4.1 %, functor parse/compare (`memrchr`/`CharSearcher`/`functor_of`/`memcmp`) 6.7 % |
| iter/vec + hashing | 4.7 % | map/cloned/`Vec::clone`, SipHash |
| **Dispatch machinery** | **3.3 %** | `step`/`run`/`backtrack` self 1.0 %, `save`/`restore_regs` 1.2 %, reg access 0.9 %, trail 0.1 % |

### Inclusive-Ir of the named functions (for direct before/after)

`run` 97.2 % · **`backtrack` 71.8 %** · **`deref_heap` 54.2 %** · `step` 31.5 % ·
`deref_var` 11.5 % · `functor_of` 5.8 % · `execute_builtin` 4.0 % · `restore_regs`
3.2 % · `trail_binding` 1.7 % · `unify` 1.0 % · every `region_*_dispatch` < 0.4 %.

**Reading.** B2 is a nondeterministic search: 2,600 small catalogs, each query
enumerating candidates/versions/conflicts and backtracking. `backtrack`'s 71.8 %
inclusive is **not dispatch** — its self-Ir is 0.2 %; the cost under it is
`saved_args` cloning (a `Vec<(usize,Value)>` of live registers, cloning `Value`
trees) on each of **479,531** backtracks, plus `drop_in_place<Value>` on
heap/trail truncation. `deref_heap` is the hottest single Rust function
(**142,796,050 calls**), and it plus the allocator it feeds is where the cycles
go. The lowered regions are essentially free here (all < 0.4 % inclusive) — they
successfully moved their predicates' work to near-zero.

## B3 — hotspot breakdown (5,000-package `resolve_layered`)

### Phase rollup (callgrind self-Ir, 4.01 B Ir)

| phase | self-Ir share | components |
|---|---:|---|
| **Memory management** | **49.3 %** | alloc/free 39.5 %, drop `Value` trees 6.7 %, memcpy 3.2 % |
| **Term deref** | **23.7 %** | `deref_heap` 14.9 %, `deref_var` 5.5 %, `same_cell` 3.3 % |
| **"f/N" functor strings + compare** | **21.9 %** | functor parse (`memrchr` 5.2 %/`CharSearcher` 4.9 %/`functor_of` 1.5 %/`memcmp` 1.4 %) 14.6 %, `String` clone 5.7 %, `term_compare` self 1.7 % |
| region native (self) + json load | 1.0 % | |
| **Dispatch machinery** | **0.3 %** | `step`/`run`/`backtrack`/`restore_regs` combined |

### Inclusive-Ir of the named functions

`run` 89.1 % · `step` 87.1 % (self ~0.4 %; the outer dispatcher) · **`term_compare`
63.1 %** · **`deref_heap` 62.1 %** · **`sort_by`/driftsort 46.5 %** ·
`drop_in_place<Value>` 24.0 % · `deref_var` 13.5 % · `functor_of` 7.1 % ·
`region_group_keyed_dispatch` 6.1 % · `region_key_dep_rows_dispatch` 2.1 % ·
`backtrack` 1.8 % · `restore_regs` 0.3 % · `unify` 0.06 %.

**Reading.** B3 barely backtracks (**1,296** total) and barely dispatches
(**1,190** interpreter Call/Execute for the whole resolve). Its cost is one
dominant primitive: the **`msort/2` / `sort/2` builtin** (`resolver.pl` uses
`sort/2` extensively for the index build and topo order at lines 372/376/824/…).
That builtin **eagerly `deref_heap`s every element** and then
`sort_by(term_compare)`; `term_compare` recurses the terms, re-dereferencing and
comparing functors as `"name/arity"` strings — so the sort's comparator is where
`deref_heap` (62 %), `term_compare` (63 %) and the functor-string parse (`memrchr`
+ `CharSearcher` ≈ 10 %) all pile up. It is **not** a lowering problem.

## Before / after vs the old 65 / 21 / 12

The old baseline (`WAM_RUST_STATUS.md`, callgrind of the **363-package resolve**,
post-D52 structural-sharing, **pre**-lowered-tier — a B3-shaped workload) was
**inclusive** Ir:

| function (inclusive Ir) | OLD 363-pkg (pre-lowering) | NOW B2 | NOW B3 (5k) |
|---|---:|---:|---:|
| `WamState::step` | **65 %** | 31.5 % | 87.1 %\* |
| `WamState::backtrack` | **21 %** | 71.8 % | 1.8 % |
| `restore_regs` | **12 %** | 3.2 % | 0.3 % |
| `deref_var`+`deref_heap` | 12 % | ~55 % | ~62 % |
| `unify` | 7 % | 1.0 % | 0.06 % |
| malloc/free (**flat/self**) | **31 %** | **~51 %** | **~40 %** |

\* B3 `step` is inclusive-of-the-whole-resolve (its *self* is ~0.4 %); it is the
outer dispatcher wrapping the native regions and the sort builtin, not
interpreted recursion.

**What the lowering shifted:**

- **Dispatch is no longer dominant.** The interpreter's own machinery
  (`step`+`backtrack`+`restore_regs`+reg access, *self*-Ir) collapsed to **3.3 %
  (B2)** and **0.3 % (B3)**. `restore_regs` fell out of the top (12 % → 3.2 %/0.3 %).
- **Allocation rose to the top.** The comparable *flat* malloc/free share went
  **31 % → ~51 % (B2) / ~40 % (B3)**; with `drop_in_place<Value>` and memcpy the
  memory subsystem is **~61 % (B2) / ~49 % (B3)** of self-Ir. Cache misses (off
  in this run) mean the wall share is likely higher still.
- **`deref_heap` and the "f/N" functor strings rose** from a shared 12 % to the
  #1 non-allocator costs (`deref_heap` inclusive 54 %/62 %; functor-string 11 %/
  22 % self).
- **B2 and B3 diverged.** `backtrack` inclusive went **21 % → 72 % on B2**
  (nondeterministic search over many small catalogs) but **21 % → 1.8 % on B3**
  (deterministic index build + sort). The single old 65/21/12 profile no longer
  describes both.

## Attribution to the DECLINED predicates (interpreter fallback)

The declined predicates own essentially **all** of the *remaining interpreter
dispatch*, but that dispatch is a small slice of the *time*.

**B2 interpreter dispatch = 451,355 Call/Execute (vs 9,506 region entries).** The
regions absorbed the old hot recursions (`matching_deps` went **112,758 → 3,952**
region entries; `matching_versions` **82,786 → 3,081**; the recursion now runs in
native Rust with no interpreter dispatch). What remains, by family:

| declined family | interp dispatch | share of interp | lowered by genrec? |
|---|---:|---:|---|
| **mutual-recursion SCC** `lookup_held/3` ↔ `item_ver/3` | 100,539 | **22.3 %** | no — cross-predicate SCC, genrec only handles single-predicate self-recursion |
| catalog accessors `provides_list`/`conflicts_list`/`base_list`/… | 85,908 | 19.0 % | no — F11-eligible pure bodies, but F11 is opt-in and **off** (measured net-negative) |
| version/constraint chain `long_enough`/`selected_ver`/`base_ver`/… | 74,591 | 16.5 % | no — non-tail / meta |
| conflict/provides/dep bodies `direct_on`/`conflicts_in`/`no_acc_conflicts`/`provides_sat`/`scan_base_holds`/… | 90,832 | 20.1 % | no |
| everything else (drivers `pick_need`/`resolve_pending`/`collect_deps`, `blocked_from`, `pick/7` [dead], topo/`close_moving`, meta-calls, misc) | 99,485 | 22.0 % | no |

**B3 interpreter dispatch = 1,150 (vs 40 region entries).** The lowering removed
essentially all of B3's old dispatch (census had 91,559): `build_tree`,
`key_dep_rows`, `group_keyed`, `key_pkg_rows`, `same_key`, `tree_lookup`,
`filter_satisfies` all run native. The declined preds fire only a few hundred
times total (`conflicts_list` 180, `provides_list` 158, `selected_ver` 140, …,
`lookup_held` 24, `item_ver` 14).

**The decisive number.** Dispatch machinery is **3.3 % of B2** and **0.3 % of
B3** self-Ir. So eliminating the interpreter fallback for *every* declined
predicate — the whole 451,355 B2 dispatches — has a **ceiling of ≈3 % on B2 and
≈0 % on B3**. The declined predicates are *not* expensive because they are
interpreted; they are expensive (where they are) because their bodies build,
deref and free `Value` terms — the same runtime cost the native regions still
pay. Lowering `lookup_held`/`item_ver` to native would remove interpreter frames
and the ~3 % dispatch, but **not** the term-alloc/deref underneath it. This is
the answer to the "is the lever more genrec coverage, or a runtime hotspot?"
question: **it is a runtime hotspot.**

## Ranked recommendation for the next lever

Ordered by expected impact per unit risk, with which workload each moves.

### 1. Intern functors as integers (kill the "f/N" string bucket) — helps BOTH

Terms carry their functor as a `"name/arity"` **String**; the runtime repeatedly
`rsplit('/')`s it (`memrchr` + `CharSearcher`), clones it, and `memcmp`s it. That
is **10.8 % of B2** and **21.9 % of B3** self-Ir directly, and it inflates
`term_compare`, `same_cell`, hashing and the allocator on top. The FFI graph
kernels already intern atoms to u32 and measured **~7.9×** there
(`WAM_CROSS_TARGET_BENCHMARK_RESULTS.md`) — extend the same interning to the WAM
`Value` functor/atom representation.
**Expected ceiling: ~11 % B2, ~20 % B3 direct**, plus a secondary allocation cut
(fewer `String` allocs/clones). Single biggest lever for B3 after #2c; the
largest *portable* lever for B2.

### 2. Cut `Value`-term allocation churn (the 50–62 % memory bucket) — helps BOTH

The allocator dominates because terms are heap-`Value` trees that are cloned,
materialised and dropped constantly. Three concrete, independently-shippable sub-levers:

- **2a. `deref_heap` (142 M calls B2 / 5.4 M B3; hottest Rust fn).** Even after
  the D59 lazy-vector fix it still returns a cloned `Value` whenever any child
  dereferences to a new cell, and it is re-invoked on the same terms constantly.
  Memoise/cache deref results, or represent already-ground terms so `deref_heap`
  is a cheap identity. **Moves both**; largest share of the non-allocator top.
- **2b. B2 backtrack saved-state.** `backtrack` clones `saved_args` (live
  register `Value`s) on each of 479,531 backtracks and drops term trees on
  heap/trail truncation (`backtrack` inclusive 72 %). A minimal saved-locals
  representation (the plan's **P2**) and trailing only changed cells would cut
  this. **Moves B2 specifically** (B3 hardly backtracks).
- **2c. B3 `msort/2` decorate-sort-undecorate.** The builtin eager-`deref_heap`s
  every element and runs `term_compare` (deep, re-derefing, string-functor)
  O(n log n) times. Extract a cheap sort key **once per element** (O(n)) and sort
  on cached keys. **Moves B3 specifically** — with interning (#1) this directly
  attacks the `sort_by` 46 % / `term_compare` 63 % inclusive that *is* B3.

**Expected ceiling: large but bounded per sub-item.** 2c + #1 together plausibly
halve B3's dominant sort cost; 2a helps every workload; 2b is the main B2-only win.

### 3. Promote the store/seek path as the default large-catalog resolve — B3 only

B3's store lane already resolves the 5k catalog in **~40 ms vs ~385 ms** for the
term lane (≈**10×**, touching 0.90 % of the store; D89). For scale, the biggest
available B3 win is architectural: make the store/`LookupSource` path the default
for large catalogs rather than building the whole catalog as in-memory `Value`
terms. **Does nothing for B2** (small catalogs, term lane is the point) and it is
a project/architecture decision, not a codegen lever — hence ranked below the
representation fixes that help both lanes.

### 4. Extend genrec to the declined predicates — LOW ceiling now

The mutual-recursion SCC `lookup_held`↔`item_ver` (22 % of B2 dispatch) and the
catalog accessors (19 %) are the obvious remaining targets, and lowering them
would be sound. But dispatch machinery is **3.3 % of B2 / 0.3 % of B3**, so the
ceiling of removing interpreter fallback is **≤3 % B2, ~0 % B3**. Only worth
doing if the native version *also* avoids `deref_heap`/alloc (i.e. bundled with
#1/#2); on its own it repeats the F11 result (net-neutral to negative), because
the cost was never the dispatch. **Do not lead with this.**

### Which lever moves which workload

| lever | B2 (backtrack+alloc bound) | B3 (sort+deref bound) |
|---|---|---|
| #1 intern functors | ~11 % | ~20 % |
| #2a `deref_heap` churn | yes | yes |
| #2b backtrack saved-state | **yes (main B2 win)** | ~0 (no backtracking) |
| #2c `msort` key-caching | ~0 (little sorting) | **yes (main B3 win)** |
| #3 store default | ~0 | **~10× at scale** |
| #4 more genrec | ≤3 % | ~0 % |

## Appendix — raw counters (instrumented build, full corpora)

**B2 scalars:** backtrack 479,531 · restore_regs 724,741 · save_regs 694,052 ·
trail_binding 4,619,157 · unify 963,562 · **deref_heap 142,796,050** · dispatch
interp 451,355 / lowered 9,506.
**B2 region entries:** matching_deps 3,952 · matching_versions 3,081 · dep_breaks
914 · tree_lookup 792 · filter_satisfies 359 · group_keyed 136 · build_tree 136 ·
key_pkg_rows 68 · key_dep_rows 68.

**B3 scalars:** backtrack 1,296 · restore_regs 1,296 · save_regs 1,004 ·
trail_binding 10,790 · unify 1,013 · **deref_heap 5,367,137** · dispatch interp
1,150 / lowered 40.
