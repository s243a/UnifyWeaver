<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) 2026 John William Creighton (s243a) -->

# uw-resolve — cross-target benchmarks

One frozen Prolog program (`resolver.pl`, P0.5 semantics with genuine
backtracking), compiled through many WAM targets and timed on identical data
with identical queries against SWI-Prolog as the oracle. Every leg passes the
same semantics gates before it is timed:

- the **51-scenario** SWI-oracled contract corpus (`corpus 51/51 matched SWI`),
- the **2,600-case** seeded term differential with **0 divergences** (term
  legs), and
- the **503-case** seeded *store* differential on the 5k catalog with **0
  divergences** (store-backed legs).

**Measured on the coordinator container (4 cores, `nproc`=4, Linux),
2026-09-07; absolute times vary by machine — the *ratios* are the result.**
All legs were re-run on this one box so the numbers are internally
comparable. The term legs (Go, Rust, wamjs, ClojureScript) were measured
sequentially on an otherwise-idle box; every leg returns the identical
10-package selection for the 5k `resolve_layered(p30)` query.

**What changed since the 2026-09-07 (D82) table.** The Rust term lane's
*deterministic lowered tier* is now **complete**: the five fused native regions
(D78–D81, regions 1/2/3a/3b/4) are joined by **region 5** (`dep_breaks/5`, D88,
[`wam_rust_stage2_region5.md`](../../docs/reports/wam_rust_stage2_region5.md),
~−3% B2) and the **general deterministic-recursion recognizer** (D89, the
sibling-gap family `filter_satisfies`/`key_pkg_rows`/`tree_lookup`,
[`wam_rust_genrec_sibling_gap.md`](../../docs/reports/wam_rust_genrec_sibling_gap.md),
~−18% to −24% on the 5k `resolve_layered` path). All are default-**ON**, gated,
0 divergences. On this box that takes Rust term **B2 to ~19.1 s (~7.2× SWI)** and
Rust term **B3 `resolve_layered` to ~0.385 s** — the headline: the cumulative
regions-ON-vs-OFF B3 delta is now **−80.7%** (pristine interpreter ~1927 ms →
full tier ~372 ms), up from D82's −74%. See the
[Stage-2 note](#stage-2-lowered-tier-rust-deterministic-region-fusion--the-general-recognizer)
below. The store lanes are **unaffected** (the index builders are never invoked
on the store resolve path) and re-measure identical to D82 within box noise, with
byte-identical bytes-read/read-count.

## Main table (single box, 2026-09-07, full Rust lowered tier)

| Leg | B1: corpus (51 scen.) | B2: differential | B3: `resolve_layered`, 5k catalog (load / resolve) |
|---|---:|---:|---|
| SWI-Prolog (oracle) | 0.113 s | **2.6 s** (2600 term) | 0.027 s / **0.020 s** ⁷ |
| Go WAM (term) | 0.070 s | 29.5 s (2600 term) | 0.098 s / 15.12 s |
| **Rust WAM** (term) | **0.048 s** | 19.1 s (2600 term) | 0.024 s / **0.385 s** ⁸ |
| wamjs (term) | 0.209 s | **16.5 s** (2600 term) | — (store row is the wamjs B3 path)⁴ |
| ClojureScript (nbb) | 2.02 s | 107.7 s (2600 term) | 0.228 s / 28.24 s |
| **wamjs store** (D48) | 0.209 s¹ | 9.23 s (503 store)² | store-seek / **0.163 s**³ |
| **Go store** (D70) | corpus 51/51¹ | 34.8 s (503 store)² | store-seek / **0.56 s**⁵ |
| **Rust store** (D73) | corpus 51/51¹ | 3.74 s (503 store)² | store-seek / **0.040 s**⁶ |

All gates passed on this box for every measured leg (corpus 51/51; term
differential 2,600 cases / 0 divergences; store differential 503 cases /
0 divergences).

¹ Store legs share the term legs' B1 corpus binary/gate (corpus 51/51 matched
SWI, and the Go/Rust store corpus is byte-identical to their term corpus); the
store path is exercised at scale in B3, so B1 is not a separate store number.
² Store legs do **not** run the 2,600-case term differential (they read from
the indexed seek store, a different data path). Their differential gate is the
**503-case store differential** on the 5k catalog; the times shown are that
leg vs the SWI store adapter on the same 503 cases (wamjs store 9.23 s vs SWI
0.67 s; Go store 34.8 s vs SWI 0.66 s; Rust store 3.74 s vs SWI 0.69 s),
0 divergences.
³ **wamjs store B3** (`run_scale_demo.sh`, lazy materialisation): the catalog
is seek-read from the D43 indexed stores — **10,305 of 1,646,323 store bytes
touched (0.63%)**, 820 reads — instead of loading the full term catalog.
Resolve 0.163 s. Identical 10-package selection.
⁴ There is no term-catalog scale runner for wamjs; its B3 path is the
store-backed row above (this matches the historical table, where the wamjs B3
was always the store run).
⁵ **Go store B3** (`run_scale_go_store.sh`): resolve **0.56 s** reading
**11,025 of 1,142,225 store bytes (0.97%)**, 880 reads — vs the Go *term*
leg's 15.12 s that scans the full catalog. That is **~27× faster than Go term**
while touching under 1% of the store, with the identical 10-package selection.
⁶ **Rust store B3** (`run_scale_rust_store.sh`, D73, indexed backend) is the
fastest store leg: resolve **0.040 s** reading **10,305 of 1,142,225 store
bytes (0.90%)**, 820 reads — vs the Rust *term* leg's 0.385 s (full tier ON), so
**~9× faster than Rust term** while touching under 1% of the store, identical
10-package selection. The bytes-read and read-count are byte-identical to D82,
confirming the store data path is unchanged by the term-lane regions. The store
legs do not run the 2,600-case term differential; the 3.74 s figure is the
503-case store differential vs the SWI store adapter (SWI 0.69 s), 0 divergences.
⁷ **SWI B3 is carried over from the D72 table** (SWI resolve = 0.0197 s). The
frozen SWI B3 reference loader (`rust/swi_scale_ref.pl`, `store/scale_demo.pl`)
predates the Debian-style version rows and `alternatives` deps that the current
5k generator appends, so it can no longer construct the catalog without the
historical row-filter (verified on this box: it emits no `swi_term_*` lines and
exits non-zero); SWI and `resolver.pl` are both frozen, so its native resolve of
the 5k catalog (~20 ms) is unchanged. See the measurement note.
⁸ **Rust term B3 with the full lowered tier ON** (default: regions 1/2/3a/3b/4/5
+ the general recognizer). All lowering OFF (pristine interpreter) it is
~1.93 s; the drift-cancelling A/B below measures the ON-vs-OFF delta at **−80.7%**.
The 0.385 s cell is the full-tier-ON default.

## Stage 2 lowered tier (Rust): deterministic region fusion + the general recognizer

Since the D72 table the Rust term lane gained a **deterministic lowered tier** —
deterministic WAM regions fused into direct native Rust (no interpreter dispatch,
a three-scalar minimal-locals rollback snapshot), each gated, with
shape-recognizers, stress tests, and a real wall-clock A/B, all default-**ON** and
gated. The tier is now complete. First the five hand-written fused regions
(D78–D81) plus **region 5** (D88):

| region | fusion | ledger | report |
|---|---|---|---|
| 1 | `matching_deps/4 ⊕ dep_to_req/3` | D78 | [`wam_rust_stage2_region1.md`](../../docs/reports/wam_rust_stage2_region1.md) |
| 2 | `matching_versions/4 ⊕ satisfies/2 ⊕ version_lt/2` | D79 | [`wam_rust_stage2_region2.md`](../../docs/reports/wam_rust_stage2_region2.md) |
| 3a | `key_dep_rows/3 ⊕ dep_to_req/3` | D80 | [`wam_rust_stage2_region3.md`](../../docs/reports/wam_rust_stage2_region3.md) |
| 3b | `group_keyed/2 ⊕ same_key/4` | D80 | [`wam_rust_stage2_region3.md`](../../docs/reports/wam_rust_stage2_region3.md) |
| 4 | `build_tree/4` (balanced-BST builder) | D81 | [`wam_rust_stage2_region4.md`](../../docs/reports/wam_rust_stage2_region4.md) |
| 5 | `dep_breaks/5` (committed-choice recursion) | D88 | [`wam_rust_stage2_region5.md`](../../docs/reports/wam_rust_stage2_region5.md) |

Regions 1/2 are the **B2** walks (dep-list and version-list scans plus the full
Debian Policy §5.6.12 `version_lt` chain); regions 3a/3b/4 are the **B3**
index-builders that dominate `resolve_layered`; region 5 (`dep_breaks/5`) is a
committed-choice recursion that lowers into the plain deterministic tier (the
`->` commits its first solution, so it is at-most-one-solution), adding a
consistent, statistically significant **~−3.15% median on B2** (13/16 rounds
negative, t ≈ −4.6). F11 (the self-tail-recursion accessor bank) was re-assessed
and left **OFF** — shallow accessors don't amortise the machinery (D81).

**The general deterministic-recursion recognizer (D89).** On top of the six
hand-written regions, a target-agnostic compositional classifier
(`deterministic_recursion_class/2`,
`src/unifyweaver/core/deterministic_recursion.pl`, taxonomy §11) replaces the
need for one-off region recognizers and closes the taxonomy's "sibling gap" with
one structural classifier. It lowers three predicates none of the hand-written
regions covered — `filter_satisfies/3` (`list_filter`), `key_pkg_rows/3`
(`list_map_index`), `tree_lookup/3` (`bst_descent`) — into native dispatch
methods, each keeping the region G-1..G-5 discipline (minimal snapshot, no choice
point, CP-depth asserted, decline-to-interpreter on any off-shape input). These
three fire on the **indexed** `resolve_layered` path (≥64-package catalog), so
they are hot in B3 and inert in the store lane and the sub-threshold B2
differential. D89's isolated interleaved A/B put them at **−18% (scale-1000) to
−24% (scale-5000)** on top of regions 1–5, banked ON. The recognizers are
structural (term inspection + variable-sharing checks), never `=@=` against a
frozen literal, so the same classifier fires for any predicate of the shape.

**B2 (throughput).** Regions 1+2+5 take Rust term B2 to **~19.1 s** on this box,
a SWI ratio of **~7.2×** — measured here as Rust 19.1 s vs SWI 2.64 s on the same
2,600-case run (0 divergences). (Was ~9.0× at the first table, ~8.2× before
region 5.)

**B3 (`resolve_layered`, the headline) — full tier ON vs all-lowering OFF, this
box.** Two release binaries from the same codegen: **all lowering OFF** (every
`UW_REGION{1,2,3A,3B,4,5}_OFF=1` plus `UW_GENREC_OFF=1`, the pristine interpreter;
F11 is opt-in and stays off) and **all ON** (default). Binary identity verified —
distinct `sha256`, and the OFF crate's `lib.rs` carries **zero**
`region_*_dispatch` arms vs the ON crate's **nine** (regions 1–5 + the three
genrec shapes; a genuine relink, each binary copied out and hashed):

```
OFF sha256 ca3f5c51f6a47e4b8243492cc3cfd406ab19ca8df3af5ef96bcc0d1a0581d550  (0 dispatch arms)
ON  sha256 90f3b42400f49408fe814bc29924a687df1ac4b3073dced48fb1d1bff966133d  (9 dispatch arms)
```

Interleaved (OFF then ON each round) on the same 5,000-package
`resolve_layered` (`case_5000.json`), timing the `resolve_ms` leg
(`load_ms` — JSON→term, not the index path — is ~23 ms either way):

| round | OFF (ms) | ON (ms) | delta (ms) | delta (%) |
|---:|---:|---:|---:|---:|
| 1 | 2090.9 | 375.5 | −1715.4 | −82.04 |
| 2 | 2029.7 | 385.3 | −1644.4 | −81.02 |
| 3 | 1903.4 | 372.6 | −1530.8 | −80.42 |
| 4 | 1918.6 | 371.9 | −1546.7 | −80.62 |
| 5 | 1927.0 | 371.9 | −1555.1 | −80.70 |
| 6 | 1913.4 | 366.9 | −1546.5 | −80.82 |
| 7 | 1927.1 | 370.6 | −1556.6 | −80.77 |
| 8 | 2000.0 | 372.6 | −1627.4 | −81.37 |

**OFF median ≈1927 ms; ON median ≈372 ms; delta median ≈−1555 ms = −80.7%.**
All eight rounds negative; the ranges do not overlap (OFF 1903–2091 ms; ON
367–385 ms). The B3 term-lane output is **byte-identical** OFF vs ON (`cmp` of
the two binaries' stdout on `case_5000.json`), `selection_size` 10 both. This is
the **cumulative** delta pristine-interpreter → full lowered tier: deeper than
the D82 table's −74% (regions 1/2/3/4 only), the extra ~7 points coming from
region 5 and, on this indexed resolve path, mostly the general recognizer's
`key_pkg_rows`/`tree_lookup`/`filter_satisfies`.

**Gates in both configs.** With all lowering OFF and with all lowering ON the
Rust lane passes term corpus 51/51, term differential 2,600 / 0 / 0, store
corpus 51/51, and store differential 503 / 0. The **store lane is inert to the
regions** — `resolve_layered_store` never calls the index builders and never
reaches `dep_breaks/5` on its resolve path, so the store B3 (0.040 s, 10,305
bytes / 820 reads) is unchanged whether the lowering is on or off. `resolver.pl`
and `resolver_store.pl` were **not modified** for any of this; the regions and
the recognizer are pure codegen in
`src/unifyweaver/targets/wam_rust_target.pl`,
`src/unifyweaver/core/deterministic_recursion.pl`, and
`templates/targets/rust_wam/state.rs.mustache`.

## Ratios vs SWI (the legible story)

**Startup (B1, single-process corpus).** Rust's native binary starts fastest
of any leg — **0.048 s, faster than SWI's 0.113 s** — then Go 0.070 s. The
JS/CLJS legs pay interpreter/runtime start-up: wamjs 0.209 s, ClojureScript
2.02 s (dominated by nbb boot, not resolution).

**Throughput (B2, 2,600-case term differential, target/SWI on the same run):**

| leg | target time | SWI (same run) | ratio |
|---|---:|---:|---:|
| wamjs | 16.5 s | 2.63 s | **6.3×** |
| Rust | 19.1 s | 2.64 s | **7.2×** |
| Go | 29.5 s | 2.61 s | **11.3×** |
| ClojureScript | 107.7 s | 2.58 s | **41.7×** |

Rust's 7.2× is down from the first table's 8.8× and the pre-region-5 8.2× — the
Stage-2 regions 1+2 removed the dep-walk and the whole `satisfies`/`version_lt`
chain from the hot path, and region 5 shaved `dep_breaks/5` on top.

**Big-catalog resolve (B3, 5k `resolve_layered`, resolve target/SWI; SWI
resolve = 0.0197 s, carried over):**

| leg | resolve | ratio vs SWI | note |
|---|---:|---:|---|
| **Rust store** | 0.040 s | **2.0×** | reads 0.90% of the store |
| **wamjs store** | 0.163 s | **8.3×** | reads 0.63% of the store |
| **Rust (term)** | 0.385 s | **19.5×** | full term catalog, **full lowered tier ON** |
| **Go store** | 0.56 s | **28×** | reads 0.97% of the store |
| Go (term) | 15.12 s | 767× | full term catalog |
| ClojureScript | 28.24 s | 1434× | full term catalog |

The one-line reading: **SWI wins the raw resolve on the full term catalog**
(first-argument indexing plus decades of WAM engineering); among the
transpiled legs **Rust is fastest on the term catalog** (startup, B2, and — with
the full lowered tier in — B3 resolve). The lowered tier collapses the Rust
*term* B3 from the original 98× SWI → 26× (D82) → **~19.5× SWI** now, without
touching a store, and the store-backed legs close the gap further by reading an
indexed seek store instead of the whole catalog. The standout remains **Rust
store: 0.040 s, within ~2.0× of SWI's native resolve while touching under 1% of
the store** — a transpiled target in SWI's league on the 5k catalog. Go store
(28×) and wamjs store (8.3×) close the same gap on their runtimes.

## Is Rust faster than SWI?

**Only on startup.** Per axis, on this box:

- **Startup (B1):** Rust **beats** SWI — its native binary runs the whole
  51-scenario corpus in **0.048 s vs SWI's 0.113 s**.
- **Throughput (B2):** SWI still **~7.2× faster** — Rust 19.1 s vs SWI 2.64 s on
  the same 2,600-case run. (Was ~8.8× originally, ~8.2× before region 5; the gap
  narrowed but SWI still wins.)
- **Term resolve (B3):** SWI **~19.5× faster** — Rust 0.385 s vs SWI 0.0197 s on
  the 5k `resolve_layered` — but the gap **collapsed from ~98×** (pristine
  interpreter) → 26× (D82) → ~19.5× now. Much smaller, still SWI's.
- **Store resolve (B3):** SWI **~2.0× faster** — Rust store 0.040 s vs SWI
  0.0197 s — very close, within a factor of two, touching 0.90% of the store.

The **Rust-vs-Rust** improvement percentages elsewhere in this document (e.g.
the B3 −80.7% ON-vs-OFF delta) are Rust-lowered-tier vs Rust-pristine-interpreter
— they measure how much the lowered tier sped Rust up, **not** Rust overtaking
SWI. On resolution SWI still wins every axis; Rust beats SWI only on startup. The
achievement is that the gaps closed dramatically: term-resolve from ~98× to
~19.5×, and store-resolve down to ~2×.

## Post-pruning note (G1 catalog index + G2 conflict-first)

`resolver.pl` builds a per-call index over the catalog lists at the
`resolve/3` / `resolve_layered/3` edge (guard G1 of
[`docs/proposals/RESOLVER_PRUNING_DESIGN.md`](../../docs/proposals/RESOLVER_PRUNING_DESIGN.md)),
gated behind a size threshold (`index_threshold/1`, 64 rows). **The figures in
this section are historical, measured in the implementing round on one
contributor box (4-core) and were _not_ re-measured for the single-box tables
above.** They are per-leg before/after numbers, comparable to each
other but not to the main table; they are retained because the *mechanism*
they describe is still in effect.

| leg / workload | before | after | note |
| --- | ---: | ---: | --- |
| SWI, 5k `resolve_layered(p30)`, 50 reps | 286,804 inf / 16.50 ms | 142,476 inf / 26.12 ms | −50 % inferences, **+58 % wall** — SWI's native scans beat building a tree |
| wamjs, 250 pkgs (363 rows / 811 deps) | 421,879 instr | 54,394 instr | 7.8× |
| wamjs, 500 pkgs (736 / 1,549) | 808,155 | 96,200 | 8.4× |
| wamjs, 1,000 pkgs (1,503 / 3,036) | 1,590,479 | 180,219 | 8.8× |
| wamjs, 2,000 pkgs (3,014 / 6,058) | **2,000,000-step cap → `fail`** | 350,033, correct | cap cleared |
| wamjs, 5,000 pkgs (7,514 / 15,000) | cap → `fail` (7,843,227 uncapped) | **854,887, correct** | **9.2×**; needs `node --stack-size=200000` |
| wamjs, B2 2,600 cases (threshold 64) | 11,903,199 instr / 22.1 s | 10,952,778 / 21.5 s | −8.0 % instr, −2.6 % wall |
| wamjs, B2 2,600 cases (index forced on) | 11,903,199 / 22.1 s | 8,881,347 / 24.3 s | −25.4 % instr but **+9.9 % wall** — why the threshold exists |

**Historical (other-box, D61) Go post-P3-port note:** with the index active,
Go's B1 corpus was 0.053 s, B2 differential 17.2 s vs SWI 1.7 s (0
divergences), B3 5k `resolve_layered` load 0.065 s / resolve 10.62 s
(~1.04× vs the pre-index 11.0 s). The difference is honest arithmetic: Go's
baseline could already finish (linear scans), and building the index
interpreted nearly cancels the lookup savings. The index is semantically
active (identical selection); on that runtime it was not a wall-time win.
These are not the single-box numbers — on this box Go term B2 is 29.5 s and
B3 resolve 15.12 s, both slower in absolute terms than the D61 contributor box,
which is why only same-box ratios are trustworthy.

## Readings

- **SWI wins resolution outright** (first-argument indexing + decades of WAM
  engineering). Every transpiled leg is an interpreter-shaped runtime hosting
  the same bytecode; none has clause indexing on par with SWI yet. On the 5k
  term catalog SWI resolves in ~20 ms; the nearest transpiled term leg (Rust,
  full lowered tier ON) is now ~19.5× that (was ~98× at D72, ~26× at D82).
- **Rust is the fastest transpiled leg across the board** on this box —
  fastest startup (beating SWI), best B2 among the compiled term legs after
  wamjs, and the fastest term B3 resolve (0.385 s, full tier ON). `Value`
  structural sharing (one refcounted spine per compound/list, allocation-free
  list peeling) keeps choice points O(live registers) rather than O(term size).
  The **deterministic lowered tier** now removes the biggest interpreter
  cost on the B2 and B3 hot paths by fusing the deterministic regions into
  native Rust (regions 1/2/3a/3b/4/5 + the general recognizer's
  `filter_satisfies`/`key_pkg_rows`/`tree_lookup`, default-ON, 0 divergences);
  the residual is the remaining interpreter machinery on the genuinely
  nondeterministic drivers (`pick/7`, `blocked_from/4`), the target of the later
  nondet round (`docs/WAM_RUST_STATUS.md`).
- **The store-backed legs are the resolution story.** By seek-reading an
  indexed store instead of scanning the full catalog, **Rust store resolves the
  5k catalog in 0.040 s touching 0.90 % of the store**, **wamjs store in
  0.163 s touching 0.63 %**, and **Go store in 0.56 s touching 0.97 %** — the
  Go store number being ~27× its own term leg. All have a bytes-read proof; all
  return the identical selection.
- **Go is the best small-scale transpiled term leg** (0.070 s corpus) but its
  term B3 scan is 15.12 s; the store lane is where Go's B3 becomes competitive.
- **ClojureScript is correct, not fast.** It executes the full switch family
  and choice-point-free list walks, and matches SWI on every gate, but at
  ~1434× SWI on the 5k resolve it is bounded by per-instruction interpretation
  cost under nbb's SCI interpreter — the query is millions of honest WAM
  instructions and each costs microseconds. See `cljs/README.md`.
- **The cut-semantics probe corpus (35 probes) passes on JavaScript, Go, and
  Rust** (`tests/test_wam_{javascript,go,rust}_cut_semantics.pl`), per
  `docs/WAM_BACKEND_CONVENTIONS.md` §9. The Clojure lane carries its own
  5-probe backtracking suite; a full §9 port is future work.

## Reproduce

```sh
export LC_ALL=C.UTF-8 LANG=C.UTF-8   # SWI mangles UTF-8 in the C locale

# term legs: build once, then corpus (B1) + differential (B2) + scale (B3)
bash examples/pkg_resolver/go/build.sh
bash examples/pkg_resolver/go/run_corpus_go.sh
bash examples/pkg_resolver/go/run_differential_go.sh
bash examples/pkg_resolver/go/run_scale_go.sh

bash examples/pkg_resolver/rust/build.sh                 # ~3–4 min from scratch (full tier ON)
bash examples/pkg_resolver/rust/run_corpus_rust.sh
bash examples/pkg_resolver/rust/run_differential_rust.sh
bash examples/pkg_resolver/rust/run_scale_rust.sh 5000
# Full-tier A/B: rebuild the all-OFF binary (pristine interpreter) and compare B3:
UW_REGION1_OFF=1 UW_REGION2_OFF=1 UW_REGION3A_OFF=1 UW_REGION3B_OFF=1 \
  UW_REGION4_OFF=1 UW_REGION5_OFF=1 UW_GENREC_OFF=1 \
  bash examples/pkg_resolver/rust/build.sh
#   (copy each binary out, sha256, interleave OFF/ON on rust/.scale/case_5000.json)

bash examples/pkg_resolver/wamjs/build.sh
bash examples/pkg_resolver/wamjs/run_corpus_wamjs.sh
bash examples/pkg_resolver/run_differential.sh          # wamjs term leg

bash examples/pkg_resolver/cljs/build.sh
bash examples/pkg_resolver/cljs/run_corpus_cljs.sh
bash examples/pkg_resolver/run_differential_cljs.sh
bash examples/pkg_resolver/cljs/bench_scale.sh

# store-backed legs (bytes-read proof + 5k B3)
bash examples/pkg_resolver/run_scale_demo.sh            # wamjs store
bash examples/pkg_resolver/run_store_differential.sh    # wamjs store gate
bash examples/pkg_resolver/go_store/run_corpus_go_store.sh
bash examples/pkg_resolver/go_store/run_differential_go_store.sh
bash examples/pkg_resolver/go_store/run_scale_go_store.sh
bash examples/pkg_resolver/rust_store/build.sh
bash examples/pkg_resolver/rust_store/run_corpus_rust_store.sh
bash examples/pkg_resolver/rust_store/run_differential_rust_store.sh
bash examples/pkg_resolver/rust_store/run_scale_rust_store.sh
```

**Measurement note.** The SWI B3 reference (`rust/swi_scale_ref.pl`,
`store/scale_demo.pl`) uses a simplified hand-written term loader that predates
the Debian-style version rows and `alternatives` deps the 5k generator now
appends (≈8 packages / 3 deps of the 7,522 / 15,003 total). Those rows are not
on the `resolve_layered(p30)` path, and the WAM targets parse them via
`resolver.pl`'s own loader, but the frozen SWI loader can no longer construct
the catalog from the current generator output without the historical row-filter
(confirmed on this box: it emits no `swi_term_*` lines and exits non-zero) — so
the SWI B3 cell (0.027 s / 0.020 s) is **carried over from the D72 table** and
marked as such; SWI and `resolver.pl` are frozen, so SWI's native resolve of the
5k catalog is unchanged. Rust B3 with and without those extra rows differs by
~5% (measured at D72: 1.93 s vs 2.02 s), confirming the exclusion is immaterial
to the ratio. Run SWI under a UTF-8 locale.

**Full re-measure report:**
[`docs/reports/wam_rust_bench_refresh_full_tier.md`](../../docs/reports/wam_rust_bench_refresh_full_tier.md).
