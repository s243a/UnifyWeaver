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

**What changed since the 2026-09-06 (D72) table.** The Rust term lane gained
the Stage-2 *deterministic lowered tier* — five fused native regions (D78–D81,
regions 1, 2, 3a, 3b, 4; default-ON, gated, 0 divergences). It takes Rust term
**B2 from ~23 s to ~20.7 s (~9.0×→~8.2× SWI)** and, the headline, Rust term
**B3 `resolve_layered` from ~1.94 s to ~0.52 s (~−73%)** on this box. See the
[Stage-2 note](#stage-2-lowered-tier-rust-deterministic-region-fusion) below.
The store lanes are **unaffected** (the index builders are never invoked on the
store resolve path) and re-measure identical to D72 within box noise.

## Main table (single box, 2026-09-07)

| Leg | B1: corpus (51 scen.) | B2: differential | B3: `resolve_layered`, 5k catalog (load / resolve) |
|---|---:|---:|---|
| SWI-Prolog (oracle) | 0.107 s | **2.5 s** (2600 term) | 0.027 s / **0.020 s** ⁷ |
| Go WAM (term) | 0.077 s | 27.0 s (2600 term) | 0.084 s / 14.78 s |
| **Rust WAM** (term) | **0.072 s** | 20.7 s (2600 term) | 0.024 s / **0.52 s** ⁸ |
| wamjs (term) | 0.209 s | **13.6 s** (2600 term) | — (store row is the wamjs B3 path)⁴ |
| ClojureScript (nbb) | 2.04 s | 101.0 s (2600 term) | 0.225 s / 25.35 s |
| **wamjs store** (D48) | 0.209 s¹ | 8.45 s (503 store)² | store-seek / **0.159 s**³ |
| **Go store** (D70) | corpus 51/51¹ | 29.3 s (503 store)² | store-seek / **0.564 s**⁵ |
| **Rust store** (D73) | corpus 51/51¹ | 3.99 s (503 store)² | store-seek / **0.047 s**⁶ |

All gates passed on this box for every measured leg (corpus 51/51; term
differential 2,600 cases / 0 divergences; store differential 503 cases /
0 divergences).

¹ Store legs share the term legs' B1 corpus binary/gate (corpus 51/51 matched
SWI, and the Go/Rust store corpus is byte-identical to their term corpus); the
store path is exercised at scale in B3, so B1 is not a separate store number.
² Store legs do **not** run the 2,600-case term differential (they read from
the indexed seek store, a different data path). Their differential gate is the
**503-case store differential** on the 5k catalog; the times shown are that
leg vs the SWI store adapter on the same 503 cases (wamjs store 8.45 s vs SWI
0.69 s; Go store 29.3 s vs SWI 0.69 s; Rust store 3.99 s vs SWI 0.67 s),
0 divergences.
³ **wamjs store B3** (`run_scale_demo.sh`, lazy materialisation): the catalog
is seek-read from the D43 indexed stores — **10,305 of 1,646,323 store bytes
touched (0.63%)**, 820 reads — instead of loading the full term catalog.
Resolve 0.159 s. Identical 10-package selection.
⁴ There is no term-catalog scale runner for wamjs; its B3 path is the
store-backed row above (this matches the historical table, where the wamjs B3
was always the store run).
⁵ **Go store B3** (`run_scale_go_store.sh`): resolve **0.564 s** reading
**11,025 of 1,142,225 store bytes (0.97%)**, 880 reads — vs the Go *term*
leg's 14.78 s that scans the full catalog. That is **~26× faster than Go term**
while touching under 1% of the store, with the identical 10-package selection.
⁶ **Rust store B3** (`run_scale_rust_store.sh`, D73, indexed backend) is the
fastest store leg: resolve **0.047 s** reading **10,305 of 1,142,225 store
bytes (0.90%)**, 820 reads — vs the Rust *term* leg's 0.52 s (regions ON), so
**~11× faster than Rust term** while touching under 1% of the store, identical
10-package selection. The bytes-read and read-count are byte-identical to D72,
confirming the store data path is unchanged. The store legs do not run the
2,600-case term differential; the 3.99 s figure is the 503-case store
differential vs the SWI store adapter (SWI 0.67 s), 0 divergences.
⁷ **SWI B3 is carried over from the D72 table** (SWI resolve = 0.0197 s). The
frozen SWI B3 reference loader (`rust/swi_scale_ref.pl`, `store/scale_demo.pl`)
predates the Debian-style version rows and `alternatives` deps that the current
5k generator appends, so it can no longer construct the catalog without the
historical row-filter; SWI and `resolver.pl` are both frozen, so its native
resolve of the 5k catalog (~20 ms) is unchanged. See the measurement note.
⁸ **Rust term B3 with the Stage-2 regions ON** (default). Regions OFF (pristine
interpreter) it is ~1.94 s; the drift-cancelling A/B below measures the
regions-ON-vs-OFF delta at **−74%**. The 0.52 s cell is the regions-ON default.

## Stage 2 lowered tier (Rust): deterministic region fusion

Since the D72 table the Rust term lane gained **Stage 2 of the lowered
throughput tier** — five *deterministic* WAM regions fused into direct native
Rust (no interpreter dispatch, a three-scalar minimal-locals rollback
snapshot), each behind its own flag, shape-recognizer, stress tests, and a real
wall-clock A/B, all default-**ON** and gated. They are:

| region | fusion | ledger | report |
|---|---|---|---|
| 1 | `matching_deps/4 ⊕ dep_to_req/3` | D78 | [`wam_rust_stage2_region1.md`](../../docs/reports/wam_rust_stage2_region1.md) |
| 2 | `matching_versions/4 ⊕ satisfies/2 ⊕ version_lt/2` | D79 | [`wam_rust_stage2_region2.md`](../../docs/reports/wam_rust_stage2_region2.md) |
| 3a | `key_dep_rows/3 ⊕ dep_to_req/3` | D80 | [`wam_rust_stage2_region3.md`](../../docs/reports/wam_rust_stage2_region3.md) |
| 3b | `group_keyed/2 ⊕ same_key/4` | D80 | [`wam_rust_stage2_region3.md`](../../docs/reports/wam_rust_stage2_region3.md) |
| 4 | `build_tree/4` (balanced-BST builder) | D81 | [`wam_rust_stage2_region4.md`](../../docs/reports/wam_rust_stage2_region4.md) |

Regions 1/2 are the **B2** walks (dep-list and version-list scans plus the full
Debian Policy §5.6.12 `version_lt` chain); regions 3a/3b/4 are the **B3**
index-builders that dominate `resolve_layered` (the census put
`same_key`/`key_dep_rows`/`dep_to_req`/`group_keyed` at ~68% and `build_tree` at
~21.9% of B3 dispatches). F11 (the self-tail-recursion accessor bank) was
re-assessed under the same snapshot and left **OFF** — shallow accessors don't
amortise the machinery (D81).

**B2 (throughput).** Regions 1+2 take Rust term B2 from ~23 s (D72) to
**~20.7 s** on this box, moving the SWI ratio from **~9.0× to ~8.2×** — measured
here as Rust 20.7 s vs SWI 2.52 s on the same 2,600-case run (0 divergences).

**B3 (`resolve_layered`, the headline) — regions ON vs OFF, this box.** Two
release binaries from the same codegen: **all five regions OFF** (every
`UW_REGION{1,2,3A,3B,4}_OFF=1`, the pristine interpreter) and **all ON**
(default). Binary identity verified — distinct `sha256`, and the OFF crate's
`lib.rs` carries **zero** `region_*_dispatch` arms vs the ON crate's five (a
genuine relink, each binary copied out and hashed):

```
OFF sha256 cc1a6d2a44881b3532cf7dd98331258fea324aebfcc9a4c0538576e9ba86f544  (0 region arms)
ON  sha256 2d4474d99988524adc83c0ea8bb4eceed11d9976405ed840c70bcb51231da6c4  (5 region arms)
```

Interleaved (OFF then ON each round) on the same 5,000-package
`resolve_layered` (`case_5000.json`), timing the `resolve_ms` leg
(`load_ms` — JSON→term, not the index path — is ~23 ms either way):

| round | OFF (ms) | ON (ms) | delta (ms) | delta (%) |
|---:|---:|---:|---:|---:|
| 1 | 1924.8 | 504.7 | −1420.1 | −73.8 |
| 2 | 1935.3 | 492.2 | −1443.1 | −74.6 |
| 3 | 1911.3 | 510.0 | −1401.3 | −73.3 |
| 4 | 1995.6 | 504.8 | −1490.8 | −74.7 |
| 5 | 2019.5 | 524.4 | −1495.1 | −74.0 |
| 6 | 1940.1 | 501.0 | −1439.1 | −74.2 |

**OFF median ≈1938 ms; ON median ≈505 ms; delta median ≈−1433 ms = −74.0%.**
All six rounds negative. The B3 term-lane output is **byte-identical** OFF vs
ON (`diff` of the two binaries' stdout), `selection_size` 10 both. This
reproduces the ledger's cumulative claim (D81: pristine interpreter ≈2000 ms →
regions-1/2/3/4 ON ≈530 ms, ≈−73%).

**Gates in both configs.** With every region OFF and with every region ON the
Rust lane passes term corpus 51/51, term differential 2,600 / 0 / 0, store
corpus 51/51, and store differential 503 / 0. The **store lane is inert to the
regions** — `resolve_layered_store` never calls the index builders, so the store
B3 (0.047 s, 10,305 bytes / 820 reads) is unchanged whether the regions are on
or off. `resolver.pl` and `resolver_store.pl` were **not modified** for any of
this; the regions are pure codegen in `src/unifyweaver/targets/wam_rust_target.pl`
+ `templates/targets/rust_wam/state.rs.mustache`.

## Ratios vs SWI (the legible story)

**Startup (B1, single-process corpus).** Rust's native binary starts fastest
of any leg — **0.072 s, faster than SWI's 0.107 s** — then Go 0.077 s. The
JS/CLJS legs pay interpreter/runtime start-up: wamjs 0.209 s, ClojureScript
2.04 s (dominated by nbb boot, not resolution).

**Throughput (B2, 2,600-case term differential, target/SWI on the same run):**

| leg | target time | SWI (same run) | ratio |
|---|---:|---:|---:|
| wamjs | 13.6 s | 2.49 s | **5.5×** |
| Rust | 20.7 s | 2.52 s | **8.2×** |
| Go | 27.0 s | 2.64 s | **10.2×** |
| ClojureScript | 101.0 s | 2.51 s | **40×** |

Rust's 8.2× is down from D72's 8.8× — the Stage-2 regions 1+2 removed the
dep-walk and the whole `satisfies`/`version_lt` chain from the hot path.

**Big-catalog resolve (B3, 5k `resolve_layered`, resolve target/SWI; SWI
resolve = 0.0197 s, carried over):**

| leg | resolve | ratio vs SWI | note |
|---|---:|---:|---|
| **Rust store** | 0.047 s | **2.4×** | reads 0.90% of the store |
| **wamjs store** | 0.159 s | **8.1×** | reads 0.63% of the store |
| **Rust (term)** | 0.52 s | **26×** | full term catalog, **Stage-2 regions ON** |
| **Go store** | 0.564 s | **29×** | reads 0.97% of the store |
| Go (term) | 14.78 s | 750× | full term catalog |
| ClojureScript | 25.35 s | 1287× | full term catalog |

The one-line reading: **SWI wins the raw resolve on the full term catalog**
(first-argument indexing plus decades of WAM engineering); among the
transpiled legs **Rust is fastest on the term catalog** (startup, B2, and — now
that the Stage-2 lowered tier is in — B3 resolve). The Stage-2 regions collapse
the Rust *term* B3 from D72's 98× to **26× SWI** without touching a store, and
the store-backed legs close the gap further by reading an indexed seek store
instead of the whole catalog. The standout remains **Rust store: 0.047 s,
within ~2.4× of SWI's native resolve while touching under 1% of the store** — a
transpiled target in SWI's league on the 5k catalog. Go store (29×) and wamjs
store (8.1×) close the same gap on their runtimes.

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
These are not the single-box numbers — on this box Go term B2 is 27.0 s and
B3 resolve 14.78 s, both slower in absolute terms than the D61 contributor box,
which is why only same-box ratios are trustworthy.

## Readings

- **SWI wins resolution outright** (first-argument indexing + decades of WAM
  engineering). Every transpiled leg is an interpreter-shaped runtime hosting
  the same bytecode; none has clause indexing on par with SWI yet. On the 5k
  term catalog SWI resolves in ~20 ms; the nearest transpiled term leg (Rust,
  Stage-2 regions ON) is now ~26× that (was ~98× at D72).
- **Rust is the fastest transpiled leg across the board** on this box —
  fastest startup (beating SWI), best B2 among the compiled term legs after
  wamjs, and the fastest term B3 resolve (0.52 s, regions ON). `Value`
  structural sharing (one refcounted spine per compound/list, allocation-free
  list peeling) keeps choice points O(live registers) rather than O(term size).
  The **Stage-2 deterministic lowered tier** now removes the biggest interpreter
  cost on the B2 and B3 hot paths by fusing the deterministic regions into
  native Rust (regions 1/2/3a/3b/4, default-ON, 0 divergences); the residual is
  the remaining interpreter machinery on the genuinely nondeterministic drivers
  (`pick/7`, `blocked_from/4`, `dep_breaks/5`), the target of the later nondet
  round (`docs/WAM_RUST_STATUS.md`).
- **The store-backed legs are the resolution story.** By seek-reading an
  indexed store instead of scanning the full catalog, **Rust store resolves the
  5k catalog in 0.047 s touching 0.90 % of the store**, **wamjs store in
  0.159 s touching 0.63 %**, and **Go store in 0.564 s touching 0.97 %** — the
  Go store number being ~26× its own term leg. All have a bytes-read proof; all
  return the identical selection.
- **Go is the best small-scale transpiled term leg** (0.077 s corpus) but its
  term B3 scan is 14.78 s; the store lane is where Go's B3 becomes competitive.
- **ClojureScript is correct, not fast.** It executes the full switch family
  and choice-point-free list walks, and matches SWI on every gate, but at
  ~1287× SWI on the 5k resolve it is bounded by per-instruction interpretation
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

bash examples/pkg_resolver/rust/build.sh                 # ~3–4 min from scratch (regions ON)
bash examples/pkg_resolver/rust/run_corpus_rust.sh
bash examples/pkg_resolver/rust/run_differential_rust.sh
bash examples/pkg_resolver/rust/run_scale_rust.sh 5000
# Stage-2 A/B: rebuild the OFF binary (pristine interpreter) and compare B3:
UW_REGION1_OFF=1 UW_REGION2_OFF=1 UW_REGION3A_OFF=1 UW_REGION3B_OFF=1 \
  UW_REGION4_OFF=1 bash examples/pkg_resolver/rust/build.sh
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
— so the SWI B3 cell (0.027 s / 0.020 s) is **carried over from the D72 table**
and marked as such; SWI and `resolver.pl` are frozen, so SWI's native resolve of
the 5k catalog is unchanged. Rust B3 with and without those extra rows differs
by ~5% (measured at D72: 1.93 s vs 2.02 s), confirming the exclusion is
immaterial to the ratio. Run SWI under a UTF-8 locale.
