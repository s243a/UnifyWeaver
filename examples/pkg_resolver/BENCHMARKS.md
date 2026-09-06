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
2026-09-06; absolute times vary by machine — the *ratios* are the result.**
All legs were re-run on this one box so the numbers are internally
comparable. The term legs (Go, Rust, wamjs, ClojureScript) were measured
sequentially on an otherwise-idle box; every leg returns the identical
10-package selection for the 5k `resolve_layered(p30)` query.

## Main table (single box, 2026-09-06)

| Leg | B1: corpus (51 scen.) | B2: differential | B3: `resolve_layered`, 5k catalog (load / resolve) |
|---|---:|---:|---|
| SWI-Prolog (oracle) | 0.109 s | **2.5 s** (2600 term) | 0.027 s / **0.020 s** |
| Go WAM (term) | 0.086 s | 28.3 s (2600 term) | 0.090 s / 15.28 s |
| **Rust WAM** (term) | **0.059 s** | 23.1 s (2600 term) | 0.023 s / **1.93 s** |
| wamjs (term) | 0.238 s | **16.1 s** (2600 term) | — (store row is the wamjs B3 path)⁴ |
| ClojureScript (nbb) | 1.93 s | 110.6 s (2600 term) | 0.275 s / 28.98 s |
| **wamjs store** (D48) | 0.238 s¹ | 9.69 s (503 store)² | store-seek / **0.186 s**³ |
| **Go store** (D70) | corpus 51/51¹ | 28.9 s (503 store)² | store-seek / **0.593 s**⁵ |
| **Rust store** (D73) | corpus 51/51¹ | 3.93 s (503 store)² | store-seek / **0.038 s**⁶ |

All gates passed on this box for every measured leg (corpus 51/51; term
differential 2,600 cases / 0 divergences; store differential 503 cases /
0 divergences).

¹ Store legs share the term legs' B1 corpus binary/gate (corpus 51/51 matched
SWI, and the Go store corpus is byte-identical to the Go term corpus); the
store path is exercised at scale in B3, so B1 is not a separate store number.
² Store legs do **not** run the 2,600-case term differential (they read from
the indexed seek store, a different data path). Their differential gate is the
**503-case store differential** on the 5k catalog; the times shown are that
leg vs the SWI store adapter on the same 503 cases (wamjs store 9.69 s vs SWI
0.73 s; Go store 28.9 s vs SWI 0.73 s), 0 divergences.
³ **wamjs store B3** (`run_scale_demo.sh`, lazy materialisation): the catalog
is seek-read from the D43 indexed stores — **10,305 of 1,646,323 store bytes
touched (0.63%)**, 820 reads — instead of loading the full term catalog.
Resolve 0.186 s. Identical 10-package selection.
⁴ There is no term-catalog scale runner for wamjs; its B3 path is the
store-backed row above (this matches the historical table, where the wamjs B3
was always the store run).
⁵ **Go store B3** (`run_scale_go_store.sh`) is the store standout: resolve
**0.593 s** reading **11,025 of 1,142,225 store bytes (0.97%)**, 880 reads —
vs the Go *term* leg's 15.28 s that scans the full catalog. That is **~26×
faster than Go term** while touching under 1% of the store, with the identical
10-package selection.
⁶ **Rust store B3** (`run_scale_rust_store.sh`, D73, indexed backend) is the
fastest store leg: resolve **0.038 s** reading **10,305 of 1,142,225 store
bytes (0.90%)**, 820 reads — vs the Rust *term* leg's 1.93 s that scans the
full catalog, so **~51× faster than Rust term** while touching under 1% of
the store, identical 10-package selection. The store legs do not run the
2,600-case term differential; the 3.93 s figure is the 503-case store
differential vs the SWI store adapter (SWI 0.72 s), 0 divergences.

## Ratios vs SWI (the legible story)

**Startup (B1, single-process corpus).** Rust's native binary starts fastest
of any leg — **0.059 s, faster than SWI's 0.109 s** — then Go 0.086 s. The
JS/CLJS legs pay interpreter/runtime start-up: wamjs 0.238 s, ClojureScript
1.93 s (dominated by nbb boot, not resolution).

**Throughput (B2, 2,600-case term differential, target/SWI on the same run):**

| leg | target time | SWI (same run) | ratio |
|---|---:|---:|---:|
| wamjs | 16.1 s | 2.55 s | **6.3×** |
| Rust | 23.1 s | 2.64 s | **8.8×** |
| Go | 28.3 s | 2.50 s | **11.3×** |
| ClojureScript | 110.6 s | 2.50 s | **44×** |

**Big-catalog resolve (B3, 5k `resolve_layered`, resolve target/SWI; SWI
resolve = 0.0197 s):**

| leg | resolve | ratio vs SWI | note |
|---|---:|---:|---|
| **Rust store** | 0.038 s | **1.9×** | reads 0.90% of the store |
| **wamjs store** | 0.186 s | **9.4×** | reads 0.63% of the store |
| **Go store** | 0.593 s | **30×** | reads 0.97% of the store |
| Rust (term) | 1.93 s | **98×** | full term catalog |
| Go (term) | 15.28 s | 776× | full term catalog |
| ClojureScript | 28.98 s | 1471× | full term catalog |

The one-line reading: **SWI wins the raw resolve on the full term catalog**
(first-argument indexing plus decades of WAM engineering); among the
transpiled legs **Rust is fastest on the term catalog** (startup, B2, and B3
resolve). The store-backed legs are where the gap closes — by reading an
indexed seek store instead of the whole catalog. With the D73 Rust store lane,
the standout is now **Rust store: 0.038 s, within ~1.9× of SWI's native
resolve while touching under 1% of the store** — a transpiled target landing
in SWI's league on the 5k catalog. Go store (30×) and wamjs store (9.4×)
close the same gap on their runtimes.

## Post-pruning note (G1 catalog index + G2 conflict-first)

`resolver.pl` builds a per-call index over the catalog lists at the
`resolve/3` / `resolve_layered/3` edge (guard G1 of
[`docs/proposals/RESOLVER_PRUNING_DESIGN.md`](../../docs/proposals/RESOLVER_PRUNING_DESIGN.md)),
gated behind a size threshold (`index_threshold/1`, 64 rows). **The figures in
this section are historical, measured in the implementing round on one
contributor box (4-core) and were _not_ re-measured for the 2026-09-06 single-
box table above.** They are per-leg before/after numbers, comparable to each
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
These are not the 2026-09-06 numbers — on this box Go term B2 is 28.3 s and
B3 resolve 15.28 s, both slower in absolute terms than the D61 contributor box,
which is why only same-box ratios are trustworthy.

## Readings

- **SWI wins resolution outright** (first-argument indexing + decades of WAM
  engineering). Every transpiled leg is an interpreter-shaped runtime hosting
  the same bytecode; none has clause indexing on par with SWI yet. On the 5k
  term catalog SWI resolves in ~20 ms; the nearest transpiled term leg (Rust)
  is ~98× that.
- **Rust is the fastest transpiled leg across the board** on this box —
  fastest startup (beating SWI), best B2 among the compiled term legs after
  wamjs, and the fastest term B3 resolve (1.93 s). `Value` structural sharing
  (one refcounted spine per compound/list, allocation-free list peeling) keeps
  choice points O(live registers) rather than O(term size); the 5k resolve
  fits in well under a gigabyte. The residual cost is interpreter machinery —
  dispatch, register file, trail, binding hash — not copying; the lowered tier
  that would remove it is not reachable from the interpreter yet
  (`docs/WAM_RUST_STATUS.md`).
- **The store-backed legs are the resolution story.** By seek-reading an
  indexed store instead of scanning the full catalog, **wamjs store resolves
  the 5k catalog in 0.186 s touching 0.63 % of the store** and **Go store in
  0.593 s touching 0.97 %** — the Go store number being ~26× its own term leg.
  Both have a bytes-read proof; both return the identical selection.
- **Go is the best small-scale transpiled term leg** (0.086 s corpus) but its
  term B3 scan is 15.28 s; the store lane is where Go's B3 becomes competitive.
- **ClojureScript is correct, not fast.** It executes the full switch family
  and choice-point-free list walks, and matches SWI on every gate, but at
  ~1471× SWI on the 5k resolve it is bounded by per-instruction interpretation
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

bash examples/pkg_resolver/rust/build.sh                 # ~4 min from scratch
bash examples/pkg_resolver/rust/run_corpus_rust.sh
bash examples/pkg_resolver/rust/run_differential_rust.sh
bash examples/pkg_resolver/rust/run_scale_rust.sh 5000
# SWI B3 reference on the same single-case JSON:
swipl -q -g main -t halt examples/pkg_resolver/rust/swi_scale_ref.pl -- \
      examples/pkg_resolver/rust/.scale/case_5000.json

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
```

**Measurement note.** The SWI B3 reference (`rust/swi_scale_ref.pl`,
`store/scale_demo.pl`) uses a simplified hand-written term loader that predates
the Debian-style version rows and `alternatives` deps the 5k generator now
appends (≈8 packages / 3 deps of the 7,522 / 15,003 total). Those rows are not
on the `resolve_layered(p30)` path, and the WAM targets parse them via
`resolver.pl`'s own loader; for the same-box SWI B3 above they were filtered
out so the frozen SWI loader could construct the catalog. Rust B3 with and
without those rows differs by ~5% (1.93 s vs 2.02 s), confirming the exclusion
is immaterial to the ratio. Run SWI under a UTF-8 locale.
