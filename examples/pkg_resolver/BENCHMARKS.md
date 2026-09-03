<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) 2026 John William Creighton (s243a) -->

# uw-resolve — cross-target benchmarks

One frozen Prolog program (`resolver.pl`, P0.5 semantics, genuine
backtracking), compiled through four targets, timed on identical data with
identical queries. Every leg passes the same semantics gates before it is
timed: the 39-scenario SWI-oracled corpus and the 2,400-case seeded
differential with **0 divergences**. Numbers below were measured on the
coordinator container (4-core, Linux) at the D51 merge except where noted;
absolute times vary by machine — the *ratios* are the result.

| Leg | B1: corpus (39 scenarios) | B2: differential leg (2400 cases) | B3: `resolve_layered`, 5k-pkg catalog (load / resolve) |
|---|---:|---:|---:|
| SWI-Prolog (oracle) | ~0.1 s | 2.8 s | 0.64 s / **0.011 s** |
| **Go WAM** | **0.034 s** | 19.5 s | 0.095 s / 11.0 s |
| **Rust WAM** | **0.036 s** | 16.5 s | 0.023 s / **4.73 s** (680 MB peak RSS)¹ |
| **wamjs** (mixed/lowered) | ~0.5 s | 5.7 s | — / 0.154 s² |
| **ClojureScript WAM** (nbb) | 1.55 s | 76.7 s | 0.35 s / 39.0 s³ |

¹ Rust row re-measured on the contributor box (4-core) after `Value` gained
structural sharing (D55); the SWI leg on that same box is B2 2.12 s and B3
0.031 s load / 0.017 s resolve, so the ratios are B2 ≈ 7.8× and B3 resolve
≈ 279× SWI. **B3 now completes the 5000-package catalog** — it previously
OOM-killed at 8.5 GB. The ladder is linear where it used to be quadratic:
resolve 62 ms at 54 packages → 274 ms at 363 → 4.73 s at 7514, with peak RSS
13.8 MB → 43.5 MB → 680 MB. Before/after on the same box: B1 102→36 ms,
B2 332.6→16.5 s, 363-package resolve 79.2 s→0.27 s. Details and the full
ladder: `rust/README.md`.
³ ClojureScript re-measured after first-argument indexing landed in its
runtime. Before/after on the contributor box: B1 1.611 s → 1.58 s (unchanged;
it is nbb start-up), B2 111.0 s → 79.0 s (**1.3–1.4×**), B3 resolve
58.22 s → 38.6 s (**1.47×**). On this table's reference box the B3 gain was
smaller: 41.8 s → 39.0 s (~7%; corpus B1 1.55 s). Same answers throughout on
both. The gap between boxes is unexplained (nbb/node version suspected) and
recorded rather than averaged away.
² wamjs B3 is the **store-backed** run (D48): the catalog is seek-read from
D43 indexed stores, 8,242 of 1,645,394 bytes touched (0.50%). The other legs
load the full term catalog. All legs return the identical 10-package
selection.

## Readings

- **SWI wins resolution outright** (first-argument indexing + decades of WAM
  engineering). Every transpiled leg is an interpreter-shaped runtime hosting
  the same bytecode; none has clause indexing on par with SWI yet.
- **Go is the best small-scale transpiled leg** and finishes B3 at 11 s;
  its flat register file copies cheaply and its 5k resolve is bounded by
  linear clause scans, not memory. (It is no longer the only native B3
  finisher — see the Rust bullet.)
- **Rust is now the fastest transpiled leg across the board.**
  Giving `Value` structural sharing — one refcounted spine per compound/list,
  and a list tail that is the same spine one element along — made
  `Value::clone` O(1) and list peeling allocation-free, so choice points cost
  O(live registers) instead of O(term size). That turned the quadratic into a
  linear and the OOM into 680 MB. It is still ~280× SWI on the 5k resolve:
  the remaining cost is interpreter machinery — instruction dispatch,
  register file, trail, binding hash — not copying, and the lowered tier
  that would remove it is not reachable from the interpreter yet
  (`docs/WAM_RUST_STATUS.md`). Cross-leg comparisons in this table mix two
  boxes; only the SWI/Rust ratios in the footnote are same-box.
- **wamjs is the best all-around leg**: near-Go differential speed, the only
  store-backed catalog path, and the only leg with a bytes-read proof.
- **ClojureScript is correct, not fast — and indexing was not the reason.**
  The Clojure runtime now executes the whole switch family it used to skip
  (`switch_on_term` / `switch_on_constant` / `switch_on_structure`, A1 and A2,
  plus `try`/`retry`/`trust` dispatch chains), and its list walks are
  choice-point-free. That bought 1.47× on B3, not the order of magnitude
  expected. A step census of the 5k resolve says why: the query is 7.56 M WAM
  instructions — *fourteen honest scans of a 15,000-row `depends` list*, which
  SWI performs too — and each instruction costs ~5 µs under nbb's SCI
  interpreter. `resolver.pl`'s 5,000 packages live in lists inside a `catalog/9`
  term, not in 5,000 clauses, so there was never a clause scan to index away.
  The lane's top residual is now per-instruction interpretation cost; see
  `cljs/README.md` for the measurements and the next lever.
- **The cut-semantics probe corpus (35 probes) now passes on JavaScript, Go,
  and Rust** (`tests/test_wam_{javascript,go,rust}_cut_semantics.pl`), per
  `docs/WAM_BACKEND_CONVENTIONS.md` §9. The Clojure lane carries its own
  5-probe backtracking suite; a full §9 port is future work.

## Reproduce

```sh
# per-leg corpus + differential + scale runners
bash examples/pkg_resolver/wamjs/run_corpus_wamjs.sh
bash examples/pkg_resolver/run_differential.sh          # wamjs leg
bash examples/pkg_resolver/go/run_corpus_go.sh
bash examples/pkg_resolver/go/run_differential_go.sh
bash examples/pkg_resolver/go/run_scale_go.sh
bash examples/pkg_resolver/rust/run_corpus_rust.sh
bash examples/pkg_resolver/rust/run_differential_rust.sh
bash examples/pkg_resolver/cljs/run_corpus_cljs.sh
bash examples/pkg_resolver/run_differential_cljs.sh
bash examples/pkg_resolver/cljs/bench_scale.sh
bash examples/pkg_resolver/run_scale_demo.sh            # store-backed wamjs
```

Run SWI under a UTF-8 locale (`LC_ALL=C.UTF-8`); the C-locale default
mangles UTF-8 in generated code.
