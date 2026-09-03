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
| **Rust WAM** | 0.102 s | 301.7 s | — / **OOM** (8.5 GB)¹ |
| **wamjs** (mixed/lowered) | ~0.5 s | 5.7 s | — / 0.154 s² |
| **ClojureScript WAM** (nbb) | 1.55 s | 76.7 s | 0.35 s / 39.0 s³ |

¹ Rust B3 measured on the contributor box; not reproduced here (deliberate —
an 8.5 GB allocation). Rust also wins B1 compile-excluded small-scale runs on
that box (93–103 ms) and its ladder shows quadratic blowup: 3.5 s at 54
packages → 79 s at 363 → OOM at 5k.
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
- **Go is the best small-scale transpiled leg and the only native one that
  finishes B3.** Its flat register file copies cheaply; its 5k resolve
  (11 s) is bounded by linear clause scans, not memory.
- **Rust is fastest when it fits and dies when it doesn't.** `save_regs()`
  deep-clones every live register into every choice point and `Value` has no
  structural sharing, so a register holding the catalog is copied per CP.
  `Rc`/`Arc` argument vectors are the recorded prerequisite
  (`docs/WAM_RUST_STATUS.md`) for any Rust speed claim on symbolic loads.
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
