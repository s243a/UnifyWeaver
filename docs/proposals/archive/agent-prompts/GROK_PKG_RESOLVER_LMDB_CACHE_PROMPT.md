<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) 2026 John William Creighton (s243a) -->

# Grok prompt — uw-resolve: LMDB catalog backend with a caching tier

> Branch from the coordinator tip as `grok/pkg-resolver-lmdb-cache`. Three
> concurrent rounds own `examples/pkg_resolver/{cljs,rust}/`, the clojure /
> rust / haskell / python / r / fsharp target files, and may touch
> `examples/pkg_resolver/BENCHMARKS.md` rows — do not create or touch any of
> those (BENCHMARKS.md included; report numbers in your README instead).
> This round makes `lmdb(Dir)` a first-class, measured catalog backend for
> store-backed resolution, with a read-through cache.

---

## Copy from here ↓↓↓

You are contributing to **UnifyWeaver** (`github.com/s243a/unifyweaver`).
Branch from **`claude/peerhailer-exploratory-docs-aodas5`** (the coordinator
tip) as `grok/pkg-resolver-lmdb-cache`. SWI-Prolog (`swipl`); Node v18+;
the `lmdb` npm package for the gated arm (via a /tmp-prefixed install as in
D43 — NO repo `package.json` dependency, per the D43 policy sentence).

### The mission
D48 made resolution store-backed on `indexed(Prefix)` (seek stores,
bytes-read ∝ query). D43 shipped `lmdb(Dir)` as the opt-in tier but it has
never carried the RESOLVER. This round:

1. **LMDB catalog backend.** Extend the store build path
   (`examples/pkg_resolver/store/`) so the same JSONL dump also builds the
   four resolver stores (`pkg`/`dep`/`conflict`/`revdep`, same key layout as
   D48's README documents) as LMDB databases via
   `scripts/js_wam/uw_fact_lmdb.js` (extend additively). A declaration-level
   switch selects `indexed(Prefix)` vs `lmdb(Dir)` for the wamjs store
   adapter — same fact-source predicates, same 10 queries, no resolver or
   adapter logic change beyond the source declarations.
2. **Read-through cache.** Add an in-process L1 cache to the JS fact-source
   layer: per-key memoisation of store lookups (both backends), bounded
   (configurable entry cap, default documented), read-only-catalog semantics
   (no invalidation needed — say so explicitly; a future write path must
   flush it, note that). Cache ON/OFF must be switchable per run
   (env var or option) so it can be measured. The cache must be
   semantics-invisible: same results, same order, cache on or off — assert
   that in a test, don't just claim it.
3. **Measurement.** On the 5k-package catalog (seed `0xc0ffee01`) report a
   2×2: {indexed, lmdb} × {cache off, cache on} for (a) one bound
   `resolve_layered` (wall + the bytes-read/`n_reads` counter where the
   backend exposes it; for lmdb report operation counts), and (b) the
   500-case store-backed differential wall time. Plus repeated-query
   behavior: the SAME query run 100× in one process, cache on vs off — the
   cache's actual value proposition.

### Gates
- All D48 gates unchanged and green: term corpus 38 plunit + dump 39/39,
  wamjs 39/39, term differential 2400/0, store corpus 39/39 both arms,
  store differential 500/0 on `indexed`.
- New, gated on lmdb availability exactly as D43 (loud actionable error when
  absent, never a silent backend swap): store corpus 39/39 on the LMDB
  backend; store differential ≥500/0 on the LMDB backend; the cache
  on/off equivalence test (both backends).
- Ungated regardless of lmdb presence: the missing-package error path and
  the cache-equivalence test on `indexed`.
- Shared-lane sanity re-run: the three `tests/test_wam_javascript_*.pl`
  suites + cut-semantics + `CONFORMANCE_TARGETS=javascript` + cli_args
  17/17 + 5067/0/0.

### Guardrails
- You may edit ONLY: `examples/pkg_resolver/store/`, `wamjs_store/`,
  `resolver_store.pl`, `store_diff_runner.pl`, `run_store_differential.sh`,
  `run_scale_demo.sh`, `test_resolver_store.pl` (append-only),
  `examples/pkg_resolver/README.md` (backend section),
  `scripts/js_wam/uw_fact_lmdb.js` + `uw_fact_index.js` (additive; D43
  store tests stay green), and the `javascript_wam` runtime's fact-source
  arm ONLY if the cache/lmdb path forces it (probe-pinned, D44 precedent).
- FROZEN: `resolver.pl`, `test_resolver.pl`, `cli/`, `wamjs/` (the term
  build), `cljs/`, `rust/`, `go/` (concurrent rounds — do not create or
  touch, even if absent on your branch), `examples/pkg_resolver/BENCHMARKS.md`,
  `examples/cli_args/` (all), every non-javascript target, `wam_target.pl`,
  `wam_text_parser.pl`, shared harness/registry/matrix/glue.
- No repo `package.json` dependency; lmdb stays opt-in with a loud error.
- Wrong output is worse than refusal; the differentials are the gate.

### Acceptance (must pass before handoff)
1. Store corpus 39/39 on BOTH backends (lmdb arm gated, state whether it ran).
2. Store differential ≥500/0 on BOTH backends (same gating).
3. Cache equivalence test green, both backends.
4. The 2×2 measurement table + the 100×-repeat numbers.
5. Full D48 + shared-lane gate list re-run green.

### Handoff format
Return: the backend-switch declaration shape, the cache design (bound,
keying, why read-only semantics make it safe, the write-path caveat), the
2×2 table + repeat-query numbers with an honest reading (when does lmdb
beat indexed? when does the cache matter?), whether the gated arm ran,
any runtime fix forced (probe-pinned), and residuals (secondary indexes,
incremental updates, write paths, CSR).

## ↑↑↑ Copy to here
