<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) 2026 John William Creighton (s243a) -->

# Rust WAM (uw-resolve): size-gated auto-switch to the store lane (D91 lever #3)

The D91 full-tier hotspot profile (`wam_rust_hotspot_profile_full_tier.md`)
ranks "promote the store/seek path as the default large-catalog resolve" as the
top B3-scale lever: the store lane already resolves the 5k catalog in ~40 ms vs
~385 ms for the term lane (~10x, touching 0.90 % of the store). This report
assesses the routing seam and the crossover, then implements a size-gated
auto-switch so large-catalog resolution gets the store win **by default**, with
**byte-identical selections** to the term path on both sides of the switch.

**Base.** `origin/claude/peerhailer-exploratory-docs-aodas5` @ `f4678837b`
(ledger through D91). Push branch `claude/wam-rust-store-default-aodas5`.
`LC_ALL=C.UTF-8`, release builds, this VM. **`resolver.pl` and
`resolver_store.pl` are UNMODIFIED** (frozen); all routing lives in a new
driver.

## STEP 1 — Assessment

### How is term-vs-store selected today?

**Two separate binaries, no chooser.** The Rust target ships two lanes:

| lane | crate | binary | catalog source | input case |
|---|---|---|---|---|
| term | `rust/uw_resolve_wam/` (committed) | `uw_resolve` | whole catalog loaded as WAM `Value` terms | `{catalog:{packages,depends,…}, query, args}` |
| store | `rust_store/uw_resolve_wam_store/` (gitignored, per-build) | `uw_resolve_store` | D43 indexed UWFI/UWIX seek store, keyed `CatId\|Name` | `{env:{…}, query, args}` (no catalog) |

Each lane has its own `build.sh` and `run_*.sh`; nothing picks between them.
Both shims reuse the same term↔JSON readers, and — decisively for routing — the
store shim's `env_term` (rust_store/shim/main.rs:185) **reads `env` when present,
else falls back to the `catalog` env fields**. So the store binary accepts the
**same term-format case line** the term binary does; it simply ignores the
embedded catalog facts and serves them from the baked store. That means an
auto-switch does not have to transform the payload — it only chooses which
binary reads each case.

**Where the switch lives.** In a new driver
(`examples/pkg_resolver/resolve_auto.mjs`), never in the frozen `resolver*.pl`.
The `pkg` CLI (`cli/pkg.mjs`) is the *wamjs* lane and does not touch Rust, so
the Rust seam is the binaries themselves; the driver sits in front of them.

### Pre-built vs build-on-demand + store build cost

**The store is a PRE-BUILT (package-manager) artifact, and must be.** The
generated store crate **bakes the absolute store-dir path** into
`setup_foreign_predicates` at compile time (rust_store/build.pl:129,
`rust_wam_fact_sources(indexed(Prefix))`). Store build cost, 5k catalog (7,522
packages), this VM:

| phase | cost | needs recompile? |
|---|---:|---|
| `gen_scale_catalog` | 108 ms | no |
| `rich_to_p2` (P/2 JSONL) | 133 ms | no |
| `build_stores.sh` (index the seek store) | 451 ms | no |
| **store DATA total** | **~0.69 s** | no |
| crate codegen + `cargo build --release` | **~67 s** (warm) / ~253 s (cold) | **yes, per store-dir path** |

So the **data** store (pkg.data/.idx …) is cheap and re-indexable in
sub-second; only the **binary** is expensive, and only when the store-dir path
changes. Because store keys are `CatId|Name`, **one store dir holds many
catalogs**, so the normal lifecycle is: build the binary **once** against a
fixed store dir, then re-index catalogs/updates into that dir (~0.7 s) without
recompiling — the package-manager model (build once, resolve many).

**Build-on-demand per resolve is a non-starter:** ~67 s of `cargo` dwarfs any
single resolve (≤0.5 s), so a per-resolve build never wins; it only pays off
amortized over ≳150 resolves against the same store. The driver therefore
assumes a pre-built/cached store and **gates on store availability**, falling
back to the term lane when no store is present (never a wrong/failed resolve).

### Crossover

Term cost grows ~linearly with catalog size (it loads the whole catalog as
terms); store cost is ~**flat** (a bound-key seek reads only the few KB a query
touches, independent of N). Measured (internal timers; `--bench` for term,
`--scale-probe` for store):

| catalog (packages) | term load+resolve | store resolve | store bytes read / total |
|---:|---:|---:|---:|
| 62   | 39 ms  | — | — |
| 142  | 42 ms  | — | — |
| 371  | 50 ms  | — | — |
| 744  | 71 ms  | — | — |
| **1,511** | **103 ms** | **40 ms** | 8,870 / 219,735 (4.0 %) |
| 3,770 | 227 ms | — | — |
| **7,522** | **403 ms** | **40 ms** | 10,305 / 1,142,225 (0.90 %) |

The store point is **40 ms at both 1,511 and 7,522 packages** — empirically
flat, confirming a single crossover. End-to-end wall (whole process):

| catalog | term wall | store wall | speedup |
|---:|---:|---:|---:|
| 371  | ~65 ms  | ~50 ms | ~1.3x |
| 1,511 | ~130 ms | ~50 ms | ~2.6x |
| 7,522 | ~535 ms | ~50 ms | **~10.7x** |

**Crossover (break-even):** the store starts winning at roughly **200–300
packages** (where term load+resolve passes the store's ~40 ms floor). Below that
the two are within ~1.3x and the term lane needs no pre-built store. The chosen
**default threshold is 500 packages** — safely past the noisy break-even band,
where the store is a clear ≥1.5x win that grows to 10x at 5k — and it is
overridable down to the raw break-even for callers who want it.

## STEP 2 — What was implemented

`examples/pkg_resolver/resolve_auto.mjs` — a size→backend **ladder** router:

- **Ordered ladder** `[{minPackages:0 → term}, {minPackages:threshold →
  indexed}]`, walked top-down: the highest rung whose catalog size is met **and
  whose backend is available** wins; otherwise it demotes to the next lower
  available rung. `term` (rung 0) is always available when its binary exists, so
  routing never yields a wrong/failed path.
- **Backend registry** is the extension point: each backend is an enum entry
  with `bin` + `available()`. A future **LMDB** rung slots in as a third tier
  (`term → indexed → lmdb`) by adding one registry entry and one ladder rung —
  **no routing rework**. LMDB is deferred this round (it fails loud on the Rust
  lane today); the exact slot is marked in the source.
- **Default** below 500 packages → term; at/above → indexed store.
- **Override:** `--threshold N` / `UW_RESOLVE_STORE_THRESHOLD`; force a lane with
  `--backend term|indexed` / `UW_RESOLVE_BACKEND` (a forced-but-unavailable
  backend exits 2 rather than silently swapping).
- **Fallback:** if the store is unavailable (no binary, or store dir without
  `pkg.data`), auto demotes to term — never a failed resolve. It never silently
  swaps a *small* catalog into a store the caller didn't expect (small stays
  term by the threshold).
- **catalog_id:** store keys are `CatId|Name`, so a store-routed case is stamped
  with the pre-built store's catalog id (discovered from
  `<store-dir>/probe.json`, or `--catalog-id` / `UW_RESOLVE_CATALOG_ID`). The
  term lane ignores `catalog_id`, so this is pure routing metadata and changes
  no selection.
- **Batched dispatch:** cases are grouped by chosen backend, each binary is run
  **once** over its group, and results are merged back in input order.

## Gate results

All four required gates pass, plus byte-identical selection across the switch.

| gate | path | result |
|---|---|---|
| corpus | term | **51/51** matched SWI |
| differential (2,600) | term | **2600 / 0 / 0** |
| corpus | store | **51/51**, byte-identical to the term corpus |
| differential (503) | store (5k) | **503 / 0** |

**Byte-identical across the switch** (`run_auto_switch_test.sh`, all PASS):

- LARGE (5k catalog, 6 queries, ≥ threshold): `auto(→store)`, `--backend
  indexed`, and `--backend term` all **byte-identical to the term oracle**; auto
  routed to the store.
- SMALL (tiny catalog, < threshold): `auto(→term)` byte-identical; kept on term.
- MIXED (small + large in one stream): byte-identical, input order preserved.
- FALLBACK (store dir absent): auto demotes to term, byte-identical.

## Crossover A/B (the switch delivers the win)

Underlying binaries, single `resolve_layered` on the 5k catalog (whole
process): term **~535 ms** vs store **~50 ms** = **~10.7x**.

Through the driver (adds ~110 ms fixed Node startup + `spawnSync`):

| workload | `--backend term` | `AUTO (→store)` | speedup |
|---|---:|---:|---:|
| single resolve, 5k | ~560 ms | ~162 ms (== forced indexed) | ~3.5x |
| 6-query 5k workload | ~1,910 ms | ~405 ms | ~4.7x |

`AUTO` equals `--backend indexed` at 5k (correct routing) and equals `--backend
term` below threshold. The Node startup dilutes the ratio on a single query;
batching many queries into one driver invocation amortizes it (6 queries in
405 ms ≈ 67 ms/query vs term 318 ms/query).

## Files changed

- `examples/pkg_resolver/resolve_auto.mjs` — **new** size→backend ladder router.
- `examples/pkg_resolver/run_auto_switch_test.sh` — **new** byte-identical
  across-the-switch test (+ routing, mixed-stream, and fallback checks).
- `examples/pkg_resolver/.gitignore` — ignore the test's `.auto_out/`.
- `docs/reports/wam_rust_store_default_autoswitch.md` — this report.

No target/spec/crate source changed; the generated crates are rebuilt from the
unmodified committed sources (term-crate codegen is deterministic modulo a
date-stamp comment).

## resolver.pl / resolver_store.pl — unmodified

`git diff` against the base shows **no change** to
`examples/pkg_resolver/resolver.pl` or `examples/pkg_resolver/resolver_store.pl`.
Routing lives entirely in the driver.

## Generalizing + follow-up

- **Go / wamjs stores.** The same seam exists there: `go_store/` and
  `wamjs_store/` are separate store lanes whose shims also accept the shared case
  format. The ladder in `resolve_auto.mjs` is backend-agnostic — add Go/wamjs
  store binaries as backends (their own `bin` + `available()`), or port the
  ladder into their drivers. The crossover would need re-measuring per target
  (each runtime's term-load curve differs), but the flat-store vs linear-term
  shape and the pre-built assumption carry over.
- **LMDB rung (the deferred third tier).** When a real LMDB tier lands, add
  `lmdb` to the backend registry (`available()` = `<dir>/lmdb/pkg` exists, built
  with `UW_STORE_BACKEND=lmdb`) and one ladder rung above `indexed` with its own
  `minPackages`. Its crossover measurement needs: LMDB resolve wall vs the
  indexed seek store at increasing N (LMDB's mmap/page-cache win is expected only
  at large N and/or high query concurrency), plus LMDB build/refresh cost, to
  set the second threshold. No routing code changes — only a registry entry, a
  rung, and a measured threshold.
- **Node startup.** For latency-sensitive single queries, a thin native/shell
  router (or teaching each shim a `--auto` mode) would drop the ~110 ms Node
  overhead; batched workloads already amortize it.
