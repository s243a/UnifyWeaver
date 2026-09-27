<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) 2026 John William Creighton (@s243a) -->

# uw-resolve — Rust WAM, store-backed (D43 indexed seek)

The store-backed sibling of [`../rust/`](../rust). Same shared adapter
([`../resolver_store.pl`](../resolver_store.pl)), same 10 queries, but the
catalog is **not** loaded as a term: `store_pkg/2`, `store_dep/2`,
`store_conflict/2`, `store_revdep/2` and `store_provides/2` are served from the
D43 **indexed seek stores** (`pkg.data`/`pkg.idx`, …) that the
language-agnostic [`../store/`](../store) builder writes. A bound-key lookup
binary-searches the sorted key index and reads only the records that key
touches — so a query reads a few KB of a multi-MB store instead of the whole
thing. This is the Rust lane of [`../go_store/`](../go_store) and
[`../wamjs_store/`](../wamjs_store).

## Layout

- `build.pl` — compiles `resolver.pl` + `resolver_store.pl` through `wam_rust`
  with `rust_wam_fact_sources(...)` declaring each store predicate as
  `indexed(Prefix)` (default) or `lmdb(Dir)` (opt-in). The two source files each
  carry a private copy of the shared helpers (`item_ver/3`, `lookup_held/3`, …);
  the loader drops the exact-duplicate second copy so the Rust WAM's
  multi-clause dispatch is not fed doubled clauses.
- `build.sh` — dumps the corpus P/2 JSONL, indexes it, compiles, `cargo build`.
- `shim/main.rs` — term ↔ JSON IO only (env term + the 10 queries). No resolver
  logic, no catalog. Reuses the term↔JSON readers of `../rust/shim/main.rs`.
- `run_corpus_rust_store.sh` — corpus vs SWI, and asserts byte-identical to the
  `../rust` term corpus.
- `run_differential_rust_store.sh` — the 5k differential (seed `0xc0ffee01`) vs
  the SWI store oracle.
- `run_scale_rust_store.sh` — the B3 bytes-read + wall-time payoff.

The generated `uw_resolve_wam_store/` crate bakes the absolute store path into
`setup_foreign_predicates`, so it is a build artifact and is `.gitignore`d; run
`build.sh` to regenerate it.

## How the seek reader plugs in

The store predicates compile to a two-instruction body
(`call_foreign store_pkg/2 2` + `proceed`), so a caller reaches them by label
like any other predicate. `call_foreign` dispatches to
`execute_foreign_predicate`, whose `seek_fact` native-kind arm calls
`execute_seek_fact_source`: it reads A1/A2, and for a bound atomic A1 does a
keyed seek over the UWFI/UWIX store (`src/seek_fact_source.rs`, a byte-for-byte
port of the Go `seekFactSource`) — binary-searching the sorted `.idx` and
reading only the matching `.data` records, maintaining a global bytes-read
counter — while an unbound A1 falls back to a full scan (the provides walk).
Matching rows stream back through the standard foreign-result choice-point
machinery, so bound-A2 filtering and multi-row backtracking work exactly as for
a term-catalog fact predicate.

## Backends

Default `UW_STORE_BACKEND=indexed` reads the dependency-free UWFI/UWIX seek
store. `UW_STORE_BACKEND=lmdb` is opt-in and **fails loudly** on the Rust lane:
the Rust binary carries no repo dependency and does not read the npm-`lmdb`
(`uw_fact_lmdb`) store format — it never silently falls back to indexed.

## Numbers (this VM)

- Corpus **51/51**, byte-identical to the `../rust` term corpus.
- 5k store differential (`0xc0ffee01`): **503 cases, 0 divergences** vs SWI.
- B3 (bound `resolve_layered` on the 5k catalog): resolve **~0.041 s** reading
  **~10 KB of the ~1.14 MB store (0.90%)** in 820 reads, same 10-package
  selection as the term build — whose B3 loads the whole catalog and resolves in
  ~1.98 s (~48× slower). See
  [`../../docs/WAM_RUST_STATUS.md`](../../../docs/WAM_RUST_STATUS.md).
