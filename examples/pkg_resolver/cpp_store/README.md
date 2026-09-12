<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) 2026 John William Creighton (@s243a) -->

# uw-resolve — C++ WAM, store-backed (D43 indexed seek)

The store-backed sibling of [`../cpp/`](../cpp). Same shared adapter
([`../resolver_store.pl`](../resolver_store.pl)), same 10 queries, but the
catalog is **not** loaded as a term: `store_pkg/2`, `store_dep/2`,
`store_conflict/2`, `store_revdep/2` and `store_provides/2` are served from the
D43 **indexed seek stores** (`pkg.data`/`pkg.idx`, …) that the
language-agnostic [`../store/`](../store) builder writes. A bound-key lookup
binary-searches the sorted key index and reads only the records that key
touches — so a query reads a few KB of a multi-MB store instead of the whole
thing. This is the C++ lane of [`../rust_store/`](../rust_store),
[`../go_store/`](../go_store) and [`../wamjs_store/`](../wamjs_store).

## Layout

- `build.pl` — compiles `resolver.pl` + `resolver_store.pl` through `wam_cpp`
  with `cpp_wam_fact_sources(...)` declaring each store predicate as
  `indexed(Prefix)` (default) or `lmdb(Dir)` (opt-in). The two source files each
  carry a private copy of the shared helpers (`item_ver/3`, `lookup_held/3`, …);
  the loader drops the exact-duplicate second copy so the C++ WAM's multi-clause
  dispatch is not fed doubled clauses. `emit_main(false)` — the binary is the
  JSON diff driver, not the argv-goal CLI shim.
- `build.sh` — dumps the corpus P/2 JSONL, indexes it, compiles the WAM project,
  then `g++` compiles+links the diff driver. (`CXX`/`CXXFLAGS` overridable;
  default `-O2`.)
- `diff_main_store.cpp` — the JSONL differential driver: reads `{id, env-or-flat
  fields, query, args}` per line, builds the machine-local `env(...)` term with
  `env_build`, calls the `*_store/N` predicate, and emits one result line. Reuses
  `query_capture` from `../cpp/diff_main.cpp` (captures the output cell BEFORE
  `run()`), and — because one `vm` is reused across all cases — also resets the
  `foreign_iters`/`dynamic_iters` seek-iterator stacks between cases.
- `env_build.hpp/.cpp` — the one store-only adapter: `env_to_term(row)` builds
  `env(CatId, Base, Installed, Requested, Layers, Excluded, Aliases)`, mirroring
  `store_diff_runner.pl`'s `json_to_env/2` (incl. the top-level-vs-nested
  `catalog_id` fallback). Every field shape (`hold_term`, `pair_term`,
  `layer_term`, `alias_term`) is reused from `../cpp/term_build`.
- `json.*`, `term_build.*`, `term_to_json.*` — reused verbatim from `../cpp/`
  (self-contained lane, same as the rust/go lanes).
- `run_corpus_cpp_store.sh` — the P3 contract corpus vs the SWI store oracle.
- `run_differential_cpp_store.sh` — the 5k differential (seed `0xc0ffee01`) vs
  the SWI store oracle.

The generated `uw_resolve_wam_cpp_store/` project bakes the absolute store path
into `register_seek_fact_source`, so it is a build artifact and is `.gitignore`d;
run `build.sh` to regenerate it.

## How the seek reader plugs in

The store predicates compile to a two-instruction body
(`call_foreign store_pkg/2 2` + `proceed`), so a caller reaches them by label
like any other predicate. `Op::CallForeign` dispatches to
`dispatch_foreign_call`: it reads A1/A2, and for a bound atomic A1 does a keyed
seek over the UWFI/UWIX store (a port of the Go/Rust `seekFactSource` — binary
search over the sorted `.idx`, reading only the matching `.data` records) while
an unbound A1 falls back to a full scan (the provides walk). Matching rows stream
back through the choice-point machinery (`ForeignIterator` +
`Op::ForeignNextClause`), so bound-A2 filtering and multi-row backtracking work
exactly as for a term-catalog fact predicate.

## Runtime builtins this lane exercised (fixed in `wam_cpp_target.pl`)

The store adapter drives two C++ WAM builtins that the term-catalog lane never
called, exposing two mode gaps (each fixed with the store adapter as witness):

- `number_string/2` was unimplemented — `pack_ver`/`unpack_ver` use it for
  version (de)serialization in all 10 queries.
- `atom_concat/3` only implemented forward `(+,+,-)` concat; `unpack_constraint`
  needs the reverse `(+,-,+)` prefix-strip mode (e.g. strip `gte:` off
  `gte:1.0.0`), so every non-`any` constraint silently dropped its requirement.
  Added the deterministic `(+,-,+)` and `(-,+,+)` decomposition modes.

## Backends

- **`UW_STORE_BACKEND=indexed`** (default) reads the dependency-free UWFI/UWIX
  seek store with positioned `ifstream` reads and NO application cache — it leans
  entirely on the OS page cache.
- **`UW_STORE_BACKEND=lmdb`** (Stage 2) is the lazy + two-level-cached LMDB
  reader (compiled under `WAM_CPP_ENABLE_LMDB`, auto-#defined when an `lmdb(Dir)`
  seek source is declared; links system `liblmdb`). Each bound-key lookup is a
  keyed range-scan over the a1Range band; results are cached in an **L1**
  direct-mapped slot table (mirrors Rust's `L1_CACHE`) and an **L2** FIFO map
  (mirrors Rust's `CacheShard`; NOT Haskell's LRU). L2's default cap auto-sizes
  from live `/proc/meminfo`; env overrides `UW_WAM_LMDB_L2_CAP` /
  `UW_WAM_LMDB_L1_SLOTS` tune it (used by the benchmark sweep). It **fails
  loudly**, never silently falling back to indexed.

  **liblmdb format note:** the default `lmdb` npm prebuilt is a Symas fork whose
  page format vanilla system `liblmdb` rejects (`MDB_INVALID`). Set
  `UW_LMDB_DATA_V1=1` (wired into this lane's `build.sh`/run scripts) so
  `ensure_lmdb.sh` builds the OpenLDAP-0.9.29-lineage module from source, which
  vanilla `liblmdb` reads. See `ensure_lmdb.sh`.

## Numbers (this VM)

- Corpus **51/51** and the 5k store differential (`0xc0ffee01`) **503/503**, 0
  divergences vs the SWI store oracle, on **both** the `indexed` and `lmdb`
  backends (indexed on `-O0`+`-O2`).
- I/O attribution over the 503 cases (`UW_WAM_CACHE_ATTRIBUTION=1`): `indexed`
  reads **~2.46 MB in ~196k reads** (no cache); `lmdb` reads **~56 KB in ~1.25k
  reads** — the L1 cache absorbs ~16.3k repeat lookups (308 disk misses). This is
  an **I/O-volume** win, not by itself a wall-time win: with ample memory
  `indexed`'s reads are cheap page-cache hits, so the I/O gap only converts to a
  wall-time advantage under memory pressure (see below).
- `run_scale_cpp_store.sh` is the **memory×scale crossover** harness: it runs the
  workload through both backends under a `systemd-run` `MemoryMax` sweep and an
  in-process `UW_WAM_LMDB_L2_CAP` sweep, reporting wall-time + the D43 byte
  counters + cache hit/miss. The 5k store (~1 MB) fits in cache under any cap so
  the crossover is muted at this scale; a dramatic crossover needs a much larger
  catalog — the natural fit is loading real repo snapshots with structural
  sharing of unchanged packages (a large, realistic store that is also the
  multi-snapshot substrate).
