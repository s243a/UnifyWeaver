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

Default `UW_STORE_BACKEND=indexed` reads the dependency-free UWFI/UWIX seek
store. `UW_STORE_BACKEND=lmdb` is opt-in and **fails loudly** on the C++ lane:
the lazy+cached LMDB reader is not built yet (planned as the M3 Stage-2
comparison tier, where the memory×scale crossover is measured) — it never
silently falls back to indexed.

## Numbers (this VM)

- Corpus **51/51**, 0 divergences vs the SWI store oracle.
- 5k store differential (`0xc0ffee01`): **503 cases, 0 divergences** vs SWI, on
  both `-O0` and `-O2` builds.
- The B3 bytes-read / wall-time payoff demo (a `--scale-probe` driver mode +
  `run_scale_cpp_store.sh`, as in `../rust_store`) lands with the Stage-2 LMDB
  comparison, where seek-vs-LMDB is measured against available memory.
