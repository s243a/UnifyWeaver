# uw-resolve — Go WAM, store-backed (D43 indexed seek)

The store-backed sibling of [`../go/`](../go). Same shared adapter
([`../resolver_store.pl`](../resolver_store.pl)), same 10 queries, but the
catalog is **not** loaded as a term: `store_pkg/2`, `store_dep/2`,
`store_conflict/2`, `store_revdep/2` and `store_provides/2` are served from the
D43 **indexed seek stores** (`pkg.data`/`pkg.idx`, …) that the language-agnostic
`../store/` builder writes. A bound-key lookup binary-searches the sorted key
index and reads only the records that key touches — so a query reads a few KB of
a multi-MB store instead of the whole thing. This is the Go lane of
[`../wamjs_store/`](../wamjs_store).

## Layout

- `build.pl` — compiles `resolver.pl` + `resolver_store.pl` through `wam_go`
  (`prefer_wam(true)`) with `go_wam_fact_sources(...)` declaring each store
  predicate as `indexed(Prefix)` (default) or `lmdb(Dir)` (opt-in).
- `build.sh` — dumps the corpus P/2 JSONL, indexes it, compiles, `go build`.
- `shim_store.go` — term ↔ JSON IO only (env term + the 10 queries). No
  resolver logic, no catalog.
- `run_corpus_go_store.sh` — corpus vs SWI, and asserts byte-identical to the
  `../go` term corpus.
- `run_differential_go_store.sh` — the 5k differential (seed `0xc0ffee01`) vs
  the SWI store oracle.
- `run_scale_go_store.sh` — the B3 bytes-read + wall-time payoff.

The generated `.go`/`go.mod` files are build artifacts (they bake the absolute
store path) and are `.gitignore`d; run `build.sh` to regenerate them.

## Backends

Default `UW_STORE_BACKEND=indexed` reads the dependency-free UWFI/UWIX seek
store. `UW_STORE_BACKEND=lmdb` is opt-in and **fails loudly** on the Go lane:
the Go binary carries no repo dependency and does not read the npm-`lmdb`
(`uw_fact_lmdb`) store format — it never silently falls back to indexed.

## Numbers (this VM)

- Corpus **51/51**, byte-identical to the `../go` term corpus.
- 5k store differential (`0xc0ffee01`): **503 cases, 0 divergences** vs SWI.
- B3 (bound `resolve_layered` on the 5k catalog): resolve **~0.77s** reading
  **~11 KB of the ~1.14 MB store (<1%)**, same 10-package selection as the term
  build — whose B3 loads the whole catalog and resolves in ~14.6s. See
  [`../../docs/WAM_GO_STATUS.md`](../../../docs/WAM_GO_STATUS.md).
