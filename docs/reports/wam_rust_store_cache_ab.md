<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) 2026 John William Creighton (s243a) -->

# Rust WAM store L1 decoded-row cache — A/B (D106)

**Date:** 2026-09-18. **Ledger:** D106. **Author:** Sonnet (implementer).
**What:** an L1 decoded-row cache on `SeekFactSource` (the D43 store-backed
`indexed(Prefix)` seek reader) — a bound-key seek that has already been decoded
once is served from memory on every later seek for the same key, in the same
process, without touching the `.idx`/`.data` files again. Default-ON, gated,
and byte-identical to the pre-change (OFF) path by construction.

**Honest framing up front:** on the scales this cloud env runs, the indexed
store reads well under 1% of a B3 resolve and the store is RAM-resident/warm
(no cold-cache misses to amortize), so this is fundamentally a
correctness-preserving port whose real payoff is at cold/at-scale regimes —
the same regime the C++ store agent's mmap+cache work and the deferred LMDB
tier target. That said, the D43 bytes-read counters DID show a real,
reproducible reduction on the 5000-package B3 scale probe (see below) — some
keys are seeked more than once within a single resolve, and the cache collapses
those repeats to one real read. Wall-clock time did **not** move (noise-
dominated at sub-millisecond scale); I am reporting the honest number, not a
speedup that isn't there.

## The change

`SeekFactSource::rows(&self, key: Option<Vec<u8>>)` is the seek path's one
public entry point (reached from `WamState::execute_seek_fact_source`, the
"seek_fact" foreign native kind): a bound key does an `.idx` binary search
(`lookup_offsets`) then decodes each matching `.data` record
(`read_record`); an unbound key (`None`) does the unindexed `scan_all` full
scan (the unbound-arg1 "provides" walk).

Under the `store_cache` feature (default ON), `SeekFactSource` gains one new
field:

```rust
pub struct SeekFactSource {
    kind: String,
    path: String,
    inner: Mutex<Option<OpenStore>>,
    #[cfg(feature = "store_cache")]
    cache: Mutex<HashMap<Vec<u8>, Vec<(Value, Value)>>>,
}
```

keyed by the tagged lookup key exactly as `encode_store_key` produces it (atom
`0x41` / int `0x49` / float `0x46` + payload — the same key the `.idx` binary
search already keys on). `rows()`'s bound-key arm now does:

1. On entry, check the cache for `target`. A hit returns `cached.clone()`
   immediately — no `.idx` binary search, no `.data` reads, no decode.
2. On a miss, fall through to the EXACT pre-D106 seek (`lookup_offsets` +
   `read_record` per offset, in seek order), then insert the freshly decoded
   `rows.clone()` into the cache before returning `rows`.

The unbound arm (`scan_all`) is **left completely untouched** — no cache
lookup, no cache write. It is only ever run once per unbound-arg1 query (the
"provides" walk), its order is the #4272 key-clustered `.data` layout (answer-
invariant, per D105, but not necessarily the SAME order a bound-key cache
entry for one of its keys would need to match), and it is not the lever this
change targets — caching it would add risk for zero benefit.

**Why this is safe:** the store is read-only for the entire life of a
resolve — facts never change, and the store path is baked into
`setup_foreign_predicates` at BUILD time, not read at runtime. So a given key
always yields the same decoded rows, in the same order, for the life of the
process. Caching the exact `Vec<(Value, Value)>` a fresh seek would produce
(verbatim, never reconstructed or reordered) is byte-safe by construction —
there is no invalidation problem to get wrong because there is nothing that
ever invalidates it.

## Interior mutability: `Mutex`, not `RefCell` — and why

`rows()` takes `&self`, so the cache needs interior mutability. A `RefCell`
would be the natural (and slightly cheaper) choice **if** the resolve were
guaranteed single-threaded. It is not, and the evidence is concrete, not
hypothetical:

- `WamState` derives `Clone`, and `seek_fact_sources: HashMap<String,
  Arc<SeekFactSource>>` is one of its fields. Cloning `WamState` clones the
  `HashMap`, which clones each `Arc` — a shallow clone that still points at
  the SAME underlying `SeekFactSource`.
- `templates/targets/rust_wam/par_aggregate.rs.mustache` (the T7
  parallel-aggregate runtime) calls `base.clone()` **once per worker thread**
  in `map_bodies`/`map_bodies_labeled`, then runs each worker's clone via
  `thread::scope(|s| { ... s.spawn(move || { let mut m = base.clone(); ... })
  ... })` — **plain `std::thread`**, not gated behind the Cargo `parallel`
  feature (that feature only gates state.rs's separate rayon-based
  aggregate path). So whenever a parallel-eligible aggregate fires with
  enough inputs (`n >= 2 && cores >= 2`), multiple OS threads each hold an
  `Arc<SeekFactSource>` pointing at the same instance, and any of them can
  call `rows()` concurrently.

Given that, `self.inner` (the lazily-opened file handles) was ALREADY a
`Mutex<Option<OpenStore>>` for exactly this reason. The new `cache` field
matches it: `Mutex<HashMap<Vec<u8>, Vec<(Value, Value)>>>`. A `RefCell` here
would be **unsound** (a live `RefCell` borrow-checked at runtime is not
`Sync`; the compiler would in fact refuse to let `SeekFactSource` cross a
`thread::spawn` boundary with a `RefCell` field, since `Arc<SeekFactSource>`
needs `Sync` and `RefCell` breaks that) — so this was a compile-time-enforced
choice, not just a judgment call. Each `rows()` call takes the cache lock only
for the lookup and, on a miss, again (separately, after releasing `self.inner`)
for the insert — two short, non-nested critical sections; a benign race on a
miss (two threads both miss and both insert) simply overwrites with an
identical value, since the store is read-only.

## `scan_all` and seek order — explicitly not touched

Per scope, only the bound-key path is memoized:

- **`scan_all` is never cached.** Its walk order is the #4272 (D105)
  key-clustered `.data` physical layout — answer-invariant (the resolver
  doesn't depend on `scan_all`'s row order, only its row SET) but not
  necessarily equal to what a bound-key cache entry for one of its keys would
  need to replay. Leaving it alone removes any chance of the two paths ever
  needing to agree.
- **Bound-key seek order is preserved exactly.** A cache entry stores the
  seek's own `rows` `Vec` verbatim (`rows.clone()`, taken right where the
  uncached code already returns it) — never rebuilt, resorted, or
  deduplicated. A cache hit returns `cached.clone()`, an element-wise clone of
  that same `Vec` in the same order. The new unit test
  `cache_hit_matches_fresh_uncached_seek_same_order` asserts this directly
  against a two-row key ("alice" → `bob` then `carol`, in that order).

## Real A/B (D43 bytes-read proof, 5000-package B3 scale probe)

`examples/pkg_resolver/rust_store/run_scale_rust_store.sh` runs one bound
`resolve_layered` against the 5000-package scale catalog and reports the D43
bytes-read/reads counters plus wall time. Same store, same query, same binary
minus the `store_cache` feature — three repeats each, fully deterministic
(`resolve_ms`/timings vary by noise only; byte/read counts did not vary at
all across repeats):

| | OFF | ON | Δ |
|---|---:|---:|---:|
| `rust_store_n_reads` | 820 | 612 | **−208 (−25.4%)** |
| `rust_store_bytes_read` | 10,305 | 7,752 | **−2,553 (−24.8%)** |
| `rust_store_read_fraction` (of total store bytes) | 0.9022% | 0.6787% | — |
| `rust_store_resolve_s` | 0.023 | 0.024 | flat (noise) |
| `rust_store_selection_size` | 10 | 10 | unchanged (same answer) |

**Honest reading.** The read/byte-count reduction is real and reproducible
(not noise — identical across three repeats of each build): this 5000-package
resolve does re-seek at least one bound key more than once (e.g. a
package's `provides`/`dependents` row queried from more than one place in the
layered resolve), and the cache collapses each repeat to a single real read.
That is a genuine ~25% cut in the D43 proof's own numerator at THIS scale —
better than the "unmeasurable here" expectation this task was scoped under.
**But it does not show up in wall time**: both OFF and ON resolve in
~23–24ms, because the store is RAM-resident (page-cache-warm `pread` calls,
no disk latency) and even OFF's 820 reads / 10.3KB is a trivial cost next to
the rest of the resolve. The `read_fraction` numbers also confirm the
scoping note independent of this change: even OFF reads under 1% of the total
store (1.14MB) for this query. **The wall-clock payoff this lever is really
for is cold storage** (a cold page-cache miss costs microseconds to
milliseconds, not nanoseconds) **and/or a workload with more key repeats than
this scale probe's single query exercises** — both outside what this
environment can exercise or measure honestly. I am reporting the I/O-count
win because it is real and measured, and explicitly NOT claiming a wall-clock
speedup, because there isn't one here.

The B3 5000-package **answer** (`out_5000.json`, term lane) is byte-identical
ON vs OFF (see gate matrix) — the cache changes nothing about what is
returned, only how many times the same bytes are re-fetched from disk.

## Gate matrix (`LC_ALL=C.UTF-8`, ON = default, OFF = `--no-default-features --features "decorate_sort intern deref_memo trail_enum"`)

| Gate | ON | OFF |
|---|---|---|
| Term corpus (`run_corpus_rust.sh`) | 51/51 | 51/51 |
| Term differential (`run_differential_rust.sh`) | 2600 / 0 div / 0 crash | 2600 / 0 div / 0 crash |
| Store corpus (`run_corpus_rust_store.sh`) | 51/51 matched SWI, identical to term corpus | 51/51 matched SWI |
| Store differential (`run_differential_rust_store.sh`) | 503 / 0 div | 503 / 0 div |
| `cargo test --lib` (term crate) | 232 passed | 232 passed |
| `cargo test --lib` (store crate) | 232 passed | 232 passed |

**Byte-identity ON ≡ OFF (`cmp`-clean):** term differential (2600 cases),
term corpus (51 cases), store differential (503 cases), store corpus (51
cases), and the 5000-package B3 selection (`out_5000.json`) — all identical.

One process-level gotcha worth recording for future gates on this lane: the
generated store crate **bakes the absolute store directory path** into
`setup_foreign_predicates` at `swipl`/`build.pl` generation time (`build.sh`'s
own comment says so). `run_corpus_rust_store.sh` and
`run_differential_rust_store.sh` each call `build.sh` with a DIFFERENT
`STORE_DIR` (the small corpus store vs. the 5000-package scale store), so
running one after the other silently re-bakes the path — a plain
`cargo build --features ...` afterward (no re-run of `build.sh`) still
targets whichever store was baked in LAST. Doing the OFF corpus build right
after the differential run (which last baked the scale store) produced 41/51
spurious divergences purely from querying the wrong catalog with the wrong
`expected` answers — not a code bug. Re-running `build.sh` with the correct
`STORE_DIR` before each `cargo build --no-default-features ...` fixed it
immediately (51/51, byte-identical). Recorded here so the next person gating
this lane doesn't spend time chasing a phantom regression.

## Unit tests (new, `seek_fact_source.rs` `store_cache_tests` module)

Both pass under ON (miss-then-hit is exercised) and OFF (every call is
identically uncached, so they are trivially self-consistent) — same pattern
as `trail_enum_tests`/`deref_memo_tests`: one test file proving both configs
correct. They build a tiny on-disk UWFI/UWIX store pair by hand (the exact
bytes `ensure_open`/`lookup_offsets`/`read_record` decode) under a fresh
temp-dir prefix and drive `SeekFactSource::rows` directly — the real seek
path, not a mock:

- `cache_hit_matches_fresh_uncached_seek_same_order` — one `SeekFactSource`,
  two calls for the same key: the first (miss, populates) and second (hit)
  must return identical rows in identical order; a brand-new instance's
  single (ground-truth fresh, uncached) call must match too; and the decoded
  content itself is checked against the hand-built fixture (`alice → bob`
  then `alice → carol`, in that order — proving the miss populates correctly
  AND the hit doesn't perturb order).
- `cache_is_per_key_and_does_not_cross_contaminate` — two real keys plus one
  absent key, re-fetched after caching, confirm no cross-key leakage and that
  a miss on an absent key stays an (cached) empty result.

**Transactional alias test** (`tests/test_wam_rust_foreign_tuple_aliases.pl`):
exit 0, zero warnings, `1 pass` (`% PL-Unit: wam_rust_foreign_tuple_aliases ...
passed 4.710 sec`).

## Feature gating

Cargo feature `store_cache`, wired into the generated `[features]` table in
`src/unifyweaver/targets/wam_rust_target.pl` exactly like
`intern`/`decorate_sort`/`deref_memo`/`trail_enum`:
`default = ["decorate_sort", "intern", "deref_memo", "trail_enum",
"store_cache"]`. OFF via `--no-default-features --features "decorate_sort
intern deref_memo trail_enum"` compiles `SeekFactSource` with no `cache`
field at all — the exact pre-D106 seek path, a perfect A/B baseline. All
features work in combination (every gate above ran under the full default
stack).

## Deliverables / scope

- `templates/targets/rust_wam/seek_fact_source.rs.mustache` — the cache field
  + constructors + `rows()` cache check/populate + the two new unit tests.
- `src/unifyweaver/targets/wam_rust_target.pl` — `[features]` table entry.
- Term crate (`examples/pkg_resolver/rust/uw_resolve_wam/`) regenerated via
  `build.sh` and re-committed; regeneration-only churn (timestamp headers on
  every OTHER generated file, and the pre-existing `lib.rs` WAM
  label/bytecode-address drift — `resolver.pl` unchanged) was reverted back
  to HEAD, so the committed diff is confined to `Cargo.toml` (the feature
  entry) and `seek_fact_source.rs` (the real change); the committed crate
  (reverted `lib.rs`/etc. + new `Cargo.toml`/`seek_fact_source.rs`) was
  rebuilt directly from that exact tree and re-gated (51/51 corpus, 2600/0/0
  differential, byte-identical to the freshly-regenerated build).
- Store crate (`examples/pkg_resolver/rust_store/uw_resolve_wam_store/`) is
  gitignored (a per-build artifact, absolute store path baked in) — no
  tracked change there; it was built and gated from the same template.
- `resolver.pl` / `resolver_store.pl` UNMODIFIED.
- L2 (a raw block/page cache under the `.data`/`.idx` reads themselves,
  independent of key decode) is explicitly **not** built here — noted as a
  follow-up for whoever picks up the cold/at-scale work, alongside the
  deferred LMDB tier.
