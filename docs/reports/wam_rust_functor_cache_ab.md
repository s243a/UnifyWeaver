<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) 2026 John William Creighton (s243a) -->

# Rust WAM functor-decomposition cache — A/B (D101)

**Date:** 2026-09-11. **Ledger:** D101. **Author:** Opus (implementer).
**What:** replace the per-call `"name/arity"` functor reverse-parse
(`inner.rfind('/')` + `parse::<usize>()`) with an O(1), id-keyed, lazily-filled
**functor-decomposition cache**, without changing the term representation
(`Value::Str(Sym, Args)` stays) or the `"name/arity"` key convention. The D100
re-profile ranked that reverse-parse the #1 remaining own-code B3 lever
(`memrchr` 6.06% + `next_match_back` 5.81% ≈ 11.9%, `functor_of` self +2.8%).

## Design

- Feature-gated exactly like interning. **`intern` OFF** (`type Sym = String`,
  no ids): the EXACT pre-change `functor_of` `rfind`/`parse` path, unchanged —
  the cache is ON-only, so OFF is a perfect pre-D101 baseline.
- **`intern` ON**: a new `functor_of_sym(&Sym, arity)` looks the functor's
  decomposition up by the functor's interned `Sym` u32 id and returns the name
  when the parsed arity matches, else the `str(...)`-stripped inner — **byte
  identical** to `functor_of` in every branch (matching arity, non-matching
  arity, no `/`, `str(...)` wrapper, unparsable/absent numeric suffix, a name
  that itself contains `/`). The ~26 mustache dispatch sites and
  `heap_node_shallow` were threaded the `&Sym` they already hold.
- The cache (`value.rs` interner module) stores, per id, a `Decomp { name,
  arity: Option<usize>, inner }` whose `&'static str`s are slices of the same
  canonical interned text `functor_of` would have sliced. It is **lazily
  filled** on the first `decomp(id)` for that id (one `AtomicPtr<Decomp>` per id
  slot, null = uncomputed; miss computes from the canonical text and publishes
  by CAS). Because the decomposition is a deterministic function of the id,
  racing fillers compute byte-identical bytes; the CAS keeps one and frees the
  losers. Lock-free / wait-free reads, no global `Mutex`, and — crucially —
  nothing on the interner's own hot path: only ids actually used as functors
  ever compute a decomposition, and each at most once.

Why lazy, not intern-time: the first cut precomputed the decomposition inside
`intern` for every newly interned name. A resolve interns a huge number of
distinct fresh **variable** names, each of which then paid a full-string
`rfind`/`parse` for a decomposition it never needed. That regressed
`interner::intern` by **+42M Ir** (48.7M → 90.6M) and nearly cancelled the
functor-parse savings (net only −3.5%). Lazy fill eliminates that entirely.

## Method

- Binary: committed `examples/pkg_resolver/rust/uw_resolve_wam`, `--release`.
- Workload: **B3** = one `resolve_layered` on the 5000-package scale catalog
  (`store/gen_scale_catalog.mjs`, seed `0xc0ffee01`) →
  `rust/.scale/case_5000.json` (via `scale_to_case.mjs`), the D94/D100 reference
  (`packages=7522`, `depends=15003`, `selection_size=10`).
- `valgrind --tool=callgrind --cache-sim=no` (Ir; LL-miss ≈ 0 on B3 per D94, so
  Ir is the faithful, deterministic proxy), `callgrind_annotate` self-Ir.
- **before** = pre-D101, `intern` ON, no cache (parent commit's crate, built ON).
  **after** = post-D101, `intern` ON, lazy cache. Same box, same case file.

## B3 callgrind Ir (before → after)

| | before (no cache) | after (lazy cache) | Δ |
| --- | ---: | ---: | ---: |
| **PROGRAM TOTALS** | **1,150,097,562** | **1,088,784,173** | **−61,313,389 (−5.33%)** |
| `core::slice::memchr::memrchr` | 69,849,797 | 39,462,731 | −30,387,066 |
| `CharSearcher::next_match_back` | 66,971,071 | 37,410,986 | −29,560,085 |
| `WamState::functor_of` (self) | 32,328,567 | 14,208,096 | −18,120,471 |
| `interner::intern` (self) | 48,751,030 | 48,234,400 | −516,630 (flat) |
| `interner::decomp` (self, new) | 0 | 19,928,701 | +19,928,701 |
| `BuildHasher::hash_one` | 32,811,711 | 32,812,269 | ~0 |
| `sip::Hasher::write` | 27,021,618 | 27,022,032 | ~0 |

The functor reverse-parse is roughly **halved** and the re-intern/hash of the
extracted name did not grow. The new `decomp` machinery costs 19.9M Ir (the
per-lookup atomic loads + first-touch compute), which is the price of the O(1)
lookup and is well under what the removed `rfind`/`parse` cost at those sites.

**Why not the full ~11.9%.** The mustache dispatch sites + `heap_node_shallow`
now use the cache, but the two hottest remaining `functor_of` callers —
`term_compare_derefed` and `terms_identical` — are **generator-emitted**
(`src/unifyweaver/targets/wam_rust_target.pl`, emitted into `state.rs`), not in
the templates, and still call `functor_of(&str)` (they hold a `&Sym` that
deref-coerces to `&str`, so they cannot key the id cache without a signature
change in the generator). They account for the residual `memrchr` (39.5M) +
`next_match_back` (37.4M). This change is deliberately **template-only** (per the
task scope: "the change lives in the TEMPLATES … NOT the .pl generator"), so
threading `functor_of_sym` through those generator-emitted comparators is the
natural follow-up (D102) that would reach the rest of the 11.9%.

## Wall time (native `--bench`, this box)

`resolve_ms`, 6 runs each, 5000-pkg case:

- **before**: 79.4, 88.0, 88.6, 87.9, 73.4, 70.0 → median ~83.7 ms
- **after** : 91.6, 82.1, 84.4, 83.0, 82.2, 81.8 → median ~82.6 ms

Wall is noise-dominated on this fast box (the two distributions overlap); the
deterministic callgrind Ir (−5.33%) is the reliable signal, consistent with
D100's note that B3 is instruction-bound and Ir is the faithful proxy.
`load_ms` and `selection_size=10` unchanged.

## Correctness gates (LC_ALL=C.UTF-8) — both configs, both lanes

Config ON = default (`decorate_sort` + `intern`); config OFF =
`--no-default-features --features decorate_sort` (pre-intern String path, the
cache not even compiled).

| gate | ON | OFF | ON ≡ OFF |
| --- | --- | --- | --- |
| term corpus (`run_corpus_rust.sh`) | 51/51 | 51/51 | `cmp` byte-identical |
| term differential (`run_differential_rust.sh`) | 2600/0/0 | 2600/0/0 | `cmp` byte-identical |
| store corpus (`run_corpus_rust_store.sh`) | 51/51 | 51/51 | `cmp` byte-identical |
| store differential (`run_differential_rust_store.sh`) | 503/0 | 503/0 | `cmp` byte-identical |
| 5000-pkg selection (`--bench`) | selection_size=10 | selection_size=10 | `cmp` byte-identical |

- `cargo test --lib`: **222/222** (ON) — includes the 2 new
  `functor_cache_tests` proving `functor_of_sym ≡ functor_of` over a spread of
  functor strings (`a/1`, `foo/2`, `str(bar/3)`, `[|]/2`, `-/2`, an atom with no
  slash, `x/2` at the wrong arity, `a/b/2` with a name containing `/`,
  `foo/bar` with an unparsable suffix, …) × arities `0..=13`, plus exact-value
  spot checks. The 2 tests also pass OFF (delegating path).
- Transactional alias test
  (`tests/test_wam_rust_foreign_tuple_aliases.pl`): **exit 0**, clean
  transactional `cargo test`, zero warnings (the test that previously caught
  interning breakage).

`examples/pkg_resolver/resolver.pl` and `resolver_store.pl` **UNMODIFIED**.

## Files changed

- `templates/targets/rust_wam/value.rs.mustache` — `Decomp`, `compute_decomp`,
  lazy `decomp(id)`, per-slot `AtomicPtr<Decomp>` dchunks in the interner.
- `templates/targets/rust_wam/state.rs.mustache` — `functor_of_sym` (cached fast
  path + OFF delegate), `heap_node_shallow` cfg-gated to use the cache, ~26
  dispatch sites threaded the `&Sym`, and the `functor_cache_tests` module.
- Regenerated committed crate `examples/pkg_resolver/rust/uw_resolve_wam`
  (`src/value.rs`, `src/state.rs`); regeneration-only timestamp/label churn in
  the other generated `.rs` reverted to keep the diff focused. `Cargo.toml`
  unchanged (the cache rides the existing `intern` feature).
- The `rust_store` crate (`uw_resolve_wam_store`) is a gitignored per-build
  artifact; it is regenerated and gated but not committed.
