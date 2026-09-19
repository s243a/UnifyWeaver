<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) 2026 John William Creighton (s243a) -->

# Rust WAM interner Sym-threading — A/B (D113)

**Date:** 2026-09-19. **Ledger:** D113. **Author:** Opus (coordinator).
**What:** thread interned `Sym` ids through term construction so the hot
deref/rebuild sites build a `Value::Str` functor from a **cached id** instead of
de-interning an existing `Sym` to `&str` and RE-interning it (a round-trip that
re-hashes the name on every rebuild). Feature-gated `intern_sym_thread`
(default ON). The follow-on to D112 (which cut the *hash function* cost);
D113 removes the residual *re-hash calls* on the construction path.

**Motivation (D111/D112):** even after D112 (FxHash) the interner's `intern`
still shows on the profile because the hot construction path
(`deref_heap`'s `Str` and `Ref` arms, `heap_node_shallow`, `deref_shallow`)
takes a `Sym` already in hand, de-interns it to `&str` via `functor_of_sym`,
and re-interns that `&str` (`functor.into()` / `strv(functor, …)`) to build the
rebuilt node — hashing a name whose id was already known.

## The change

- **`Decomp` gains `name_sym` / `inner_sym`** (`value.rs`): the interned ids of
  the functor's `name` and `str(...)`-stripped `inner`, computed **once per
  functor id** in `compute_decomp` (the `Decomp` cache is already id-keyed,
  immutable, lazily filled — D101), not per rebuild.
- **`functor_sym(f: Sym, arity) -> Sym`** (`value.rs`, mod interner): the
  Sym-producing analog of `functor_of_sym` — returns `name_sym` when the arity
  matches, else `inner_sym`. O(1) slot read, no hash.
- **`WamState::functor_sym_for(&Sym, arity) -> Sym`** (`state.rs`): the single
  cfg point. ON → `crate::value::functor_sym` (cached id); OFF →
  `functor_of_sym(f, arity).into()` (the exact pre-D113 intern). The six hot
  construction sites all route through it:
  - `deref_heap` `Str` arms (deref-memo short-circuit, None, Some);
  - `deref_heap` `Ref` arm (also drops the manual re-intern of its parsed
    substring);
  - `heap_node_shallow` (intern path);
  - `deref_shallow` (also drops a wasted `f.to_string()` alloc).

Only `value.rs` + `state.rs` + the feature list change (wired in
`wam_rust_target.pl`'s `[features]` block); no `Cargo.toml.mustache` /
`template_system.pl` change.

## Byte-identity — by construction, and verified

`functor_sym(f, arity)` returns `name_sym` = `intern(name)` (or `inner_sym` =
`intern(inner)`), and `functor_of_sym(f, arity)` returns that same `name`/`inner`
`&str`; interning is canonical (same name ⟺ same id), so the id `functor_sym`
returns **is** the id the old `functor.into()` produced. Every routed site is
therefore byte-identical by construction.

Verified empirically, ON (default) vs OFF (isolated: `--no-default-features
--features "decorate_sort intern intern_fxhash deref_memo trail_enum
store_cache mimalloc"` — everything else, including D112's FxHash, held ON;
only `intern_sym_thread` toggled):

| gate | ON vs SWI | ON vs OFF |
| --- | --- | --- |
| term corpus (`run_corpus_rust.sh`) | 51/51 matched SWI | `cmp`-clean (51 lines) |
| term differential (`run_differential_rust.sh`, 2600 cases) | 0 divergences, 0 crashes | `cmp`-clean (2600 lines) |

## A/B measurement (isolated)

Callgrind Ir (B3, 5000-pkg `resolve_layered`, `--bench`; deterministic):
- ON (Sym-threaded): **683,302,765 Ir**
- OFF (re-intern): 698,097,477 Ir
- **−14,794,712 Ir, −2.12%.**

Smaller than D112's −5.95%, as expected: D112 already replaced the expensive
SipHash with FxHash, so the re-hash calls D113 eliminates were already cheap.
D113 removes them entirely on the construction path (plus the `deref_shallow`
`.to_string()` and the `Ref`-arm substring re-intern).

**Note on the baseline:** `functor_sym_for`'s OFF arm calls `functor_of_sym`
a second time (the deref sites already computed the `&str` `functor` for their
`is_cons`/verbatim-guard checks), so the isolated OFF binary (698.1 M) carries a
small redundant-`decomp` overhead vs the true pre-D113 default (D112's 694.0 M).
That overhead is present in **both** arms, so the −2.12% delta cleanly isolates
the hashing removal; the honest *default-to-default* trajectory is **694.0 M
(D112) → 683.3 M (D113)**, −1.5%. Either reading is a real, byte-identical win.

## Why this lever (beyond the −2%): it ports to the other targets

The interner + `Decomp` id-cache + "carry the id, don't re-intern the name"
pattern is target-agnostic. The Go and wamjs WAM targets do the same functor
`"name/arity"` round-tripping on their construction paths; D112 (hash choice)
and D113 (thread the id) are the two halves of the interner-cost fix that those
targets can reuse directly. That cross-target reuse — not the B3 −2% here — is
the main reason to land it (it makes the backlog "reuse the term levers on
Go/wamjs" item concrete: there is now a worked, byte-verified reference).

## Verdict

`intern_sym_thread` removes the construction-path re-intern round-trips,
byte-identical by construction and verified `cmp`-clean on both lanes and
ON-vs-OFF: **−2.12% B3 Ir** isolated (−1.5% default-to-default). B3 callgrind Ir
trajectory: D100 1,148 M → D109 967.6 M → D110 737.3 M → D112 694.0 M → **D113
683.3 M**.

The term path is now near its floor: the remaining own-code is the deref walk
itself (structural — D103 already memoises it) and the `intern` first-sight
insert (unavoidable). Further term-path work has diminishing returns; the higher
value from here is the backlog — porting the D112/D113 interner levers to
Go/wamjs (this report is the reference), and mutual-recursion emission.
`resolver.pl`/`resolver_store.pl` UNMODIFIED.
