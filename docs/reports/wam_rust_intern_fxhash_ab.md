<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) 2026 John William Creighton (s243a) -->

# Rust WAM interner FxHash — A/B (D112)

**Date:** 2026-09-18. **Ledger:** D112. **Author:** Opus (coordinator).
**What:** swap the interner's `name -> id` lookup map from std's default SipHash
to the crate's existing FxHash (`state.rs` `FxBuildHasher`), feature-gated
(`intern_fxhash`, default ON). The D109 runner-up lever, confirmed by the D111
post-mimalloc re-profile as the top tractable own-code cost.

**Motivation (D111):** after mimalloc (D110) took the allocator bucket from
37.1% to 18.8%, the term path is interner-/deref-bound. Interner **hashing**
was 8.1% of B3 Ir — `hash_one` 4.5% + `sip::write` 3.7%. Every `intern(s)`
SipHashes the name string to probe the map, and the hot construction path
(`deref_heap` functor normalisation, `deref_var`, `copy_term`) re-interns name
strings constantly. SipHash is DoS-resistant but ~5–10× slower than a
non-cryptographic hash for these short, program-internal names.

## The change

- `value.rs` interner module: the `map: RwLock<HashMap<&'static str, u32>>`
  becomes a `NameMap` type alias — `HashMap<&'static str, u32, FxBuildHasher>`
  when `intern_fxhash` is on, the plain (SipHash) `HashMap` when off. It reuses
  the crate's existing `crate::state::FxBuildHasher` (defined for the kernel
  path's hot integer-keyed maps), so no new hasher code and no duplication.
- One line each: the field type and its initialiser (`HashMap::new()` →
  `NameMap::default()`). `resolve`/`decomp` are untouched (they index the
  chunked id→name arrays, not the map).
- Feature `intern_fxhash = []`, declared in every generated crate's
  `[features]` and added to `default` (after `intern`), exactly like the
  D95–D106 levers. Wired in `wam_rust_target.pl`'s `[features]` `format/2`
  block; no `Cargo.toml.mustache` / `template_system.pl` change (this lever is
  entirely inside `value.rs` + the feature list).

## Byte-identity — by construction, and verified

A name's id is assigned from the interner's `len` counter in **first-sight
order** (see `intern`: read-lock probe → on miss, `id = len; len += 1`), never
from the hash. The map is only ever probed with `get`/`insert`; its iteration
order is never observed anywhere. So the hasher **cannot** change any id, hence
cannot change any output — byte-identical by construction.

Verified empirically, ON (default) vs OFF (isolated: `--no-default-features
--features "decorate_sort intern deref_memo trail_enum store_cache mimalloc"`
— mimalloc held ON so only this lever toggles):

| gate | ON vs SWI | ON vs OFF |
| --- | --- | --- |
| term corpus (`run_corpus_rust.sh`) | 51/51 matched SWI | `cmp`-clean (51 lines) |
| term differential (`run_differential_rust.sh`, 2600 cases) | 0 divergences, 0 crashes | `cmp`-clean (2600 lines) |

## A/B measurement

**Isolated** (mimalloc ON both sides; only `intern_fxhash` toggled).

**Callgrind Ir (B3, 5000-pkg `resolve_layered`, `--bench`; deterministic, the
faithful instrument here — D94 established this path is instruction-bound,
LL-miss ≈ 0):**
- ON (FxHash): **693,964,358 Ir**
- OFF (SipHash): 737,900,158 Ir
- **−43,935,800 Ir, −5.95%.**

The hashing bucket was ~60 M Ir (8.1% of 737 M); FxHash removed ~44 M of it
(≈73%), the residual being FxHash's own (much cheaper) per-byte mixing. The
`sip::write` frame disappears entirely; `hash_one` shrinks to the FxHash inline.

**Wall-clock (B3, same workload, interleaved ON/OFF, n=12 reps each, warm-up
excluded, shared/noisy host so median is the headline):**
- ON resolve_ms median **58.02**, OFF **60.95** — **−2.93 ms, −4.8%.**

Consistent in direction with the Ir delta (a hair smaller, as expected — not
every instruction converts to wall-time 1:1, and ~5% is near this host's noise
floor, so the deterministic Ir number is the honest headline).

## Verdict

`intern_fxhash` delivers the D111-recommended lever: **−5.95% B3 Ir** (−4.8%
wall-clock), byte-identical by construction and verified `cmp`-clean on both
lanes vs the SWI oracle and ON-vs-OFF. It reuses the crate's existing FxHash,
adds ~5 lines of real change behind a clean feature gate, and carries no
correctness risk (a lookup-table hasher cannot affect insertion-ordered ids).

Trajectory: B3 callgrind Ir D100 1,148 M → D109 967.6 M → D110 737.3 M → **D112
694.0 M**. The remaining own-code costs are `intern` itself (the map probe +
first-sight insert, 6.5%) and the deref family (15.6%, already D103-optimised);
both are structural from here. The next-biggest tractable step would be
**threading `Sym`s through construction** so already-interned names are not
re-looked-up at all (attacks the residual `intern` probe, not just its hash) —
higher effort; a reasonable stopping point for the term path, with the backlog
(cross-target reuse to Go/wamjs; mutual-recursion emission) the better use of
effort next. `resolver.pl`/`resolver_store.pl` UNMODIFIED.
