<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) 2026 John William Creighton (s243a) -->

# Rust WAM re-profile after D101–D104 — the term path is now allocator-bound

**Date:** 2026-09-18. **Ledger:** D109. **Author:** Opus (coordinator).
**Why:** the last per-site profile (D100) predated the whole D101–D104 lever
series (functor-decomposition cache, generator-comparator cache, deref
memoization, trail enum-tag). Before choosing the next optimization — or
declaring the hot path done — the rankings had to be regenerated against the
current (post-D104) binary.

## Method
- Binary: committed `examples/pkg_resolver/rust/uw_resolve_wam` at main
  (`cc21eccaf`, D108), all default features ON, `--release`.
- Workload: **B3** = one `resolve_layered` on the 5000-package scale catalog
  (`store/gen_scale_catalog.mjs`, seed `0xc0ffee01`), `--bench` (load = index
  build + JSON parse, then resolve). Same shape as D94/D100.
- `valgrind --tool=callgrind` (Ir only; cache-sim off — LL-miss ≈ 0, so Ir is
  a faithful, deterministic proxy). `callgrind_annotate` self-Ir.

Plain timing (this box): `load_ms=24.8`, `resolve_ms=73.2`, `selection_size=10`.
Trajectory: **D100 165 ms → 73 ms** (the D101–D104 term levers ~halved it
again). Callgrind total: **967,589,185 Ir** (D100 was 1,148,034,614 → **−15.7%**;
the deref/functor/trail levers landed as projected).

## B3 self-Ir buckets (967.6 M Ir)

| bucket | Ir | share | notes |
| --- | ---: | ---: | --- |
| **allocator** (`_int_malloc`+`_int_free`+`malloc`+`free`+`malloc_consolidate`+`unlink_chunk`+`__rdl_alloc`+`_int_free_merge_chunk`) | **359.3 M** | **37.1 %** | glibc malloc internals; downstream of Value/Vec/Arc/String construction |
| deref family (`deref_heap'2`+`deref_heap`+`deref_var`+`deref_shallow`+`deref_chain`) | 114.9 M | 11.9 % | post-D103 memo; the residual walk |
| **interner + hashing** (`intern`+`hash_one`+`sip::write`) | **108.3 M** | **11.2 %** | hashing a name string to find/insert its `u32` id, per intern call |
| comparator (`term_compare_derefed` ×2) | 52.0 M | 5.4 % | healthy post-decorate-sort |
| decomp cache (`interner::decomp`, D101) | 36.9 M | 3.8 % | the functor `(name,arity)` cache lookup |
| json load (`Parser::value`+`string`) | 33.5 M | 3.5 % | load-time, not resolve |
| functor `"/"` reverse-parse residue (`memrchr`+`next_match_back`) | 23.1 M | 2.4 % | D101/D102 killed the bulk (was ~11.9 %); residue from `functor_of(&str)` callers not migrated to `functor_of_sym` |

## Reading — the bottleneck has shifted to allocation

The own-code hot paths D95–D104 targeted (functor parse, deref re-walk, sort
comparator) have all shrunk; what's left standing is **allocation churn**. The
allocator is now **37 % of B3** and grew as a *share* (it was ~32 % at D100)
because everything around it got cheaper. It is downstream of term construction:
`Value`/`Args` `Vec` allocations during `deref_heap` rebuilds and `strv`,
`Arc<Spine>` churn (`Arc::drop_slow` 1.4 %, `finish_grow`/`reserve`/`from_iter`
~3 %), and name `String`s (`String::clone` 0.9 %, `format_inner` 0.7 %). D94's
cache-sim already established this cost is **instruction-bound** (LL-miss ≈ 0),
so the lever is cutting allocator *instructions*, not latency.

Second: the **interner + its hashing is 11.2 %** — every `intern()` hashes the
name string (SipHash via `hash_one`/`sip::write`) to find-or-insert its id. D96
made a *name* a `u32`, but constructing terms still re-interns name strings on
the hot path (deref_heap functor normalisation, deref_var, copy_term), each
paying a full string hash.

## Recommended next levers (ranked)

1. **Drop-in global allocator (mimalloc or jemalloc)** — the single
   highest-leverage, lowest-risk lever. A `#[global_allocator]` swap changes
   *how* memory is allocated, not *what* is computed, so it is **byte-identical
   by construction** (a perfect A/B), feature-gateable, and a few lines +
   one vendored-C-source crate. glibc's `_int_malloc`/`malloc_consolidate`/
   `unlink_chunk` (a big slice of the 37 %) are exactly the binning/coalescing
   overhead mimalloc's segregated free-lists avoid; allocation-heavy Rust
   programs routinely see a large chunk of that bucket evaporate. **Recommended
   first** — it attacks the dominant cost with essentially zero correctness risk.
2. **Interner hashing reduction (~11 %)** — avoid re-hashing already-interned
   names on the hot path: thread `Sym`s through instead of re-`intern()`ing
   strings, and/or cache the functor-normalisation → id so `deref_heap` doesn't
   re-hash the same functor name every rebuild. Medium complexity; byte-identical.
3. **Allocation-count reduction** — fewer `Value`/`Vec` allocs in `deref_heap`
   / `strv` (e.g. small-vec / arena for the resolve). Highest ceiling but
   structural and risky (touches the core term representation); do only if (1)
   and (2) don't get us where we want.

## Caveat / verdict
The term path is now allocator-bound; the biggest remaining win is not another
site-specific micro-opt but **the allocator itself**. Start with the mimalloc
`#[global_allocator]` swap (byte-identical, feature-gated), re-profile, then
decide whether the interner-hashing lever is still worth it. If the allocator
swap under-delivers, the remaining site work has diminishing returns and the
better use of effort is the backlog (cross-target reuse of these levers to
Go/wamjs; mutual-recursion emission) rather than squeezing the term path further.
