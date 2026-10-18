<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) 2026 John William Creighton (s243a) -->

# Rust WAM re-profile after D110 (mimalloc) — the allocator bucket collapsed; interner-hashing is the top remaining own-code lever

**Date:** 2026-09-18. **Ledger:** D111. **Author:** Opus (coordinator).
**Why:** D110 shipped the mimalloc `#[global_allocator]` that D109 recommended.
Before choosing the next lever, the B3 ranking had to be regenerated on the
post-D110 (mimalloc-ON) binary — both to see how much of the D109 allocator
bucket actually evaporated and to decide whether interner-hashing (the D109
runner-up) is still worth doing or the term path has bottomed out.

## Method
- Binary: `examples/pkg_resolver/rust/uw_resolve_wam` rebuilt at main after the
  D110 merge (`c68f67b1a`), `mimalloc(true)` → mimalloc default-ON, `--release`.
- Workload: **B3** = one `resolve_layered` on the 5000-package scale catalog
  (`.scale/case_5000.json`, seed `0xc0ffee01`), `--bench`. Identical shape to
  D94/D100/D109.
- `valgrind --tool=callgrind` (Ir only; cache-sim + branch-sim off).
  `callgrind_annotate` self-Ir.
- Plain wall-clock sanity (this box, native): resolve_ms median **66.2** (reps
  2–9, warm-up excluded), load_ms median **17.1** — matches the D110 report's
  mimalloc-ON B3 median (65.6 ms). Trajectory: D100 165 → D109 73 → **~66 ms**.

## Callgrind total: 967.6 M → **737.3 M Ir (−23.8%)**

**The mimalloc swap IS visible to callgrind** — and it is essentially the whole
story of the −230 M drop.

**Correction to the D110 report.** The D110 A/B report stated callgrind "can't
see the mimalloc swap" (valgrind's own allocator substitution) and measured the
win by wall-clock only. That caveat is **wrong**: callgrind does *not* replace
`malloc` the way memcheck does — it just counts instructions, so mimalloc's
`mi_*` allocation code runs and is counted normally. The D110 wall-clock A/B
(−13.2% B3, −17.2% B2) stands and was correct; only the "callgrind is blind to
it" note was mistaken. Callgrind sees a −23.8% Ir reduction, and it isolates
cleanly to the allocator (below).

## B3 self-Ir buckets — D109 vs post-D110

| bucket | D109 Ir (share) | post-D110 Ir (share) | Δ absolute |
| --- | ---: | ---: | ---: |
| **allocator** | 359.3 M (37.1%) | **138.4 M (18.8%)** | **−220.9 M** (glibc `_int_malloc`/`consolidate`/`unlink` → mimalloc `mi_*` fast path) |
| deref family (`deref_heap`/`_var`/`_shallow`/`_chain`) | 114.9 M (11.9%) | 114.9 M (15.6%) | ~0 |
| **interner + hashing** (`intern`+`hash_one`+`sip::write`) | 108.3 M (11.2%) | 108.0 M (14.6%) | ~0 |
| comparator (`term_compare_derefed` ×2) | 52.0 M (5.4%) | 52.0 M (7.1%) | ~0 |
| decomp cache (`interner::decomp`, D101) | 36.9 M (3.8%) | 36.9 M (5.0%) | ~0 |
| json load (`Parser::value`+`string`) | 33.5 M (3.5%) | 33.4 M (4.5%) | ~0 |
| functor `/` reverse-parse residue (`memrchr`+`next_match_back`) | 23.1 M (2.4%) | 23.1 M (3.1%) | ~0 |
| libc `memcpy`+`memcmp` | — | 26.2 M (3.6%) | — |
| **TOTAL** | **967.6 M** | **737.3 M** | **−230.3 M (−23.8%)** |

Every **own-code** bucket is unchanged in **absolute** Ir (D110 touched zero
computation — a global allocator changes *how* memory is allocated, not *what*
is computed), so each grew as a **share** only because the total shrank. The
entire −230 M reduction is the allocator bucket (−220.9 M); the ~9 M remainder
is minor libc/rounding. mimalloc did exactly, and only, what D109 predicted.

## Reading — the term path is now interner-/deref-bound, not allocator-bound

With the allocator down to 18.8% (and it is now mimalloc's own already-lean
fast path, not glibc's binning/coalescing), the two largest **own-code** costs
are:

1. **deref family — 15.6% (114.9 M).** Already the target of D103 (deref memo)
   and D103's Arc-spine stability flag; the residual is the unavoidable canonical
   walk. Further gains here are **structural** (changing the term representation)
   and risky.
2. **interner + hashing — 14.6% (108.0 M):** `intern` 6.5% (48.2 M) + `hash_one`
   4.5% (32.8 M) + `sip::write` 3.7% (27.0 M). Every `intern()` SipHashes the
   name string to find-or-insert its `u32` id. D96 made a *name* a `u32`, but
   term construction on the hot path (deref_heap functor normalisation,
   `deref_var`, copy_term) still re-interns name **strings**, each paying a full
   SipHash. This is the **D109 runner-up, now confirmed as the top tractable
   own-code lever.**

The allocator is still 18.8%, but it is now mimalloc — a second allocator swap
buys nothing; cutting it further means **allocation-count** reduction (fewer
`Value`/`Vec`/`Arc` allocs in `deref_heap`/`strv`), which is the same structural,
high-risk lever as (1).

## Recommended next lever

**Interner-hashing reduction (byte-identical, medium complexity).** Attack the
`hash_one` + `sip::write` portion (8.1%, ~60 M Ir) — the pure find-or-insert
hashing cost — by not re-hashing already-interned names on the hot path:
- thread `Sym`s through construction instead of re-`intern()`ing strings, and/or
- cache the functor-normalisation (`"f/N"` name → `Sym`) so `deref_heap` doesn't
  re-hash the same functor name on every rebuild.

Realistic target: reclaim a good fraction of the 60 M hashing Ir, byte-identical
(the interned id for a name is stable, so caching/threading it changes nothing
observable). This is a clean A/B behind a feature gate, exactly like D95–D106.

**If it under-delivers**, the remaining term-site work is structural (term-repr
allocation-count reduction) with diminishing returns, and the better use of
effort is the backlog: cross-target reuse of these levers to Go/wamjs, or
mutual-recursion emission.

## Verdict

D110 delivered the D109 allocator lever exactly as predicted — the allocator
bucket fell from 37.1% to 18.8% (−220.9 M Ir), the whole of the −23.8% total.
The term path is not bottomed out: **interner-hashing (14.6%) is the top
tractable own-code lever** and is the recommended next step, with deref-family
(15.6%) and allocation-count both gated behind structural term-repr work. B3
wall-clock is now ~66 ms (D100 was 165 ms). `resolver.pl`/`resolver_store.pl`
UNMODIFIED (profiling only; no generator or crate change committed).
