<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) 2026 John William Creighton (s243a) -->

# Rust WAM re-profile after D95 (decorate-sort) + D96 (interning) — B3

**Date:** 2026-09-11. **Ledger:** D100. **Author:** Opus (coordinator).
**Why:** the last per-site profile (D94, `wam_rust_hotpath_deep_profile.md`)
predates D95 and D96, which landed the two biggest levers it ranked. Before
choosing the next optimization — the open question being "LMDB catalog tier vs.
a hotter path" — the rankings had to be regenerated against the current
(post-D96) binary.

## Method

- Binary: committed `examples/pkg_resolver/rust/uw_resolve_wam` (post-D96,
  `decorate_sort` + `intern` default-ON), `--release`.
- Workload: **B3** = one `resolve_layered` on the 5000-package scale catalog
  (`store/gen_scale_catalog.mjs`, seed `0xc0ffee01`), the D94/scale reference.
  `--bench` reports `load_ms` + `resolve_ms`; this profile covers the whole
  `--bench` run (load = index build + JSON parse, then resolve).
- `valgrind --tool=callgrind` (Ir only; cache-sim off — D94 already established
  LL-miss ≈ 0 %, so Ir is a faithful proxy and the cost is instruction-bound).
- `callgrind_annotate` self-Ir ranking.

Plain timing (this box): `load_ms=106`, `resolve_ms=165`, `selection_size=10`.
The D94 baseline was `resolve_ms≈312`; **D95+D96 roughly halved B3 resolve**,
as their A/B rows claimed. Callgrind total: **1.148 B Ir** (matches the D96 A/B
B3 figure 1,149 M — the profile is on the same shape).

## B3 self-Ir (callgrind, 1.148 B Ir)

| self-Ir | share | function | bucket |
| ---: | ---: | --- | --- |
| 124.8 M | 10.87 % | `_int_malloc` (libc) | allocator |
| 95.4 M | 8.31 % | `WamState::deref_heap'2` | deref |
| 83.8 M | 7.30 % | `_int_free` (libc) | allocator |
| **69.5 M** | **6.06 %** | `core::slice::memchr::memrchr` | **functor "/" reverse-parse** |
| **66.7 M** | **5.81 %** | `CharSearcher::next_match_back` | **functor "/" reverse-parse** |
| 54.3 M | 4.73 % | `malloc` (libc) | allocator |
| 48.6 M | 4.23 % | `value::interner::intern` | interner |
| 40.3 M | 3.51 % | `malloc_consolidate` (libc) | allocator |
| 38.3 M | 3.33 % | `term_compare_derefed'2` | comparator |
| 34.7 M | 3.02 % | `free` (libc) | allocator |
| 33.6 M | 2.93 % | `deref_var` | deref |
| 32.8 M | 2.86 % | `BuildHasher::hash_one` | interner-hash |
| 32.2 M | 2.80 % | `functor_of` | functor parse |
| 27.0 M | 2.35 % | `sip::Hasher::write` | interner-hash |
| 24.1 M | 2.09 % | `uw_resolve::json::Parser::value` | JSON load |
| 23.4 M | 2.04 % | `deref_heap` | deref |
| 14.1 M | 1.23 % | `Arc::drop_slow` | allocator |
| 11.6 M | 1.01 % | `term_compare_derefed` | comparator |
| (tail: `deref_shallow` 0.89 %, `interner::resolve` 0.79 %, `Value::strv` 0.64 %, `deref_chain` 0.45 %, region kernels ~1 %) | | | |

### Buckets

- **Allocator family** (`_int_malloc`+`_int_free`+`malloc`+`malloc_consolidate`+`free`+`unlink_chunk`+`__rdl_alloc`): **≈ 32 %.** Still the single largest cost — downstream of term (`Value`/`Vec`/`Arc`) construction. D96 removed the *name-String* allocs; the remaining churn is structural (heap nodes, arg vecs).
- **Functor `"name/arity"` string reverse-parse** (`memrchr` 6.06 % + `next_match_back` 5.81 % + the `functor_of` 2.80 % that calls them): **≈ 14.7 %**, and it *feeds* the interner (below) because the parsed name substring is re-interned each time.
- **Deref subsystem** (`deref_heap'2` 8.31 % + `deref_var` 2.93 % + `deref_heap` 2.04 % + `deref_shallow` 0.89 % + `deref_chain` 0.45 %): **≈ 14.6 %.**
- **Interner + its hashing** (`intern` 4.23 % + `hash_one` 2.86 % + `sip write` 2.35 % + `resolve` 0.79 %): **≈ 10.2 %** — largely driven by re-interning the name substring the functor parse just extracted.
- **Comparator** (`term_compare_derefed` ×2): **≈ 4.3 %** (post-decorate-sort; healthy).
- **JSON load** (`json::Parser::value`+`string`): **≈ 2.9 %** — load-time, not resolve.

## Reading — the next lever, and why it is not LMDB

**The `"name/arity"` functor representation is the new #1 own-code lever.**
`Value::Str(f, args)` stores the functor as a string `f = "name/arity"` (or
`str(name/arity)`); `functor_of` and `heap_node_shallow` recover the name and
arity with `inner.rfind('/')` + `parse::<usize>()` on **every** deref /
materialise / compare (`state.rs:4377`, `:4399`, `:4750`). That reverse search
is the `memrchr` + `next_match_back` ≈ 11.9 %, `functor_of` adds 2.8 %, and the
extracted substring is then re-interned/hashed (≈ 10 % more). D96 interned the
*name* but left the *composite functor* as a string that is re-parsed hot.

The indicated fix (a real optimization, D96-scale blast radius): **carry the
functor as a structured `(Sym, arity)` instead of a `"name/arity"` string** —
`Value::Str` holds the interned name id + a `u16`/`usize` arity, so `functor_of`
becomes a field read, the `rfind('/')`/`parse` vanish, and the re-intern/hash
per deref collapses. Estimated reachable: the ~11.9 % parse + most of the
functor-driven ~10 % interner/hash, i.e. a **~15–20 % B3** envelope, plus a B2
win (B2 was even more name/parse-bound in D94). Scale-independent — fully
demonstrable at the sizes this environment supports. Byte-identity risk is real
(functor equality/normalisation and standard order must be preserved exactly,
as D96 required for `functor_of`), so it gates the same way: both Cargo configs,
both lanes, term 2600/0 + store 503/0, `cmp`-clean vs the pre-change build.

**Runner-up:** the deref subsystem (~14.6 %) — D94 lever #3 (ground-bit / memo
so a ground subtree is not re-walked/re-materialised). Higher risk, and it
partly overlaps the functor change (deref_heap is hot *because* of the functor
parse), so do the functor lever first and re-profile.

**LMDB catalog tier is NOT the next lever.** The indexed seek store already
reads <1 % on B3 (D93/D94), so LMDB's only hypothesised win is mmap behaviour
at a scale well beyond the 5000-package point — and this cloud environment caps
the scale at which that utility could even be *measured* (per the maintainer).
Building it here would test functionality, not utility. Defer LMDB until it can
be benchmarked at a warranting scale; take the functor-representation lever now.

## Deferred / backlog after this

1. **Functor `(Sym, arity)` representation** — the lever above (next).
2. **deref ground-bit / memo** (D94 #3) — re-profile after (1).
3. **enum-tag the trail** (D94 #4, B2 bind path) — B2-specific.
4. **LMDB catalog tier** — only when a warranting scale is available to prove
   utility; measure vs the indexed seek store first.
