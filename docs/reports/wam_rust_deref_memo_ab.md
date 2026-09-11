<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) 2026 John William Creighton (s243a) -->

# Rust WAM deref-memoization (deref-stability cache) — A/B (D103)

**Date:** 2026-09-11. **Ledger:** D103. **Author:** Opus (implementer).
**What:** the D94 #3 lever — cache the predicate "`deref_heap` returns this node
structurally unchanged" (call it *deref-stable*) on the `Arc`-shared `Args`
spine, and short-circuit an already-canonical term instead of recursively
re-walking (and, where a sub-term moves, rebuilding) the entire term tree on
every top-level deref. The resolver derefs the same ground catalog rows
(`package/…`, `depends/4` …) over and over; a spine proven stable skips the
walk on every subsequent deref.

## The hazard this had to respect

`deref_heap` performs transformations even on terms with no unbound vars, so
"ground" is NOT sufficient to skip it. Every branch where `deref_heap` returns
something OTHER than a verbatim clone of the input node is a hazard:

1. `Value::Unbound` → follows `self.bindings`;
2. `Value::Ref(addr)` → materialises from `self.heap`;
3. `Value::Str` with a cons functor + arity 2 → rebuilt into a list via
   `deref_cons_chain`;
4. `Value::Str` functor **normalisation**: even when no arg moved it returns
   `Value::Str(functor, args.clone())` when the stored functor `!= functor_of`
   (i.e. `"f/N"` → `"f"`, or a `str(...)` wrapper);
5. any arg/item that derefs to a different cell → rebuilds the spine.

So the cache stores **deref-stability of the SPINE**, set ONLY from
`deref_heap`'s own verified identity path — never guessed from a separate
groundness walk. A spine is marked *stable* (state `1`) exactly when, in
`deref_heap`, it took the nothing-moved (`None`) branch on a **full view**
(`off == 0`) AND every element is itself deref-stable; it is marked *not stable*
(state `2`) when an element moved or is not deref-stable. `child_is_deref_stable`
classifies an element: atomic leaves (Atom/Integer/Float/Bool) are always
stable; `Str`/`List` iff their full spine is already stable (checked bottom-up —
the recursion has marked each stable child before the parent tests it);
`Unbound`/`Ref`/`Uninit` are NEVER stable. A deref-stable spine therefore
contains no `Unbound`, no `Ref`, no `"f/N"` functor and no cons-`Str`.

## Correctness invariants (relied upon)

- **Immutability.** `Value`s are immutable — a deref rebuilds, a binding lives
  in `self.bindings`, a spine is never mutated in place — so a spine's stability,
  once decided, is decided **forever**. Both states `1` and `2` are permanent
  properties of the immutable spine (the state-`2` case: whether an element slot
  holds `Unbound`/`Ref`/a non-stable sub-spine is fixed structure; whether that
  `Unbound` is currently bound only flips the node between "moves" and
  "returns-verbatim-but-unclassifiable", never to *markable-stable*).
- **Backtracking-safe.** A deref-stable node has no `Unbound`, no `Ref`, so
  bindings, the trail and heap changes cannot affect it. `child_is_deref_stable`
  reads immutable structure + permanent child flags, never `self.bindings`.
- **Byte-identical BY CONSTRUCTION.** The flag is set only where `deref_heap`
  itself returned the verbatim identity, so the short-circuit's `val.clone()` (or,
  for a shared spine reached through a normalising node, `Str(functor,
  args.clone())`) is exactly what the walk would have produced.
- **Idempotence.** `deref_heap(v) == deref_heap(deref_heap(v))` holds before and
  after; a unit test asserts it over `"f/N"` functors, cons cells, nested
  compounds, unbound + bound vars, and a heap `Ref`.
- **Offset soundness.** The flag describes the FULL spine (`items[0..]`); a
  window `[off..]` of an all-stable spine is all-stable, so the short-circuit is
  sound at any `off`, but the flag is only ever SET from a full-spine
  (`off == 0`) walk. A list `tail()` shares the parent's `Arc`, so once the full
  list is marked (by its `off == 0` deref) every tail short-circuits for free; a
  window whose full spine was never walked at `off == 0` simply falls back to the
  normal walk (correct, unoptimised).

## Representation

`value.rs`: the spine is now `Arc<Spine>` where
`struct Spine { items: Vec<Value>, stable: AtomicU8 }` (the `stable` field is
`deref_memo`-only). The tristate is `0 = unknown / 1 = stable / 2 = not`, read
and written with **`Relaxed`** atomics — a race recomputes the same
deterministic value, so a benign double-write is harmless and no lock is needed.
`Args::{from_vec,tail,cons,same_ref}`, `Deref` and `PartialEq` updated to go
through `spine.items`; new `is_full_view`/`stability`/`is_full_stable`/
`mark_stable`/`mark_not_stable` (all `deref_memo`-gated no-ops when OFF).

`state.rs` `deref_heap`: a `deref_memo` short-circuit at the top of the `Str`
arm (guarded `!is_cons`, returns `val.clone()` for a verbatim functor or
`Str(functor, args.clone())` for a normalising one) and the `List` arm; the
nothing-moved branches mark the spine `1`/`2`; the moved branches mark it `2`.
New `child_is_deref_stable`.

## Feature gate

Cargo feature **`deref_memo`**, default ON, added to
`default = ["decorate_sort", "intern", "deref_memo"]` in
`src/unifyweaver/targets/wam_rust_target.pl`. OFF (`--no-default-features
--features "decorate_sort intern"`) compiles the spine with no `stable` field
and every memo block vanishes — the EXACT pre-D103 `deref_heap`, a perfect A/B
baseline. ON must be byte-identical to OFF, and is.

## What is left unoptimised (for safety / by construction)

- A node whose functor **normalises** (`"depends/4"`) is itself a *move*, so it
  is never marked stable AND it disqualifies its ancestors (a parent holding it
  rebuilds). Its own **argument spine** is still marked stable, so the win — 
  skipping the N-argument re-walk — applies to the resolver's ground `depends/4`
  catalog rows; only the O(1) top functor re-normalisation + `args.clone()`
  remains per deref. (Unit-tested explicitly.)
- Terms containing an `Unbound` (a partial/query term) are never stable
  (correct — the var may bind).
- Cons-`Str` cells (the rebuild path) and heap `Ref`s are never memoised.
- A compound argument that is a list **tail** (`off > 0`) is only classifiable
  once its full spine was walked at `off == 0`; otherwise the parent is left
  unmarked (conservative).
- `deref_shallow` (the `unify` fast path) is unchanged — already O(1) per level.

## Method

- Binary: committed `examples/pkg_resolver/rust/uw_resolve_wam`, `--release`.
- Workload: **B3** = one `resolve_layered` on the 5000-package scale catalog
  (`store/gen_scale_catalog.mjs`, seed `0xc0ffee01`) →
  `rust/.scale/case_5000.json` (`packages=7522`, `depends=15003`,
  `selection_size=10`), the D94/D100/D101/D102 reference.
- `valgrind --tool=callgrind --cache-sim=no` (Ir; LL-miss ≈ 0 on B3 per D94, so
  Ir is the deterministic proxy), `callgrind_annotate` self-Ir.
- **before** = D102 HEAD (`deref_memo` not present), rebuilt and measured on
  **this** box. **after** = D103 ON. Same box, same `case_5000.json`.

## B3 callgrind Ir (before D102 → after D103)

| | before (D102) | after (D103) | Δ |
| --- | ---: | ---: | ---: |
| **PROGRAM TOTALS** | **1,021,351,227** | **979,505,663** | **−41,845,564 (−4.10%)** |
| `deref_heap'2` (mono copy) | 92,873,592 (9.09%) | 61,076,721 (6.24%) | **−31,796,871** |
| `deref_heap` (self) | 22,758,298 (2.23%) | 18,089,914 (1.85%) | **−4,668,384** |
| `deref_var` (self) | 33,633,195 (3.29%) | 20,766,937 (2.12%) | **−12,866,258** |
| `interner::decomp` (self) | 38,703,685 | 37,131,777 | −1,571,908 |
| `deref_shallow` (self) | 9,850,584 | 9,850,728 | 0 (untouched) |
| `deref_chain` (self) | 5,197,465 | 5,202,047 | 0 (untouched) |
| `memrchr` (comparator path) | 12,061,403 | 12,059,189 | 0 |
| `next_match_back` (comparator) | 11,024,522 | 11,022,754 | 0 |
| `RawVecInner::finish_grow` | 11,375,512 | 11,375,512 | 0 (see below) |

`deref_heap` (both copies) **115.63M → 79.17M = −31.5%**; `deref_heap` +
`deref_var` **149.27M → 99.93M = −49.3M**. The short-circuit retires the
recursive descent (fewer `deref_heap'2` calls) and, with it, the per-node
`deref_var` call on each argument — that is the whole −41.8M.

**Honest note on allocation:** `finish_grow` / `grow_one` (spine `Vec`
allocation) are **flat**. The pre-existing deferred-allocation path in
`deref_heap` (build the `derefed` vector only when a sub-term actually moves)
already avoided *rebuilding* ground spines; D103 additionally skips the
*re-walk* of them. So this lever cuts the instruction count of the walk
(`deref_heap`/`deref_var` self-Ir), **not** the allocation count — allocation
was already minimal on B3 for these terms.

`resolve_ms` (uninstrumented, this box): ON ~74.8–81.0 ms across runs;
noise-dominated and overlapping OFF, exactly as D101/D102 saw on this variable
box — Ir is the deterministic proxy. `load_ms`/`selection_size=10` unchanged.

## Gate matrix (both configs, both lanes, `LC_ALL=C.UTF-8`)

ON = default. OFF = `--no-default-features --features "decorate_sort intern"`.

| gate | ON | OFF | ON≡OFF |
| --- | --- | --- | --- |
| term corpus (`run_corpus_rust.sh`) | 51/51 | 51/51 | — |
| term differential (`run_differential_rust.sh`) | 2600 / 0 / 0 | 2600 / 0 / 0 | `cmp`-clean |
| term 5000-pkg selection | selection_size=10 | selection_size=10 | `cmp`-clean |
| store corpus (`run_corpus_rust_store.sh`) | 51/51 | 51/51 | `cmp`-clean |
| store differential (`run_differential_rust_store.sh`) | 503 / 0 | 503 / 0 | `cmp`-clean |
| `cargo test --lib` | 227/227 | 224/224 | (3 flag tests `deref_memo`-gated) |
| alias test (`test_wam_rust_foreign_tuple_aliases.pl`) | exit 0, 0 warnings | — | — |

New tests (`state.rs` `deref_memo_tests`): `deref_heap_is_idempotent`,
`transform_hazards_are_correct_cold_and_warm` (both run OFF too),
`stable_flag_semantics`, `nested_ground_marks_bottom_up`,
`normalising_functor_marks_arg_spine_but_not_ancestors` (the last three
`deref_memo`-gated). resolver.pl / resolver_store.pl **UNMODIFIED**.
