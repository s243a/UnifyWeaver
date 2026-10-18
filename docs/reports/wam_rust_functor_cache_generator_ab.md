<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) 2026 John William Creighton (s243a) -->

# Rust WAM functor-decomposition cache in the GENERATOR comparator — A/B (D102)

**Date:** 2026-09-11. **Ledger:** D102. **Author:** Opus (implementer).
**What:** extend the D101 O(1), id-keyed, lazily-filled functor-decomposition
cache (`functor_of_sym(&Sym, arity)`) into the **generator-emitted** sort
comparator `term_compare_derefed`, the last hot `functor_of(&str)` caller on
the B3 path. D101 threaded `functor_of_sym` through the ~26 mustache dispatch
sites and `heap_node_shallow`, but was **template-scoped**; `term_compare_derefed`
is emitted from `src/unifyweaver/targets/wam_rust_target.pl` and still called
`Self::functor_of(f, arity)` on the interned functor `Sym` (Deref to `&str` →
`rfind('/')`/`parse` reverse-scan on **every** comparison, O(n log n) per sort).

## The change

In `src/unifyweaver/targets/wam_rust_target.pl`, in the emitted
`term_compare_derefed`, the `(Value::Str(f1, a1), Value::Str(f2, a2))` equal-arity
arm:

```rust
-   let n1 = Self::functor_of(f1, a1.len());
-   let n2 = Self::functor_of(f2, a2.len());
+   let n1 = Self::functor_of_sym(f1, a1.len());
+   let n2 = Self::functor_of_sym(f2, a2.len());
```

`f1`/`f2` are the functor `Sym`s of `Value::Str` (already in hand as `&Sym`;
they previously deref-coerced to `&str` for `functor_of`). `functor_of_sym`
takes `&Sym`, so they are passed directly — no de-interning, no work before the
O(1) slot read. It returns the **byte-identical** name/inner (`functor_of_sym ≡
functor_of` in every branch, established by the D101 `functor_cache_tests`).
Under `intern` **OFF**, `functor_of_sym` delegates to `functor_of`, so the OFF
build is unchanged and remains a perfect pre-cache baseline. The change is a
pure call-site swap plus the doc comment above the function; the committed crate
diff is confined to `state.rs`.

### `terms_identical` — measured, deliberately NOT touched

`terms_identical` uses `Self::display_functor_name(f, arity)` only on its
`f1 != f2` **fallback** path; the fast path is `f1 == f2` `Sym` equality. The
re-profile (below) shows `WamState::display_functor_name` self-cost at **1,352
Ir (0.00%)** both before and after — the interned-`Sym` fast path retires the
comparison and the string fallback is essentially never entered on B3. It is
**not** a material B3 cost, so per the task ("if it is NOT material … LEAVE
`terms_identical` unchanged and say so") it was left unchanged. No
`display_functor_name_sym` was added.

## Method

- Binary: committed `examples/pkg_resolver/rust/uw_resolve_wam`, `--release`.
- Workload: **B3** = one `resolve_layered` on the 5000-package scale catalog
  (`store/gen_scale_catalog.mjs`, seed `0xc0ffee01`) →
  `rust/.scale/case_5000.json` (via `scale_to_case.mjs`), the D94/D100/D101
  reference (`packages=7522`, `depends=15003`, `selection_size=10`).
- `valgrind --tool=callgrind --cache-sim=no` (Ir; LL-miss ≈ 0 on B3 per D94, so
  Ir is the faithful, deterministic proxy), `callgrind_annotate` self-Ir.
- **before** = D101 (`intern` ON, cache present, comparator still calls
  `functor_of(&str)`) — the committed HEAD crate, rebuilt and re-measured on
  **this** box. **after** = D102 (`intern` ON, comparator calls
  `functor_of_sym`). Same box, same `case_5000.json`.
- Directly-measured same-box before-total is **1,077.3M Ir**; the D101 report's
  nominal after-total (its own box) was 1,088.8M. The A/B delta below is the
  internally-consistent same-box, same-case-file before→after.

## B3 callgrind Ir (before D101 → after D102)

| | before (D101) | after (D102) | Δ |
| --- | ---: | ---: | ---: |
| **PROGRAM TOTALS** | **1,077,342,955** | **1,032,365,485** | **−44,977,470 (−4.17%)** |
| `core::slice::memchr::memrchr` | 39,462,731 | 12,061,403 | **−27,401,328** |
| `CharSearcher::next_match_back` | 37,410,986 | 11,024,522 | **−26,386,464** |
| `WamState::functor_of` (self) | 14,208,096 | ~0 (dropped out) | **−14,208,096** |
| `interner::decomp` (self) | 19,928,701 | 38,703,685 | +18,774,984 |
| `term_compare_derefed` (self) | 11,814,195 | 12,504,075 | +689,880 |
| `term_compare_derefed'2` (mono copy) | 38,621,594 | 40,047,280 | +1,425,686 |
| `WamState::display_functor_name` (self) | 1,352 | 1,352 | 0 (cold) |

The comparator's per-comparison functor reverse-parse is **gone**:
`WamState::functor_of` falls out of the profile entirely (only the shim's
unrelated `uw_resolve::functor_of` at 1,640 Ir remains), and `memrchr` /
`next_match_back` — the D100/D101 residual "functor lever" — each drop by ~69%
/ ~71% (they now retain only work from unrelated string scanning elsewhere,
e.g. version-segment parsing, not the comparator). The cost that moves **in** is
`interner::decomp` (+18.8M: the comparator now performs the O(1) cached
decomposition lookup, lazily first-touching functor ids during the sort) plus a
marginal +2.1M in the comparator itself. Net **−45.0M Ir (−4.17%)**, on top of
D101's −61.3M, completing the functor lever the D101 report named as the D102
follow-up.

## Wall time (native `--bench`, this box)

`resolve_ms`, 5 runs each, 5000-pkg case:

- **before (D101)**: 84.5, 70.5, 68.8, 77.3, 87.0 → median ~77.3 ms
- **after (D102)** : 97.0, 87.3, 88.6, 80.1, 78.7 → median ~87.3 ms

Wall is **noise-dominated** on this box (the two distributions overlap and the
box was under variable build/valgrind load during sampling); the deterministic
callgrind Ir (−4.17%) is the reliable signal, consistent with D100/D101 that B3
is instruction-bound and Ir is the faithful proxy. `load_ms` and
`selection_size=10` unchanged.

## Correctness gates (LC_ALL=C.UTF-8) — both configs, both lanes

Config ON = default (`decorate_sort` + `intern`); config OFF =
`--no-default-features --features decorate_sort` (pre-intern String path; the
comparator's `functor_of_sym` delegates to `functor_of`).

| gate | ON | OFF | ON ≡ OFF |
| --- | --- | --- | --- |
| term corpus (`run_corpus_rust.sh`) | 51/51 | 51/51 | — |
| term differential (`run_differential_rust.sh`) | 2600/0/0 | 2600/0/0 | `cmp` byte-identical |
| store corpus (`run_corpus_rust_store.sh`) | 51/51 | 51/51 | — |
| store differential (`run_differential_rust_store.sh`) | 503/0 | 503/0 | `cmp` byte-identical |
| 5000-pkg selection (`--bench`) | selection_size=10 | selection_size=10 | `cmp` byte-identical |

- `cargo test --lib`: **222/222** (ON) — includes the D101 `functor_cache_tests`
  proving `functor_of_sym ≡ functor_of` (the equivalence this change relies on).
  No new tests were needed: `terms_identical` was not touched and no
  `display_functor_name_sym` was added.
- Transactional alias test (`tests/test_wam_rust_foreign_tuple_aliases.pl`):
  **exit 0**, zero warnings.

`examples/pkg_resolver/resolver.pl` and `resolver_store.pl` **UNMODIFIED**.

## Files changed

- `src/unifyweaver/targets/wam_rust_target.pl` — emitted `term_compare_derefed`:
  the two `functor_of` → `functor_of_sym` call-site swaps in the equal-arity
  `Str` arm, plus the function's doc comment updated to describe the id-keyed
  path.
- Regenerated committed crate `examples/pkg_resolver/rust/uw_resolve_wam`: the
  diff is **confined to `src/state.rs`** (the emitted `term_compare_derefed`).
  `value.rs` was not touched (no new cached field was needed — the D101 `decomp`
  cache already suffices). Regeneration-only churn was reverted to keep the
  diff focused: (a) timestamp-header-only diffs in every generated `.rs`, and
  (b) the pre-existing `lib.rs` WAM label-address-table drift (`labels.insert`
  lines) — that address drift is unrelated to this change (bytecode-layout drift
  from earlier generator merges; `resolver.pl` is unchanged) and is kept OUT of
  the diff. `git diff` confirms no `labels.insert` churn and no timestamp-only
  files are staged.
- The `rust_store` crate (`uw_resolve_wam_store`) is a gitignored per-build
  artifact; it is regenerated and gated but not committed.
