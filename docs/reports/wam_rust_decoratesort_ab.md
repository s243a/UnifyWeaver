<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) 2026 John William Creighton (s243a) -->

# Rust WAM (uw-resolve) round #1 — decorate-sort the sort comparator: A/B

This is the implementation + A/B report for **worklist item #1** of the deep
hot-path profile (`wam_rust_hotpath_deep_profile.md`, D94): decorate-sort the
`sort`/`msort` comparator so it stops re-dereferencing and re-allocating on
every comparison. D94 ranked this first — the sort comparator path is
**≈61.6 % of all B3 instructions** — as a low-risk, single-function change.

**Result up front:** decorate-sort cuts **B3 by 60.0 % of total instructions**
(callgrind Ir) and **63.0 % of wall time** (2.70× faster on the 5 000-package
`resolve_layered`), byte-identical output, no B2 regression. This **exceeds**
D94's ~30–40 % projection. **Default: ON.**

## What changed

Base content SHA: `origin/main` @ `ef148c548` + D94 docs (the deep profile).
Branch `claude/wam-rust-decoratesort-aodas5`.

### The problem (from D94)

`msort/2` and `sort/2` already pre-deref each element **once** (O(n)):

```rust
let mut sorted: Vec<Value> = list.iter()
    .map(|v| self.deref_heap(&self.deref_var(v)))   // pre-deref ONCE, O(n)
    .collect();
sorted.sort_by(|a, b| self.term_compare(a, b));      // but term_compare RE-derefs
```

…but the comparator `term_compare` opened by re-dereferencing **both** operands
on **every** comparison:

```rust
let da = self.deref_heap(&self.deref_var(a));   // redundant — a is already derefed
let db = self.deref_heap(&self.deref_var(b));
```

so all ~84 000 comparisons re-walked and re-cloned both operands (O(n log n))
and, at each `Value::Str` node, allocated a fresh functor `String` via
`display_functor_name`. That comparator path (driftsort + `dedup_by`) was 61.6 %
of B3.

### The fix — `term_compare_derefed` + decorate-sort

- **`term_compare_derefed(a, b)`**: identical ordering to `term_compare`, but it
  **trusts that both operands are already fully dereferenced** — the invariant
  the pre-dereffing sort builtins establish. It calls **no**
  `deref_heap`/`deref_var` at any level (`deref_heap` recurses, so a pre-deref'd
  term has fully-deref'd children too, and the recursion stays in
  `term_compare_derefed`).
- **No per-comparison functor `String` alloc, without interning.** In the
  compound branch it compares functor names via **`functor_of`** (which returns a
  borrowed `&str`) instead of `display_functor_name` (which returns an owned
  `String`). `functor_of` and `display_functor_name` apply **byte-identical**
  normalisation (strip a `str(...)` wrapper, then strip a trailing `/N` iff it
  matches the arity), and `deref_heap` has already normalised every compound
  functor to its bare form — so this is byte-identical *and* allocation-free.
  This is **not** functor interning (that is round #2); it only removes the
  per-comparison `String` churn in the comparator. Atom-class compares likewise
  use a new borrowing `value_atom_name_ref` (`&str`) instead of the
  `String`-cloning `value_atom_name`.
- **Decorate-sort wiring.** The pre-dereffing builtins — `sort/2`, `msort/2`,
  `sort/4`, `keysort/2`, and the `setof` aggregate reduce — sort/dedup on their
  pre-deref'd keys through a `sort_cmp` dispatch. `compare/3` and the
  `@<`/`@=<`/`@>`/`@>=` family keep the raw `term_compare` (their operands come
  straight from registers, un-deref'd, so they legitimately need the deref).

### The flag (for a clean A/B)

`sort_cmp` is gated on a new Cargo feature **`decorate_sort` (default ON)**:

```rust
#[cfg(feature = "decorate_sort")]
fn sort_cmp(&self, a, b) -> Ordering { self.term_compare_derefed(a, b) }
#[cfg(not(feature = "decorate_sort"))]
fn sort_cmp(&self, a, b) -> Ordering { self.term_compare(a, b) }   // original re-deref path
```

so the ON build and the OFF build are **sha-distinct binaries** with
byte-identical output. The feature is appended to the generated `Cargo.toml`
inside the wam_rust target (`wam_rust_target.pl`), not the shared cargo template.

## Gates — both configs

Run with `LC_ALL=C.UTF-8`. ON = default features; OFF = `--no-default-features`.

| gate | ON | OFF | ON≡OFF byte-identical |
|---|---|---|---|
| term corpus | **51/51** | **51/51** | ✅ (`cmp` clean) |
| term differential | **2600 / 0 / 0** | **2600 / 0 / 0** | ✅ (sha match) |
| store corpus | **51/51** | **51/51** | ✅ (`cmp` clean) |
| store differential | **503 / 0** | **503 / 0** | ✅ (`cmp` clean) |
| 5 k `resolve_layered` selection | selection_size=10 | selection_size=10 | ✅ (sha match) |

Every gate passes both configs, and the ON output is byte-identical to the OFF
output across corpus, differential, and the 5 000-package selection.

## Stress tests (direct comparator equivalence)

`examples/pkg_resolver/rust/uw_resolve_wam/tests/comparator_equiv.rs` (5 tests,
all pass): a direct `term_compare_derefed == term_compare` check over a corpus
covering every standard-order class (var / number / atom / `[]` / list /
compound), the named edge cases (Float before an equal Integer, list-prefix
ordering, compound arity-then-name-then-args, embedded-slash functor
normalisation), and deep/nested terms; plus antisymmetry + reflexivity of the
new comparator, and sort + dedup identity under both comparators. Both functions
are feature-independent, so the result holds for ON and OFF.

## Real B3 A/B — decorate-sort ON vs OFF

Workload: 5 000-package `resolve_layered --bench` (`selection_size=10`,
byte-identical selection ON vs OFF). Binaries are sha-distinct, verified-relink.

### Instruction count (callgrind Ir — deterministic, drift-immune)

| build | total Ir | vs OFF |
|---|---:|---:|
| **OFF** (original re-deref) | 4,028,841,168 | — (≈ D94's 4.184 B B3 baseline) |
| **ON** (decorate-sort) | 1,609,745,729 | **−60.0 %** |

Attribution (callgrind self-Ir), the mechanism, OFF → ON:

| function | OFF self-Ir | ON self-Ir |
|---|---:|---:|
| `deref_heap` (incl. `'2` split) | ~599 M | ~119 M |
| `deref_var` | 222 M | 43 M |
| `term_compare` (self) | driven via callees | ~0 (only `@<`/`compare/3`) |
| `term_compare_derefed` (new) | — | ~50 M |

The redundant per-comparison re-deref is gone: `deref_heap` self-Ir drops ~80 %,
`deref_var` ~80 %, and the malloc/free family (the functor-`String` churn) falls
with it. The new comparator itself is cheap (~50 M Ir).

### Wall time (6-round interleaved, drift-cancelling)

| round | ON resolve_ms | OFF resolve_ms |
|---|---:|---:|
| 1 | 136.4 | 351.5 |
| 2 | 132.8 | 347.4 |
| 3 | 124.0 | 355.7 |
| 4 | 127.2 | 343.2 |
| 5 | 118.2 | 349.6 |
| 6 | 132.5 | 359.3 |
| **median** | **129.8** | **350.5** |

**B3 wall reduction (median): 63.0 %  → 2.70× faster.** The wall result tracks
the Ir result (D94's cache-sim showed the working set is L2/L3-resident, so Ir
is a faithful proxy — confirmed here).

### Confirm / refute the ~30–40 % projection

**Confirmed and exceeded.** D94 projected ~30–40 % of B3 for decorate-sort
alone, assuming roughly two-thirds of the 61.6 % comparator cost was the
head-deref + functor-`String`. In practice essentially the *whole* comparator
cost was the re-deref/alloc: `term_compare_derefed` on already-flat terms is
nearly free, so removing the re-deref removes close to the full 61.6 %, landing
at **60.0 % Ir / 63.0 % wall**.

### B2 spot-check (no regression)

B2 barely sorts (D94: "~0 there"). 2 600-case term differential corpus wall
time, 4-round interleaved: ON median 18 600 ms vs OFF median 18 796 ms
(**+1.0 %**, i.e. ON marginally faster, within noise). No regression.

### Default decision

**ON by default.** 60 % B3 Ir / 2.70× wall for zero correctness cost
(byte-identical, all gates green both lanes), and B2 unaffected. The OFF path is
retained behind `--no-default-features` for future A/Bs.

## Files changed

Owned scope only:

- `src/unifyweaver/targets/wam_rust_target.pl` — `term_compare_derefed`,
  `value_atom_name_ref`, `sort_cmp` (cfg-gated); route `sort/2`, `msort/2`,
  `sort/4`, `keysort/2`, `setof` through `sort_cmp`; append the `decorate_sort`
  Cargo feature to the generated `Cargo.toml`.
- `examples/pkg_resolver/rust/uw_resolve_wam/src/state.rs` — regenerated.
- `examples/pkg_resolver/rust/uw_resolve_wam/Cargo.toml` — regenerated (feature).
- `examples/pkg_resolver/rust/uw_resolve_wam/tests/comparator_equiv.rs` — new.
- `docs/reports/wam_rust_decoratesort_ab.md` — this report.

The store crate (`examples/pkg_resolver/rust_store/uw_resolve_wam_store`) is a
per-build gitignored artifact; it was regenerated + built (both configs) to
prove the store lane builds and passes, but is not committed.

`resolver.pl` and `resolver_store.pl` are **unmodified** (`git status` clean).
No functor interning was done (that is round #2).

## Bearing on round #2 (functor interning)

- Decorate-sort already removed the *comparator's* functor-`String` allocs via
  `functor_of` (borrowed). The remaining functor-`String` churn D94 flagged
  (76 % of B3 alloc blocks) lives in the **general** `deref_heap` /
  `deref_var` / `display_functor_name` paths that decorate-sort does not touch —
  interning still stands to remove those. But note the B3 denominator has
  shrunk: with decorate-sort ON, `deref_heap`/`deref_var`/malloc are now a much
  smaller absolute cost (deref_heap ~119 M vs ~599 M Ir), so interning's B3 win
  will be measured against the **post-decorate-sort** 1.61 B baseline, not the
  4.18 B one. Its B2 win (where sorting is negligible and the name-`String`
  churn dominates) is unaffected by this round and remains the stronger case.
- `functor_of` is now a load-bearing equality-normaliser in the comparator; if
  round #2 changes the `Value` functor representation to a `u32` id, the
  `functor_of`/`display_functor_name` equivalence used here must be preserved (or
  replaced by an integer compare on the interned ids), keeping sort output
  byte-identical.
