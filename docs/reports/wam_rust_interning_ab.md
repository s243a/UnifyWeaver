<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) 2026 John William Creighton (s243a) -->

# Rust WAM (uw-resolve) round #2 — intern functor/atom/var names to u32: A/B

Implementation + A/B report for **worklist item #2** of the deep hot-path
profile (`wam_rust_hotpath_deep_profile.md`, D94): intern functor / atom /
variable-name Strings to integer ids so term construction, `deref_var` and the
`"f/N"` functor parse stop allocating tiny name Strings on the hot paths. D94
found these are **63% of B2 and 76% of B3 allocation blocks**.

**Result up front:** interning cuts **B3 by 29% of instructions (callgrind Ir)
and 29% of wall time**, and **B2 by 41% of Ir / 32% of wall time**, byte-identical
output across every gate. Both **confirm and exceed** D94's projection
(~15–25% B3, ~11% B2). **Default: ON** (alongside decorate-sort).

Base content SHA: `origin/claude/peerhailer-exploratory-docs-aodas5` @
`bf69aa776` (D95, full lowered tier + regions 1–5 + general recognizer +
decorate-sort default-ON). Branch `claude/wam-rust-interning-aodas5`.

## The interning design

### `Sym`: the name representation, chosen by a Cargo feature

Names carried by `Value::Atom(Sym)`, `Value::Str(Sym, Args)` and
`Value::Unbound(Sym)` are represented by a `Sym` type selected at compile time by
a new **`intern` Cargo feature** (default ON), so the ON build and the OFF build
are sha-distinct binaries producing byte-identical output — a clean A/B and a
retained pre-intern path.

- **feature OFF** → `pub type Sym = String`. This is the pre-intern path,
  verbatim: every name is an owned `String` exactly as before. All of `String`'s
  inherent methods and trait impls (`Deref<str>`, `Display`, `PartialEq<&str>`,
  `Ord`, …) serve directly, so the rest of the runtime compiles against `Sym`
  with **no cfg noise** and **zero behaviour change**. This makes OFF a perfect
  A/B baseline.

- **feature ON** → `Sym(u32)`, a `Copy` index into a global, append-only,
  **canonical** interner:
  - `resolve(id) -> &'static str` is **lock-free / wait-free**: a chunked array
    of `AtomicPtr` chunks that, once published, never move or free; the names are
    `Box::leak`'d `&'static str`, so de-intern hands back a `'static` slice with
    zero lifetime friction (functor slices, Display, comparator names all borrow
    it freely).
  - `intern(&str) -> u32` takes a read lock on a `name -> id` map on the common
    "already interned" path (the construction fast path), and a write lock only
    when a brand-new name is first seen.
  - Cloning a name becomes a `u32` copy (no malloc); `deref_var` is
    allocation-free; `deref_heap`'s functor normalisation interns the bare name
    instead of `to_string()`-cloning it.

De-interning happens only at the **output / comparison boundaries**: `Display`,
`writeq`/format helpers, the JSON shims (`atom()`/`atom_json()`), and the sort
comparators (for the name-order decision only).

### How the generator emits ids

This is a **code-generator change**, not a hand-edit of `lib.rs`. The runtime is
emitted from `templates/targets/rust_wam/*.mustache` plus inline Rust in the
generator `src/unifyweaver/targets/wam_rust_target.pl`; the per-predicate
`lib.rs` is emitted by the generator. Construction sites (~650: ~425 `Atom` +
~230 `Str`/`Unbound`, the majority in generated code) route the name through
`.into()` or the `Value::atom`/`unbound`/`str_sym`/`make_str`/`strv`/`new_unbound`
helpers, which take `impl Into<Sym>`:

- Under OFF, `EXPR.into()` is `String`→`String` (identity) and the helpers are
  `impl Into<String>` — a no-op, so OFF output is unchanged.
- Under ON, `EXPR.into()` interns via `From<&str>/<String>/<&String> for Sym`.

The lib.rs atom emitter (`Value::Atom("~w".to_string())` in the generator) gained
`.into()`, fixing all generated-predicate atoms at one site. Consumption sites
mostly compiled unchanged via `Sym: Deref<str>` + `PartialEq<&str>`; the sites
that genuinely need a `String`/`&str` (HashMap keys, env calls, comparator names,
`copy_term`/variant var-maps, `Vec<String>` collectors) de-intern with
`.as_str()` / `.as_str().to_string()`. The whole change compiles under **both**
feature settings; the Rust compiler enumerated every site to fix.

## The sort-order trap, and how it is kept name-correct

SWI standard order compares atoms/functors **by name**. Interned ids are assigned
in **first-seen order**, not name order, so a naive integer-id compare would order
by id and silently corrupt every sort (and every order-dependent gate). This is
handled by **approach (a): the comparators de-intern to `&str` for the ordering
decision.**

- `term_compare_derefed` (the decorate-sort comparator) already compared atom
  names via the borrowing `value_atom_name_ref` (`&str`) and functor names via
  `functor_of` (`&str`); under ON both return the de-interned `&'static str`, so
  the LESS/GREATER decision is a **name** compare — byte-identical ordering, no
  allocation.
- `term_compare` (used by `@<`/`compare/3` and the OFF sort path) had its
  atom-class and var-class arms changed from `s.clone()` / `n.clone()` (which
  under ON would compare `Sym`s) to `s.as_str().to_string()` / `n.as_str()...`,
  so it too orders by name.

To make the guarantee **load-bearing rather than a convention**, the ON `Sym`
**deliberately implements no `Ord`/`PartialOrd`**: any code that tries to order
symbols by id is a **compile error**, not a wrong answer. (`Sym`'s `PartialEq` —
an id compare — stays sound because interning is canonical: the same name always
maps to the same id, so id-equality iff name-equality; `Value: PartialEq` and the
`==`/`\==` path keep their exact semantics.)

`functor_of` / `display_functor_name` equivalence D95 depends on is preserved:
both take a de-interned `&str` (via `Sym: Deref<str>`) and apply the identical
`str(...)`/`/N` normalisation, so the comparator's functor compares are
byte-identical to D95.

## Gates — both configs, both lanes

Run with `LC_ALL=C.UTF-8`. ON = `--features intern` (default); OFF =
`--no-default-features --features decorate_sort`. Decorate-sort + regions 1–5 +
genrec all ON.

| gate | ON | OFF | ON≡OFF byte-identical |
|---|---|---|---|
| term corpus | **51/51** | **51/51** | ✅ (`cmp` clean) |
| term differential | **2600 / 0 / 0** | **2600 / 0 / 0** | ✅ (`cmp` clean) |
| store corpus | **51/51** | **51/51** | ✅ (`cmp` clean) |
| store differential | **503 / 0** | **503 / 0** | ✅ (`cmp` clean) |
| 5 k `resolve_layered` selection | selection_size=10 | selection_size=10 | ✅ (`cmp` clean) |

Every gate passes both configs, and the ON output is byte-identical to the OFF
output across the 2600 term differential cases, the 51 term + 51 store corpus
scenarios, the 503 store differential cases, and the 5000-package
`resolve_layered` selection. The committed default-ON binary was re-verified via
the standard `run_corpus_rust.sh` / `run_differential_rust.sh` scripts (corpus
51/51, differential 2600/0/0).

## Stress tests

`examples/pkg_resolver/rust/uw_resolve_wam/tests/intern_stress.rs` (5 tests, pass
under both `intern` and default):

- **`sort_orders_by_name_not_intern_id`** / **`compound_functor_orders_by_name_not_id`**
  — the sort-order-trap proof: the interner is forced to assign ids in the
  OPPOSITE order to the names, then the decorate-sort comparator (and the raw
  comparator) must still sort by NAME. If ordering used ids the result would be
  reversed; it is not.
- **`display_round_trips_names`** — de-intern round-trip for names first seen at
  runtime (emoji, spaces, empty string, non-ASCII).
- **`atom_equality_is_by_name`** — canonical interning: id-equality iff
  name-equality, for atoms and compounds.
- **`comparator_is_a_total_order_over_interned_names`** — reflexivity +
  antisymmetry + `ordering == name ordering` over an intern-scrambled corpus.

The D95 comparator-equivalence suite (`comparator_equiv.rs`, 5 tests) and the
generated lib unit tests (220, region/genrec) also pass under `--features intern`.

## Real A/B — interning ON vs OFF

Sha-distinct, verified-relink binaries (ON `7e31856e…`, OFF `f3ebede5…`).

### B3 — 5000-package `resolve_layered --bench`

Instruction count (callgrind Ir — deterministic, drift-immune):

| build | total Ir | vs OFF |
|---|---:|---:|
| **OFF** (pre-intern) | 1,617,957,821 | — (≈ the ~1.61B post-decorate-sort baseline) |
| **ON** (intern) | 1,149,491,479 | **−29.0%** |

Wall time (`resolve_ms`, 6-round interleaved, drift-cancelling):

| round | ON | OFF |
|---|---:|---:|
| 1 | 97.5 | 129.0 |
| 2 | 95.8 | 132.6 |
| 3 | 96.1 | 134.7 |
| 4 | 91.6 | 144.5 |
| 5 | 93.5 | 137.6 |
| 6 | 92.9 | 120.0 |
| **median** | **94.6** | **133.7** |

**B3 wall reduction (median): −29.2% → 1.41× faster.** Ir and wall agree (D94's
cache-sim showed the working set is L2/L3-resident, so Ir is a faithful proxy).

### B2 — 2600-case term differential corpus

Wall time (whole 2600-case corpus in one process, 6-round interleaved):

| round | ON (ms) | OFF (ms) |
|---|---:|---:|
| 1 | 13002 | 19093 |
| 2 | 12875 | 18510 |
| 3 | 13083 | 18813 |
| 4 | 12445 | 18192 |
| 5 | 12674 | 18667 |
| 6 | 12607 | 19079 |
| **median** | **12775** | **18740** |

**B2 wall reduction (median): −31.8%.** Deterministic cross-check (callgrind Ir,
600-case subset): OFF 33.13B → ON 19.62B = **−40.8%** — larger than the wall
because Ir counts the removed `malloc`/`free` instructions directly, while wall
carries fixed JSON-parse/IO overhead.

### Confirm / refute the D94 projection

**Confirmed and exceeded on both workloads.** D94 projected ~15–25% B3 (against
the post-decorate-sort 1.61B baseline) and ~11% B2. B3 landed at −29% (Ir and
wall). B2 landed at −32% wall / −41% Ir — far above the ~11% D94 counted, because
D94's B2 estimate was only the functor-parse self-Ir plus the name-String clones,
whereas interning makes **every** Atom/Unbound/Str name clone a `u32` Copy
(`deref_var` 24% of B2 alloc blocks, `deref_heap` functor 28%, and the
`put_reg`/`backtrack` String clones), collapsing the whole
malloc/free/String::clone/drop family that D94 measured at ~62% of B2 self-Ir /
~50% of B3 self-Ir. `load_ms` rises slightly under ON (~25→~32 ms on B3) because
the shim interns names at JSON parse; resolve — the hot metric — falls.

### Default decision

**ON by default** (`default = ["decorate_sort", "intern"]`). Confirmed win on both
workloads, all gates green both lanes, byte-identical. The pre-intern path is
retained via `--no-default-features --features decorate_sort` for future A/Bs.

## Files changed

Owned scope only:

- `src/unifyweaver/targets/wam_rust_target.pl` — comparators (`value_atom_name`,
  `term_compare` atom/var arms) de-intern to names; `deref_heap` interns the
  normalised functor from `&str`; copy_term / `variant_terms` var-maps, env
  builtins, region native kernels, read/format builtins de-intern at String
  boundaries; the lib.rs atom emitter routes through `.into()`; append the
  `intern` Cargo feature and flip default to `["decorate_sort", "intern"]`.
- `templates/targets/rust_wam/value.rs.mustache` — the `Sym` type + interner +
  helper constructors + enum + trait impls.
- `templates/targets/rust_wam/{state,dynamic_db_methods,par_aggregate,
  seek_fact_source,os_error_builtin,os_utility_builtin,process_builtin,
  process_context_builtin,process_resource_builtin,stream_builtin,time_builtin,
  main}.rs.mustache` — construction sites routed through `.into()`/helpers;
  name-consuming sites de-interned where a `String`/`&str` is required.
- `examples/pkg_resolver/rust/shim/main.rs`,
  `examples/pkg_resolver/rust_store/shim/main.rs` — intern at the JSON boundary
  (`atom()`, `OUT_VAR`), de-intern at output (`atom_json`).
- `examples/pkg_resolver/rust/uw_resolve_wam/` — regenerated crate (committed).
- `examples/pkg_resolver/rust/uw_resolve_wam/tests/intern_stress.rs` — new.
- `examples/pkg_resolver/rust/uw_resolve_wam/tests/comparator_equiv.rs` — the two
  helper constructors updated for `Sym`.
- `docs/reports/wam_rust_interning_ab.md` — this report;
  `docs/proposals/JS_TARGETS_PARITY_PUNCHLIST.md` — ledger D96.

The store crate (`examples/pkg_resolver/rust_store/uw_resolve_wam_store`) is a
per-build gitignored artifact; it was regenerated + built (both configs) to prove
the store lane builds and passes, but is not committed.

`resolver.pl` and `resolver_store.pl` are **unmodified** (`git status` clean). The
frozen spec was not modified.

## What remains (D94 worklist)

- **#3 `deref_heap` ground-bit / memo** (`state.rs` deref_heap): a
  ground/resolved bit so an already-ground term's `deref_heap` is an O(1) identity
  (no spine rebuild, no re-walk across backtracks). Attacks the top non-allocator
  self-Ir (9.5% B2 / 13.8% B3) in both workloads.
- **#4 enum-tag the trail** (`bind_var`/`trail_binding`): replace the
  `format!("__binding__{}")` + `key.to_string()` String-tagged trail with an
  enum (`Binding(u32) | Reg(u16)`) — with interning the var id is already a
  `u32`. A focused B2 bind-path win (4.6M binds / 479K backtracks).
