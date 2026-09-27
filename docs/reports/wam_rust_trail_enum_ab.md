<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) 2026 John William Creighton (s243a) -->

# Rust WAM enum-tagged trail — A/B (D104)

**Date:** 2026-09-17. **Ledger:** D104. **Author:** Opus (implementer).
**What:** the D94 #4 lever (B2 bind path) — replace the stringly-typed trail key
with an enum so `bind_var` stops concatenating a `"__binding__"` prefix via
`format!` on every variable bind and `unwind_trail_to` discriminates a binding
entry from a register entry by matching a variant instead of `strip_prefix`-
parsing on every unwind. Strictly more correct (a register name can no longer
collide with the `"__binding__"` prefix) and default-ON; ON is byte-identical to
the OFF pre-change baseline.

## The change

`TrailEntry.key` was a `String`. `bind_var` did
`key: format!("__binding__{}", var_name)` (a fmt allocation per bind) and
`unwind_trail_to` recovered the kind with `entry.key.strip_prefix("__binding__")`
(a parse per unwind); register-trail entries stored a raw register name as the
same `String`, and several sites discriminated the two kinds by that prefix.

Under the `trail_enum` feature (default ON) the key becomes:

```rust
pub enum TrailKey {
    Binding(String),   // an unbound variable was bound in self.bindings
    Register(String),  // a register / Yi slot was overwritten
}
pub struct TrailEntry { pub key: TrailKey, pub old_value: Option<Value> }
```

`bind_var` pushes `TrailKey::Binding(var_name.to_string())` (no prefix concat),
`trail_binding` pushes `TrailKey::Register(name.to_string())`, and every
discrimination site matches the variant. With the feature OFF the key is the
exact pre-change `String` (`format!("__binding__{}")` + `strip_prefix` /
`starts_with`), a perfect A/B baseline — the two builds are byte-identical.

### Keeping the call sites uniform

The ON/OFF split lives in one place: `TrailEntry`'s type + a small `impl` block,
both `#[cfg]`-gated once. Every construction and discrimination site calls a
config-independent helper, so no call site carries a `#[cfg]`:

| Helper | ON | OFF (byte-identical) |
|---|---|---|
| `TrailEntry::binding(name, old)` | `TrailKey::Binding(name.to_string())` | `format!("__binding__{}", name)` |
| `TrailEntry::register(name, old)` | `TrailKey::Register(name.to_string())` | `name.to_string()` |
| `entry.binding_name() -> Option<&str>` | variant match | `key.strip_prefix("__binding__")` |
| `entry.register_name() -> Option<&str>` | variant match | `!starts_with("__binding__")` → `Some(key)` |
| `entry.classify() -> (TrailUndo, Option<Value>)` | move name out of variant | `strip_prefix` → `Binding(name.to_string())` / move → `Register(key)` |

`classify()` is the one consuming path (used by `unwind_trail_to`, which moves
`old_value`); the OFF branch reproduces the pre-change work exactly — a
`to_string()` on the binding branch (where the old code did `var.to_string()`
for the `bindings.insert`) and a plain move on the register branch (where the old
code did `put_reg(&entry.key, …)` with no allocation). So the OFF **allocation
profile is unchanged**, not just the output.

### Sites touched

Two constructors and five discrimination/read sites (see the ledger row for the
exact list): `bind_var`, `trail_binding`, `unwind_trail_to` (all in
`state.rs.mustache`); `unwind_trail_bindings_only`, `unifiable/3`, and the genrec
Yi-register-drop (all generator-emitted from `wam_rust_target.pl`).

## Correctness

- **Byte-identical by construction.** OFF is the verbatim pre-change path; ON
  produces the same undo semantics — a binding entry only ever touches
  `self.bindings`, a register entry only ever touches `put_reg`, and the
  discrimination is total (every entry is exactly one variant / prefix state).
- **Strictly more correct.** Under the enum a register named, e.g.,
  `"__binding__X"` could never be misread as a binding; the OFF string scheme
  relied on real register names (`A1`, `X1`, `Y1`, …) never starting with the
  reserved prefix. No such name exists today, so the two are behaviourally
  identical here, but the enum removes the latent hazard.
- **`bindings` map unchanged.** It stays `HashMap<String, Value>`. The bonus
  `Binding(Sym)` (Copy u32, zero trail-key alloc) was **declined**: the map is
  String-keyed, so a `Sym` trail key would have to de-intern back to a `String`
  on every unwind `bindings.insert(var, old)` — moving the allocation rather than
  removing it, and risking byte-identity for a speculative win. Per scope
  ("prefer the clean enum with `String` if the Sym route adds risk"), kept simple.

## Real A/B (B2)

This is a B2 lever: the bind + backtrack path is exercised by the 2,600-case
differential, not B3. Profiled a representative **600-case B2 slice** (the first
600 of the differential's seeded `cases.jsonl`) through the term binary in a
single process under `valgrind --tool=callgrind --cache-sim=no`, OFF (pre-change
baseline) vs ON.

**Total Ir (600-case B2 slice):**

| | Ir | Δ |
|---|---:|---:|
| OFF (before) | 16,618,606,510 | — |
| ON (after) | 16,592,979,525 | **−25,626,985 (−0.154%)** |

**Key function deltas (OFF → ON, callgrind self-Ir):**

| function | OFF | ON | Δ |
|---|---:|---:|---:|
| `alloc::fmt::format::format_inner` | 44,332,549 | 35,144,461 | **−9,188,088** |
| `unwind_trail_bindings_only` | 13,316,164 | 10,838,573 | **−2,477,591** |
| `bind_var` (self) | 11,016,568 | 11,297,894 | +281,326 |
| `trail_binding` | 59,988,156 | 61,954,980 | +1,966,824 |
| `libc malloc` | 915,614,093 | 915,334,514 | −279,579 |
| `__rustc::__rust_alloc` | 21,019,870 | 21,019,870 | 0 |

**Honest reading.** The measured win is the removal of the `format!` fmt
machinery from the bind path (`format_inner` −9.19M) plus the cheaper unwind
discrimination (variant match vs `strip_prefix`, −2.48M), netting −25.6M Ir
(−0.154%). It is **modest and I am not dressing it up**: the per-bind String
allocation itself is *unchanged* — `__rust_alloc` and `malloc` are flat, because
ON still allocates one String for the trail key (`var_name.to_string()`) just as
OFF did (`format!`), only shorter and without the format layer. The allocation
count could only be cut by the declined `Sym` route (see Correctness), which the
scope rules out. B2 is dominated by term-construction malloc/free (`_int_malloc`
12.2% + `_int_free` 8.8% + `malloc` 5.5% + `free` 3.5% ≈ 30%), so the bind key,
at ~0.27% of B2, bounds the reachable win — the lever collects the fmt-overhead
slice of that and leaves the term-churn allocator untouched. The `trail_binding`
+1.97M is the enum making `TrailEntry` one word wider (a `String` + discriminant
vs a bare `String`), a slightly larger `push` memcpy; it is real and reported.

**B3 (expected flat).** B3 binds little (it resolves ground catalog rows). The
5000-package B3 selection is byte-identical ON vs OFF (`out_5000.json` `cmp`-
clean), and no bind-path Ir change is expected there; not separately profiled per
scope.

## Gate matrix (`LC_ALL=C.UTF-8`, ON = default, OFF = `--no-default-features --features "decorate_sort intern deref_memo"`)

| Gate | ON | OFF |
|---|---|---|
| Term corpus (`run_corpus_rust.sh`) | 51/51 | 51/51 |
| Term differential (`run_differential_rust.sh`) | 2600 / 0 div / 0 crash | 2600 / 0 div / 0 crash |
| Store corpus (`run_corpus_rust_store.sh`) | 51/51 | 51/51 |
| Store differential (`run_differential_rust_store.sh`) | 503 / 0 div | 503 / 0 div |
| `cargo test --lib` | 230 passed | 230 passed |

**Byte-identity ON ≡ OFF (`cmp`-clean):** term differential (2600), store
differential (503), 5000-package B3 selection (`out_5000.json`), and the 600-case
B2 slice output — all identical.

**Unit tests (new `trail_enum_tests`, pass ON and OFF):**
`bind_unwind_round_trip_with_register_entry` (bind a var, rebind it, bind a fresh
var, and mutate a register in the SAME trail, then unwind to a mark and assert
every binding and the register restored exactly — proving the two entry kinds are
never confused), `full_unwind_clears_all_query_bindings`, and
`binding_and_register_entries_discriminate` (variant/prefix discrimination +
`classify()` round-trip).

**Transactional alias test** (`test_wam_rust_foreign_tuple_aliases.pl`): pass,
exit 0, zero warnings.

## Feature gating

Cargo feature `trail_enum`, wired into the generated `[features]` table in
`wam_rust_target.pl` exactly like `intern` / `decorate_sort` / `deref_memo`:
`default = ["decorate_sort", "intern", "deref_memo", "trail_enum"]`. OFF via
`--no-default-features --features "decorate_sort intern deref_memo"` is the exact
pre-change `format!` / `strip_prefix` String path. All features work in
combination (gates run under the full default stack).

`resolver.pl` / `resolver_store.pl` UNMODIFIED.
