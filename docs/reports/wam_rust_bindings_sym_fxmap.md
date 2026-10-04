<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) 2026 John William Creighton (s243a) -->

# Rust WAM: the binding table keyed by `Sym` on FxHash (D125)

**Date:** 2026-10-04. **Ledger:** D125. **Author:** Opus.
**What:** `WamState::bindings`, the variable-binding table, was a
`HashMap<String, Value>` with std's default SipHash. It is now a small
`Bindings` wrapper around `HashMap<Sym, Value, FxBuildHasher>`. The key is the
variable's interned name, which every caller already holds from
`Value::Unbound`. Output is byte-identical.
**Result:** about **−2.0 M Ir per warm resolve at N=40, and more at larger N**:
−7.1% at N=40 (27.85 M → 25.87 M Ir) and −2.2% at N=5000 (399.94 M → 391.09 M).
The warm in-process N=40 resolve drops from 4.80 ms to 4.57 ms (−5%, noisy box).

## Profile first: what the map cost

The probe, catalogs and method are the same as in D120–D124
(`docs/reports/wam_rust_register_decode.md`). The base is origin/main
`485e65b` (D124 merged). It reproduces D124: 27.85 M Ir per warm resolve at
N=40 (D124 reported 27.86 M).

std's SipHash showed up as `core::hash::BuildHasher::hash_one` (1.86 M Ir per
warm resolve at N=40, inclusive) plus `sip::Hasher::write` inside it. Not all
of it was `bindings`. The callers, per warm resolve:

| caller of `hash_one` | hashes / resolve | Ir / resolve | map |
| --- | ---: | ---: | --- |
| `deref_var` | 4,458 | 0.71 M | `bindings` |
| `deref_chain` | 1,214 | 0.19 M | `bindings` |
| `bind_var` (old-value `get`) | 894 | 0.14 M | `bindings` |
| `bind_var` (`insert`) | 895 | 0.14 M | `bindings` |
| `unwind_trail_bindings_only` | 477 | 0.08 M | `bindings` |
| `step` (`labels.get`, 6 sites) | 2,922 | 0.60 M | `labels: HashMap<String, usize>` |
| **`bindings` total** | **7,938** | **1.26 M (4.5%)** | |

That is about 160 Ir per SipHash of a 3–6 byte name like `_V123`. On top of
the hashing:

- each probe compared the key bytes (`memcmp`) after the hash matched;
- `bind_var` allocated two `String`s per bind: `var_name.to_string()` for the
  map key and another for the trail entry (`TrailKey::Binding(String)`). Both
  were freed again on unwind or truncate. That is ~1.8 K allocations per
  resolve;
- every caller held the name as an interned `Sym` (a `u32` under the default
  `intern` feature) and turned it back into `&str` (`name.as_str()`) only to
  hash the text.

Inclusive costs at the base: `deref_var` 1.81 M, `bind_var` 0.59 M,
`deref_chain` 0.48 M, `unwind_trail_bindings_only` 0.25 M.

The other SipHash user, `labels`, is a separate map with its own public type
and constructor, used by generated `lib.rs` code and test harnesses. It is
left for a later item (see "What dominates next").

## Choosing the fix

1. **FxHash on the `String` key.** This is the D112 pattern, and the smallest
   diff. But FxHash on bytes still walks the name, every bind still allocates
   two Strings, and every probe still compares bytes.
2. **Key by the interned `Sym`.** The caller already has the `u32`. Hashing it
   with Fx is one multiply, a bind allocates nothing, and the probe compares
   two `u32`s. Interning is canonical (same name ⟺ same `Sym`, which D96/D112
   already rely on), so a `Sym` key identifies exactly the variable the name
   string did.

Option 2 was taken. Its cost is API compatibility. About 30 test harnesses,
`par_aggregate.rs` and `rust_target.pl`'s generated wrappers read the table
by string: `vm.bindings.get("N")`, `.get(name)` with `name: &str`,
`.get(&temp: &String)`, `.remove(&temp)`, `.is_empty()`. A
`HashMap<Sym, _>` cannot serve `get(&str)`: `Sym` cannot implement
`Borrow<str>`, because a `u32` hash differs from the text's hash. So the map
sits behind a small wrapper that keeps those calls compiling unchanged.

### Byte-identity hazard: iteration

SipHash's seed is random per process, so if any output had depended on the
map's iteration order, it would already have been nondeterministic. To make
sure nothing does, every use of `bindings` in the generator, the templates,
`rust_target.pl`, the tests and the fixtures was listed (`.iter`, `.keys`,
`.values`, `.drain`, `.retain`, `.extend`, `.entry`, `.clone`, `.len`, plus
any whole-map assignment). The only iteration is the D124 test-only
`fingerprint`, which sorts first. Everything else probes: `get`, `insert`,
`remove`, `clear`. The wrapper's `iter()` documents the order as unspecified.

## The change

### `Bindings` and `BindKey` (`templates/targets/rust_wam/state.rs.mustache`)

```rust
#[derive(Clone, Default)]
pub struct Bindings { map: HashMap<crate::value::Sym, Value, FxBuildHasher> }

pub trait BindKey {
    fn bind_sym(&self) -> Sym;                                       // for insert
    fn with_sym<R>(&self, f: impl FnOnce(&Sym) -> R) -> Option<R>;  // for lookup
}
```

`BindKey` is implemented for:

- `Sym`: the hot path. It copies the id and does no hashing of text.
- `str` and `String`: lookups go through the new `value::lookup_sym`, which
  finds an already-interned name without interning it. A name that was never
  interned cannot be bound, so the result is `None`, and a lookup does not grow
  the interner. Inserts intern the name, as building `Value::Unbound` already
  does.
- `&T`, through a blanket impl.

With `intern` OFF, `Sym` is `String`, so the `String` impl is the `Sym` impl
and the map is a `String`-keyed FxHash map.

`Bindings` exposes `get`, `contains_key`, `insert(Sym, Value)`, `remove`,
`clear`, `len`, `is_empty` and `iter`. All of them have the same meaning as
the `HashMap` methods they replace.

### Callers

- `deref_var`, `deref_chain`, `deref_heap`, `deref_match_atom` and
  `deref_atom_str` call `self.bindings.get(name)` on the `&Sym` they already
  matched, instead of `get(name.as_str())`.
- `bind_var` is now `bind_var<K: BindKey + ?Sized>(&mut self, var_name: &K, val)`.
  Every runtime caller passes the `Sym` from `Value::Unbound`. Tests may still
  pass a `&str`. The body does the same three steps in the same order: read the
  old value, push the trail entry, insert.
- The trail's binding entry carries the `Sym`: `TrailKey::Binding(Sym)` and
  `TrailUndo::Binding(Sym)`. The new `TrailEntry::binding_from_sym` builds one,
  and the new `entry.binding_sym()` reads it back. `binding_name()` still
  returns `&str`, and the `trail_enum`-OFF path still uses the
  `"__binding__"` string key, unchanged.
- `unwind_trail_bindings_only` (generated by `compile_unwind_trail_to_rust`
  in `wam_rust_target.pl`) and `unwind_trail_to` restore or remove by the
  entry's `Sym`. They visit the same entries in the same reverse order and do
  the same insert or remove for each.
- `src/unifyweaver/targets/wam_rust_target.pl`'s other `bindings.get(name)`
  site (the `unifiable/3`-style alias collector) passes a `&str` from
  `binding_name()` and works unchanged through `BindKey for str`.

### Why it is exact

The map's observable behaviour is the set of (variable, value) pairs and the
result of each probe. For each operation, the new key identifies the same
variable as the old one, because interning is canonical. The operations
happen in the same order with the same values. Nothing observes iteration
order, hash values or capacity. The trail entries keep their position, kind
and old value, and only their name representation changes.

### Tests (`mod d125_bindings_tests`, generated crate, `cargo test` only)

- **`bindings_match_the_string_keyed_model`**: the pre-D125 table, rebuilt in
  the test as a `HashMap<String, Value>` plus a `(String, Option<Value>)`
  binding trail, runs side by side with the real machine. Each of 12 seeds
  runs 1,500 pseudo-random steps: binds through all three key forms, register
  trail entries, marks, `unwind_trail_to`, backtrack-style
  `unwind_trail_bindings_only` + truncate, and `reset_query`. The names
  include `_V*`, `_H*`, `_`, a non-ASCII name, a name with a space,
  `__binding__x` and `A1`. Bound values include atoms, integers, lists and
  acyclic chains to other variables. After every step, every name is looked up
  by `&str`, `&String` and `&Sym`, and `deref_var`, `len`, the trail length
  and the sorted `iter()` must all match the model.
- **`binding_trail_entries_from_str_and_sym_agree`**: `TrailEntry::binding`
  and `binding_from_sym` give the same `binding_name`, `binding_sym` and
  `classify` result, and a register entry has no `binding_sym`.
- **`string_lookup_does_not_intern`** (`intern` only): `get`, `remove` and
  `contains_key` on a never-seen name return nothing and leave it un-interned.
  A string bind interns it, and the binding is then visible under every key
  form.

The generated-crate lib suite passes: **253/253** (D124's 250 plus these 3).
`cargo test --release --lib` also passes with the feature sets
`decorate_sort`, `decorate_sort trail_enum` and `decorate_sort intern`
(`intern` and/or `trail_enum` OFF).

## Gates

Run from the repo root with `LANG=C.utf8 LC_ALL=C.utf8`. The term binaries are
a fresh `build.sh` build of this tree. The store gates rebuild the store crate
themselves, so they ran in a clean export of this commit. The base outputs come
from an origin/main `485e65b` build, which is `cmp`-identical to D124's
recorded outputs.

| gate | result |
| --- | --- |
| `rust/run_differential_rust.sh` | `cases: 2600 / divergences: 0 / crashes: 0` |
| `rust/run_corpus_rust.sh` | `corpus-under-rust: 51/51 matched SWI` |
| `rust_store/run_differential_rust_store.sh` | `cases: 503 / divergences: 0 / crashes: 0` |
| `rust_store/run_corpus_rust_store.sh` | `corpus-under-rust-store: 51/51 matched SWI`; `rust_store corpus IDENTICAL to term corpus (51/51)` |
| byte identity vs base build | `cmp`-identical: term diff `rust.jsonl` (2600 lines), term corpus `rust.jsonl` (51), store diff `rust.jsonl` (503), store corpus `rust_store.jsonl` (51); scale `--bench` stdout identical at N=40/250/1000/5000 |
| generated crate `cargo test --release --lib` | `test result: ok. 253 passed; 0 failed` |
| Rust WAM plunit (`tests/test_wam_rust_*.pl`, `tests/core/*rust*.pl`, 56 files) | same per-file rc as base (28 rc=0, 28 rc≠0) and the same 42 failing test names; `test_wam_rust_target.pl` has the same failure lines |
| CI rust conformance (`CONFORMANCE_TARGETS=rust`, `CONFORMANCE_PROGRAMS=member,builtins`) | rc=0, both unsampled and with `CONFORMANCE_SAMPLE=2` |

One note on plunit. In the first full run, `test_wam_rust_par_aggregate.pl`
failed two timing assertions ("cheap workload should not fan out", "parallel
(0.1256s) not faster than sequential (0.1230s)"). That run overlapped two
cargo builds on the 4-core box. The file tests the standalone
`src/unifyweaver/targets/rust_runtime/par_aggregate.rs`, which this change
does not touch. Rerun alone, it passes (rc=0). The table above uses the rerun.

## Measurements

The method is the same as D120–D124. Ir comes from callgrind on an uncommitted
scratch shim that repeats `call_pred` in-process and checks every answer.
Ir per warm resolve = (Ir(1+k) − Ir(1)) / k, with k=10 at N=40 and k=4
elsewhere.

### Ir per warm resolve (deterministic)

| N | base (main `485e65b`) | D125 | Δ | Δ absolute |
| ---: | ---: | ---: | ---: | ---: |
| 40 | 27.85 M | **25.87 M** | **−7.1%** | −1.98 M |
| 250 | 43.86 M | **41.55 M** | −5.3% | −2.31 M |
| 1000 | 99.86 M | **96.53 M** | −3.3% | −3.32 M |
| 5000 | 399.94 M | **391.09 M** | −2.2% | −8.85 M |

The saving grows with N, unlike D120–D124. Larger catalogs make more binds
and longer fresh-variable names (`_V12345`), and SipHash cost grows with the
name length while a `Sym` hash does not.

Where the N=40 saving comes from (inclusive, per warm resolve):

| site | base | D125 |
| --- | ---: | ---: |
| `hash_one` (all SipHash) | 1.86 M | 0.60 M (only `labels` is left) |
| `deref_var` | 1.81 M | 0.83 M |
| `get_reg` (mostly `deref_var`) | 2.11 M | 1.44 M |
| `unify` | 1.20 M | 0.55 M |
| `bind_var` | 0.59 M | 0.14 M |
| `deref_chain` | 0.48 M | inlined |
| `unwind_trail_bindings_only` | 0.25 M | 0.13 M |

### Wall clock (shared 4-core container, noisy)

Base and new were interleaved, 9 rounds per size. "Cold" is the shim's own
`resolve_ms` from a fresh process. "Warm" is the median of 11 in-process
repeats, then the median over the rounds.

| N | cold base | cold new | warm base | warm new | Δ warm |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 40 | 6.13 ms | 6.22 ms | 4.80 ms | **4.57 ms** | −5% |
| 250 | 8.76 ms | 8.35 ms | 7.14 ms | **6.86 ms** | −4% |
| 1000 | 16.96 ms | 15.91 ms | 15.35 ms | 15.20 ms | (noise) |
| 5000 | 60.97 ms | 61.00 ms | 66.28 ms | 67.28 ms | (noise) |

The warm medians at N=40/250 track the Ir change. At N=1000/5000 the expected
−3%/−2% is below this box's noise. Ir is the reliable metric.

## What dominates next (post-D125 callgrind, N=40, 25.87 M Ir/resolve)

| cost | share | notes |
| --- | ---: | --- |
| `execute_builtin` (inclusive) | 15.0% | builtin work (sort, list deref, unify) |
| `backtrack` (inclusive) | 13.8% | `restore_ax_regs` 6.4% self |
| `trail_binding` (inclusive) | 8.5% | `get_reg` + a `String` key per register entry (D127) |
| `format!` in `step` | 6.8% | functor keys built only to compare, fresh-variable names (D126) |
| `labels` SipHash | 2.3% + probe | `labels: HashMap<String, usize>`, ~2.9 K lookups per resolve in `Call`/`Execute`/`TryMeElse`/`SwitchOnStructure`; same fix as here, but `labels` is part of `WamState::new`'s public signature |
| `intern` | 3.1% | fresh-variable names and the `"__struct_arg__"` heap placeholders re-intern their text on every construction |

## Files

- `templates/targets/rust_wam/state.rs.mustache`: `Bindings`, `BindKey`, the
  `bindings` field, `bind_var`, `TrailKey::Binding(Sym)` /
  `TrailUndo::Binding(Sym)` / `binding_from_sym` / `binding_sym`,
  `unwind_trail_to`, the five deref sites, and `mod d125_bindings_tests`.
- `templates/targets/rust_wam/value.rs.mustache`: `interner::lookup_sym`
  (re-exported as `value::lookup_sym`).
- `src/unifyweaver/targets/wam_rust_target.pl`: `compile_unwind_trail_to_rust`
  (unwind by `binding_sym()`).

`resolver.pl` and `resolver_store.pl` are unmodified. The regenerated
checked-in crate is not committed.
