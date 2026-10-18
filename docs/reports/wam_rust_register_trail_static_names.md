<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) 2026 John William Creighton (s243a) -->

# Rust WAM: register trail entries without a `String` (D127)

**Date:** 2026-10-04. **Ledger:** D127. **Author:** Opus.
**What:** `trail_binding` records a register's old value before an
instruction overwrites it. Each entry used to store `reg_name.to_string()`,
one heap `String` per trailed register. A canonical register name (`A<n>`,
`X<n>`, `Y<n>`, n ≤ 99, which is every name the compiler emits) now borrows
the same text from a static table. Any other name still gets an owned copy.
Output is byte-identical.
**Result:** about **−0.95 M Ir per warm resolve at every catalog size**: −3.9%
at N=40 (24.24 M → 23.30 M Ir) and −0.2% at N=5000 (389.47 M → 388.52 M).
The warm in-process N=40 resolve drops from 4.52 ms to 4.32 ms (−4%, noisy
box).

## Profile first

The base is the D126 commit `902ec0d`, measured with the same probe and
method as D120–D126. `trail_binding` was **2.20 M Ir per warm resolve at
N=40 (9.1%)**, inclusive, over 10,790 calls (10,678 from `step`). Its callees,
per warm resolve:

| part of `trail_binding` | Ir / resolve | notes |
| --- | ---: | --- |
| `get_reg` (the old value, dereferenced) | 0.87 M | needed: the entry must hold exactly this value |
| `__rust_alloc` for the name `String` | 0.47 M | one per call (10,790) |
| `memcpy` of the name bytes | 0.19 M | one per call |
| self (entry construction, trail `push`) | ~0.67 M | |

Each `String` was freed again later, when `backtrack` truncated the trail or
`unwind_trail_to` consumed the entry. That cost shows up under `mi_free` and
`backtrack`, not under `trail_binding`.

### What D104 already did

`docs/reports/wam_rust_hotpath_deep_profile.md` (D94 #4, "enum-tag the
trail") is the origin of this lever. D104 (`docs/reports/wam_rust_trail_enum_ab.md`,
feature `trail_enum`, default ON) made the trail key an enum,
`TrailKey::Binding(String) | Register(String)`. That removed the
`"__binding__"` prefix `format!` from every bind and the `strip_prefix` parse
from every unwind. It kept a `String` per entry. D125 then made the binding
variant a `Sym`. D127 handles the register variant.

## Choosing the fix

Where is a register entry's name read?

- `unwind_trail_to` calls `put_reg(&name, old)`. `put_reg` routes by the name
  text: a Y name goes to the topmost frame's `YRegs::insert(name)`, and any
  other name goes to `set_reg_str(name)` → `reg_index(name)`.
- `register_name()` returns it as `&str`. This is used by the nested-run
  filter in `call_goal_once`, which drops entries whose name starts with `Y`,
  and by tests.
- `classify()` hands it out as `TrailUndo::Register`.
- `backtrack` only truncates the trail (registers are restored from the
  choice point), so there the name is just dropped.

Every reader looks at the text. So the exact fix is to keep the same text and
only avoid allocating it. Storing a decoded register index instead would have
changed what `register_name()` can return for non-canonical spellings such as
`A01`, which decodes to the same slot as `A1` but is different text.

The compiler only ever emits a register name as a prefix plus 1–2 decimal
digits. Those 300 names have one canonical spelling each, so a static
`[&str; 300]` table can supply `&'static str`s with the same bytes.

## The change

All in `templates/targets/rust_wam/state.rs.mustache`:

```rust
pub type RegName = std::borrow::Cow<'static, str>;
pub static REG_NAMES: [&str; 300] = ["A0", "A1", …, "A99", "X0", …, "X99", "Y0", …, "Y99"];
pub fn canonical_reg_slot(name: &str) -> Option<usize>;   // A/X/Y + "0".."9" or "1".."9" + digit
pub fn reg_name_of(name: &str) -> RegName {
    match canonical_reg_slot(name) {
        Some(i) => Cow::Borrowed(REG_NAMES[i]),
        None => Cow::Owned(name.to_string()),
    }
}
```

- `TrailKey::Register(String)` becomes `TrailKey::Register(RegName)`, and
  `TrailUndo::Register` likewise.
- `TrailEntry::register(name, old)` stores `reg_name_of(name)`.
  `trail_binding` is unchanged: it still calls `get_reg(key)` and pushes the
  entry, in the same order.
- `register_name()` returns `name.as_ref()`. `unwind_trail_to` still calls
  `put_reg(&reg, …)`, since `&Cow<str>` derefs to the same `&str`.
- With `trail_enum` OFF, the key stays the pre-D104 `String`, and `classify`
  wraps it as `Cow::Owned`.

**Why it is exact.** `canonical_reg_slot` accepts a name only if it is
`A`/`X`/`Y` followed by one digit, or by a nonzero digit and a digit. For
such a name, `REG_NAMES[slot]` has exactly the same bytes: the table is the
decimal spelling with no leading zero, and a test checks every entry. Every
other name, including `A01`, `X100`, `Y`, `+1`, non-ASCII and empty, keeps an
owned copy. So every reader of the entry sees the same `&str` as before.
Entry order, kind and old value are unchanged. `backtrack`, cut/barrier,
`lo_restore_clause` (via `unwind_trail_to`), the nested-run Y filter and
par_aggregate (which never touches trail names) all go through
`register_name()`, `classify()` or `put_reg(&str)`, so they see the same
text.

### Tests (`mod d127_register_trail_tests`, generated crate, `cargo test` only)

- **`reg_names_table_is_the_canonical_text`**: entry `i` is
  `format!("{}{}", ["A","X","Y"][i/100], i%100)`, and it maps back to slot
  `i`.
- **`reg_name_of_keeps_the_exact_text`**: about 9 K names: 9 prefixes
  (`A X Y B Z a y`, empty, `É`) × 0..999, the zero-padded forms, and odd
  suffixes (`+1`, `-1`, spaces, a letter, an Arabic-Indic digit, a 20-digit
  overflow). For each, `reg_name_of` keeps the text. The result borrows
  exactly when the name is canonical, and exactly 300 distinct names borrow.
  A register entry built from the name reports, classifies and filters on the
  same text.
- **`unwind_matches_the_string_keyed_trail`**: a reference machine keeps the
  pre-D127 register trail by hand, as `(name.to_string(), get_reg(name))`, and
  undoes it with the same `put_reg` calls. Each of 10 seeds runs 2,000 random
  steps over A/X/Y names, including non-canonical names that decode to live
  registers (`A01`, `X007`, `Y01`, `A100`), with a Y frame present, plus marks
  and `unwind_trail_to`. After every step, every register must be equal on
  both machines, and every trail entry must have the same name and old value.

The generated-crate lib suite passes: **260/260** (D126's 257 plus these 3).
It also passes with the feature sets `decorate_sort`,
`decorate_sort trail_enum` and `decorate_sort intern`.

## Gates

Run from the repo root with `LANG=C.utf8 LC_ALL=C.utf8`. The term binaries are
a fresh `build.sh` build of this tree. The store gates, plunit and conformance
ran in a clean export of this commit. The base outputs are the D126 build's.

| gate | result |
| --- | --- |
| `rust/run_differential_rust.sh` | `cases: 2600 / divergences: 0 / crashes: 0` |
| `rust/run_corpus_rust.sh` | `corpus-under-rust: 51/51 matched SWI` |
| `rust_store/run_differential_rust_store.sh` | `cases: 503 / divergences: 0 / crashes: 0` |
| `rust_store/run_corpus_rust_store.sh` | `corpus-under-rust-store: 51/51 matched SWI`; `rust_store corpus IDENTICAL to term corpus (51/51)` |
| byte identity vs base build | `cmp`-identical: term diff `rust.jsonl` (2600 lines), term corpus `rust.jsonl` (51), store diff `rust.jsonl` (503), store corpus `rust_store.jsonl` (51); scale `--bench` stdout identical at N=40/250/1000/5000 |
| generated crate `cargo test --release --lib` | `test result: ok. 260 passed; 0 failed` |
| Rust WAM plunit (56 files) | same per-file rc as base (28 rc=0, 28 rc≠0) and the same 42 failing test names; `test_wam_rust_target.pl` has the same failure lines |
| CI rust conformance (`CONFORMANCE_TARGETS=rust`, `CONFORMANCE_PROGRAMS=member,builtins`) | rc=0, both unsampled and with `CONFORMANCE_SAMPLE=2` |

## Measurements

The method is the same as D120–D126.

### Ir per warm resolve (deterministic)

| N | base (D126) | D127 | Δ | Δ absolute |
| ---: | ---: | ---: | ---: | ---: |
| 40 | 24.24 M | **23.30 M** | **−3.9%** | −0.94 M |
| 250 | 39.92 M | **38.99 M** | −2.3% | −0.94 M |
| 1000 | 94.90 M | **93.95 M** | −1.0% | −0.95 M |
| 5000 | 389.47 M | **388.52 M** | −0.2% | −0.95 M |

The saving is flat across N. At N=40, per warm resolve:

| site | base | D127 |
| --- | ---: | ---: |
| `trail_binding` (inclusive) | 2.20 M | 1.55 M (`get_reg` 0.87 M + self 0.68 M) |
| `__rust_alloc` from `trail_binding` | 0.47 M | 0 (no register name allocates on this workload) |
| `memcpy` from `trail_binding` | 0.19 M | 0 |
| `mi_free` (all) | 0.78 M | 0.54 M |
| `backtrack` (inclusive, includes dropping truncated entries) | 3.58 M | 3.38 M |

### Wall clock (shared 4-core container, noisy)

Base and new were interleaved, 21 rounds at N=40/250 and 9 at N=1000/5000.
"Cold" is the shim's own `resolve_ms` from a fresh process. "Warm" is the
median of 11 in-process repeats, then the median over the rounds.

| N | rounds | cold base | cold new | warm base | warm new | Δ warm |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 40 | 21 | 6.12 ms | 5.77 ms | 4.52 ms | **4.32 ms** | −4% |
| 250 | 21 | 7.66 ms | 7.51 ms | 6.63 ms | **6.34 ms** | −4% |
| 1000 | 9 | 16.80 ms | 16.17 ms | 14.34 ms | 14.08 ms | (noise) |
| 5000 | 9 | 72.96 ms | 67.00 ms | 67.44 ms | 66.31 ms | (noise) |

At N=40, the five fastest warm rounds were 4.18–4.26 ms for base and
4.08–4.15 ms for new. Ir is the reliable metric.

Cumulative wall, main `485e65b` vs D127, measured the same way: warm N=40
4.81 → **4.40 ms** (−8%), N=250 7.07 → 6.58 ms (−7%), N=1000
15.14 → 13.95 ms, N=5000 63.74 → 64.79 ms (noise).

## Cumulative (main `485e65b` → D127)

| N | main | D125 | D126 | D127 | Δ vs main |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 40 | 27.85 M | 25.87 M | 24.25 M | **23.30 M** | **−16.4%** (−4.56 M) |
| 250 | 43.86 M | 41.55 M | 39.92 M | 38.99 M | −11.1% |
| 1000 | 99.86 M | 96.53 M | 94.90 M | 93.95 M | −5.9% |
| 5000 | 399.94 M | 391.09 M | 389.48 M | **388.52 M** | −2.9% (−11.42 M) |

## What dominates next (post-D127 callgrind, N=40, 23.30 M Ir/resolve)

| cost | share | notes |
| --- | ---: | --- |
| `execute_builtin` (inclusive) | 16.6% | builtin work: `execute_ext_builtin` 11.8% (`sort_by` over `term_compare_derefed`, `deref_list_arg`, `dedup_by`) |
| `backtrack` (inclusive) | 14.5% | `restore_ax_regs` 7.3%: dropping and cloning register `Value`s |
| `trail_binding` (inclusive) | 6.7% | now `get_reg` (3.7%) plus the entry push. The old value has to stay the dereferenced value, so this is close to its floor without a different trail design |
| `set_heap_or_list` (inclusive) | 5.8% | structure/list construction |
| `save_regs` + `Vec::clone` | 4.2% + 4.0% | choice-point register snapshot |
| `intern` | 3.8% | constant placeholder atoms and fresh names re-interned per construction (see D126) |
| `labels` SipHash | 2.6% + probe | `labels: HashMap<String, usize>` (see D125) |

## Files

- `templates/targets/rust_wam/state.rs.mustache`: `RegName`, `REG_NAMES`,
  `canonical_reg_slot`, `reg_name_of`, `TrailKey::Register(RegName)`,
  `TrailUndo::Register(RegName)`, `TrailEntry::register`, `register_name`,
  the `trail_enum`-OFF `classify`, and `mod d127_register_trail_tests`.

`resolver.pl` and `resolver_store.pl` are unmodified. The regenerated
checked-in crate is not committed.
