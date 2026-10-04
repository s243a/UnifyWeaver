<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) 2026 John William Creighton (s243a) -->

# Rust WAM: no `format!` on the `step` hot path (D126)

**Date:** 2026-10-04. **Ledger:** D126. **Author:** Opus.
**What:** `step` called `format!` about 2.5 K times per warm resolve. Most of
these calls built a `"f/N"` functor key only to compare it, and the rest built
the fresh-variable names `_V<n>`/`_H<n>`. The key comparisons now check the
bytes in place, and the names are built in a stack buffer. Output is
byte-identical.
**Result:** about **−1.6 M Ir per warm resolve at every catalog size**: −6.3%
at N=40 (25.87 M → 24.25 M Ir) and −0.4% at N=5000 (391.10 M → 389.48 M).
The warm in-process N=40 resolve drops from 4.55 ms to 4.42 ms (−3%, noisy
box).

## Profile first: which `format!` calls

The base is the D125 commit `c3c38cf`, measured with the same probe and
method as D120–D125. `alloc::fmt::format::format_inner` was **1.77 M Ir per
warm resolve at N=40 (6.8%)**, inclusive. Of that, 1.75 M came from `step`,
over 2,493 calls, which is about 700 Ir per call: the `fmt` machinery, a heap
allocation and a free.

To tie each call back to its source line, the base crate was built with
line tables (`CARGO_PROFILE_RELEASE_DEBUG=line-tables-only`). The
`format_inner` call sites in `step` were then taken from
`callgrind --dump-instr=yes` and run through `addr2line -i`. Per warm resolve:

| call site | calls / resolve | what it builds |
| --- | ---: | --- |
| `GetStructure`, read mode on a compound held in a register | 958 | `format!("{}/{}", f, args.len()) == *fn_str`: a key built only to compare |
| `PutVariable` | 642 | `format!("_V{}", var_counter)`, the fresh variable's name |
| `SwitchOnStructure` / `SwitchOnStructurePc` | 476 | `let key = format!("{}/{}", f, args.len())`, compared against each table entry |
| `SetVariable` | 413 | `format!("_H{}", var_counter)` |
| `UnifyVariable` (write mode) | 4 | `format!("_H{}", var_counter)` |

These five sites account for all of `step`'s `format_inner` calls. The
fresh-variable names then went through `Sym: From<String>`, which interns
the text and drops the `String`.

Other `format!` calls in `step`'s arms were left alone: either they were not
reached on this workload, or they do more than build a comparison key.
`GetStructure`'s heap-`Ref` arm, for example, compares against
`format!("str({})", fn_str)` only after `s == fn_str` has failed, and that
path had 0 calls.

## The change

All five sites live in step-arm bodies in
`src/unifyweaver/targets/wam_rust_target.pl`. Three small helpers were added
to `impl WamState` in `templates/targets/rust_wam/state.rs.mustache`:

```rust
pub fn usize_decimal(n: usize, buf: &mut [u8; 20]) -> &[u8];
pub fn functor_key_eq(f: &str, arity: usize, key: &str) -> bool;
pub fn fresh_var_sym(tag: u8, n: usize) -> crate::value::Sym;
```

- **`usize_decimal`** writes `n`'s decimal digits right-aligned into a stack
  buffer and returns them. That is the same text as `n.to_string()`: no sign,
  no leading zeros, `"0"` for zero. Twenty bytes hold `u64::MAX`.
- **`functor_key_eq(f, n, key)`** stands in for
  `format!("{}/{}", f, n) == key`. The formatted string is `f`, then `/`,
  then `usize_decimal(n)`. So the two are equal exactly when `key` is at least
  `f.len() + 2` bytes long, starts with `f`, has `/` at byte `f.len()`, and the
  rest equals `usize_decimal(n)`. That is what the helper checks. The early
  length test only rejects keys too short to hold even one digit, and
  `format!` can never produce such a key.
- **`fresh_var_sym(tag, n)`** builds `_`, `tag`, `usize_decimal(n)` in a
  22-byte stack buffer and turns it into a `Sym` with `From<&str>`. Under
  `intern` this calls `intern` on the same text that `From<String>` passed to
  it before. Interning is canonical, so the result is the same `Sym`. With
  `intern` OFF, `Sym` is `String`, and the result is the same `String`.

The call sites:

- `GetStructure`: `|| format!("{}/{}", f, args.len()) == *fn_str` becomes
  `|| Self::functor_key_eq(f, args.len(), fn_str)`. It is still the right-hand
  side of the same `||`, so it is still evaluated only when
  `f == fn_str.as_str()` fails.
- `SwitchOnStructure` / `SwitchOnStructurePc`: the old arm built `key` once
  and took the first table entry `k` with `*k == key`. The new arm tests each
  entry with `functor_key_eq(f, n, k)`. That is the same predicate, applied to
  the same entries in the same order, so the first match is the same.
- `PutVariable`, `SetVariable`, `UnifyVariable` and the kernel arm
  `RecurseCategoryAncestorPc` (which has the same `_V` line) call
  `Value::Unbound(Self::fresh_var_sym(b'V' | b'H', self.var_counter))`.
  `var_counter` is read and incremented exactly as before.

The fresh names matter beyond comparisons. They can appear in output, for
example when an unbound variable is printed, and they decide `Sym` identity.
This is why the change keeps the exact text and does not invent a new
naming scheme.

### Tests (`mod d126_format_free_step_tests`, generated crate, `cargo test` only)

Each test compares the new helper with the old `format!` expression, kept
verbatim in the test:

- **`usize_decimal_matches_to_string`**: compares `usize_decimal(n)` with
  `n.to_string()` for `n` = 0..200 000, every power of ten ±1, `u32::MAX`,
  `2^40` and `usize::MAX`.
- **`functor_key_eq_matches_the_format_compare`**: 19 functors × 136 arities ×
  about 40 keys each, more than 100 K comparisons. The functors include `""`,
  `[|]`, `.`, `a/b`, `f/2`, `/`, `f/`, `str(f`, digit names and a non-ASCII
  name. The keys include the exact key and near misses: leading zero (`f/02`),
  extra digit, `+N`, a trailing space, `//`, no slash, `f/`, bare `f`, the
  `str(...)` wrapper, N±1 (with wrap-around at `usize::MAX`), a prefix, the
  empty string, every other functor at the same arity, and small arities.
- **`fresh_var_sym_matches_the_format_name`**: `fresh_var_sym` vs
  `format!("_V{}")`/`format!("_H{}")` turned into a `Sym`, for 50 000+ counter
  values. It compares the `Sym`, its text and the `Value::Unbound`.
- **`switch_on_structure_picks_the_same_entry`**: the old "build the key, take
  the first equal entry" search vs the new per-entry test, on a table with
  duplicates, a leading-zero key, a cons key and an empty-name key.

The generated-crate lib suite passes: **257/257** (D125's 253 plus these 4).
It also passes with the feature sets `decorate_sort` and
`decorate_sort intern`.

## Gates

Run from the repo root with `LANG=C.utf8 LC_ALL=C.utf8`. The term binaries are
a fresh `build.sh` build of this tree. The store gates, plunit and conformance
ran in a clean export of this commit. The base outputs are the D125 build's.

| gate | result |
| --- | --- |
| `rust/run_differential_rust.sh` | `cases: 2600 / divergences: 0 / crashes: 0` |
| `rust/run_corpus_rust.sh` | `corpus-under-rust: 51/51 matched SWI` |
| `rust_store/run_differential_rust_store.sh` | `cases: 503 / divergences: 0 / crashes: 0` |
| `rust_store/run_corpus_rust_store.sh` | `corpus-under-rust-store: 51/51 matched SWI`; `rust_store corpus IDENTICAL to term corpus (51/51)` |
| byte identity vs base build | `cmp`-identical: term diff `rust.jsonl` (2600 lines), term corpus `rust.jsonl` (51), store diff `rust.jsonl` (503), store corpus `rust_store.jsonl` (51); scale `--bench` stdout identical at N=40/250/1000/5000 |
| generated crate `cargo test --release --lib` | `test result: ok. 257 passed; 0 failed` |
| Rust WAM plunit (56 files) | same per-file rc as base (28 rc=0, 28 rc≠0) and the same 42 failing test names; `test_wam_rust_target.pl` has the same failure lines |
| CI rust conformance (`CONFORMANCE_TARGETS=rust`, `CONFORMANCE_PROGRAMS=member,builtins`) | rc=0, both unsampled and with `CONFORMANCE_SAMPLE=2` |

## Measurements

The method is the same as D120–D125. Ir per warm resolve =
(Ir(1+k) − Ir(1)) / k, with k=10 at N=40 and k=4 elsewhere, on the
uncommitted in-process repeat shim.

### Ir per warm resolve (deterministic)

| N | base (D125) | D126 | Δ | Δ absolute |
| ---: | ---: | ---: | ---: | ---: |
| 40 | 25.87 M | **24.25 M** | **−6.3%** | −1.62 M |
| 250 | 41.55 M | **39.92 M** | −3.9% | −1.63 M |
| 1000 | 96.53 M | **94.90 M** | −1.7% | −1.63 M |
| 5000 | 391.10 M | **389.48 M** | −0.4% | −1.62 M |

The saving is flat across N, because the number of these instructions per
resolve does not depend on the catalog size for this probe. At N=40,
`format_inner` drops from 1.77 M to 0.02 M (the remaining calls are outside
`step`), `step` inclusive drops from 21.32 M to 19.70 M, and `mi_free` from
0.86 M to 0.78 M. `intern` is unchanged (0.88 M): the fresh names are still
interned once per construction, now from a `&str`.

### Wall clock (shared 4-core container, noisy)

Base and new were interleaved. N=40/250 were run for 21 rounds, because a
first 9-round run at N=40 was dominated by outliers (warm new ranged
4.34–6.37 ms). N=1000/5000 were run for 9 rounds. "Cold" is the shim's own
`resolve_ms` from a fresh process. "Warm" is the median of 11 in-process
repeats, then the median over the rounds.

| N | rounds | cold base | cold new | warm base | warm new | Δ warm |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 40 | 21 | 6.72 ms | 6.25 ms | 4.55 ms | **4.42 ms** | −3% |
| 250 | 21 | 7.62 ms | 8.59 ms | 6.82 ms | **6.69 ms** | −2% |
| 1000 | 9 | 15.65 ms | 16.15 ms | 15.05 ms | 14.76 ms | (noise) |
| 5000 | 9 | 67.72 ms | 63.46 ms | 68.72 ms | 64.64 ms | (noise) |

At N=40, the five fastest warm rounds were 4.29–4.48 ms for base and
4.15–4.32 ms for new. The cold medians are noisy in both directions. Ir is
the reliable metric.

## What dominates next (post-D126 callgrind, N=40, 24.25 M Ir/resolve)

| cost | share | notes |
| --- | ---: | --- |
| `execute_builtin` (inclusive) | 16.0% | builtin work (sort, list deref, unify) |
| `backtrack` (inclusive) | 14.8% | `restore_ax_regs` 6.9% self |
| `trail_binding` (inclusive) | 9.1% | `get_reg` + a `String` key per register entry (D127) |
| `intern` (inclusive) | 3.6% | 0.88 M over ~3.9 K calls: 1,059 from `fresh_var_sym`, 1,510 from `Value::strv` (functor keys), and 1,293 inlined into `step` itself. The step arms also build their heap placeholder atoms (`"__struct_arg__"`, `"__list_head__"`, …) with `.to_string().into()` on every construction; those calls were not traced to lines. Since interning is canonical and append-only, a cached `Sym` per constant name, and a per-counter cache for fresh names, would both be exact |
| `labels` SipHash | ~2.5% + probe | see D125 |

## Files

- `src/unifyweaver/targets/wam_rust_target.pl`: the `GetStructure`,
  `SwitchOnStructure`, `SwitchOnStructurePc`, `PutVariable`, `SetVariable`,
  `UnifyVariable` and `RecurseCategoryAncestorPc` arm bodies.
- `templates/targets/rust_wam/state.rs.mustache`: `usize_decimal`,
  `functor_key_eq`, `fresh_var_sym`, and `mod d126_format_free_step_tests`.

`resolver.pl` and `resolver_store.pl` are unmodified. The regenerated
checked-in crate is not committed.
