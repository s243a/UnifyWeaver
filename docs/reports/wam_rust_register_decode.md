<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) 2026 John William Creighton (s243a) -->

# Rust WAM: cheaper register access (D123)

**Date:** 2026-10-04. **Ledger:** D123. **Author:** Opus.
**What:** make register access in the Rust WAM target cheaper without
changing what it returns. The register name is still a `String` operand, but
the names the compiler emits are now decoded inline instead of going through
`str::parse::<usize>`. `get_reg` also no longer clones the register before
dereferencing it. Output is byte-identical.
**Result:** about **−1.3 M Ir per warm resolve at every catalog size**: −4.5%
at N=40 (29.29 M → 27.98 M Ir) and −0.3% at N=5000 (401.34 M → 400.09 M).
The warm in-process N=40 resolve drops from 5.08 ms to 4.92 ms (−3%, noisy
box).

## Profile first: where `get_reg`'s 10.2% went

The probe and catalogs are the same as in D120–D122
(`docs/reports/wam_rust_yreg_slot_frames.md`): `resolve_layered([p30])` on
`gen_scale_catalog.mjs` catalogs, capped by `scale_to_case.mjs` at N = 40 /
250 / 1000 / 5000. The base is origin/main `6649944` (D122 merged). It
reproduces D122's numbers: 29.29 M Ir per warm resolve at N=40 (D122 reported
29.30 M).

To split `get_reg` into its parts, the base crate was rebuilt once more with
`CARGO_PROFILE_RELEASE_DEBUG=line-tables-only` (same optimisation level) and
run under `callgrind --dump-instr=yes`. Self cost was then grouped by
function and source line, inlined library lines included. Per warm resolve at
N=40:

| part of register access | Ir / resolve | share of 29.3 M |
| --- | ---: | ---: |
| `get_reg` inclusive | 2.99 M | 10.2% |
| — `deref_var` (bindings SipHash lookup + result clone) | 1.15 M | 3.9% |
| — `get_reg_raw` called from `get_reg` (parse + clone) | 0.68 M | 2.3% |
| — `get_reg` self (Y-frame walk, slot lookup, copy drop, frame) | 1.17 M | 4.0% |
| A/X name parse (`reg_index` + inlined `core::num` `from_str`) in `get_reg_raw` | 0.58 M | 2.0% |
| same, in `set_reg` | 0.31 M | 1.1% |
| Y name parse (`YRegs::slot`) and Y test in `get_reg`/`put_reg` | 0.17 M | 0.6% |
| `trail_binding` inclusive (calls `get_reg` on every trailed register) | 3.15 M | 10.8% |

The full cost of decoding names, across every caller, was about **1.1 M Ir
per resolve (≈3.8%)**. Almost all of it was the generic `parse::<usize>`
(`from_str_radix` with its sign and overflow handling), run on every access.
`get_reg` added a second cost of about the same size: it cloned the slot
(`y_regs.get(name).cloned()` or `get_reg_raw`), `deref_var` cloned again, and
then the first copy was dropped. The remaining `get_reg` cost is
`deref_var`'s bindings lookup, which is the SipHash-on-`String` `bindings` map
and a separate lever.

Call counts per resolve: about 15.6 K `get_reg` calls (10.8 K from
`trail_binding`, 4.8 K from `step`), 8.7 K `get_reg_raw` calls under
`get_reg`, and 8.4 K `set_reg` calls (5.3 K from `step`, 3.1 K via `put_reg`).

**None of the `format!` calls in `step` (6.1%) builds a register name.** The
`format_inner` call sites were mapped back to `step`'s source lines (callgrind
call-site addresses, then `addr2line -i` on the line-table build). Of the
~2.5 K calls per resolve:

- 958 come from `GetStructure` read mode on an in-register compound:
  `format!("{}/{}", f, args.len()) == *fn_str`, which builds a functor key
  only to compare it.
- 476 come from `SwitchOnStructure`'s `format!("{}/{}", f, args.len())`
  table key.
- 1,059 are fresh-variable names: `_V{n}` in `PutVariable` (642), `_H{n}` in
  `SetVariable` (413) and `UnifyVariable` (4).

They are functor keys and variable names, so they are out of scope for D123.
See "What dominates next".

## Choosing the fix

Three options, as listed in the brief:

1. **A decoded operand side table** (a parallel `Vec` indexed by pc, built at
   load). This would remove the parse only for operand-sourced names. The
   builtins (`get_reg_raw("A1")`), the trail unwind (`put_reg(&reg)`) and the
   lowered tier all pass literal or stored names, and would still parse.
   There is also a structural problem: `step(&Instruction)` is also called on
   temporaries (the lowered tier, regions, `RecurseCategoryAncestor`
   re-dispatch), so `step` cannot assume the decoded entry at `pc` belongs to
   the instruction it was given. Every register-touching arm would need an
   index-taking twin, with a decode-on-the-fly fallback for temporaries.
2. **Changing the operand type in the generator** (`GetVariable(Reg, Reg)`).
   This has the largest blast radius: generated `lib.rs` code tables, the
   lowered emitter, regions, foreign shims, hand-written test harnesses and
   fixtures.
3. **Making the parse and the lookup cheaper, in place.** This reaches every
   caller, literal names included, with no type or API change.

Options 1 and 2 could save at most what option 3 leaves behind. A two-byte
inline decode is a handful of instructions, so over the ~25 K name decodes per
resolve that is an estimated ~0.2 M Ir, against a much larger blast radius.
So option 3 was taken, in two pieces.

## The change

All in `templates/targets/rust_wam/state.rs.mustache`.

### 1. `reg_index` decodes the emitted names inline

```rust
pub fn reg_index(name: &str) -> usize {
    let num = match name.as_bytes() {
        [p @ (b'A' | b'X' | b'Y'), d0] if d0.is_ascii_digit() =>
            (*p, (d0 - b'0') as usize),
        [p @ (b'A' | b'X' | b'Y'), d0, d1]
            if d0.is_ascii_digit() && d1.is_ascii_digit() =>
            (*p, (d0 - b'0') as usize * 10 + (d1 - b'0') as usize),
        _ => return Self::reg_index_parse(name),
    };
    match num {
        (b'A', n) => n.wrapping_sub(1),
        (b'X', n) => n.wrapping_add(99),
        (_, n) => n.wrapping_add(199),
    }
}
```

`reg_index_parse` is the pre-D123 body, unchanged and marked
`#[inline(never)]`.

**Why it is exact.** The fast path is taken only for an `A`/`X`/`Y` byte
followed by exactly one or two ASCII digits. For such a suffix,
`usize::from_str` returns exactly its decimal value: no sign, no overflow,
leading zeros allowed (`"A01"` gives 1 on both paths). The prefix arithmetic
is copied as is, including `A0 → usize::MAX`. Every other name goes through
the old code byte for byte: three or more digits, `+1`, an empty or non-digit
suffix, any other prefix, non-ASCII. That includes the old panic on a
multi-byte first char (`&name[1..]` off a char boundary), which nothing emits.

### 2. `get_reg` dereferences the slot in place

```rust
pub fn get_reg(&self, name: &str) -> Option<Value> {
    if Self::is_y_reg_name(name) {
        for entry in self.stack.iter().rev() {
            if let StackEntry::Env(_, y_regs) = entry {
                return y_regs.get(name).map(|v| self.deref_var(v));
            }
        }
        None
    } else {
        match self.regs.get(Self::reg_index(name)) {
            Some(Value::Uninit) | None => None,
            Some(v) => Some(self.deref_var(v)),
        }
    }
}
```

`deref_var(&Value) -> Value` only reads its argument. Calling it on the slot
gives the same value as calling it on a clone of the slot, minus one clone and
one drop. The lookup rules do not change:

- A Y register comes from the topmost `Env` frame, skipping
  `UnifyCtx`/`WriteCtx` entries.
- With no frame, the result is `None`.
- A stored `Uninit` in a Y slot is still returned as `Some(Uninit)`, as the
  old `.cloned()` did.
- An A/X slot that is `Uninit` or out of range is `None`. That is
  `get_reg_raw`'s `idx < len` + `Uninit` rule, written as `regs.get(idx)`.

### 3. One Y-name test, as a byte test

`get_reg`, `put_reg` and `get_reg_ref` each spelled out
`name.starts_with('Y') && name.len() > 1 && name[1..].chars().next().map_or(false, |c| c.is_ascii_digit())`.
They now call `is_y_reg_name`:
`b.len() > 1 && b[0] == b'Y' && b[1].is_ascii_digit()`. `'Y'` is ASCII, so
byte 1 is a char boundary. If byte 1 is ASCII, it is the first char. If it is
not, it starts a non-ASCII char, which is never an ASCII digit. The two tests
are equal on every string.

Unchanged: `get_reg_raw`, `set_reg`, `set_reg_str`, `put_reg`, `YRegs`, the
trail, and every caller. They get the faster `reg_index` automatically.

### Tests

`mod d123_register_decode_tests` (compiled only under `cargo test`) keeps the
old `reg_index`, the old Y test and the old `get_reg` verbatim, and compares:

1. **`reg_index_matches_the_parse_for_every_name_form`**: over 600 K names.
   The set is every ASCII prefix byte × every suffix of 0–3 chars over
   `0-9 / : + - a space` (`/` and `:` are the bytes next to the digits), plus
   `A`/`X`/`Y` × every 1- and 2-byte ASCII suffix. It also adds long and odd
   forms: `A100`, `Y4096`, `A18446744073709551615`, `…616` (overflow), `X+12`,
   `A-1`, `Aé`, `A١` (a non-ASCII digit), `Y01`, `a1`, `AA1`, ….
2. **`reg_index_keeps_the_known_values_and_the_panic`**: the documented
   indices, `A0 → usize::MAX`, and the same panic on `"é1"` from both
   versions.
3. **`y_name_test_matches_the_old_predicate`**: every ASCII (and `é`) prefix ×
   every suffix of up to 2 ASCII chars, plus `Yé`, `Y١`, an emoji, `Y`, `""`.
4. **`get_reg_matches_the_old_clone_then_deref`**: about 50 names compared on
   six machine states, by `Debug` string. The names are `A0..A7`, `X0..X7`,
   `Y0..Y7`, `Y01`, `A+1`, `X100`, `Y4096`, `Yé`, `""` and others. The states
   are: no frame; A/X registers holding an atom, a two-link binding chain to a
   list, an unbound var, a compound, an integer, and `A99`; an empty top frame
   over a populated lower one; a top frame with a bound var, an atom, a stored
   `Uninit` and a non-canonical `Y01`; `UnifyCtx`/`WriteCtx` entries above the
   frame; and a binding added afterwards.

The full generated-crate lib suite passes: **248/248** (D122's 244 plus these
4).

## Gates

Run from the repo root with `LANG=C.utf8 LC_ALL=C.utf8` on fresh `build.sh`
builds (term and store) of this tree and of an untouched origin/main `6649944`
export.

| gate | result |
| --- | --- |
| `rust/run_differential_rust.sh` | `cases: 2600 / divergences: 0 / crashes: 0` |
| `rust/run_corpus_rust.sh` | `corpus-under-rust: 51/51 matched SWI` |
| `rust_store/run_differential_rust_store.sh` | `cases: 503 / divergences: 0 / crashes: 0` |
| `rust_store/run_corpus_rust_store.sh` | `corpus-under-rust-store: 51/51 matched SWI`; `rust_store corpus IDENTICAL to term corpus (51/51)` |
| byte identity vs base build | `cmp`-identical: term diff `rust.jsonl` (2600 lines), term corpus `rust.jsonl` (51), store diff `rust.jsonl` (503), store corpus `rust_store.jsonl` (51); scale `--bench` stdout identical at N=40/250/1000/5000 |
| generated crate `cargo test --release --lib` | `test result: ok. 248 passed; 0 failed` |
| Rust WAM plunit (`tests/test_wam_rust_*.pl`, `tests/core/*rust*.pl`, 55 files) | same per-file rc as base (27 rc=0, 28 rc≠0); failing-test-name sets identical (42 = 42) |
| CI rust conformance (`CONFORMANCE_TARGETS=rust`, `CONFORMANCE_PROGRAMS=member,builtins`) | rc=0, both unsampled and with `CONFORMANCE_SAMPLE=2` |

The 42 failing plunit tests are the same pre-existing breakage D120–D122
recorded: the test harnesses' hand-written Rust still builds
`Value::Atom(String)` against the interned `Sym`. `tests/test_wam_rust_target.pl`
has the same failure lines in base and new. No test's expected strings
changed.

## Measurements

The method is the same as D120–D122. Binaries are `--release` with the
crate's default features (mimalloc on via the pkg_resolver `build.pl`). Ir
comes from `valgrind --tool=callgrind` on an uncommitted scratch copy of the
shim that repeats `call_pred` in-process. Each run checks that every repeat
gives the same answer. Ir per warm resolve = (Ir(1+k) − Ir(1)) / k, with k=10
at N=40 and k=4 elsewhere.

### Ir per warm resolve (deterministic)

| N | base (D122) | D123 | Δ | Δ absolute |
| ---: | ---: | ---: | ---: | ---: |
| 40 | 29.29 M | **27.98 M** | **−4.5%** | −1.32 M |
| 250 | 45.30 M | **43.98 M** | −2.9% | −1.32 M |
| 1000 | 101.30 M | **99.99 M** | −1.3% | −1.31 M |
| 5000 | 401.34 M | **400.09 M** | −0.3% | −1.26 M |

As in D122, the saving is flat because it scales with the number of register
accesses, and that is about the same at every N for this probe.

Where the N=40 saving comes from (per warm resolve):

| site | base | D123 |
| --- | ---: | ---: |
| `get_reg` (inclusive) | 2.99 M | 2.11 M |
| `get_reg` self | 1.17 M | 0.83 M |
| `get_reg_raw` (inclusive; now only its direct callers) | 1.20 M | 0.30 M |
| `set_reg` (inclusive) | 0.70 M | 0.48 M |
| `trail_binding` (inclusive) | 3.15 M | 2.52 M |
| `deref_var` (inclusive) | 1.81 M | 1.81 M |
| `reg_index_parse` (the slow path) | — | not reached |

`deref_var` is unchanged, as expected: the same lookups still run. The slow
path is never taken on this workload, because every emitted name is a prefix
plus one or two digits.

### Wall clock (shared 4-core container, noisy)

Base and new were interleaved, 9 rounds per size. "Cold" is the shim's own
`resolve_ms` from a fresh process. "Warm" is the median of 11 in-process
repeats, then the median over the 9 rounds.

| N | cold base | cold new | warm base | warm new | Δ warm |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 40 | 6.72 ms | 8.32 ms | 5.08 ms | **4.92 ms** | −3% |
| 250 | 8.54 ms | 8.94 ms | 7.50 ms | **7.11 ms** | −5% |
| 1000 | 18.09 ms | 18.53 ms | 15.32 ms | 15.85 ms | (noise) |
| 5000 | 65.97 ms | 61.60 ms | 65.37 ms | 65.59 ms | (noise) |

The warm medians at N=40/250 track the Ir change. At N=40, warm base was
4.91–7.16 ms and warm new 4.55–6.93 ms; six of nine new rounds were below the
fastest base round but one. The cold medians are bimodal on this box and
should not be read as a change:

- At N=40, cold base ranged 6.11–43.4 ms and cold new 6.13–34.5 ms, with
  matching minima (6.11 vs 6.13 ms).
- At N=1000/5000, the expected −1.3%/−0.3% is far below the box's noise.

Ir is the reliable metric here.

## What dominates next (post-D123 callgrind, N=40, 27.98 M Ir/resolve)

| cost | share | notes |
| --- | ---: | --- |
| `execute_builtin` (inclusive) | 16.0% | almost all of it is the builtins' own work: `execute_ext_builtin` 10.0% (`sort_by` 4.5%, `deref_list_arg`), `unify` for `=/2`, `deref_heap` in term builtins. The string-name family cascade itself is ~0.2 M (see D124) |
| `backtrack` (inclusive) | 13.9% | `restore_ax_regs` 6.1% |
| `trail_binding` (inclusive) | 9.0% | `get_reg` (deref'd old value) + a `String` key alloc per trailed register |
| `get_reg` (inclusive) | 7.6% | now mostly `deref_var`, the `bindings` lookup |
| `bindings` SipHash | 6.7% + 3.1% | `hash_one` + `sip::write` on `bindings: HashMap<String, Value>` (keys are interned `Sym`s turned back into `&str`) |
| `format!` in `step` | 6.3% | ~2.5 K per resolve: 1.4 K build a `"f/N"` key only to compare it (`GetStructure` read mode, `SwitchOnStructure`), 1.1 K are fresh-variable names `_V{n}`/`_H{n}` |
| mimalloc malloc/free | ~11% incl. | spread across the sites above |

The next levers, roughly in order of size:

1. **`bindings` keyed by the interned id with an Fx hasher** (the D112
   pattern). This cuts `deref_var`, `bind_var` and the trail unwind at once.
2. **The `format!`s in `step`.** Compare a functor key without building it:
   `fn_str` is `f`, then `/`, then the decimal arity, checked by slicing. For
   fresh-variable names, write `_V`/`_H` + the counter with a small integer
   formatter. The bytes are identical in both cases, so this is exact.
3. **`trail_binding` without a key allocation.** A register trail entry could
   carry the decoded index instead of a `String`, as long as non-canonical
   names keep a String fallback.

## Files

- `templates/targets/rust_wam/state.rs.mustache`: `reg_index` fast path +
  `reg_index_parse`, `is_y_reg_name`, `get_reg` in-place deref,
  `get_reg_ref`/`put_reg` use `is_y_reg_name`, and
  `mod d123_register_decode_tests`.

`resolver.pl` and `resolver_store.pl` are unmodified. The regenerated
checked-in crate is not committed.
