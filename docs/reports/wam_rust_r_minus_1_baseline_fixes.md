<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) 2026 John William Creighton (s243a) -->

# Rust WAM: R−1 baseline fixes (D129–D134)

**Date:** 2026-10-06. **Ledger:** D129 (R−1f) and following rows.
**Design:** `docs/proposals/wam_rust_heap_cell_rewrite_design.md` §12.1.
**What:** phase R−1 of the heap-cell rewrite fixes, on main and before the
rewrite, the places where today's Rust WAM target differs from SWI-Prolog.
Each item is its own commit and ledger row and changes only what its row
says. This report has one section per item, in landing order.

**Regression harness:** `tests/test_wam_rust_baseline_fixes.pl`. Each program
is held once (`bf_clause/2`), asserted into `user:` for SWI and handed to
`write_wam_rust_project/3`. Three checks:

- **all solutions, interpreted entry**: a failure-driven driver
  `bf_<name> :- Setup, ( Query, write(Out), nl, fail ; true )` is compiled
  with the program and run by its WAM label. Its stdout and its success flag
  must equal SWI's. Run in `emit_mode(interpreter)` and
  `emit_mode(functions)`.
- **first solution, lowered entry**: when `emit_mode(functions)` lowered the
  query predicate, the Rust function is called directly with the query
  arguments; the oracle is SWI's `once/1`.
- the functions-mode crate is also built and run with
  `--no-default-features --features decorate_sort` (`Sym = String`).

Every item's cases were added before its fix and confirmed failing on the
unfixed code; earlier items' cases stay green.

## R−1f (D129): lowered emitter literals compile under `intern`

**Defect.** Since D96, `Value::Atom` and `Value::Unbound` hold a `Sym`, which
is a `u32` interner id under the default `intern` feature and a plain
`String` without it. The lowered emitter still wrote `String`s:

| site (`wam_rust_lowered_emitter.pl`) | emitted before | emitted now |
| --- | --- | --- |
| `get_constant` atom, unbound-register arm (:901) | `Value::Atom("x".to_string())` | `Value::Atom("x".into())` |
| `get_nil`, unbound-register arm (:937) | `Value::Atom("[]".to_string())` | `Value::Atom("[]".into())` |
| `put_variable` (:978) | `Value::Unbound(format!("_V{}", vm.var_counter))` | `Value::Unbound(WamState::fresh_var_sym(b'V', vm.var_counter))` |
| `rust_val_literal` atom (:1114; `put_constant`, `unify_constant`, `set_constant`) | `Value::Atom("x".to_string())` | `Value::Atom("x".into())` |

So any lowered predicate with an atom constant or a fresh variable failed to
compile with default features. `.into()` converts `&str` to either `Sym`.
`fresh_var_sym` is the interpreter's own allocation-free name builder (D126)
and yields the same text, `_V<n>`, as the old `format!`. No other emitted
`String`-for-`Sym` site exists: the shared instruction table already writes
`Value::Atom("x".to_string().into())`, and every other `Value::Atom(` /
`Value::Unbound(` construction in `wam_rust_target.pl` and `rust_target.pl`
is a pattern or already converts. The resolver crates contain one lowered
function with no atom constant or fresh variable, so their generated code
is unchanged.

**Tests.** The harness's R−1f programs (`lf_atoms`: atom `get_constant` /
`put_constant` in an ITE; `lf_fresh`: `put_variable` and a compound result;
`lf_nil`: `get_nil` over two clauses) are lowered in functions mode. On the
unfixed code the functions crate did not build (10 `E0308` mismatches in
`src/lib.rs`); now it builds and runs under both feature sets and every
answer equals SWI's. The Rust sources of the lowered execution tests
(`test_wam_rust_lowered_ite_exec.pl`, `_t4`, `_t5`, `_t6`) and the
cut-semantics probe driver built their own inputs with `Value::Atom(s.to_string())`
/ `Value::Unbound(OUT.to_string())`, which also fails under `intern`; they
now use `.into()`. Two expected-text assertions in `test_wam_rust_target.pl`
(the quoted-numeric atom regression for the lowered `put_constant`) were
updated deliberately from `"42".to_string()` to `"42".into()`.

**Gates** (clean export of the commit, as for every item below). Term
differential `cases: 2600 / divergences: 0 / crashes: 0`, term corpus 51/51,
store differential 503/0/0, store corpus 51/51 and identical to the term
corpus, generated-crate `cargo test --release --lib` 260/260, CI rust
conformance smoke (`member,builtins`) rc=0 unsampled and with sample 2.
**Byte identity:** both generated resolver crates are textually identical to
the base `634c244` (only generation timestamps and store paths differ), and
the four output JSONLs and the scale `--bench` stdout at N=40 and N=5000 are
`cmp`-identical, so the frozen baseline is unchanged by this item.
**Perf:** callgrind on one `--bench` run at N=40 (load plus one resolve):
34.41 M → 34.43 M Ir, +0.05% with identical generated code, which is the
measurement noise floor used for the later items. **Rust WAM plunit** (52
files, every `tests/test_wam_rust_*.pl`): base 26 passing / 26 failing files
and 38 failing tests; after this item 32 / 20 and 29. The failing set is a
strict subset of the base's: `test_wam_rust_cut_semantics` (all four modes;
its probe driver no longer fails to build), `test_wam_rust_lowered_dispatch`,
`_lowered_ite_exec` and `_lowered_t4`/`_t5`/`_t6` now pass, and nothing newly
fails.
