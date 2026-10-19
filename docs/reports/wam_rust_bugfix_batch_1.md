<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) 2026 John William Creighton (s243a) -->

# Rust WAM: bug-fix batch 1 (D140–)

**Date:** 2026-10-18. **Ledger:** D140 and following rows of
`docs/proposals/JS_TARGETS_PARITY_PUNCHLIST.md`.
**What:** correctness bugs in the Rust WAM target that the R−1, R0a, R0b and
R1 work found and left open (`wam_rust_r_minus_1_baseline_fixes.md` "Found,
not fixed", `wam_rust_r0a_arity_only_clause_cps.md` "pre-existing",
`wam_rust_r0b_register_trail_audit.md` §4 and §6.2, `wam_rust_r1_numeric_encoding.md`
"Surprise"). Each item is its own commit and ledger row; this report has one
section per item, in landing order. The reference is SWI-Prolog 9.

**Harness:** `tests/test_wam_rust_bugfix_batch_1.pl`, the harness of
`tests/test_wam_rust_baseline_fixes.pl`. Each program is held once
(`bb_clause/2`), asserted into `user:` for SWI and handed to
`write_wam_rust_project/3`. Checks:

- **all solutions, interpreted entry**: a failure-driven driver
  `bb_drv_<name> :- Setup, ( Query, write(Out), nl, fail ; true )` is compiled
  with the program and run by its WAM label; stdout and the success flag must
  equal SWI's, in `emit_mode(interpreter)` and `emit_mode(functions)`;
- **first solution, lowered entry**: when `emit_mode(functions)` lowered the
  query predicate, its Rust function is called directly and compared with
  SWI's `once/1`, on the default build and on
  `--no-default-features --features decorate_sort`.

Every item's cases were added before its fix and run on the unfixed code; the
"before" results below are from those runs.

**Gates** (every commit; base = `a56714b`, built the same way):
`rust/build.sh`, `run_differential_rust.sh`, `run_corpus_rust.sh`,
`rust_store/build.sh`, `run_differential_rust_store.sh`,
`run_corpus_rust_store.sh`; `cmp` of the four output JSONLs and the scale
`--bench` stdout at N=40 and N=5000 against base; the harnesses listed in each
section; the generated crate's `cargo test --release --lib`; the Rust WAM
plunit files; the CI rust conformance smoke
(`CONFORMANCE_TARGETS=rust CONFORMANCE_PROGRAMS=member,builtins`); Ir per warm
resolve at N=40 (callgrind, as in `wam_rust_r1_numeric_encoding.md` §3).

## B1 (D140): names with `\` or `"` in emitted Rust string literals

**Symptom.** A clause body that builds a `\+` term
(`G = (\+ q(X)), call(G)`) produced a crate that `rustc` rejects
(`unknown character escape: '+'`). The same happens for every functor whose
name contains a backslash or a double quote (`\=`, `=\=`, `\==`, `'q"r'`), for
a predicate whose name contains one (its `labels.insert` line), and, in
lowered code, for atom constants containing a `"`.

**Root cause.** Several emitters wrote a name from the WAM text straight into
a Rust `"…"` literal without `escape_rust_string/2` (base `a56714b`):

| site | what |
| --- | --- |
| `wam_rust_target.pl:8734`, `:8776` | `get_structure`/`put_structure`: `Functor::new("~w")` |
| `wam_rust_target.pl:8692` | `labels.insert("~w"…)` (predicate and ITE labels) |
| `wam_rust_target.pl:9034-9059` | the by-name `TryMeElse*`/`Try`/`Trust`/`Retry*`/`Jump` forms |
| `wam_rust_target.pl:9225-9237` | switch-table fallback entries |
| `wam_rust_target.pl:10324`, `:10526-10531`, `:10553` | `lowered_call` dispatch keys (genrec, lowered, fact-table arms) |
| `wam_rust_lowered_emitter.pl:1095`, `:1130` | lowered `get_structure`/`put_structure` |
| `wam_rust_lowered_emitter.pl:1196`, `:1209` | lowered `call`/`execute`: `vm.labels.get("~w")` |
| `wam_rust_lowered_emitter.pl:960`, `:968`, `:1226` | lowered foreign calls |
| `wam_rust_lowered_emitter.pl:1269` | `rust_val_literal/2`: `Value::Atom("~w".into())` |

The last one only worked for backslashes by accident: the lowered emitter
split WAM lines on spaces and commas (`wam_rust_lowered_emitter.pl:44`,
`:120`), so a quoted constant kept its WAM escapes (`'c\\d'` stayed `c\\d`,
which is the right Rust spelling), while a constant containing a space or a
comma (`'say "hi"'`, `'a,b'`) was cut into several tokens, no longer matched
`instr_from_parts/2`, and the instruction was **silently dropped** (the then
branch of `esc_atoms/2` below lost its `put_constant` and returned the else
branch's value).

**Fix.** Every site above escapes the name with `escape_rust_string/2`.
`rust_lowered_dispatch_keys/2`, which reads the dispatch keys back from the
emitted arms, unescapes them (`rust_unescape_string/2`). The lowered emitter
now tokenizes with the shared quote-aware `wam_tokenize_line/2`
(`wam_text_parser.pl`), as the instruction-table path does: a quoted constant
stays one token with its escapes resolved, and the conjunction functor
`,/2` stays one token (the R−1 report recorded it being emitted as `"/2"`).

**Quoted atoms containing a newline: a different root cause, not fixed.**
The shared compiler's `quote_wam_constant/2` (`wam_target.pl:1226`) quotes an
atom only for space, tab, comma, colon, quote and backslash, and escapes only
`\` and `'`. An atom with a newline is written raw (`get_constant a<LF>b, A1`),
so the line-based WAM text is broken before any target reads it: the Rust
generator sees `get_constant a` and a stray `b, A1` line. The fix belongs in
`wam_target.pl` (quote and escape control characters) together with every
target's tokenizer (decode `\n`), so it is left for a shared change.

**Tests.** `esc_naf_no`/`esc_naf_yes` (the R0b case: a `\+` term built and
meta-called), `esc_neq` (`\=`, after a choice point), `esc_functor` (head
structures `\+/1`, `\=/2`, `=\=/2`, `\==/2`, `'q"r'/1` read back with
`functor/3`), `esc_head` (lowered `put_structure`/`get_structure \+/1`),
`esc_atoms_then`/`esc_atoms_else` (lowered constants `'a\\b'`, `'say "hi"'`,
`'c\\d'`), `esc_quote_const` (`'x"y'`), `esc_pred_name` (a predicate named
`'esc\\pred'`).

| | interpreter | functions | lowered entry (default / `Sym = String`) |
| --- | --- | --- | --- |
| before | crate does not compile (12 `rustc` errors) | crate does not compile | — |
| after (rustc fixed, lowered tokenizer not yet) | all pass | `esc_atoms_then`, `esc_atoms_else` wrong | `esc_atoms_*` wrong |
| after | 12/12 | 12/12 | 6 lowered, all match; 3 not checked (`call/1` in a lowered ITE condition: the label-less builtin defect noted by the R−1 harness) |

**Gates.** Harness 40/40; `test_wam_rust_baseline_fixes.pl` 92/92,
`test_wam_rust_r0a_clause_cps.pl` 90/90, `test_wam_rust_register_restore.pl`
120/120, `test_wam_rust_first_arg_dispatch.pl` 217/217,
`test_wam_rust_r1_encoding.pl` 13/13, `test_wam_rust_stream_swi_parity.pl` 9
passed / 12 blocked (as base); lib `cargo test --release --lib` 273/273;
conformance smoke rc=0. Term differential 2600/0/0, term corpus 51/51, store
differential 503/0/0, store corpus 51/51 and IDENTICAL to term; all four
JSONLs and `--bench` stdout at N=40 and N=5000 `cmp`-identical to base. The
generated resolver crate is identical to base's apart from the date lines (no
resolver name needs escaping, and the resolver has no lowered predicates), so
its Ir is unchanged.
