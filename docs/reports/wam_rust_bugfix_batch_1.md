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

## B2 (D141): `ignore/1`

**Symptom.** Every call of `ignore/1` failed: `ign_fail(R) :- ignore(fail_goal), R = ok`
printed nothing (SWI `ok`), and a `findall/3` whose goal contains `ignore/1`
returned `[]`.

**Root cause.** Nothing implemented it. The shared compiler does not inline
`ignore/1` (it inlines `once/1`, `forall/2` and `\+/1`), so it arrives as
`call ignore/1` / `execute ignore/1`; the `Call`/`Execute` arms
(`wam_rust_target.pl:735-783`, `:835-893`) find no label, no dynamic clauses,
no foreign registration, and fall back to `execute_builtin`, whose meta family
(`execute_meta_builtin`, `:7077`) has no `ignore/1` arm, so the goal fails
(silently, unless `UW_WAM_WARN_UNKNOWN` is set). Meta-calls
(`call(ignore(G))`) take `call_goal_key`, which ends in the same table.

**Fix.** An `ignore/1` arm next to `call/N` in `execute_meta_builtin`:
`ignore(G) :- (call(G) -> true ; true)`. The goal runs exactly as `call/1`
runs it (`call_goal_once`: an opaque cut scope with a backtrack floor at the
call's entry depth, first solution, the caller's environment stack restored).
When it fails, the bindings it made are undone (`unwind_trail_to` to the mark
taken before the call, heap truncated) and `ignore/1` succeeds; a ball thrown
inside it is left in flight. Like `call/1` in this runtime, a conjunction goal
is meta-called structurally and cannot retry its left conjunct (shared
limitation of `call_goal_value`, unchanged).

**Tests.** `ign_fail` (failing goal), `ign_first` (first solution only),
`ign_undo` (bindings of the failed goal undone), `ign_cut` (`ignore(!)` keeps
the caller's choice points), `ign_findall` (inside a `findall/3` goal),
`ign_meta` (`call(ignore(G))`), `ign_tail` (last goal), `ign_write` (a side
effect happens once), `ign_throw` (a ball propagates to `catch/3`).

| | interpreter | functions | lowered entry |
| --- | --- | --- | --- |
| before | 9/9 wrong (no output, or `[]`) | 9/9 wrong | — |
| after | 9/9 | 9/9 | not checked: every B2 query predicate the emitter lowers reaches `ignore/1` through a lowered `call ignore/1`, and a lowered `call`/`execute` of a label-less runtime builtin returns false (the defect the R−1 harness records); the functions-mode check above runs the same predicates through `lowered_call`, which falls back to the interpreter when the lowered function declines |

**Gates.** Harness 76/76; the other harnesses, lib tests, conformance,
differentials, corpora and byte identity as listed in the ledger row. The
generated resolver crate gains only the new match arm (the resolver never
calls `ignore/1`).

## B3 (D142): `aggregate_all/3` (and `findall/3`) as runtime builtins

**Symptom.** `findall(c(K, C), (member(K, [p, q, r]), aggregate_all(count, e(K, _), C)), L)`
gave `L = []` (SWI `[c(p,3),c(q,1),c(r,0)]`); the R0b audit's
`nested_agg_inline`. The same for `aggregate_all/3` in an if-then-else
condition or branch, a disjunction, a `\+`, or a meta-call
(`call(aggregate_all(...))`), and for a meta-called `findall/3`.

**Root cause.** The shared compiler inlines `aggregate_all/3`
(begin/end_aggregate) only as a clause-body goal (`compile_goals`,
`wam_target.pl:1738`). Inner goal positions go through
`compile_inner_call_goals/4` (`wam_target.pl:2415`), which inlines `findall/3`
(`:2434`), `\+`, `once/1`, `forall/2`, if-then-else and disjunction but sends
every other goal, `aggregate_all/3` included, to `compile_goal_call`
(`:2469`): `call aggregate_all/3`. The Rust runtime had no such predicate
(no label; `execute_meta_builtin`, `wam_rust_target.pl:7077`, had no arm), so
the call failed, and with it the findall goal for every element. The shared
compiler is not changed: the call is a correct compilation of the goal, and
the other WAM targets have their own runtimes; the gap is the Rust runtime's.

**Fix.** `aggregate_all/3` and `findall/3` are runtime builtins (meta family).
They need every solution of an arbitrary goal, which the runtime could not
produce: `call_goal_value` is first-solution and cannot retry the left
conjunct of a conjunction, and the dynamic body solver re-ran goals on clones
(B4). So this item adds a resumable goal solver, `BodyGen`
(`state.rs.mustache`), used here and by B4:

- control constructs are taken apart in Rust: conjunction, disjunction,
  if-then-else (condition opaque, first solution), `\+`, `call/N`, `once/1`,
  `ignore/1`, `M:G`, and `!` (a cut prunes the alternatives to its left in its
  clause body or meta-call; `call/N`, `once/1`, `\+` and the if-then-else
  condition stop it);
- every other goal is a `GoalGen`: a machine that meta-called it
  (`meta_call_first`, the dispatch of `call_goal_key`) and, on the next
  request, backtracks into the choice points the goal left above its floor;
  a compiled predicate then runs on to its next exit (the `cp == 0` sentinel
  of its call), a builtin's or dynamic predicate's redo is self-contained in
  `resume_builtin`;
- a solution is a copy of the machine with the goal's choice points removed
  and the caller's control registers put back;
- the goal runs on a copy of the caller's machine (without its choice
  points), and the state a nested run must share with its caller rather than
  copy (`SideState`: the dynamic database, the label table, the `read/1`
  input and cursor, the fresh-variable counter, the step counters, the
  aggregate accumulator, the R0a full-save flag) is moved into the running
  machine before each step and back out after it, so side effects happen once,
  in order, and reach the caller, and copying a machine does not copy the
  database or the label table.

`builtin_aggregate_all` follows SWI-Prolog's `library(aggregate)`: `count`,
`sum(E)`, `max(E)`, `min(E)` (E evaluated; the first maximal value is kept),
`max(X, W)`, `min(X, W)` (result `max(X, W)`/`min(X, W)`, X must be a number),
`bag(T)`, `set(T)` (sorted, duplicates removed). `count` and `sum` of no
solutions are 0, `bag` and `set` are `[]`, `max` and `min` fail. Each
solution's template is copied with fresh variables (findall semantics); a
ball thrown in the goal propagates.

**The inline path had two wrong answers, fixed in the same item.** The
begin/end_aggregate finalisation (`resume_builtin` `"aggregate_frame"`,
`wam_rust_target.pl:5267` on base) failed for `aggregate_all(bag(T), G, L)`
and `aggregate_all(set(T), G, S)` with no solutions (`"bagof" | "bag"` shared
bagof's empty-fails rule, `:5318`; SWI gives `[]`), and returned `[]` for
`max`/`min` of no solutions (`:5349`, `:5368`; SWI fails). Both now follow
SWI; `bagof`/`setof` keep failing on no solutions. The T7 parallel-aggregate
reductions (the `ParAggregate` arm, `:1128`/`:1145`, and the injected
wrapper's `rust_agg_minmax_reduce/2`) still give `[]` for an empty `max`/`min`;
they run only under `parallel(true)` and are left for the parallel work.

**Not done.** `bagof/3` and `setof/3` in inner positions (also compiled as
calls) have no runtime builtin yet; with free variables they need ISO
grouping, which the inline path refuses too. `call/1` itself stays
first-solution over conjunctions (unchanged).

**Tests.** `agg_in_findall` (the R0b case), `agg_templates` (sum, max, min,
bag, set), `agg_witness` (`max(X, W)`, `min(X, W)`), `agg_expr` (evaluated
`sum(V*2)`, `max(V+1)`), `agg_empty` (count 0, sum 0, bag/set `[]`),
`agg_empty_max`, `agg_inline_empty` / `agg_inline_max_empty` /
`agg_inline_min_empty` (the inline path), `agg_control` (nondeterministic left
conjunct, cut, disjunction, `\+`, if-then-else inside the goal), `agg_ite`
(in an if-then-else condition and branch), `agg_meta`, `findall_meta`,
`agg_assert` (assertz in the goal persists), `agg_throw` (ball to
`catch/3`), `agg_fresh` (fresh variables per solution).
`tests/test_wam_rust_register_restore.pl`: `nested_agg_inline` is now
compared with SWI in both modes (its `rr_known_divergence/3` lines removed).

| | interpreter | functions | lowered entry |
| --- | --- | --- | --- |
| before | 14 of 16 wrong (`[]`, no output, or the inline empty cases; `agg_empty_max` and `agg_fresh` matched by accident: the failing inner call gave the same printout) | 14 of 16 wrong | — |
| after | 16/16 | 16/16 | not checked (the query predicates the emitter lowers contain a findall/aggregate, which the lowered emitter drops, or reach `aggregate_all/3` through a lowered call) |
