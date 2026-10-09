<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) 2026 John William Creighton (s243a) -->

# Rust WAM: clause-entry choice points save only A1..An (R0a, D135)

**Date:** 2026-10-09. **Ledger:** D135. **Base:** origin/main `5f08375`
(R−1 complete, D129–D134).
**What:** phase R0a of `docs/proposals/wam_rust_heap_cell_rewrite_design.md`
(§12, §5.5). The clause-entry `try_me_else` of a multi-clause predicate now
pushes a choice point that saves only the argument registers A1..A<arity>
(`Instruction::TryMeElseArgs(label, arity)`, emitted by the generator).
Everything else about that choice point is unchanged: the environment-stack
snapshot, the trail/heap marks, B0 and the parked clause barrier, the
`levels`, and the clear-then-restore of `restore_ax_regs` (every other dirty
A/X slot becomes `Uninit` on a retry, as before). ITE guards and
disjunctions, aggregate frames, builtin/redo, fact-table, foreign and
dynamic choice points, `lo_clause_snapshot` and the other lowered snapshots
keep the full `save_regs`.
**Result:** −2.04 M Ir per warm resolve at N=40 (23.10 M → 21.06 M, −8.8%),
−2.01 M at N=5000 (388.25 M → 386.24 M, −0.5%). All gates green and every
resolver output byte-identical.

The spike (D128, variant s2) measured this lever on top of heap-cell
variables. The spike's code was not kept; this is a re-derivation on main,
with one difference that matters: **the spike's premise does not hold for
every program.** This report states the safety argument, the shapes where it
fails, and how the change excludes them.

## The safety argument

A clause choice point C for P/n is pushed at P's entry with register file R0.
A retry restores the stack, trail, heap, B0 and continuation from C and jumps
to clause k (`L_P_n_k`). With the full save, the A/X registers are R0; with
the arity-only save, they are R0 ∩ {A1..An} and everything else is `Uninit`.
The two runs differ only if some instruction reads a register r ∉ {A1..An}
after the retry before anything writes r. Such a read can only be

1. **in clause k itself, before it writes r.** The WAM convention: a clause
   reads only its argument registers before it writes a temporary
   (temporaries beyond the arity are dead at clause entry);
2. **in the continuation of the call that entered P** (P returns through
   `proceed`, or through `execute` chains, to the `cp` saved in C). The WAM
   convention: no A/X register is live across a call, because the callee may
   overwrite any of them;
3. **in Rust code that ran P inside a nested `run()`** and reads registers
   after it returns.

This argument is specific to clause-entry choice points. An ITE guard is
pushed mid-clause, where temporaries are live and its else branch reads them;
a builtin or fact choice point resumes the builtin's own continuation, which
can read temporaries live across an inline builtin call. Those keep the full
save.

### Where the convention does not hold in this compiler

The shared compiler's permanent-variable rule (`find_permanent_vars/2`) makes a
variable permanent if it occurs in any goal after the first, and it exposes
the goals inside a first-position ITE, disjunction or aggregate. It does not
look inside `forall/2`, `\+/1` or `once/1`, so a variable used only inside
such a first goal stays in an X register even when it is read after a call:

```prolog
p(A, B) :- forall(q(A), s(B)).
%   get_variable X2, A2 ... put_value X1, A1 / call q/1 / ... / put_value X2, A1 / call s/1
```

`X2` is live across `call q/1`. With the full save, a retry of q's clause
choice point restores it; with the arity-only save (q/1 saves only A1) it is
`Uninit`, `put_value X2, A1` fails, and `forall` answers wrong. The same holds
for `\+ (q(A), t(A, B))` and `once((q(A), t(A, B)))`. A count (or constant
template) aggregate is another case: the compiler collects the `A1`
placeholder at `end_aggregate A1` after the inner goal; a retried 0-arity
choice point inside the goal clears it and `EndAggregate` then fails.

In the lowered tier, an ITE whose condition calls a predicate restores the
else branch's registers through the trail, not through a guard choice point,
so the else branch's registers are live across that call too.

On the Rust side, four builtins read an argument register **after** running
their goals: `include/3`, `exclude/3` and `partition/4` read A3 (and A4), and
`foldl/4,5` read A4 (A5). A goal clause retried inside them with arity below
that index would clear it. They can also be reached through `call/N` with a
goal built at run time, which no static check sees. The benchmark `main.rs`
template reads A1 back after running `dimension_n/1`.

Every other Rust site that runs a goal in a nested run was checked: `call/N`,
`catch/3` (restores its own full `save_regs` snapshot on failure), the `\+`
builtin, `maplist`, `predsort` (captures A3 first and saves/restores around
the comparator), `call_goal_value`/`call_goal_key`, the dynamic-clause
solver (works on clones; the solution continues at the compiled
continuation), lowered `call`/`execute` and `lowered_dispatch` (lowered code
reads registers exactly as its WAM does, so the WAM check covers it),
`par_aggregate` and `run_parallel` (clones), the parser's
`run_named_label` (a separate machine), the pkg_resolver shim (reads the
answer through a named variable, not a register). After returning, each reads
only its Rust locals or a snapshot it took, so its continuation is the
compiled code after its `call`/`builtin_call`: case 2.

### How the change keeps the argument true

1. **The generator checks cases 1 and 2** (`rust_r0a_instr_options/4` in
   `wam_rust_target.pl`). It runs a backward A/X liveness pass over every
   shared-table predicate's WAM (interpreted and lowered; after the R−1c
   rewrite and the embedded-aggregate splice, i.e. the code that is
   emitted) and emits `TryMeElseArgs` only if
   * at the predicate's entry and at every retry label `L_P_n_k` (k ≥ 2)
     only A1..An are live, and
   * after every instruction that can run predicate code (`call`,
     `builtin_call` other than `fail/0`, `call_foreign`) no A/X register is
     live; in a lowered predicate a call inside an ITE condition also counts
     the else branch's live registers.

   Modelling choices: `call P/n`, `execute`, `builtin_call` read A1..An;
   `begin_aggregate` reads its result register and has the code after the
   matching `end_aggregate` as successor (finalisation backtracks into the
   full-save aggregate frame); `end_aggregate` reads its value register and
   falls nowhere; `switch_on_*` read A1 (A2) and jump to every label they
   name; an instruction or label the pass does not know makes it give up.
   The decision is per program: one violation keeps the full save for every
   clause choice point and is printed
   (`R0a: clause choice points keep the full register save: <reason>`).
   `clause_cp_arity_save(false)` turns R0a off; `clause_cp_arity_save(unchecked)`
   skips the check (tests only).
2. **Case 3 by a runtime flag.** `WamState::clause_cp_full_save` makes
   `TryMeElseArgs` take the full `save_regs()`. `include/3`, `exclude/3`,
   `partition/4` and `foldl/4,5` run each goal through
   `call_goal_once_full_save`, which sets it for the duration of the goal
   and restores it; the benchmark `main.rs` sets it around its
   `dimension_n/1` run. Inside those extents every clause choice point saves
   exactly what it saved before R0a. `reset_query` clears the flag.

### The R−1 changes now on main

* **Backtrack floors (D130).** A floor limits which choice point a nested run
  may resume; a resumed clause choice point restores the same state as
  before except the registers argued above. The floor-protected meta-calls
  are case 3 and were audited.
* **Per-frame aggregate accumulators (D131).** Aggregate frames keep the
  full save; the value register read after the inner goal is a use in the
  liveness pass (the count-aggregate shape above is how it shows up).
* **ITE Y-initialisation rewrite (D132).** The pass runs on the rewritten WAM;
  it touches only Y registers, which live in env frames and are restored by
  the stack snapshot.
* **B0 in env frames (D133).** `TryMeElseArgs` saves `cut_barrier` and parks
  `pending_cut_barrier` exactly as the clause `TryMeElse` did.
* **catch/3 opaque to cut (D134).** `catch/3` keeps its own full register
  snapshot for its failure and throw paths; its success path continues at the
  compiled continuation (case 2).
* **Lowered/functions mode.** Lowered predicates are in the pass (their WAM is
  the same shared-table code), with the else-branch rule above;
  `lo_clause_snapshot` and the lowered ITE marks are unchanged.

### What the resolvers get

* `resolver.pl` (term lane) passes the check: **63** clause-entry choice
  points use `TryMeElseArgs`; 98 ITE/disjunction guards and 3 dispatch-chain
  `try` choice points keep the full save.
* `resolver_store.pl` (store lane) does not:
  `live_across_call(assert_catalog_store/2, 85, "call", ["X23"])` — a
  `forall/2` at the first goal of a clause. The store crate keeps the full
  save everywhere, so its Ir does not move. A per-callee exclusion (or making
  the compiler classify those variables as permanent) would let it qualify;
  both are follow-ups, not part of R0a.

## Tests

`tests/test_wam_rust_r0a_clause_cps.pl` (new, 90 tests; the R−1 harness
pattern: each program compiled through `write_wam_rust_project/3`, run by a
failure-driven driver, compared with SWI; interpreter and functions mode, plus
the direct lowered entry). Two groups, each its own crate:

* **safe** (14 programs): clause choice points retried after head
  unification wrote temporaries and failed (arity 3), after the caller
  continued (through a tail call), arity 0 and arity 6, behind first-argument
  indexing, inside and around ITEs, with a cut, in a first-position
  disjunction, under findall/aggregate_all/setof and a nested findall, inside
  `catch/3` (fail, throw, succeed), under `call/N`, `maplist`, `\+`,
  `between`, and under `include`/`exclude`/`partition`/`foldl` (case 3). The
  test also asserts the generated crate uses `TryMeElseArgs` for every clause
  chain of the group and that ITE guards keep `TryMeElse`.
* **full_save** (4 programs): the forall, `\+`-conjunction,
  `once`-conjunction and count-over-0-arity shapes. The test asserts the
  crate has no `TryMeElseArgs`, matches SWI, and that a third build with
  `clause_cp_arity_save(unchecked)` **diverges from SWI on every one of them**.

Template model tests (`mod d135_clause_cp_arg_save_tests`, 6): `save_arg_regs`
equals `save_regs` restricted to slots `< arity` over 3,000 random register
operations and 12 arities; a retry through `TryMeElseArgs` restores A1..An and
leaves a temporary clause 1 wrote, and registers set before the entry,
`Uninit`; `TryMeElse` with a clause label and with an `L_ite_else_` label
still restores every register set before it; the PC-resolved form saves the
same; the flag makes the arity-only form save everything; a compiled
three-clause search gives the same answer either way.

**Do the tests exercise the choice point path?** On the base commit the
SWI comparisons of both groups pass (the programs are correct today) and the
R0a-specific assertions fail (no `TryMeElseArgs` exists). Mutation runs, each
on a fresh copy of this tree:

| mutation | result |
| --- | --- |
| clause choice points save A1..A(n−1) | 12 of the 14 safe programs diverge from SWI, in both modes (24 failures); 3 template tests fail |
| include/exclude/partition/foldl without the flag | 2 plunit failures (`list_meta`, both modes) |
| ITE guards save no registers | template test `full_save_choice_points_still_restore_every_live_register` fails (the compiler keeps ITE-visible variables in Y registers, so the plunit programs do not see it) |
| generator skips the liveness check | all 4 full_save programs diverge from SWI in both modes (8 failures), and the 4 "keeps the full save" assertions fail |

Found while writing the tests, pre-existing on base and outside R0a (not
changed): a first-argument group that does not contain clause 1 is reached
through a `try/retry/trust` dispatch chain, which this runtime does not run
(`p(1,p). p(2,q). p(2,r).` gives no answer for `p(2,Y)`); `include/3` with a
goal of arity ≥ 3 reads its A3 back after `call_goal_key` overwrote it
(`include(btw(1,5), [2,7], L)` fails, SWI `[2]`); two lowered direct entries
are wrong on base too (a lowered ITE condition that needs to retry a lowered
call; lowered aggregates), and are excluded from the lowered check with that
reason.

## Gates

`LANG=C.utf8`, from the repo root; base = a clean export of `5f08375` built
the same way.

| gate | result |
| --- | --- |
| `rust/run_differential_rust.sh` | 2600 / 0 / 0 |
| `rust/run_corpus_rust.sh` | 51/51 |
| `rust_store/run_differential_rust_store.sh` | 503 / 0 / 0 |
| `rust_store/run_corpus_rust_store.sh` | 51/51, IDENTICAL to term |
| byte identity vs base | `cmp`-identical: term diff `rust.jsonl` (2600), term corpus (51), store diff (503), store corpus (51); scale `--bench` stdout at N=40/250/1000/5000 |
| generated crate `cargo test --release --lib` | 266 passed, 0 failed (260 + 6 new) |
| `tests/test_wam_rust_baseline_fixes.pl` | 92/92 |
| `tests/test_wam_rust_r0a_clause_cps.pl` | 90/90 |
| Rust WAM plunit (`tests/test_wam_rust_*.pl`) | 53 base files: same per-file rc (32 rc=0, 21 rc≠0) and the same 30 failing test names as base; plus the new file, 90/90. One file (`test_wam_rust_par_aggregate.pl`) had a timing-dependent cargo test (`expensive_parallel_is_faster` / `gate_stays_sequential_when_cheap`) fail in both runs under load, a different one each time; rerun idle it passes on base and new |
| CI rust conformance (`CONFORMANCE_TARGETS=rust`, `member,builtins`) | rc=0 unsampled and with `CONFORMANCE_SAMPLE=2` |

## Measurements

### Ir per warm resolve (callgrind, deterministic)

In-process repeats in a scratch copy of the shim (not committed):
(Ir(1+k) − Ir(1)) / k, k = 10 at N=40 and 2 at N=5000, every repeat checked
equal to the first.

| N | base | R0a | Δ |
| ---: | ---: | ---: | ---: |
| 40 | 23,100,296 | **21,061,770** | **−2.04 M (−8.8%)** |
| 5000 | 388,246,801 | **386,239,950** | −2.01 M (−0.5%) |

Where it went at N=40 (inclusive, per warm resolve):

| function | base | R0a |
| --- | ---: | ---: |
| `save_regs` (+ `save_arg_regs`) | 0.97 M | 0.29 M + 0.08 M |
| `restore_ax_regs` | 1.70 M | 0.46 M |
| `backtrack` | 3.32 M | 2.05 M |
| `drop_in_place<ChoicePoint>` | 0.38 M | 0.23 M |
| `step` | 18.96 M | 18.19 M |

The saving is flat in N, as the spike predicted (the choice-point counts do
not depend on N). It is a little larger than the spike's s1→s2 step
(−1.84 M), which was measured on top of heap-cell variables.

### Wall clock

Base and new interleaved (alternating order per round), 21 rounds at
N=40/250 and 9 at N=1000/5000. "Cold" is the shim's own `resolve_ms` from a
fresh process; "warm" is the median of 11 in-process repeats, then the median
over the rounds. The box is a shared 4-core container (load average 3–5
during the run), so treat these as indicative; Ir is the reliable metric.

| N | rounds | cold base | cold new | warm base | warm new | Δ warm |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 40 | 21 | 5.03 ms | 4.61 ms | 3.51 ms | **3.04 ms** | −13% |
| 250 | 21 | 6.43 ms | 5.95 ms | 5.47 ms | **5.02 ms** | −8% |
| 1000 | 9 | 12.98 ms | 12.81 ms | 12.31 ms | 11.59 ms | −6% |
| 5000 | 9 | 50.64 ms | 52.26 ms | 53.61 ms | 53.37 ms | (noise) |

## Files

* `templates/targets/rust_wam/instructions.rs.mustache`: `TryMeElseArgs`,
  `TryMeElsePcArgs`.
* `templates/targets/rust_wam/state.rs.mustache`: `save_arg_regs`,
  `clause_cp_full_save`, `mod d135_clause_cp_arg_save_tests`.
* `templates/targets/rust_wam/main.rs.mustache`: the flag around the
  `dimension_n/1` read-back.
* `src/unifyweaver/targets/wam_rust_target.pl`: the two step arms, the
  `GetLevel` look-ahead, the clause-entry `try_me_else` emission, the liveness
  check (`rust_r0a_*`), `call_goal_once_full_save` for
  include/exclude/partition/foldl.
* `examples/benchmark/generate_wam_*_benchmark.pl`: resolve `TryMeElseArgs`
  to `TryMeElsePcArgs`.
* `tests/test_wam_rust_r0a_clause_cps.pl` (new).

`resolver.pl` and `resolver_store.pl` are unmodified. The regenerated crate is
not committed.
