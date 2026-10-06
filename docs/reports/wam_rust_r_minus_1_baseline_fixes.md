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

## R−1a (D130): a backtrack floor for every nested execution

**Defect.** `backtrack()` never resumes a choice point at or below
`backtrack_floor` (D71). `call_goal_once` sets the floor to its entry depth
around a meta-call, but four other places run nested code with no floor, so a
goal that failed inside the nested `run()` backtracked into the caller's
choice points from inside the nested run (rule L3 of the design, §5.2):

| site | nested execution | what went wrong |
| --- | --- | --- |
| `catch/3` (`wam_rust_target.pl`, `execute_builtin`) | the goal and, after a caught ball, the recovery goal, both through `call_goal_value` → `call_goal_key` → `run()` | `catch_floor`: the failing goal resumed the caller's second clause inside the nested run; the driver then failed |
| lowered `call` / `execute` (`wam_rust_lowered_emitter.pl`) | `vm.run()` of the callee | `lower_floor` (functions mode): the failing `lower_bad(2)` resumed the driver's choice points |
| `lowered_dispatch` (`state.rs.mustache`) | the whole lowered body, entered from an interpreted `call`/`execute` | raised only `cut_barrier`, which bounds `!` but not failure |
| dynamic rule bodies (`dynamic_db_methods.rs.mustache`, `dynamic_body_solutions`) | `call_goal_value` of a body goal on a cloned machine | `dyn_floor`: `dq(X) :- dbad(X)` with `dbad/1` failing resumed the caller's clause CP inside the clone (this confirms the design's suspected cause, §16) |

**Change.** Each site saves `backtrack_floor`, sets it to its entry depth
(`choice_points.len()` when the nested execution starts), and restores it
on every exit: after the call returns for `catch/3` (goal and recovery
separately) and lowered `call`/`execute`; after `f(self)` in
`lowered_dispatch`, before its decline/success split; and in the dynamic
solver each collected solution machine gets the caller's floor back (the
solver replaces `*self` with one of them).

**Not changed, with reasons.**

- The six hybrid-wrapper `vm.backtrack()` loops in `rust_target.pl` (§8 of
  the design). Every wrapper body starts with `vm.reset_query()`, which
  clears `choice_points` and sets `backtrack_floor = 0`, and then replaces
  `code` and `labels`. When the loop runs, the only choice points that exist
  are the ones the kernel pushed inside the wrapper, so an exhausted stream
  cannot resume an older caller CP: there is none. A floor at the entry depth
  would be 0, the value `reset_query` already set. This answers the design's
  open question (§16) for the current code; the rewrite's scope for the
  wrappers (§8) still applies.
- The general `\+/1` builtin arm (`execute_builtin`) runs a nested `run()`
  with a `naf_succeed` sentinel CP that `resume_builtin` does not handle, so
  a failing goal can pass the sentinel. Under the Rust target's default
  options the compiler does not emit it (`\+` is inlined as an ITE, and
  `call/N` reaches `\+` through `call_goal_once`, which is floored), so no
  compiled program in the harness or the resolver reaches it; noted here for
  the rewrite, not changed (it is outside the R−1a site list).

**Tests.** New harness programs: `catch_floor`, `catch_rec_floor` (the
recovery goal fails after a caught `throw`), `lower_floor`, `lower_floor_ex`
(the failing callee is reached by a lowered `execute`), and `dyn_floor`
(driver runs `assertz((dq(Y) :- dbad(Y)))` first). Before the fix:
`catch_floor`, `catch_rec_floor` and `dyn_floor` printed `ok` and then the
driver failed in both modes; `lower_floor` and `lower_floor_ex` did the same
in functions mode (the interpreter was already right). After: all match SWI
in both modes, and the direct lowered calls match `once/1`.

**Gates.** Term differential 2600/0/0, term corpus 51/51, store differential
503/0/0, store corpus 51/51 and identical to the term corpus, lib 260/260,
CI rust conformance smoke rc=0 (unsampled and sample 2). **Byte identity:**
the generated resolver crates differ from base only by the floor code above
(`lowered_dispatch`, `catch/3`, the dynamic solver); all four output JSONLs and
scale `--bench` stdout (N=40, 5000) are `cmp`-identical to the base, so the
frozen baseline is unchanged. **Perf:** callgrind N=40 `--bench` 34.43 M Ir,
identical to D129 (base 34.41 M; +0.05%, the noise floor). **Plunit:** same
per-file results and failing names as after D129, except
`test_wam_rust_par_aggregate`'s `expensive_parallel_is_faster`, a wall-clock
assertion (parallel vs sequential, untouched code) that failed once while the
gate builds loaded the machine and passed on two reruns of the file alone.

## R−1b (D131): one accumulator per active aggregate frame

**Defect.** Inlined `findall/3`, `bagof/3`, `setof/3` and `aggregate_all/3`
compile to `BeginAggregate … EndAggregate`. `BeginAggregate` pushes an
aggregate-frame choice point and `EndAggregate` appends each solution to the
machine's single `aggregate_acc`. `BeginAggregate` started with
`aggregate_acc.clear()` and finalisation cleared it again, so an aggregate
nested inside another aggregate's goal wiped the outer one's solutions so
far: `nested(L)` gave `[b]` where SWI gives `[a,b]`, in both emit modes.

**Change** (`wam_rust_target.pl` only: the `BeginAggregate` step arm and the
`aggregate_frame` arm of `resume_builtin`; one step implementation serves
both emit modes). `BeginAggregate` moves the enclosing accumulator out
(`std::mem::take`) and parks it in the frame's `BuiltinState.data` after the
continuation pc (`data = [ret_pc, outer…]`; with no enclosing aggregate this
is the same one-element vector as before). Finalisation first splits the
parked values off `data` and swaps them back into `aggregate_acc`, then
computes the result from the frame's own list. The swap comes before every
exit, so a `bagof`/`setof` that fails on an empty set, an unknown aggregate
type or a result that does not unify all leave the enclosing frame's list
intact. `data[0]` (the continuation pc) is read as before. The finalisation
now moves the list into the result for `collect`/`bagof` instead of cloning
it, which yields the same list.

**Not changed.** A frame removed without finalisation (an exception thrown
out of the aggregate's goal) still leaves `aggregate_acc` holding the
abandoned inner list, as before; `catch/3` does not restore it. The
parallel aggregate path (`par_aggregate.rs`) runs on forked machines and is
unaffected.

**Tests.** New harness programs: `nested` (the design's case), `nested_bag`
(the inner result is used: `findall(p(A,Bs), (nested_a(A), findall(B,
nested_b(A,B), Bs)), L)`), and `nested_fail` (an inlined inner `bagof` that
fails on an empty set, reached through a helper clause, must not lose the
outer list). Before: `[b]`, `[p(b,[two])]` and `[b]` in both modes; after:
equal to SWI in both modes. The direct lowered entries of these three
programs are not checked: the lowered emitter has no `begin_aggregate` /
`end_aggregate` support at all and drops both instructions, so a lowered
`findall` body runs its goal once as a plain conjunction (for example
`lowered_nested_1` returns `L = a`). That is a separate lowered-tier defect,
outside R−1; it is reported here and left unchanged.

**Gates.** Term differential 2600/0/0, term corpus 51/51, store differential
503/0/0, store corpus 51/51 and identical to the term corpus, lib 260/260,
CI rust conformance smoke rc=0 (unsampled and sample 2). **Byte identity:**
the generated crates differ from D130 only in the two aggregate arms; all four
output JSONLs and scale `--bench` stdout (N=40, 5000) are `cmp`-identical to
the base, so the frozen baseline is unchanged. **Perf:** callgrind N=40
`--bench` 34.41 M Ir (D130 34.43 M, base 34.41 M): no measurable change.
**Plunit:** same per-file results and failing names as after D130, including
the `expensive_parallel_is_faster` timing flake. That test compiles only
`src/unifyweaver/targets/rust_runtime/par_aggregate.rs` (a standalone toy
machine that no R−1 commit touches) and asserts parallel wall time below
sequential; it failed again under gate load and passed 1 of 2 reruns alone.

## R−1c (D132): permanent variables first seen inside an if-then-else (Rust only)

**Defect.** After `( C -> T ; E )` the shared compiler continues with the
Then branch's variable map (`compile_if_then_else/7` in `wam_target.pl`:
`Vf = V2` in both cases; `compile_disjunction/6` does the same with the left
branch's map). A permanent variable whose first occurrence is inside the
construct is therefore written by a first-occurrence instruction on one path
only. Two shapes:

- **first seen in the condition** (the design's case, §6.2 item 3):
  `ite_map(A,R) :- (ite_q(A,X) -> true ; true), ite_r(X,R)` compiles the
  condition's `X` as `put_variable Y2, A2` after the guard. When `ite_q`
  fails, backtracking to the guard restores the environment snapshot, Y2 is
  absent again, and the `put_value Y2, A1` after the ITE fails. SWI leaves
  `X` unbound and gives `[a,b]` for `ite_map(no,R)`; the interpreter gave
  `[]`. `rundoite(c,els)` failed in the interpreter for the same reason,
  while the lowered tier (whose condition writes are not undone, §6.6)
  succeeded, so the two Rust tiers disagreed.
- **first seen in the else branch** (found while writing the tests):
  `ite_els_only(A,R) :- (A == 1 -> true ; X = one), R = X` keeps the Then
  map, which lacks `X`, so the `R = X` after the ITE is compiled as another
  first occurrence (`put_variable`) and replaces the else branch's binding
  with a fresh variable. SWI gives `one` for `ite_els_only(2,R)`; both Rust
  tiers gave `_V2`. Disjunctions share the label shape and the rule:
  `dis_map(yes,R) :- (ite_q(yes,X) ; true), ite_r(X,R)` gave `[a]` for SWI's
  `[a,a,b]`.

**Change** (Rust only, as the owner decided; `wam_target.pl` is unchanged).
`classify_predicates/3` in `wam_rust_target.pl` passes each predicate's WAM
text through the new `rust_ite_init_permanent_vars/2` right after
`compile_predicate_to_wam/3`, so the shared instruction table and the lowered
emitter both see the rewritten text. Clauses are delimited by any label
that is not an `L_ite_` label (Y numbering is per clause). A construct is
`try_me_else L_ite_else_N … L_ite_else_N: … L_ite_cont_N:`, and its *scope*
is the code its guard dominates: from the guard to the end of the enclosing
construct's branch that contains it (the enclosing `L_ite_else_M:` label for
a Then/left branch, the enclosing `L_ite_cont_M:` for an Else/right branch),
or to the clause end at top level. For every `Yn` whose first occurrence
lies strictly inside a construct and that occurs again after the construct
but within its scope, it:

1. inserts `put_variable Yn, Yn` before the outermost such construct's guard
   (before its `get_level` when there is one). This self-init form is the one
   the shared compiler already emits for aggregate result variables; it puts
   a fresh variable in the Y slot only.
2. turns every first-occurrence instruction for `Yn` inside that scope
   (`get_variable`, `put_variable`, `unify_variable`, `set_variable`) into its
   `_value` form. With `Yn` already holding an unbound variable, each value
   form does what the variable form did on the path that reached it, and the
   value survives into the continuation on every path.

Occurrences outside the scope sit in a sibling branch of an enclosing
construct, on paths that never pass the guard; they keep their
first-occurrence form. The scope limit matters: a first version of this
change hoisted the init only past "after the continuation label", and in the
resolver's `resolve_pending/6` (`Ver` and `DepReqs` first seen in an ITE
nested in one branch of an outer ITE and seen again in the outer ITE's other
branch) it turned the other branch's `put_variable` into a `put_value` of an
absent slot: every resolve failed in the differential. The harness case
`ite_sibling_else` reproduces that shape and failed on the first version.

The guard's choice point is pushed after the init, so its snapshot includes
`Yn`; a failed condition restores `Yn` to the fresh variable and undoes its
bindings through the trail. Barrier registers (`get_level`/`cut` operands)
are never candidates. Predicates without an `L_ite_else_` label are returned
unchanged. This is the design's preferred form (§12.1: "initialize it before
the guard"), and it gives the lowered ITE condition the plain policy of §6.6:
no slot read after the condition is first written inside it.

For `ite_map` the rewritten clause is:

```
allocate / get_variable Y1, A1 / get_variable Y3, A2
put_variable Y2, Y2                      % R-1c init
get_level Y4 / try_me_else L_ite_else_1
put_value Y1, A1 / put_value Y2, A2      % was put_variable Y2, A2
call ite_q/2, 2 / cut Y4 / ... / L_ite_cont_1:
put_value Y2, A1 / put_value Y3, A2 / deallocate / execute ite_r/2
```

**Tests.** New harness programs: `ite_map` (`ite_map(no,R)`, the design's
query) and `ite_map_yes` (the condition succeeds), `rundoite` (query
`rundoite(c,els)`), `ite_els_only`, `dis_map`, and `ite_sibling_then` /
`ite_sibling_else` (the scope rule above). `ite_map_yes` and the two sibling
cases already pass on the base; they guard against the rewrite breaking a
working shape. Before: `ite_map` `[]` in both
modes (SWI `[a,b]`); `rundoite` no solution in the interpreter (SWI and the
lowered tier: true); `ite_els_only` `_V2` in both modes and `_V1` from the
direct lowered call (SWI `one`); `dis_map` `[a]` (SWI `[a,a,b]`). After: all
equal to SWI in interpreter and functions mode, and the direct lowered calls
equal `once/1`, so the interpreter and the lowered tier agree on `rundoite`.
`test_wam_rust_lowered_ite_exec.pl`, which pins the lowered
`rundoite(c,els)` = true and `rundoite(c,then)` = false, still passes.

**Gates.** Term differential 2600/0/0, term corpus 51/51, store differential
503/0/0, store corpus 51/51 and identical to the term corpus, lib 260/260,
CI rust conformance smoke rc=0 (unsampled and sample 2). **Byte identity:**
the rewrite touches nine resolver predicates (`audit_holds/4`,
`blocked_acc/5`, `direct_on/4`, `filter_satisfies/3`,
`keep_installed_or_base/4`, `matching_deps/4`, `matching_versions/4`,
`removal_orphans/3`, `scan_base_holds/3`: one init each, the shape where a
variable is first written in both branches of an ITE and read after it), so
instruction indices in the generated table shift; all four output JSONLs and
scale `--bench` stdout (N=40, 5000) are still `cmp`-identical to the base,
so the frozen baseline is unchanged. Those predicates already worked because
both branches wrote the variable. **Perf:** callgrind N=40 `--bench`
34.42 M Ir (base 34.41 M, +0.02%). **Plunit:** same per-file results and
failing names as after D129 (32 / 20 files, 29 failing tests; the
`par_aggregate` timing test passed with the gate builds no longer running
concurrently).
