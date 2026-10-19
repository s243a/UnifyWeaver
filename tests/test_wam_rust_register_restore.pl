:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (s243a)
%
% test_wam_rust_register_restore.pl
%
% Test net for phase R0b of docs/proposals/wam_rust_heap_cell_rewrite_design.md
% (section 12): removing A/X register entries from the Rust WAM trail. The
% audit is docs/reports/wam_rust_r0b_register_trail_audit.md.
%
% Every program below runs a goal that loads argument and temporary registers
% (through put_constant/put_value/get_variable, which pushed register trail
% entries before R0b) and then FAILS, or succeeds inside a scope that is
% rolled back, and afterwards reads an argument or a temporary. Each program
% covers one kind of non-choice-point rollback (or one choice-point kind next
% to it, for contrast). Every checked case matches SWI-Prolog. Before R0b,
% disabling the A/X register pushes in a scratch build failed exactly the
% interpreter include/exclude/partition checks: those builtins read their
% output argument register after rolling back each element call, and only
% the register trail gave it back (see the audit). R0b removed the A/X
% entries and made those builtins (and foldl/4,5) read their output
% arguments at entry, which also fixed their functions-mode and
% extended-goal divergences. Programs that differ from SWI-Prolog for other
% reasons are listed in rr_known_divergence/3 (none since bug-fix batch 1).
%
% The harness is the one of tests/test_wam_rust_baseline_fixes.pl: one source
% of truth (rr_clause/2), asserted into user: for SWI and handed to
% write_wam_rust_project/3; a failure-driven driver
%     rr_drv_<name> :- Setup, ( Query, write(Out), nl, fail ; true ).
% is compiled with the program and run by its WAM label in
% emit_mode(interpreter) and emit_mode(functions); its stdout and success
% flag must equal SWI's. When emit_mode(functions) lowered the query
% predicate, its Rust function is also called directly and compared with
% SWI's first solution (the lowered tier is first-solution by contract).
%
%   LANG=C.UTF-8 swipl -q -g run_tests -t halt tests/test_wam_rust_register_restore.pl
%
% (A UTF-8 locale is needed: the generator reads templates that contain
% non-ASCII text.)

:- module(test_wam_rust_register_restore,
          [test_wam_rust_register_restore/0]).

:- use_module(library(plunit)).
:- use_module(library(lists)).
:- use_module(library(apply)).
:- use_module(library(filesex), [make_directory_path/1,
                                 delete_directory_and_contents/1]).
:- use_module(library(process)).
:- use_module(library(readutil)).
:- use_module('../src/unifyweaver/targets/wam_rust_target',
              [write_wam_rust_project/3]).
:- use_module('../src/unifyweaver/targets/wam_rust_lowered_emitter', []).

% ---------------------------------------------------------------------
% Programs. rr_program(Name, Kind, Query, Out, Setup)
%   Kind  : the audited rollback kind the program exercises.
%   Query/Out : the query whose solutions are printed (write(Out)).
%   Setup     : goal run once by the driver before the query loop.
% ---------------------------------------------------------------------

:- discontiguous rr_program/5, rr_clause/2.

% Shared helpers. A call to rr_w/5 loads A1..A5; rr_clob/1 does that and
% fails, rr_clob_ok/1 does that and succeeds. Both overwrite every argument
% register the callers below keep live, through instructions that push
% register trail entries today (get_variable, put_constant, put_value).
rr_clause(common, rr_w(_, _, _, _, _)).
rr_clause(common, (rr_clob(X) :- rr_w(k1, k2, k3, k4, X), fail)).
rr_clause(common, (rr_clob_ok(X) :- rr_w(k1, k2, k3, k4, X))).
rr_clause(common, (rr_keep(X) :- rr_clob_ok(X), X > 1)).
rr_clause(common, (rr_pos(X) :- rr_clob_ok(X), X > 0)).

% \+ : inline (compiled to a guard choice point), meta-called (the structural
% `\+` of call_goal_value: trail-only rollback), and a temporary read after it.
rr_program(naf_inline, naf, rr_naf_inline(a, R), R, true).
rr_program(naf_meta,   naf, rr_naf_meta(a, R),   R, true).
rr_program(naf_temp,   naf, rr_naf_temp(a, b, R), R, true).
rr_clause(naf_inline, (rr_naf_inline(X, R) :- ( \+ rr_clob_ok(X) -> R = no ; R = yes(X) ))).
rr_clause(naf_meta,   (rr_naf_meta(X, R) :- rr_naf_goal(rr_clob_ok(X), G), ( call(G) -> R = no ; R = yes(X) ))).
% Builds \+(Inner) at run time. A `\+` written as a term in a compiled body
% is emitted as the Rust literal "\+/1", which does not compile (see the
% audit); building the functor from its codes avoids that literal.
rr_clause(common, (rr_naf_goal(Inner, G) :- atom_codes(F, [92, 43]), G =.. [F, Inner])).
rr_clause(naf_temp,   (rr_naf_temp(X, Y, R) :- \+ rr_clob(X), R = got(X, Y))).

% call/N: a failed left branch of a meta-called disjunction (`;` rollback in
% call_goal_value), a meta-called if-then-else whose condition fails after
% clobbering (the `->` rollback), and call/2 of a failing goal.
rr_program(call_disj, call, rr_call_disj(a, R), R, true).
rr_program(call_ite,  call, rr_call_ite(a, R),  R, true).
rr_program(call_n,    call, rr_call_n(a, R),    R, true).
rr_program(call_once, call, rr_call_once(a, R), R, true).
rr_clause(call_disj, (rr_call_disj(X, R) :- call((rr_clob(X) ; true)), R = got(X))).
rr_clause(call_ite,  (rr_call_ite(X, R) :- call(( ( rr_clob_ok(X), fail ) -> R = then ; R = els(X) )))).
rr_clause(call_n,    (rr_call_n(X, R) :- ( call(rr_clob, X) -> R = t ; R = e(X) ))).
rr_clause(call_once, (rr_call_once(X, R) :- ( once(rr_clob(X)) -> R = t ; R = e(X) ))).

% catch/3: the goal fails after clobbering (catch fails; caller's fallback),
% the goal throws after clobbering (recovery reads the caller's variable),
% and the recovery fails after clobbering.
rr_program(catch_fail,  catch, rr_catch_fail(a, R),  R, true).
rr_program(catch_throw, catch, rr_catch_throw(a, R), R, true).
rr_program(catch_rec,   catch, rr_catch_rec(a, R),   R, true).
rr_clause(catch_fail,  (rr_catch_fail(X, R) :- ( catch(rr_clob(X), _, true) -> R = caught ; R = failed(X) ))).
rr_clause(catch_throw, (rr_catch_throw(X, R) :- catch(( rr_clob_ok(X), throw(oops) ), E, R = rec(E, X)))).
rr_clause(catch_rec,   (rr_catch_rec(X, R) :- ( catch(throw(oops), _, rr_clob(X)) -> R = t ; R = e(X) ))).

% findall/3 over a goal that clobbers and then fails, or that clobbers on
% every solution; aggregate_all/3 nested inside findall/3.
rr_program(findall_fail, aggregate, rr_findall_fail(a, R), R, true).
rr_program(findall_some, aggregate, rr_findall_some(a, R), R, true).
rr_program(nested_agg,   aggregate, rr_nested_agg(a, R),   R, true).
rr_program(nested_findall, aggregate, rr_nested_findall(a, R), R, true).
rr_clause(findall_fail, (rr_findall_fail(X, R) :- findall(Y, ( rr_clob_ok(X), Y = X, fail ), L), R = f(X, L))).
rr_clause(findall_some, (rr_findall_some(X, R) :- findall(Y, ( member(Y, [1, 2]), rr_clob_ok(Y) ), L), R = f(X, L))).
rr_clause(nested_agg,   (rr_nested_agg(X, R) :- findall(c(A, C), ( member(A, [p, q]), rr_count(A, C) ), L), R = r(X, L))).
rr_clause(nested_agg,   (rr_count(A, C) :- aggregate_all(count, ( rr_clob_ok(A), member(_, [1, 2]) ), C))).
% the same with aggregate_all/3 written inside the findall/3 goal (compiled
% as `call aggregate_all/3`, the runtime builtin since bug-fix batch 1, B3)
rr_program(nested_agg_inline, aggregate, rr_nested_agg_inline(a, R), R, true).
rr_clause(nested_agg_inline, (rr_nested_agg_inline(X, R) :- findall(c(A, C), ( member(A, [p, q]), aggregate_all(count, ( rr_clob_ok(A), member(_, [1, 2]) ), C) ), L), R = r(X, L))).
rr_clause(nested_findall, (rr_nested_findall(X, R) :- findall(c(A, Bs), ( member(A, [p, q]), findall(B, ( rr_clob_ok(A), member(B, [1, 2]) ), Bs) ), L), R = r(X, L))).
% findall over a goal that clobbers and fails on every candidate, nested in
% an outer findall whose next solution reads the caller's variable
rr_program(nested_fail, aggregate, rr_nested_fail(a, R), R, true).
rr_clause(nested_fail, (rr_nested_fail(X, R) :- findall(c(A, Bs), ( member(A, [p, q]), findall(B, ( member(B, [1, 2]), rr_clob(A) ), Bs) ), L), R = r(X, L))).

% forall/2 (compiled to a double negation), holding and failing.
rr_program(forall_ok,   forall, rr_forall_ok(a, R),   R, true).
rr_program(forall_fail, forall, rr_forall_fail(a, R), R, true).
rr_clause(forall_ok,   (rr_forall_ok(X, R) :- ( forall(member(E, [1, 2]), rr_clob_ok(E)) -> R = all(X) ; R = none(X) ))).
rr_clause(forall_fail, (rr_forall_fail(X, R) :- ( forall(member(E, [1, 2]), rr_keep(E)) -> R = all(X) ; R = none(X) ))).

% predsort/3: a comparator that clobbers every call, and one that fails
% partway (predsort fails; the caller's fallback reads its argument).
rr_program(predsort_ok,   predsort, rr_predsort_ok(a, R),   R, true).
rr_program(predsort_fail, predsort, rr_predsort_fail(a, R), R, true).
rr_clause(predsort_ok,   (rr_predsort_ok(X, R) :- predsort(rr_cmp, [3, 1, 2], S), R = s(X, S))).
rr_clause(predsort_fail, (rr_predsort_fail(X, R) :- ( predsort(rr_cmp_bad, [3, 1, 2], S) -> R = s(S) ; R = failed(X) ))).
rr_clause(predsort_ok,   (rr_cmp(O, A, B) :- rr_clob_ok(A), compare(O, A, B))).
rr_clause(predsort_fail, (rr_cmp_bad(O, A, B) :- rr_clob_ok(A), A \== 1, B \== 1, compare(O, A, B))).

% include/3, exclude/3, partition/4: each element's test is meta-called and
% rolled back (trail only); the builtin reads its output argument afterwards.
% maplist/2 with an element that fails, and the caller's fallback.
rr_program(include,   filter, rr_include(R),       R, true).
rr_program(exclude,   filter, rr_exclude(R),       R, true).
rr_program(partition, filter, rr_partition(R),     R, true).
rr_program(maplist_fail, filter, rr_maplist_fail(a, R), R, true).
rr_clause(include,   (rr_include(R) :- include(rr_keep, [1, 2, 3], R))).
rr_clause(exclude,   (rr_exclude(R) :- exclude(rr_keep, [1, 2, 3], R))).
rr_clause(partition, (rr_partition(R) :- partition(rr_keep, [1, 2, 3], I, E), R = parts(I, E))).
rr_clause(maplist_fail, (rr_maplist_fail(X, R) :- ( maplist(rr_keep, [2, 1]) -> R = all ; R = some(X) ))).
% foldl/4 and include/3 whose extended goal has three or more arguments: the
% element calls overwrite the output register (A4, A3). Before R0b the
% builtin read it after the calls and diverged from SWI; R0b reads it at
% builtin entry.
rr_program(foldl,        filter, rr_foldl(R),        R, true).
rr_program(include_wide, filter, rr_include_wide(R), R, true).
rr_clause(foldl, (rr_foldl(R) :- foldl(rr_add, [1, 2, 3], 0, R))).
rr_clause(foldl, (rr_add(X, A0, A) :- rr_clob_ok(X), A is A0 + X)).
rr_clause(include_wide, (rr_include_wide(R) :- include(rr_between(1, 2), [1, 5, 2], R))).
rr_clause(include_wide, (rr_between(L, H, X) :- X >= L, X =< H)).

% Dynamic rule bodies: a body that clobbers and fails before the next clause
% is tried, and a body that runs a \+ over a clobbering goal.
rr_program(dyn_body, dynamic, rr_dq(a, R), R,
           ( assertz((rr_dq(X, Y) :- rr_clob_ok(X), Y = first, fail)),
             assertz(rr_dq(X0, second(X0))) )).
rr_program(dyn_naf, dynamic, rr_dn(a, R), R,
           ( rr_naf_goal(rr_clob_ok(X), N),
             assertz((rr_dn(X, Y) :- N, Y = no)),
             assertz(rr_dn(X0, yes(X0))) )).

% If-then-else whose condition writes temporaries (a call, an is/2) and then
% fails; a bare if-then (no else) whose condition fails inside a disjunction.
rr_program(ite_temps,  ite, rr_ite_temps(a, b, R), R, true).
rr_program(ite_is,     ite, rr_ite_is(3, R),       R, true).
rr_program(ite_nested, ite, rr_ite_nested(a, R),   R, true).
rr_program(ifthen,     ite, rr_ifthen(a, R),       R, true).
rr_clause(ite_temps,  (rr_ite_temps(X, Y, R) :- ( rr_clob_ok(X), X == zz -> R = t(X, Y) ; R = e(X, Y) ))).
rr_clause(ite_is,     (rr_ite_is(X, R) :- ( Z is X + 1, Z > 100 -> R = big(Z) ; R = small(X) ))).
rr_clause(ite_nested, (rr_ite_nested(X, R) :- ( ( rr_clob_ok(X), X == zz -> true ; rr_clob(X) ) -> R = t ; R = e(X) ))).
rr_clause(ifthen,     (rr_ifthen(X, R) :- ( ( rr_clob_ok(X), X == zz -> R = t ) ; R = e(X) ))).

% Multi-clause predicates whose first clause clobbers and fails: clause
% choice points (interpreter), T4/lo_restore_clause attempts (lowered).
rr_program(multi_call,   multi, rr_multi_call(a, b, R),   R, true).
rr_program(multi_pure,   multi, rr_multi_pure(3, b, R),   R, true).
rr_program(multi_ite,    multi, rr_multi_ite(a, b, R),    R, true).
rr_clause(multi_call, (rr_multi_call(X, Y, R) :- rr_clob_ok(X), X == zz, R = one(Y))).
rr_clause(multi_call, rr_multi_call(X, Y, two(X, Y))).
rr_clause(multi_pure, (rr_multi_pure(X, Y, R) :- Z is X * 2, Z > 100, R = big(Y))).
rr_clause(multi_pure, (rr_multi_pure(X, Y, R) :- R = small(X, Y))).
% a lowered ITE whose condition CALLS a predicate that clobbers A1/A2 and
% whose else branch reads both
rr_clause(multi_ite,  (rr_multi_ite(X, Y, R) :- ( rr_clob_ok(Y), Y == zz -> R = t ; R = e(X, Y) ))).

%% rr_dynamic(?PI)
%  Predicates the drivers create at run time with assertz/1.
rr_dynamic(rr_dq/2).
rr_dynamic(rr_dn/2).

% ---------------------------------------------------------------------
% Installation
% ---------------------------------------------------------------------

rr_driver_name(Name, D) :- atom_concat(rr_drv_, Name, D).

rr_clause_head(Clause, F/A) :-
    ( Clause = (H :- _) -> true ; H = Clause ),
    functor(H, F, A).

rr_program_pis(PIs) :-
    findall(PI, (rr_clause(_, C), rr_clause_head(C, PI)), PIs0),
    findall(D/0, (rr_program(N, _, _, _, _), rr_driver_name(N, D)), PIs1),
    append(PIs0, PIs1, PIs2),
    sort(PIs2, PIs).

install_rr_programs :-
    rr_program_pis(PIs),
    forall(member(F/A, PIs),
           ( functor(H, F, A), catch(retractall(user:H), _, true) )),
    forall(rr_dynamic(F/A),
           ( functor(H, F, A),
             catch(retractall(user:H), _, true),
             dynamic(user:F/A) )),
    forall(rr_clause(_, C), assertz(user:C)),
    forall(rr_program(N, _, Q, Out, Setup),
           ( rr_driver_name(N, D),
             assertz(user:(D :- Setup, ( Q, write(Out), nl, fail ; true ))) )).

rr_rust_preds(Preds) :-
    rr_program_pis(PIs),
    findall(user:PI, member(PI, PIs), Preds).

% ---------------------------------------------------------------------
% SWI oracle
% ---------------------------------------------------------------------

rr_reset_dynamic :-
    forall(rr_dynamic(F/A),
           ( functor(H, F, A), catch(retractall(user:H), _, true) )).

%% swi_all(+Name, -Out, -Ok)
swi_all(Name, Out, Ok) :-
    rr_reset_dynamic,
    rr_driver_name(Name, D),
    with_output_to(string(Raw),
                   ( catch(user:D, _, fail) -> Ok = true ; Ok = false )),
    normalize_output(Raw, Out).

%% swi_once(+Name, -Out)
swi_once(Name, Out) :-
    rr_reset_dynamic,
    rr_program(Name, _, Q, O, Setup),
    with_output_to(string(Raw),
                   catch(( call(user:Setup), call(user:Q) -> write(O), nl ; true ),
                         _, true)),
    normalize_output(Raw, Out).

normalize_output(Raw, Out) :-
    split_string(Raw, "\n", "\r", Lines0),
    exclude(==(""), Lines0, Lines),
    atomic_list_concat(Lines, "\n", Joined),
    split_string(Joined, " \t", "", Parts),
    atomic_list_concat(Parts, '', Out).

% ---------------------------------------------------------------------
% Rust builds
% ---------------------------------------------------------------------

cargo_available :-
    catch(
        ( process_create(path(cargo), ['--version'],
                         [stdout(null), stderr(null), process(Pid)]),
          process_wait(Pid, exit(0)) ),
        _, fail).

%% rr_mode(?Mode, ?EmitOpts, ?Dir)
rr_mode(interpreter, [emit_mode(interpreter)],
        'output/rust_wam_register_restore_interpreter').
rr_mode(functions,   [emit_mode(functions)],
        'output/rust_wam_register_restore_functions').

:- dynamic rr_built/1.
:- dynamic rr_generated/1.

rr_generate(Dir, EmitOpts) :-
    (   rr_generated(Dir)
    ->  true
    ;   ( exists_directory(Dir) -> delete_directory_and_contents(Dir) ; true ),
        make_directory_path(Dir),
        install_rr_programs,
        rr_rust_preds(Preds),
        append([module_name('rrprobe'), wam_fallback(true)], EmitOpts, Opts),
        write_wam_rust_project(Preds, Opts, Dir),
        write_rr_driver(Dir),
        assertz(rr_generated(Dir))
    ).

%% rr_ensure_built(+Mode)
rr_ensure_built(Mode) :-
    (   rr_built(Mode)
    ->  true
    ;   rr_mode(Mode, EmitOpts, Dir),
        rr_generate(Dir, EmitOpts),
        format(atom(Cmd),
               'cd ~w && CARGO_TARGET_DIR=target cargo build --bin rr_probe 2>&1',
               [Dir]),
        process_create(path(sh), ['-c', Cmd],
                       [stdout(pipe(Out)), stderr(std), process(Pid)]),
        read_string(Out, _, OutStr), close(Out),
        process_wait(Pid, Status),
        (   Status == exit(0)
        ->  assertz(rr_built(Mode))
        ;   format(user_error, "~n[cargo build output, ~w]~n~w~n", [Mode, OutStr]),
            throw(rr_build_failed(Mode, Status))
        )
    ).

rr_bin(Mode, Bin) :-
    rr_mode(Mode, _, Dir),
    format(atom(Bin), '~w/target/debug/rr_probe', [Dir]).

%% lowered_entry(+Dir, +Name, -Key)
%  Key is pred_arity when the functions-mode emitter lowered the query
%  predicate of program Name.
lowered_entry(Dir, Name, Key) :-
    rr_program(Name, _, Q, _, _),
    functor(Q, F, A),
    format(atom(Key), '~w_~w', [F, A]),
    atomic_list_concat([Dir, '/src/lib.rs'], LibRs),
    exists_file(LibRs),
    read_file_to_string(LibRs, S, []),
    format(atom(Sym), 'pub fn lowered_~w(', [Key]),
    sub_string(S, _, _, _, Sym).

write_rr_driver(Dir) :-
    atomic_list_concat([Dir, '/src/bin/rr_probe'], BinDir),
    make_directory_path(BinDir),
    atomic_list_concat([BinDir, '/main.rs'], Path),
    rr_driver_source(Head, Tail),
    findall(Arm,
            ( rr_program(N, _, _, _, _),
              lowered_entry(Dir, N, Key),
              format(atom(Arm),
                     '            "~w" => run_lowered(&mut vm, rrprobe::lowered_~w, &rest),~n',
                     [Key, Key]) ),
            Arms0),
    sort(Arms0, Arms),
    atomic_list_concat(Arms, ArmsText),
    setup_call_cleanup(open(Path, write, S),
                       format(S, '~w~w~w', [Head, ArmsText, Tail]),
                       close(S)).

rr_driver_source(
"// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 John William Creighton (s243a)
//
// rr_probe: run a 0-arity driver predicate by its WAM label (program output
// comes from the compiled write/1), or call a LOWERED function directly with
// the query arguments (`--lowered key arg..`, `_` = the output variable).
#![allow(unused_imports, dead_code, unused_mut, unused_variables)]
use rrprobe::state::WamState;
use rrprobe::value::Value;
use rrprobe::{setup_foreign_predicates, shared_wam_program};

const OUT: &str = \"_rr_out\";

fn parse_arg(s: &str) -> Value {
    if s == \"_\" {
        Value::Unbound(OUT.into())
    } else if let Ok(n) = s.parse::<i64>() {
        Value::Integer(n)
    } else if s == \"[]\" {
        Value::Atom(\"[]\".into())
    } else {
        Value::Atom(s.into())
    }
}

fn run_lowered(vm: &mut WamState, f: fn(&mut WamState) -> bool, args: &[String]) {
    vm.reset_query();
    vm.cp = 0;
    for (i, a) in args.iter().enumerate() {
        vm.set_reg(&format!(\"A{}\", i + 1), parse_arg(a));
    }
    let out_reg = args.iter().position(|a| a == \"_\").map(|i| format!(\"A{}\", i + 1));
    if f(vm) {
        match out_reg {
            None => println!(\"yes\"),
            Some(r) => {
                let bound = vm.deref_heap(&Value::Unbound(OUT.into()));
                let out = if bound.is_unbound() {
                    vm.get_reg(&r).map(|v| vm.deref_heap(&v)).unwrap_or(bound)
                } else {
                    bound
                };
                println!(\"{}\", out);
            }
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (code, labels) = shared_wam_program();
    let mut vm = WamState::new(code, labels);
    setup_foreign_predicates(&mut vm);

    if args.get(1).map(|s| s.as_str()) == Some(\"--lowered\") {
        let key = args.get(2).cloned().unwrap_or_default();
        let rest: Vec<String> = args.iter().skip(3).cloned().collect();
        match key.as_str() {
",
"            other => { eprintln!(\"NO_LOWERED {}\", other); std::process::exit(3); }
        }
        return;
    }

    let key = match args.get(1) {
        Some(k) => k.clone(),
        None => { eprintln!(\"usage: rr_probe <pred/arity>\"); std::process::exit(2); }
    };
    let target = match vm.labels.get(&key) {
        Some(pc) => *pc,
        None => { eprintln!(\"NO_LABEL {}\", key); std::process::exit(3); }
    };
    vm.reset_query();
    vm.cp = 0;
    vm.pc = target;
    let ok = vm.run();
    eprintln!(\"run={}\", ok);
}
").

%% rr_run(+Mode, +Args, -Out, -Err)
rr_run(Mode, Args, Out, Err) :-
    rr_bin(Mode, Bin),
    process_create(Bin, Args,
                   [stdout(pipe(O)), stderr(pipe(E)), process(Pid)]),
    read_string(O, _, S1), read_string(E, _, S2),
    close(O), close(E),
    process_wait(Pid, _),
    normalize_output(S1, Out),
    Err = S2.

%% rust_all(+Mode, +Name, -Out, -Ok)
rust_all(Mode, Name, Out, Ok) :-
    rr_driver_name(Name, D),
    format(atom(Key), '~w/0', [D]),
    rr_run(Mode, [Key], Out, Err),
    (   sub_string(Err, _, _, _, "run=true") -> Ok = true
    ;   sub_string(Err, _, _, _, "run=false") -> Ok = false
    ;   Ok = crashed(Err)
    ).

%% rust_once(+Mode, +Name, -Out) is semidet.
%  Fails when the query predicate was not lowered.
rust_once(Mode, Name, Out) :-
    rr_mode(Mode, _, Dir),
    lowered_entry(Dir, Name, Key),
    rr_program(Name, _, Q, O, _),
    Q =.. [_|QArgs],
    maplist(rr_arg_text(O), QArgs, ArgTexts),
    rr_run(Mode, ['--lowered', Key|ArgTexts], Out, _).

rr_arg_text(O, A, '_') :- A == O, !.
rr_arg_text(_, A, T) :- format(atom(T), '~w', [A]).

% ---------------------------------------------------------------------
% Checks
% ---------------------------------------------------------------------

rr_all_mode(interpreter).
rr_all_mode(functions).

check_all(Mode, Name) :-
    rr_ensure_built(Mode),
    (   rr_known_divergence(Mode, Name, Why)
    ->  format(user_error, '  [~w] ~w: known divergence, not checked (~w)~n',
               [Mode, Name, Why])
    ;   swi_all(Name, SOut, SOk),
        rust_all(Mode, Name, ROut, ROk),
        (   SOut == ROut, SOk == ROk
        ->  true
        ;   format(user_error,
                   'REGISTER RESTORE DIVERGENCE [~w] ~w~n  swi : ~q run=~w~n  rust: ~q run=~w~n',
                   [Mode, Name, SOut, SOk, ROut, ROk]),
            fail
        )
    ).

check_once(Name) :-
    rr_ensure_built(functions),
    (   rr_lowered_out_of_scope(Name, Why)
    ->  format(user_error, '  [functions] ~w: lowered entry not checked (~w)~n',
               [Name, Why])
    ;   rust_once(functions, Name, ROut)
    ->  swi_once(Name, SOut),
        (   SOut == ROut
        ->  true
        ;   format(user_error,
                   'LOWERED DIVERGENCE ~w~n  swi once: ~q~n  rust    : ~q~n',
                   [Name, SOut, ROut]),
            fail
        )
    ;   true
    ).

%% rr_known_divergence(?Mode, ?Name, ?Why)
%  Programs whose interpreted entry differs from SWI-Prolog today, for reasons
%  the R0b audit traced (docs/reports/wam_rust_r0b_register_trail_audit.md).
%  They are not checked against SWI; test known_divergences_still_present
%  asserts that each one still differs, so a fix must move the program into
%  the checked set by deleting its line here. (R0b did this for the
%  functions-mode include/exclude/partition and both modes of foldl and
%  include_wide: those builtins now read their output registers at entry.
%  Bug-fix batch 1, B3, did it for nested_agg_inline: aggregate_all/3 written
%  inside a findall/3 goal is compiled as `call aggregate_all/3`, which is now
%  a runtime builtin.) None is left.
:- dynamic rr_known_divergence/3.

%% rr_lowered_out_of_scope(?Name, ?Why)
%  Programs whose DIRECT lowered entry is wrong, or does not reach the
%  rollback it is meant to exercise, for reasons outside R0b. Their
%  interpreted entries in emit_mode(functions) are still checked.
%
%  A lowered `call`/`execute` of a runtime builtin that has no label (call/N,
%  catch/3, predsort/3, aggregate_all/3) returns false instead of running the
%  builtin (also noted in tests/test_wam_rust_baseline_fixes.pl). Inside an
%  if-then-else condition that makes the lowered entry take the else branch,
%  which may coincide with SWI's answer without exercising anything.
rr_lowered_out_of_scope(naf_meta,      lowered_call_of_builtin_fails).
rr_lowered_out_of_scope(call_disj,     lowered_call_of_builtin_fails).
rr_lowered_out_of_scope(call_ite,      lowered_call_of_builtin_fails).
rr_lowered_out_of_scope(call_n,        lowered_call_of_builtin_fails).
rr_lowered_out_of_scope(catch_fail,    lowered_call_of_builtin_fails).
rr_lowered_out_of_scope(catch_throw,   lowered_call_of_builtin_fails).
rr_lowered_out_of_scope(catch_rec,     lowered_call_of_builtin_fails).
rr_lowered_out_of_scope(predsort_ok,   lowered_call_of_builtin_fails).
rr_lowered_out_of_scope(predsort_fail, lowered_call_of_builtin_fails).
rr_lowered_out_of_scope(nested_agg,    lowered_call_of_builtin_fails).
rr_lowered_out_of_scope(nested_agg_inline, lowered_call_of_builtin_fails).
%  The lowered emitter drops begin_aggregate/end_aggregate.
rr_lowered_out_of_scope(findall_fail,   lowered_aggregates_unsupported).
rr_lowered_out_of_scope(findall_some,   lowered_aggregates_unsupported).
rr_lowered_out_of_scope(nested_findall, lowered_aggregates_unsupported).
rr_lowered_out_of_scope(nested_fail,    lowered_aggregates_unsupported).
%  A lowered if-then-else condition whose `call` fails leaves the callee's
%  environment frame on the stack (the lowered `call` restores neither the
%  stack nor E), so the else branch reads its Y registers from the callee's
%  frame. A lowered-tier defect independent of the register trail.
rr_lowered_out_of_scope(naf_temp, lowered_ite_keeps_failed_callee_frame).

%% rr_lowered_required(?Name)
%  Programs whose query predicate MUST be lowered in functions mode, so the
%  lowered checks of the T4 clause snapshot and the lowered ITE cannot
%  silently become skips.
rr_lowered_required(multi_pure).
rr_lowered_required(multi_call).
rr_lowered_required(multi_ite).
rr_lowered_required(ite_is).
rr_lowered_required(ite_temps).
rr_lowered_required(ifthen).

test_wam_rust_register_restore :-
    run_tests([wam_rust_register_restore, wam_rust_r0b_ite_else_check]).

:- begin_tests(wam_rust_register_restore, [condition(cargo_available),
                                          setup(install_rr_programs)]).

test(all_solutions_match_swi,
     [forall(( rr_all_mode(Mode), rr_program(Name, _, _, _, _) ))]) :-
    once(check_all(Mode, Name)).

test(lowered_first_solution_matches_swi,
     [forall(rr_program(Name, _, _, _, _))]) :-
    once(check_once(Name)).

test(lowered_entries_present, [forall(rr_lowered_required(Name))]) :-
    rr_ensure_built(functions),
    rr_mode(functions, _, Dir),
    once(lowered_entry(Dir, Name, _)).

test(known_divergences_still_present,
     [forall(rr_known_divergence(Mode, Name, _))]) :-
    rr_ensure_built(Mode),
    swi_all(Name, SOut, SOk),
    rust_all(Mode, Name, ROut, ROk),
    \+ ( SOut == ROut, SOk == ROk ).

:- end_tests(wam_rust_register_restore).

% ---------------------------------------------------------------------
% R0b emit-time check (no cargo needed): the else path of a lowered
% if-then-else (its else branch plus the instructions after the ITE) must
% write every A/X register before reading it, because unwind_trail_to no
% longer restores A/X. wam_rust_lowered_emitter:rust_ite_else_ax_check/2 is
% called by emit_instrs/3 on every ITE it emits.
% ---------------------------------------------------------------------

ite_check(Else, Rest) :-
    wam_rust_lowered_emitter:rust_ite_else_ax_check(Else, Rest).

ite_check_rejects(Else, Rest, Reg) :-
    catch(( ite_check(Else, Rest), Got = accepted ),
          error(lowered_ite_else_reads_ax_before_write(R, _), _),
          Got = rejected(R)),
    Got == rejected(Reg).

:- begin_tests(wam_rust_r0b_ite_else_check).

test(else_loads_its_arguments) :-
    ite_check([put_value("Y2", "A1"), put_constant("e", "A2"),
               builtin_call("=/2", "2")],
              [deallocate, proceed]).

test(else_reads_an_argument_it_did_not_load) :-
    ite_check_rejects([put_value("Y2", "A2"), builtin_call("=/2", "2")],
                      [proceed], "A1").

test(after_the_ite_reads_a_temporary_the_else_did_not_write) :-
    ite_check_rejects([put_constant("e", "A1")],
                      [get_value("X3", "A1"), proceed], "X3").

test(nested_ite_keeps_only_registers_both_branches_wrote) :-
    % the nested condition and then branch write A1 and A2, its else branch
    % only A3: after the nested ITE, A1 is not known to be written.
    ite_check_rejects([ite([put_constant("a", "A1"), builtin_call("atom/1", "1")],
                           [put_constant("b", "A2")],
                           [put_constant("c", "A3")]),
                       builtin_call("atom/1", "1")],
                      [], "A1").

test(a_call_ends_the_path) :-
    ite_check([put_value("Y1", "A1"), call("q/1", "1"), get_value("X1", "A1")],
              [proceed]).

test(execute_reads_its_arguments) :-
    ite_check_rejects([put_value("Y1", "A1"), execute("q/2")], [], "A2").

test(y_registers_are_not_checked) :-
    ite_check([get_value("Y1", "Y2"), put_value("Y3", "A1"),
               builtin_call("atom/1", "1")],
              []).

:- end_tests(wam_rust_r0b_ite_else_check).
