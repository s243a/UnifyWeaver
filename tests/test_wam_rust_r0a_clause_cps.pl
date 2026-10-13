:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (s243a)
%
% test_wam_rust_r0a_clause_cps.pl
%
% Phase R0a (D135) of docs/proposals/wam_rust_heap_cell_rewrite_design.md:
% a clause-entry choice point saves only A1..A<arity>
% (Instruction::TryMeElseArgs). The generator emits that form only when its
% liveness check finds no other A/X register live where a retry resumes
% (rust_r0a_instr_options/4 in wam_rust_target.pl); otherwise every clause
% choice point of the program keeps the full save.
%
% Same harness pattern as test_wam_rust_baseline_fixes.pl: each program's
% clauses are asserted into user: (SWI runs them) and compiled through
% write_wam_rust_project/3; a failure-driven driver
%     ra_drv_<name> :- Setup, ( Query, write(Out), nl, fail ; true ).
% prints every solution, and stdout plus the driver's success flag must equal
% SWI's, in emit_mode(interpreter) and emit_mode(functions). In functions
% mode a lowered query predicate is also called directly (first solution).
%
% Two program groups, each compiled into its own crate (the R0a decision is
% per program):
%
%   * safe: clause choice points retried in every context the safety
%     argument covers -- head unification failing after temporaries were
%     written, retries after the caller continued, tail calls, arity 0 and 6,
%     indexing, ITEs inside and around multi-clause predicates, cut,
%     aggregates, catch/3, call/N, maplist, \+, between, and the builtins
%     that read an output register after their goals ran (include/3,
%     exclude/3, partition/4, foldl/4). The generated crate must use
%     TryMeElseArgs for the clause chains.
%   * full_save: the shapes where the shared compiler keeps a register live
%     across a call (forall/2, \+ and once/1 over a conjunction at the first
%     goal; a count aggregate's A1 placeholder). The generator must keep the
%     full save there, and the answers must match SWI. Built a second time
%     with clause_cp_arity_save(unchecked) (arity-only without the check),
%     every one of them must DIVERGE from SWI: the shapes are real and the
%     check is what keeps them correct.
%
%   swipl -q -g run_tests -t halt tests/test_wam_rust_r0a_clause_cps.pl

:- module(test_wam_rust_r0a_clause_cps,
          [test_wam_rust_r0a_clause_cps/0]).

:- use_module(library(plunit)).
:- use_module(library(lists)).
:- use_module(library(apply)).
:- use_module(library(filesex), [make_directory_path/1,
                                 delete_directory_and_contents/1]).
:- use_module(library(process)).
:- use_module(library(readutil)).
:- use_module('../src/unifyweaver/targets/wam_rust_target',
              [write_wam_rust_project/3]).

% ---------------------------------------------------------------------
% Programs. ra_program(Group, Name, Query, Out, Setup)
% ---------------------------------------------------------------------

:- discontiguous ra_program/5, ra_clause/2.

% --- safe group -------------------------------------------------------

% Clause 1 builds temporaries in head unification and fails late; the retry
% must see every argument (arity 3).
ra_program(safe, args, ( member(T, [f(1, 2), f(5, 2), g(q)]), ra_m(T, Y, Z) ),
           [T, Y, Z], true).
ra_clause(args, (ra_m(X, Y, Z) :- X = f(A, B), A > 1, Y = B, Z = c1)).
ra_clause(args, (ra_m(X, Y, Z) :- X = f(A, _), Y = A, Z = c2)).
ra_clause(args, ra_m(g(Y), Y, c3)).

% The callee is retried after the caller continued, through a tail call.
ra_program(safe, deep, ra_deep(X, Y), [X, Y], true).
ra_clause(deep, (ra_deep(X, Y) :- ra_a(X), ra_b(X, Y))).
ra_clause(deep, ra_a(1)).
ra_clause(deep, ra_a(2)).
ra_clause(deep, ra_a(3)).
ra_clause(deep, (ra_b(X, Y) :- ra_c(X, Y))).
ra_clause(deep, (ra_c(X, two) :- X =:= 2)).
ra_clause(deep, ra_c(3, three)).

% Arity 0: the clause choice point saves nothing.
ra_program(safe, arity0, ra_z(R), R, true).
ra_clause(arity0, (ra_z(R) :- ra_z0, ra_z1(R))).
ra_clause(arity0, ra_z0).
ra_clause(arity0, (ra_z0 :- ra_a(2))).
ra_clause(arity0, ra_z1(a)).
ra_clause(arity0, ra_z1(b)).

% Arity 6, clause 2 reads all six arguments.
ra_program(safe, wide, ra_w6(b, 1, 2, 3, 4, F), F, true).
ra_clause(wide, (ra_w6(a, B, C, D, E, F) :- B = C, D = E, F = one)).
ra_clause(wide, (ra_w6(A, B, C, D, E, F) :- A \== a, F = two(A, B, C, D, E))).
ra_clause(wide, ra_w6(_, _, _, _, E, three(E))).

% First-argument indexing in front of the clause chain.
ra_program(safe, indexed, ( member(K, [a, b, c, d, _]), ra_idx(K, V) ), [K, V], true).
ra_clause(indexed, ra_idx(a, 1)).
ra_clause(indexed, ra_idx(b, 2)).
ra_clause(indexed, ra_idx(a, 3)).
ra_clause(indexed, ra_idx(c, 4)).

% If-then-else inside the clauses of a multi-clause predicate, and a cut.
ra_program(safe, ite_in, ( member(X, [1, 3, 7, 12]), ra_cls(X, R) ), [X, R], true).
ra_clause(ite_in, (ra_cls(X, R) :- X > 10, !, R = big)).
ra_clause(ite_in, (ra_cls(X, R) :- ( X > 5 -> R = mid ; ra_small(X, R) ))).
ra_clause(ite_in, (ra_cls(X, R) :- ra_small(X, R), R \== tiny)).
ra_clause(ite_in, (ra_small(X, tiny) :- X < 2)).
ra_clause(ite_in, (ra_small(X, small) :- X >= 2)).

% An ITE condition that retries a multi-clause callee.
ra_program(safe, ite_cond, ( member(X, [0, 2, 3]), ra_ite(X, R) ), [X, R], true).
ra_clause(ite_cond, (ra_ite(X, R) :- ( ra_a(Y), Y > X -> R = Y ; R = none ))).

% A cut after a retried callee.
ra_program(safe, cut, ra_cut(X, R), [X, R], true).
ra_clause(cut, (ra_cut(X, R) :- ra_a(X), X > 1, !, R = X)).
ra_clause(cut, ra_cut(_, none)).

% A disjunction at the first goal.
ra_program(safe, disj, ( member(X, [1, 2]), ra_dis(X, R) ), [X, R], true).
ra_clause(disj, (ra_dis(X, R) :- ( ra_a(X), R = a ; ra_pair(X, R) ))).
% (Clause 1 is in the multi-clause first-argument group on purpose: a group
% without clause 1 is reached through a try/retry/trust dispatch chain, which
% this runtime does not run correctly -- a separate, pre-existing defect.)
ra_clause(disj, ra_pair(2, q)).
ra_clause(disj, ra_pair(1, p)).
ra_clause(disj, ra_pair(2, r)).

% Aggregates around multi-clause calls (compound, variable and sum/max
% templates; setof; a nested findall).
ra_program(safe, agg, ra_agg(L1, L2, S, M, Set, N), [L1, L2, S, M, Set, N], true).
ra_clause(agg, (ra_agg(L1, L2, S, M, Set, N) :-
                    findall(p(X, Y), ( ra_a(X), ra_pair(X, Y) ), L1),
                    findall(Y, ( ra_a(X), ra_pair(X, Y) ), L2),
                    aggregate_all(sum(X), ra_a(X), S),
                    aggregate_all(max(X), ra_a(X), M),
                    setof(Y, ra_pair2(Y), Set),
                    findall(p(X, Ys), ( ra_a(X), findall(Y, ra_pair(X, Y), Ys) ), N))).
ra_clause(agg, (ra_pair2(Y) :- ra_pair(_, Y))).

% catch/3 around a callee whose clauses fail, throw and succeed.
ra_program(safe, catch, ( member(X, [1, 2, 3]), ra_catch(X, R) ), [X, R], true).
ra_clause(catch, (ra_catch(X, R) :- catch(ra_thrower(X, R), err(E), R = caught(E)))).
ra_clause(catch, (ra_thrower(X, R) :- X > 2, R = big)).
ra_clause(catch, (ra_thrower(X, _) :- X =:= 2, throw(err(two)))).
ra_clause(catch, (ra_thrower(X, small) :- X < 2)).

% Meta-calls into multi-clause predicates: call/N, maplist, \+, between.
% (call/N and maplist commit to the first solution in this runtime, so their
% goals have exactly one.)
ra_program(safe, meta, ra_meta(X, L, Ns, Bs), [X, L, Ns, Bs], true).
ra_clause(meta, (ra_meta(X, L, Ns, Bs) :-
                     call(ra_cnt, X),
                     maplist(ra_inc, [1, 2, 3], L),
                     findall(N, ( member(N, [1, 2, 3, 4]), \+ ra_a(N) ), Ns),
                     findall(B, ( between(1, 4, B), ra_a(B) ), Bs))).
ra_clause(meta, (ra_cnt(X) :- ra_a(X), X > 2)).
ra_clause(meta, (ra_inc(X, Y) :- X > 1, !, Y is X * 10)).
ra_clause(meta, (ra_inc(X, Y) :- Y is X + 100)).

% include/3, exclude/3, partition/4 and foldl/4 read their output register
% after the goals ran; the goals' clause choice points are retried (clause 1
% fails for every element), so those registers must survive the retries.
ra_program(safe, list_meta, ra_lists(I, E, P1, P2, F), [I, E, P1, P2, F], true).
ra_clause(list_meta, (ra_lists(I, E, P1, P2, F) :-
                          include(ra_even, [1, 2, 3, 4], I),
                          exclude(ra_even, [1, 2, 3, 4], E),
                          partition(ra_even, [1, 2, 3, 4, 6], P1, P2),
                          foldl(ra_add, [1, 2, 3], 0, F))).
ra_clause(list_meta, (ra_even(X) :- X > 100)).
ra_clause(list_meta, (ra_even(X) :- 0 is X mod 2)).
ra_clause(list_meta, (ra_add(X, A0, A) :- X > 2, !, A is A0 + X * 100)).
ra_clause(list_meta, (ra_add(X, A0, A) :- A is A0 + X)).

% --- full_save group --------------------------------------------------

% forall/2 at the first goal: B stays in an X register across the call.
ra_program(full_save, forall, rb_forall(b), yes, true).
ra_clause(forall, (rb_forall(B) :- forall(rb_q(_), rb_s(B)))).
ra_clause(forall, rb_q(1)).
ra_clause(forall, rb_q(2)).
ra_clause(forall, rb_s(b)).

% \+ over a conjunction at the first goal.
ra_program(full_save, naf_conj, rb_naf(_, y), yes, true).
ra_clause(naf_conj, (rb_naf(A, B) :- \+ ( rb_q(A), rb_t(A, B) ))).
ra_clause(naf_conj, rb_t(2, y)).

% once/1 over a conjunction at the first goal.
ra_program(full_save, once_conj, rb_once(A, y), A, true).
ra_clause(once_conj, (rb_once(A, B) :- once(( rb_q(A), rb_t(A, B) )))).

% aggregate_all(count, ...) collects its A1 placeholder after the goal; the
% retried choice point is a 0-arity one reached by a tail call.
ra_program(full_save, count0, rb_count(N), N, true).
ra_clause(count0, (rb_count(N) :- aggregate_all(count, rb_r0, N))).
ra_clause(count0, (rb_r0 :- rb_q0)).
ra_clause(count0, rb_q0).
ra_clause(count0, rb_q0).

%% ra_uses(?Name, ?Other)
%  Program Name's clauses call helpers defined with program Other.
ra_uses(ite_cond, deep).
ra_uses(cut, deep).
ra_uses(disj, deep).
ra_uses(agg, deep).
ra_uses(agg, disj).
ra_uses(meta, deep).
ra_uses(naf_conj, forall).
ra_uses(once_conj, forall).
ra_uses(once_conj, naf_conj).

% ---------------------------------------------------------------------
% Installation
% ---------------------------------------------------------------------

ra_driver_name(Name, D) :- atom_concat(ra_drv_, Name, D).

ra_clause_head(Clause, F/A) :-
    ( Clause = (H :- _) -> true ; H = Clause ),
    functor(H, F, A).

%% ra_group_program(?Group, ?Name)
ra_group_program(Group, Name) :- ra_program(Group, Name, _, _, _).

%% ra_group_pis(+Group, -PIs): every predicate compiled into Group's crate.
ra_group_pis(Group, PIs) :-
    findall(PI,
            ( ra_group_program(Group, N),
              ( M = N ; ra_uses(N, M) ),
              ra_clause(M, C),
              ra_clause_head(C, PI) ),
            PIs0),
    findall(D/0, ( ra_group_program(Group, N), ra_driver_name(N, D) ), PIs1),
    append(PIs0, PIs1, PIs2),
    sort(PIs2, PIs).

install_ra_programs :-
    findall(PI, ( ra_clause(_, C), ra_clause_head(C, PI) ), PIs0),
    findall(D/0, ( ra_program(_, N, _, _, _), ra_driver_name(N, D) ), PIs1),
    append(PIs0, PIs1, PIs2),
    sort(PIs2, PIs),
    forall(member(F/A, PIs),
           ( functor(H, F, A), catch(retractall(user:H), _, true) )),
    forall(ra_clause(_, C), assertz(user:C)),
    forall(ra_program(_, N, Q, Out, Setup),
           ( ra_driver_name(N, D),
             assertz(user:(D :- Setup, ( Q, write(Out), nl, fail ; true ))) )).

% ---------------------------------------------------------------------
% SWI oracle
% ---------------------------------------------------------------------

swi_all(Name, Out, Ok) :-
    ra_driver_name(Name, D),
    with_output_to(string(Raw), ( catch(user:D, _, fail) -> Ok = true ; Ok = false )),
    normalize_output(Raw, Out).

swi_once(Name, Out) :-
    ra_program(_, Name, Q, O, Setup),
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

%% ra_build(?Build, ?Group, ?EmitOpts, ?Dir)
ra_build(safe_interpreter,      safe,      [emit_mode(interpreter)],
         'output/rust_wam_r0a_safe_interpreter').
ra_build(safe_functions,        safe,      [emit_mode(functions)],
         'output/rust_wam_r0a_safe_functions').
ra_build(full_save_interpreter, full_save, [emit_mode(interpreter)],
         'output/rust_wam_r0a_full_save_interpreter').
ra_build(full_save_functions,   full_save, [emit_mode(functions)],
         'output/rust_wam_r0a_full_save_functions').
ra_build(full_save_unchecked,   full_save, [emit_mode(interpreter),
                                            clause_cp_arity_save(unchecked)],
         'output/rust_wam_r0a_full_save_unchecked').

:- dynamic ra_built/1.

%% ra_ensure_built(+Build)
ra_ensure_built(Build) :-
    (   ra_built(Build)
    ->  true
    ;   ra_build(Build, Group, EmitOpts, Dir),
        ( exists_directory(Dir) -> delete_directory_and_contents(Dir) ; true ),
        make_directory_path(Dir),
        install_ra_programs,
        ra_group_pis(Group, PIs),
        findall(user:PI, member(PI, PIs), Preds),
        append([module_name('r0aprobe'), wam_fallback(true)], EmitOpts, Opts),
        once(write_wam_rust_project(Preds, Opts, Dir)),
        write_ra_driver(Dir),
        format(atom(Cmd), 'cd ~w && cargo build --bin ra_probe 2>&1', [Dir]),
        process_create(path(sh), ['-c', Cmd],
                       [stdout(pipe(Out)), stderr(std), process(Pid)]),
        read_string(Out, _, OutStr), close(Out),
        process_wait(Pid, Status),
        (   Status == exit(0)
        ->  assertz(ra_built(Build))
        ;   format(user_error, "~n[cargo build output, ~w]~n~w~n", [Build, OutStr]),
            throw(ra_build_failed(Build, Status))
        )
    ).

ra_bin(Build, Bin) :-
    ra_build(Build, _, _, Dir),
    format(atom(Bin), '~w/target/debug/ra_probe', [Dir]).

ra_lib(Build, S) :-
    ra_build(Build, _, _, Dir),
    atomic_list_concat([Dir, '/src/lib.rs'], LibRs),
    read_file_to_string(LibRs, S, []).

%% lowered_entry(+Build, +Name, -Key)
lowered_entry(Build, Name, Key) :-
    ra_program(_, Name, Q, _, _),
    functor(Q, F, A),
    format(atom(Key), '~w_~w', [F, A]),
    ra_lib(Build, S),
    format(atom(Sym), 'pub fn lowered_~w(', [Key]),
    sub_string(S, _, _, _, Sym).

write_ra_driver(Dir) :-
    atomic_list_concat([Dir, '/src/bin/ra_probe'], BinDir),
    make_directory_path(BinDir),
    atomic_list_concat([BinDir, '/main.rs'], Path),
    ra_driver_source(Head, Tail),
    atomic_list_concat([Dir, '/src/lib.rs'], LibRs),
    read_file_to_string(LibRs, Lib, []),
    findall(Arm,
            ( ra_program(_, _, Q, _, _),
              functor(Q, F, A),
              format(atom(Key), '~w_~w', [F, A]),
              format(atom(Sym), 'pub fn lowered_~w(', [Key]),
              sub_string(Lib, _, _, _, Sym),
              format(atom(Arm),
                     '            "~w" => run_lowered(&mut vm, r0aprobe::lowered_~w, &rest),~n',
                     [Key, Key]) ),
            Arms0),
    sort(Arms0, Arms),
    atomic_list_concat(Arms, ArmsText),
    setup_call_cleanup(open(Path, write, S),
                       format(S, '~w~w~w', [Head, ArmsText, Tail]),
                       close(S)).

ra_driver_source(
"// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 John William Creighton (s243a)
//
// ra_probe: run a 0-arity driver predicate by its WAM label, or call a
// LOWERED function directly (`--lowered key arg..`, `_` = the output).
#![allow(unused_imports, dead_code, unused_mut, unused_variables)]
use r0aprobe::state::WamState;
use r0aprobe::value::Value;
use r0aprobe::{setup_foreign_predicates, shared_wam_program};

const OUT: &str = \"_ra_out\";

fn parse_arg(s: &str) -> Value {
    if s == \"_\" {
        Value::Unbound(OUT.into())
    } else if let Ok(n) = s.parse::<i64>() {
        Value::Integer(n)
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
    if f(vm) {
        let bound = vm.deref_heap(&Value::Unbound(OUT.into()));
        if args.iter().any(|a| a == \"_\") {
            println!(\"{}\", bound);
        } else {
            println!(\"yes\");
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
        None => { eprintln!(\"usage: ra_probe <pred/arity>\"); std::process::exit(2); }
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

ra_run(Build, Args, Out, Err) :-
    ra_bin(Build, Bin),
    process_create(Bin, Args,
                   [stdout(pipe(O)), stderr(pipe(E)), process(Pid)]),
    read_string(O, _, S1), read_string(E, _, S2),
    close(O), close(E),
    process_wait(Pid, _),
    normalize_output(S1, Out),
    Err = S2.

rust_all(Build, Name, Out, Ok) :-
    ra_driver_name(Name, D),
    format(atom(Key), '~w/0', [D]),
    ra_run(Build, [Key], Out, Err),
    (   sub_string(Err, _, _, _, "run=true") -> Ok = true
    ;   sub_string(Err, _, _, _, "run=false") -> Ok = false
    ;   Ok = crashed(Err)
    ).

%% rust_once(+Build, +Name, -Out) is semidet.
%  Fails when the query predicate was not lowered, or its arguments are not
%  all atomic or the output (the direct entry takes atoms and integers).
rust_once(Build, Name, Out) :-
    lowered_entry(Build, Name, Key),
    ra_program(_, Name, Q, O, true),
    Q =.. [_|QArgs],
    maplist(ra_arg_text(O), QArgs, ArgTexts),
    ra_run(Build, ['--lowered', Key|ArgTexts], Out, _).

ra_arg_text(O, A, '_') :- A == O, !.
ra_arg_text(_, A, T) :- atomic(A), format(atom(T), '~w', [A]).

% ---------------------------------------------------------------------
% Checks
% ---------------------------------------------------------------------

check_all(Build, Name) :-
    ra_ensure_built(Build),
    swi_all(Name, SOut, SOk),
    rust_all(Build, Name, ROut, ROk),
    (   SOut == ROut, SOk == ROk
    ->  true
    ;   format(user_error,
               'R0a DIVERGENCE [~w] ~w~n  swi : ~q run=~w~n  rust: ~q run=~w~n',
               [Build, Name, SOut, SOk, ROut, ROk]),
        fail
    ).

check_once(Build, Name) :-
    ra_ensure_built(Build),
    (   ra_lowered_out_of_scope(Name, Why)
    ->  format(user_error, '  [~w] ~w: lowered entry not checked (~w)~n',
               [Build, Name, Why])
    ;   rust_once(Build, Name, ROut)
    ->  swi_once(Name, SOut),
        (   SOut == ROut
        ->  true
        ;   format(user_error,
                   'R0a LOWERED DIVERGENCE [~w] ~w~n  swi once: ~q~n  rust    : ~q~n',
                   [Build, Name, SOut, ROut]),
            fail
        )
    ;   true
    ).

%% ra_lowered_out_of_scope(?Name, ?Why)
%  Direct lowered entries that are wrong for a reason outside R0a, the same on
%  the base commit (all full_save programs: no arity-only choice point is
%  involved). The lowered emitter drops begin_aggregate/end_aggregate (see
%  test_wam_rust_baseline_fixes.pl); a lowered ITE whose condition is a
%  conjunction runs its lowered calls first-solution, so it cannot retry
%  rb_q/1 when rb_t/2 fails. Their interpreted entries are checked.
ra_lowered_out_of_scope(count0,    lowered_aggregates_unsupported).
ra_lowered_out_of_scope(once_conj, lowered_ite_condition_is_first_solution).

%% clause_chain_pred(+Group, -F/A): a multi-clause predicate of the group.
clause_chain_pred(Group, F/A) :-
    ra_group_pis(Group, PIs),
    member(F/A, PIs),
    findall(x, ( ra_clause(_, C), ra_clause_head(C, F/A) ), Xs),
    length(Xs, NC), NC > 1.

ra_checked_build(safe_interpreter).
ra_checked_build(safe_functions).
ra_checked_build(full_save_interpreter).
ra_checked_build(full_save_functions).

ra_build_group(Build, Group) :- ra_build(Build, Group, _, _).

test_wam_rust_r0a_clause_cps :-
    run_tests(wam_rust_r0a_clause_cps).

:- begin_tests(wam_rust_r0a_clause_cps, [condition(cargo_available),
                                        setup(install_ra_programs)]).

test(all_solutions_match_swi,
     [forall(( ra_checked_build(Build), ra_build_group(Build, Group),
               ra_group_program(Group, Name) ))]) :-
    once(check_all(Build, Name)).

test(lowered_first_solution_matches_swi,
     [forall(( member(Build, [safe_functions, full_save_functions]),
               ra_build_group(Build, Group),
               ra_group_program(Group, Name) ))]) :-
    once(check_once(Build, Name)).

% The safe group's clause chains use the arity-only form, in both modes: the
% tests above exercise TryMeElseArgs, not the full save.
test(safe_group_uses_arity_only_clause_cps,
     [forall(( member(Build, [safe_interpreter, safe_functions]),
               clause_chain_pred(safe, F/A) ))]) :-
    ra_ensure_built(Build),
    ra_lib(Build, S),
    format(string(Args), 'Instruction::TryMeElseArgs("L_~w_~w_2".to_string(), ~w)', [F, A, A]),
    format(string(Full), 'Instruction::TryMeElse("L_~w_~w_2".to_string())', [F, A]),
    once(sub_string(S, _, _, _, Args)),
    \+ sub_string(S, _, _, _, Full).

% ITE guards keep the full save even when the program is verified.
test(ite_guards_keep_the_full_save) :-
    ra_ensure_built(safe_interpreter),
    ra_lib(safe_interpreter, S),
    once(sub_string(S, _, _, _, 'Instruction::TryMeElse("L_ite_else_')),
    \+ sub_string(S, _, _, _, 'Instruction::TryMeElseArgs("L_ite_else_').

% The full_save group fails the check: its clause chains keep the full-save
% TryMeElse, and no arity-only choice point is emitted anywhere.
test(full_save_group_keeps_the_full_save,
     [forall(( member(Build, [full_save_interpreter, full_save_functions]),
               clause_chain_pred(full_save, F/A) ))]) :-
    ra_ensure_built(Build),
    ra_lib(Build, S),
    \+ sub_string(S, _, _, _, 'TryMeElseArgs('),
    format(string(Full), 'Instruction::TryMeElse("L_~w_~w_2".to_string())', [F, A]),
    once(sub_string(S, _, _, _, Full)).

% Arity-only WITHOUT the check breaks every full_save shape.
test(unchecked_arity_only_breaks_the_full_save_shapes,
     [forall(ra_group_program(full_save, Name))]) :-
    ra_ensure_built(full_save_unchecked),
    ra_lib(full_save_unchecked, S),
    once(sub_string(S, _, _, _, 'TryMeElseArgs(')),
    swi_all(Name, SOut, SOk),
    rust_all(full_save_unchecked, Name, ROut, ROk),
    \+ ( SOut == ROut, SOk == ROk ).

:- end_tests(wam_rust_r0a_clause_cps).
