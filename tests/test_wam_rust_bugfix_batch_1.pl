:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (s243a)
%
% test_wam_rust_bugfix_batch_1.pl
%
% SWI-Prolog comparison harness for the Rust WAM bug-fix batch 1
% (docs/reports/wam_rust_bugfix_batch_1.md, ledger rows D140 onward). Each
% program is a small reproduction of one defect; the item tag names the
% report section:
%
%   B1  names containing `\` or `"` emitted into Rust string literals
%   B2  ignore/1
%   B3  aggregate_all/3 inside a findall/3 goal (and other inner positions)
%   B4  dynamic rule bodies resume on redo instead of re-running
%   B5  lowered ITE: a failing call in the condition leaves its frame behind
%   B6  `_XT<N>` aggregate template scratch registers (slot 0 = A1)
%
% The harness is the one of tests/test_wam_rust_baseline_fixes.pl: one source
% of truth (bb_clause/2), asserted into user: for SWI and handed to
% write_wam_rust_project/3; a failure-driven driver
%     bb_drv_<name> :- Setup, ( Query, write(Out), nl, fail ; true ).
% is compiled with the program and run by its WAM label in
% emit_mode(interpreter) and emit_mode(functions); its stdout and success
% flag must equal SWI's. When emit_mode(functions) lowered the query
% predicate, its Rust function is also called directly and compared with
% SWI's first solution (the lowered tier is first-solution by contract), on
% the default build and on the non-interned
% `--no-default-features --features decorate_sort` build.
%
%   LANG=C.UTF-8 swipl -q -g run_tests -t halt tests/test_wam_rust_bugfix_batch_1.pl
%
% (A UTF-8 locale is needed: the generator reads templates that contain
% non-ASCII text.)

:- module(test_wam_rust_bugfix_batch_1,
          [test_wam_rust_bugfix_batch_1/0]).

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
% Programs. bb_program(Name, Item, Query, Out, Setup)
%   Query/Out : the query whose solutions are printed (write(Out)).
%   Setup     : goal run once by the driver before the query loop.
% Clauses are listed under one program name each (bb_clause/2) and all of
% them are compiled into every crate.
% ---------------------------------------------------------------------

:- discontiguous bb_program/5, bb_clause/2.

% --- B1: names with `\` or `"` in Rust string literals ----------------
%
% A `\+` term built in a clause body (put_structure \+/1), meta-called.
bb_program(esc_naf_no,  'B1', esc_naf(a, R), R, true).
bb_program(esc_naf_yes, 'B1', esc_naf(b, R), R, true).
bb_clause(esc_naf_no, (esc_naf(X, R) :- G = (\+ esc_q(X)), ( call(G) -> R = yes ; R = no ))).
bb_clause(esc_naf_no, esc_q(a)).
% `\=` built in a body and meta-called, after a choice point.
bb_program(esc_neq, 'B1', esc_neq(X, R), [X, R], true).
bb_clause(esc_neq, (esc_neq(X, R) :- member(X, [a, b]), G = (X \= a), ( call(G) -> R = ne ; R = eq ))).
% Functor names with a backslash or a double quote in clause heads
% (get_structure), read back with functor/3 (write/1 prints operators in
% canonical form in this runtime, so the terms themselves are not printed).
bb_program(esc_functor, 'B1', esc_functor(N), N, true).
bb_clause(esc_functor, (esc_functor(N) :- esc_ops(T), functor(T, N, _))).
bb_clause(esc_functor, esc_ops(\+ a)).
bb_clause(esc_functor, esc_ops(a \= b)).
bb_clause(esc_functor, esc_ops(1 =\= 2)).
bb_clause(esc_functor, esc_ops(a \== b)).
bb_clause(esc_functor, esc_ops('q"r'(1))).
% Lowered code: put_structure and get_structure of `\+/1` in deterministic
% predicates, and atom constants containing `\` and `"`.
bb_program(esc_head,  'B1', esc_mk(R), R, true).
bb_clause(esc_head, (esc_mk(R) :- esc_unwrap(\+ k, R))).
bb_clause(esc_head, esc_unwrap(\+ X, X)).
bb_program(esc_atoms_then, 'B1', esc_atoms('a\\b', R), R, true).
bb_program(esc_atoms_else, 'B1', esc_atoms(x, R), R, true).
bb_clause(esc_atoms_then, (esc_atoms(X, R) :- ( X == 'a\\b' -> R = 'say "hi"' ; R = 'c\\d' ))).
bb_program(esc_quote_const, 'B1', esc_qc(R), R, true).
bb_clause(esc_quote_const, (esc_qc(R) :- esc_qv(V), R = V)).
bb_clause(esc_quote_const, esc_qv('x"y')).
% A predicate whose name contains a backslash (its label, its dispatch key).
bb_program(esc_pred_name, 'B1', esc_callp(R), R, true).
bb_clause(esc_pred_name, (esc_callp(R) :- 'esc\\pred'(R))).
bb_clause(esc_pred_name, 'esc\\pred'(ok)).

% --- B2: ignore/1 ----------------------------------------------------
%
% A failing goal: ignore/1 succeeds.
bb_program(ign_fail, 'B2', ign_fail(R), R, true).
bb_clause(ign_fail, (ign_fail(R) :- ignore(ign_none), R = ok)).
bb_clause(ign_fail, (ign_none :- fail)).
% First solution only.
bb_program(ign_first, 'B2', ign_first(X), X, true).
bb_clause(ign_first, (ign_first(X) :- ignore(ign_d(X)))).
bb_clause(ign_first, ign_d(1)).
bb_clause(ign_first, ign_d(2)).
% The bindings of a goal that fails are undone.
bb_program(ign_undo, 'B2', ign_undo(R), R, true).
bb_clause(ign_undo, (ign_undo(R) :- ignore(ign_bind_fail(X)), ( var(X) -> R = unbound ; R = bound(X) ))).
bb_clause(ign_undo, (ign_bind_fail(X) :- X = 1, fail)).
% The goal is an opaque cut scope: the caller's choice points survive.
bb_program(ign_cut, 'B2', ign_cut(X), X, true).
bb_clause(ign_cut, (ign_cut(X) :- ign_d(X), ignore(!))).
% Inside a findall/3 goal, meta-called, and as the last goal of a clause.
bb_program(ign_findall, 'B2', ign_findall(L), L, true).
bb_clause(ign_findall, (ign_findall(L) :- findall(p(X, Y), (ign_d(X), ignore(ign_e(X, Y)), ( var(Y) -> Y = none ; true )), L))).
bb_clause(ign_findall, ign_e(2, two)).
bb_program(ign_meta, 'B2', ign_meta(R), R, true).
bb_clause(ign_meta, (ign_meta(R) :- G = ignore(ign_none), call(G), R = ok)).
bb_program(ign_tail, 'B2', ign_tail(X), X, true).
bb_clause(ign_tail, (ign_tail(X) :- X = t, ignore(ign_d(_)))).
% Side effects of the goal happen once; a ball thrown inside propagates.
bb_program(ign_write, 'B2', ign_write, done, true).
bb_clause(ign_write, (ign_write :- ignore((write(a), fail)))).
bb_program(ign_throw, 'B2', ign_throw(R), R, true).
bb_clause(ign_throw, (ign_throw(R) :- catch(ignore(throw(oops)), E, R = caught(E)))).

%% bb_dynamic(?PI)
%  Predicates the drivers create at run time with assertz/1. Declared dynamic
%  in user: for SWI; not compiled for Rust (the runtime's dynamic database
%  serves them).
:- dynamic bb_dynamic/1.

%% bb_lowered_out_of_scope(?Name, ?Why)
%  Programs whose DIRECT lowered entry is wrong for a reason outside this
%  batch. Their interpreted entries in emit_mode(functions) are still
%  checked. A lowered `call`/`execute` of a runtime builtin that has no label
%  (call/N here) returns false instead of running it (noted in
%  tests/test_wam_rust_baseline_fixes.pl).
bb_lowered_out_of_scope(esc_naf_no,  lowered_call_of_builtin_fails).
bb_lowered_out_of_scope(esc_naf_yes, lowered_call_of_builtin_fails).
bb_lowered_out_of_scope(esc_neq,     lowered_call_of_builtin_fails).
% ignore/1 is such a builtin: every B2 query predicate the emitter lowers
% reaches it through a lowered `call ignore/1`.
bb_lowered_out_of_scope(Name, lowered_call_of_builtin_fails) :-
    bb_program(Name, 'B2', _, _, _).

%% bb_lowered_required(?Name)
%  Programs whose query predicate MUST be lowered in functions mode, so the
%  lowered check cannot silently become a skip.
bb_lowered_required(esc_head).
bb_lowered_required(esc_atoms_then).
bb_lowered_required(esc_quote_const).
bb_lowered_required(esc_pred_name).

% ---------------------------------------------------------------------
% Installation
% ---------------------------------------------------------------------

bb_driver_name(Name, D) :- atom_concat(bb_drv_, Name, D).

bb_clause_head(Clause, F/A) :-
    ( Clause = (H :- _) -> true ; H = Clause ),
    functor(H, F, A).

bb_program_pis(PIs) :-
    findall(PI, (bb_clause(_, C), bb_clause_head(C, PI)), PIs0),
    findall(D/0, (bb_program(N, _, _, _, _), bb_driver_name(N, D)), PIs1),
    append(PIs0, PIs1, PIs2),
    sort(PIs2, PIs).

install_bb_programs :-
    bb_program_pis(PIs),
    forall(member(F/A, PIs),
           ( functor(H, F, A), catch(retractall(user:H), _, true) )),
    forall(bb_dynamic(F/A),
           ( functor(H, F, A),
             catch(retractall(user:H), _, true),
             dynamic(user:F/A) )),
    forall(bb_clause(_, C), assertz(user:C)),
    forall(bb_program(N, _, Q, Out, Setup),
           ( bb_driver_name(N, D),
             assertz(user:(D :- Setup, ( Q, write(Out), nl, fail ; true ))) )).

bb_rust_preds(Preds) :-
    bb_program_pis(PIs),
    findall(user:PI, member(PI, PIs), Preds).

% ---------------------------------------------------------------------
% SWI oracle
% ---------------------------------------------------------------------

bb_reset_dynamic :-
    forall(bb_dynamic(F/A),
           ( functor(H, F, A), catch(retractall(user:H), _, true) )).

%% swi_all(+Name, -Out, -Ok)
swi_all(Name, Out, Ok) :-
    bb_reset_dynamic,
    bb_driver_name(Name, D),
    with_output_to(string(Raw),
                   ( catch(user:D, _, fail) -> Ok = true ; Ok = false )),
    normalize_output(Raw, Out).

%% swi_once(+Name, -Out)
swi_once(Name, Out) :-
    bb_reset_dynamic,
    bb_program(Name, _, Q, O, Setup),
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

%% bb_mode(?Mode, ?EmitOpts, ?CargoFeatureArgs, ?Dir)
%  functions_nodef shares the functions crate (same generated source) and
%  builds it into a second target dir without default features.
bb_mode(interpreter,     [emit_mode(interpreter)], '',
        'output/rust_wam_bugfix_batch_1_interpreter').
bb_mode(functions,       [emit_mode(functions)],   '',
        'output/rust_wam_bugfix_batch_1_functions').
bb_mode(functions_nodef, [emit_mode(functions)],
        '--no-default-features --features decorate_sort',
        'output/rust_wam_bugfix_batch_1_functions').

bb_target_dir(functions_nodef, 'target_nodef') :- !.
bb_target_dir(_, 'target').

:- dynamic bb_built/1.
:- dynamic bb_generated/1.

bb_generate(Dir, EmitOpts) :-
    (   bb_generated(Dir)
    ->  true
    ;   ( exists_directory(Dir) -> delete_directory_and_contents(Dir) ; true ),
        make_directory_path(Dir),
        install_bb_programs,
        bb_rust_preds(Preds),
        append([module_name('bbprobe'), wam_fallback(true)], EmitOpts, Opts),
        write_wam_rust_project(Preds, Opts, Dir),
        write_bb_driver(Dir),
        assertz(bb_generated(Dir))
    ).

%% bb_ensure_built(+Mode)
bb_ensure_built(Mode) :-
    (   bb_built(Mode)
    ->  true
    ;   bb_mode(Mode, EmitOpts, Feats, Dir),
        bb_generate(Dir, EmitOpts),
        bb_target_dir(Mode, TD),
        format(atom(Cmd),
               'cd ~w && CARGO_TARGET_DIR=~w cargo build ~w --bin bb_probe 2>&1',
               [Dir, TD, Feats]),
        process_create(path(sh), ['-c', Cmd],
                       [stdout(pipe(Out)), stderr(std), process(Pid)]),
        read_string(Out, _, OutStr), close(Out),
        process_wait(Pid, Status),
        (   Status == exit(0)
        ->  assertz(bb_built(Mode))
        ;   format(user_error, "~n[cargo build output, ~w]~n~w~n", [Mode, OutStr]),
            throw(bb_build_failed(Mode, Status))
        )
    ).

bb_bin(Mode, Bin) :-
    bb_mode(Mode, _, _, Dir),
    bb_target_dir(Mode, TD),
    format(atom(Bin), '~w/~w/debug/bb_probe', [Dir, TD]).

%% lowered_entry(+Dir, +Name, -Fn)
%  Fn is the lowered function's Rust name when the functions-mode emitter
%  lowered the query predicate of program Name.
lowered_entry(Dir, Name, Fn) :-
    bb_program(Name, _, Q, _, _),
    functor(Q, F, A),
    atomic_list_concat([Dir, '/src/lib.rs'], LibRs),
    exists_file(LibRs),
    read_file_to_string(LibRs, S, []),
    format(atom(Key), '~w_~w', [F, A]),
    format(atom(Fn), 'lowered_~w', [Key]),
    format(atom(Sym), 'pub fn ~w(', [Fn]),
    sub_string(S, _, _, _, Sym).

write_bb_driver(Dir) :-
    atomic_list_concat([Dir, '/src/bin/bb_probe'], BinDir),
    make_directory_path(BinDir),
    atomic_list_concat([BinDir, '/main.rs'], Path),
    bb_driver_source(Head, Tail),
    findall(Arm,
            ( bb_program(N, _, _, _, _),
              lowered_entry(Dir, N, Fn),
              format(atom(Arm),
                     '            "~w" => run_lowered(&mut vm, bbprobe::~w, &rest),~n',
                     [Fn, Fn]) ),
            Arms0),
    sort(Arms0, Arms),
    atomic_list_concat(Arms, ArmsText),
    setup_call_cleanup(open(Path, write, S),
                       format(S, '~w~w~w', [Head, ArmsText, Tail]),
                       close(S)).

bb_driver_source(
"// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 John William Creighton (s243a)
//
// bb_probe: run a 0-arity driver predicate by its WAM label (program output
// comes from the compiled write/1), or call a LOWERED function directly with
// the query arguments (`--lowered fn arg..`, `_` = the output variable).
#![allow(unused_imports, dead_code, unused_mut, unused_variables)]
use bbprobe::state::WamState;
use bbprobe::value::Value;
use bbprobe::{setup_foreign_predicates, shared_wam_program};

const OUT: &str = \"_bb_out\";

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
        None => { eprintln!(\"usage: bb_probe <pred/arity>\"); std::process::exit(2); }
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

%% bb_run(+Mode, +Args, -Out, -Err)
bb_run(Mode, Args, Out, Err) :-
    bb_run(Mode, Args, [], Out, Err).

%% bb_run(+Mode, +Args, +StdinText, -Out, -Err)
bb_run(Mode, Args, Stdin, Out, Err) :-
    bb_bin(Mode, Bin),
    process_create(Bin, Args,
                   [stdin(pipe(I)), stdout(pipe(O)), stderr(pipe(E)), process(Pid)]),
    ( Stdin == [] -> true ; format(I, '~w', [Stdin]) ),
    close(I),
    read_string(O, _, S1), read_string(E, _, S2),
    close(O), close(E),
    process_wait(Pid, _),
    normalize_output(S1, Out),
    Err = S2.

%% rust_all(+Mode, +Name, -Out, -Ok)
rust_all(Mode, Name, Out, Ok) :-
    bb_driver_name(Name, D),
    format(atom(Key), '~w/0', [D]),
    bb_run(Mode, [Key], Out, Err),
    (   sub_string(Err, _, _, _, "run=true") -> Ok = true
    ;   sub_string(Err, _, _, _, "run=false") -> Ok = false
    ;   Ok = crashed(Err)
    ).

%% rust_once(+Mode, +Name, -Out) is semidet.
%  Fails when the query predicate was not lowered.
rust_once(Mode, Name, Out) :-
    bb_mode(Mode, _, _, Dir),
    lowered_entry(Dir, Name, Fn),
    bb_program(Name, _, Q, O, _),
    Q =.. [_|QArgs],
    maplist(bb_arg_text(O), QArgs, ArgTexts),
    bb_run(Mode, ['--lowered', Fn|ArgTexts], Out, _).

bb_arg_text(O, A, '_') :- A == O, !.
bb_arg_text(_, A, T) :- format(atom(T), '~w', [A]).

% ---------------------------------------------------------------------
% Checks
% ---------------------------------------------------------------------

bb_all_mode(interpreter).
bb_all_mode(functions).

check_all(Mode, Name) :-
    bb_ensure_built(Mode),
    swi_all(Name, SOut, SOk),
    rust_all(Mode, Name, ROut, ROk),
    (   SOut == ROut, SOk == ROk
    ->  true
    ;   format(user_error,
               'BUGFIX-1 DIVERGENCE [~w] ~w~n  swi : ~q run=~w~n  rust: ~q run=~w~n',
               [Mode, Name, SOut, SOk, ROut, ROk]),
        fail
    ).

check_once(Mode, Name) :-
    bb_ensure_built(Mode),
    (   bb_lowered_out_of_scope(Name, Why)
    ->  format(user_error, '  [~w] ~w: lowered entry not checked (~w)~n',
               [Mode, Name, Why])
    ;   rust_once(Mode, Name, ROut)
    ->  swi_once(Name, SOut),
        (   SOut == ROut
        ->  true
        ;   format(user_error,
                   'BUGFIX-1 LOWERED DIVERGENCE [~w] ~w~n  swi once: ~q~n  rust    : ~q~n',
                   [Mode, Name, SOut, ROut]),
            fail
        )
    ;   format(user_error, '  [~w] ~w: query predicate not lowered (skipped)~n',
               [Mode, Name])
    ).

test_wam_rust_bugfix_batch_1 :-
    run_tests(wam_rust_bugfix_batch_1).

:- begin_tests(wam_rust_bugfix_batch_1, [condition(cargo_available),
                                        setup(install_bb_programs)]).

test(all_solutions_match_swi,
     [forall(( bb_all_mode(Mode), bb_program(Name, _, _, _, _) ))]) :-
    once(check_all(Mode, Name)).

test(lowered_first_solution_matches_swi,
     [forall(( member(Mode, [functions, functions_nodef]),
               bb_program(Name, _, _, _, _) ))]) :-
    once(check_once(Mode, Name)).

test(lowered_entries_present, [forall(bb_lowered_required(Name))]) :-
    bb_ensure_built(functions),
    bb_mode(functions, _, _, Dir),
    once(lowered_entry(Dir, Name, _)).

:- end_tests(wam_rust_bugfix_batch_1).
